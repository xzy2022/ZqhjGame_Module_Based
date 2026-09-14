# 修改时间：2026-09-14
# 修改目的：让个人实验脱离官方仓库的后续修改并支持独立运行。
# 修改内容：统一模块、SDK、运行资源及输出路径并保留实验行为。
# 修改时间：2026-09-13（收尾核查）。
# 修改目的：确保完成状态具有真实采集时长，并避免退出后的空间检查误判。
# 修改内容：校验单轮摘要必要字段，并在子进程退出后直接进入收尾。
# 修改时间：2026-09-13。
# 修改目的：让多天气采集按双盘预算串行运行，并保留短测和失败轮的真实占用。
# 修改内容：新增可续跑的轮次清单、整轮容量预留、资源检查和停止文件收尾。
"""双盘串行采集入口；短测、失败轮和生产轮共享同一批次预算。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path

from .paths import PROJECT_ROOT, SIM_ROOT, RUNTIME_ROOT, OUTPUT_ROOT, SCENARIO_ROOT
import shutil
import subprocess
import sys
import time


ROOT = PROJECT_ROOT
GIB = 1024 ** 3
WEATHERS = ("Clear_Skies", "Partly_Cloudy", "Rain", "Foggy", "Snow_Light", "Sand_Dust_Calm")
CHILD_MODULE = "personal_hf2026.dataset_capture"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def total_bytes(root):
    # 不跟随目录链接，避免把预算根之外的文件算入本批次。
    size = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                size += path.stat().st_size
            except FileNotFoundError:
                pass
    return size


def free_bytes(path):
    existing = path
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free


def load_plan(args):
    plan = read_json(args.plan_json) if args.plan_json else [
        {"run_id": f"run-{index:02d}", "weather": weather, "seed": seed,
         "duration": 200, "stage": "runs"}
        for index, (seed, weather) in enumerate(
            ((seed, weather) for seed in (1, 2, 3) for weather in WEATHERS), 1)
    ]
    identifiers = set()
    for item in plan:
        run_id = str(item["run_id"])
        if not run_id or run_id in (".", "..") or any(c in run_id for c in '/\\:'):
            raise ValueError(f"无效 run_id：{run_id}")
        if run_id in identifiers:
            raise ValueError(f"重复 run_id：{run_id}")
        identifiers.add(run_id)
        if item["weather"] not in WEATHERS or item["duration"] not in (20, 200):
            raise ValueError(f"{run_id} 只支持六种天气和 20/200 秒")
        if item["stage"] not in ("preflight", "runs"):
            raise ValueError(f"{run_id} stage 必须为 preflight 或 runs")
        if item.get("layout"):
            item["layout"] = str(Path(item["layout"]).resolve())
        item["fov"] = args.fov
    return plan


def reserve_200_bytes(manifest, initial):
    reserve = int(initial * GIB)
    weather_estimates = {}
    for record in manifest["runs"]:
        size = record.get("total_bytes", 0)
        actual = record.get("actual_duration_s", 0)
        if not size or not actual:
            continue
        # 按全部产物和实际采集时长外推，失败短轮也能上调下一轮预留。
        projected = math.ceil(size * 200 / actual * 1.3)
        if record.get("requested_duration_s") == 200 and record["status"] == "completed":
            projected = max(projected, math.ceil(size * 1.3))
        weather = record["weather"]
        weather_estimates[weather] = max(projected, weather_estimates.get(weather, 0))
        reserve = max(reserve, projected)
    return reserve, weather_estimates


def disk_states(roots, args):
    return [{"disk": disk, "root": str(root), "used_bytes": total_bytes(root),
             "free_bytes": free_bytes(root), "budget_bytes": int(budget * GIB)}
            for disk, root, budget in zip(("D", "E"), roots,
                                         (args.d_budget_gib, args.e_budget_gib))]


def choose_disk(states, reserve, minimum):
    for state in states:
        if (state["budget_bytes"] - state["used_bytes"] >= reserve
                and state["free_bytes"] >= minimum + reserve):
            return state
    return None


def busy_simulations(runtime_root):
    # 只检查，不结束既有 UE 或引擎；Redis 和桥接服务可继续共用。
    names = {"opensim-sim.exe", "testwl.exe", "testwl-win64-shipping.exe", "unrealeditor.exe"}
    config = runtime_root / "config" / "renderers" / "ue_testwl.json"
    if config.is_file():
        names.add(Path(read_json(config)["executable"]["launcher"]).name.lower())
    result = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True,
                            text=True, errors="replace", check=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return [{"name": row[0], "pid": row[1]} for row in csv.reader(io.StringIO(result.stdout))
            if len(row) > 1 and row[0].lower() in names]


def child_command(item, output, args):
    # 子进程重新经过独立仓库入口，明确继承 SDK 和二进制路径。
    command = [sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "tools/run_official.py"),
               "dataset_capture",
               "--runtime-root", str(args.runtime_root), "--weather", item["weather"],
               "--seed", str(item["seed"]), "--duration", str(item["duration"]),
               "--fov", str(item["fov"]), "--output", str(output),
               "--stop-file", str(output / "stop-request.json"),
               "--min-free-gib", str(args.min_free_gib)]
    if item.get("layout"):
        command.extend(["--layout", item["layout"]])
    return command


def save_manifest(manifest, roots, args):
    manifest["updated_at"] = now()
    manifest["disks"] = disk_states(roots, args)
    reserve, estimates = reserve_200_bytes(manifest, args.initial_reserve_gib)
    manifest["reserve_200_bytes"] = reserve
    manifest["weather_estimated_200_bytes"] = estimates
    for root in roots:
        write_json(root / "manifest.json", manifest)


def existing_manifest(roots, args):
    copies = [read_json(root / "manifest.json") for root in roots if (root / "manifest.json").is_file()]
    if copies and not args.resume:
        raise ValueError("已有批次清单；续跑请指定 --resume，避免覆盖")
    if copies:
        manifest = max(copies, key=lambda value: value["updated_at"])
        if manifest["roots"] != [str(root) for root in roots]:
            raise ValueError("续跑必须使用原批次的两个根目录")
        if len({value["batch_id"] for value in copies}) != 1:
            raise ValueError("两盘清单属于不同批次")
        return manifest
    return {"batch_id": datetime.now().strftime("%Y%m%d-%H%M%S-%f"), "created_at": now(),
            "roots": [str(root) for root in roots], "code_root": str(ROOT),
            "control_mode": "oracle_identity", "perception": "ideal_positions",
            "visibility": "unverified", "status": "ready", "runs": [], "invocations": []}


def finish_record(record):
    output = Path(record["output"])
    summary_path = output / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        for key in ("status", "stop_reason", "actual_duration_s", "image_bytes",
                    "control_mode", "perception"):
            if key in summary:
                record[key] = summary[key]
        record["summary_path"] = str(summary_path)
        actual = summary.get("actual_duration_s")
        if ("status" not in summary or not isinstance(actual, (float, int))
                or not math.isfinite(actual) or actual <= 0):
            record.update(status="failed", stop_reason="invalid_summary_required_fields")
    else:
        record.update(status="failed", stop_reason="missing_summary")
    if record.get("returncode", 0) != 0 and record["status"] == "completed":
        record.update(status="failed", stop_reason="child_nonzero_exit")
    if record.get("batch_stop_reason"):
        record.update(status="aborted", stop_reason=record["batch_stop_reason"])
    if record["status"] not in ("completed", "aborted", "failed"):
        record.update(status="failed", stop_reason="invalid_summary_status")
    record["total_bytes"] = total_bytes(output)
    record["finished_at"] = now()


def run_one(item, state, manifest, roots, args):
    output = Path(state["root"]) / item["stage"] / item["run_id"]
    # 目录创建为排他操作，任何已有原始记录均不覆盖。
    output.mkdir(parents=True, exist_ok=False)
    command = child_command(item, output, args)
    record = dict(item, output=str(output), disk=state["disk"], status="running",
                  requested_duration_s=item["duration"], started_at=now(), command=command)
    manifest["runs"].append(record)
    save_manifest(manifest, roots, args)
    stop_file = output / "stop-request.json"
    with (output / "batch-child.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        record["pid"] = process.pid
        save_manifest(manifest, roots, args)
        next_budget_check = time.monotonic()
        while process.poll() is None:
            try:
                time.sleep(5)
                if process.poll() is not None:
                    break
                reason = None
                if free_bytes(output) <= int((args.min_free_gib + 0.25) * GIB):
                    reason = "disk_free_below_headroom"
                if time.monotonic() >= next_budget_check:
                    if total_bytes(Path(state["root"])) >= state["budget_bytes"]:
                        reason = reason or "disk_batch_budget_reached"
                    next_budget_check = time.monotonic() + 15
                if reason and not record.get("batch_stop_reason"):
                    record["batch_stop_reason"] = reason
                    write_json(stop_file, {"reason": reason, "requested_at": now()})
                    save_manifest(manifest, roots, args)
                    print(f"请求正常收尾：{item['run_id']}，{reason}", flush=True)
            except KeyboardInterrupt:
                # 中断只向本轮发送停止文件，等待其关闭相机、索引和受管进程。
                record["batch_stop_reason"] = "operator_interrupt"
                write_json(stop_file, {"reason": "operator_interrupt", "requested_at": now()})
        record["returncode"] = process.returncode
    finish_record(record)
    save_manifest(manifest, roots, args)
    return record


def execute(manifest, plan, roots, args):
    started = datetime.fromisoformat(manifest["created_at"]).timestamp()
    invocation = {"started_at": now(), "plan": plan, "status": "running",
                  "d_budget_gib": args.d_budget_gib, "e_budget_gib": args.e_budget_gib,
                  "min_free_gib": args.min_free_gib, "child_launch_count": 0}
    manifest["invocations"].append(invocation)
    reason = "plan_completed"
    for item in plan:
        old = next((record for record in manifest["runs"] if record["run_id"] == item["run_id"]), None)
        if old:
            for key in ("weather", "seed", "duration", "stage", "fov", "layout"):
                if old.get(key) != item.get(key):
                    raise ValueError(f"{item['run_id']} 的已有参数与本次计划不同：{key}")
            if old["status"] == "running":
                finish_record(old)
            if old["status"] == "completed":
                print(f"跳过已完成轮：{item['run_id']}", flush=True)
                continue
            reason = "existing_noncompleted_run_no_retry"
            break
        if time.time() >= started + args.deadline_hours * 3600:
            reason = "batch_deadline_reached"
            break
        reserve_200, _ = reserve_200_bytes(manifest, args.initial_reserve_gib)
        reserve = math.ceil(reserve_200 * item["duration"] / 200)
        states = disk_states(roots, args)
        state = choose_disk(states, reserve, int(args.min_free_gib * GIB))
        invocation["last_capacity_check"] = {"run_id": item["run_id"],
                                               "required_bytes": reserve, "disks": states}
        # 容量拒绝必须先于资源检查和仿真子进程创建，离线即可真实验证。
        if state is None:
            reason = "no_disk_can_fit_next_run"
            break
        busy = busy_simulations(args.runtime_root)
        if busy:
            invocation["busy_processes"] = busy
            reason = "unrelated_simulation_running"
            break
        print(f"开始 {item['run_id']} {item['weather']}：{state['disk']} 盘，"
              f"预留 {reserve / GIB:.3f} GiB", flush=True)
        invocation["child_launch_count"] += 1
        record = run_one(item, state, manifest, roots, args)
        if record["status"] != "completed":
            reason = record.get("stop_reason", "run_failed")
            break
    invocation.update(status="completed" if reason == "plan_completed" else "stopped",
                      stop_reason=reason, finished_at=now())
    manifest.update(status=invocation["status"], stop_reason=reason)
    save_manifest(manifest, roots, args)
    print(json.dumps({"status": manifest["status"], "stop_reason": reason,
                      "manifest": [str(root / "manifest.json") for root in roots],
                      "child_launch_count": invocation["child_launch_count"]}, ensure_ascii=False), flush=True)
    return 0 if reason in ("plan_completed", "no_disk_can_fit_next_run", "batch_deadline_reached") else 1


def main():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--d-root", type=Path, default=OUTPUT_ROOT / "personal_v2" / f"night-capture-{stamp}")
    parser.add_argument("--e-root", type=Path, default=Path("E:/datasets/_for/_codex") / f"night-capture-{stamp}")
    parser.add_argument("--d-budget-gib", type=float, default=60)
    parser.add_argument("--e-budget-gib", type=float, default=60)
    parser.add_argument("--min-free-gib", type=float, default=20)
    parser.add_argument("--initial-reserve-gib", type=float, default=8,
                        help="200 秒整轮初始预留；20 秒轮按比例预留")
    parser.add_argument("--deadline-hours", type=float, default=8)
    parser.add_argument("--fov", type=float, default=48)
    parser.add_argument("--plan-json", type=Path, help="轮次 JSON 数组；省略时为 3 seed × 6 天气 × 200 秒")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="仅打印清单和当前容量，不创建输出或启动仿真")
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.resolve()
    roots = [args.d_root.resolve(), args.e_root.resolve()]
    if roots[0].drive.upper() != "D:" or roots[1].drive.upper() != "E:":
        parser.error("--d-root 必须在 D 盘，--e-root 必须在 E 盘")
    if not roots[1].is_relative_to(Path("E:/datasets/_for/_codex").resolve()) or roots[1] == Path("E:/datasets/_for/_codex").resolve():
        parser.error("--e-root 必须是 E:/datasets/_for/_codex 下的独立批次目录")
    plan = load_plan(args)
    if args.plan_only:
        manifest = existing_manifest(roots, args)
        reserve, estimates = reserve_200_bytes(manifest, args.initial_reserve_gib)
        print(json.dumps({"code_root": str(ROOT), "runtime_root": str(args.runtime_root),
                          "plan": plan, "disks": disk_states(roots, args),
                          "reserve_200_bytes": reserve, "weather_estimated_200_bytes": estimates,
                          "control_mode": "oracle_identity", "perception": "ideal_positions",
                          "visibility": "unverified", "starts_simulation": False}, ensure_ascii=False, indent=2))
        return 0
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    # 操作系统文件锁防止两个批处理同时占用同一批次；进程退出后自动释放。
    import msvcrt
    locks = []
    try:
        for root in roots:
            lock = (root / ".batch.lock").open("a+b")
            lock.seek(0)
            if not lock.read(1):
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            locks.append(lock)
        manifest = existing_manifest(roots, args)
        return execute(manifest, plan, roots, args)
    finally:
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
