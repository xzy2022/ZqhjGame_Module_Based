# 修改时间：2026-09-18。
# 修改目的：避免空结果或部分收尾的单轮被批次误判为成功。
# 修改内容：收紧 FOV 和种子范围，并核验两份摘要、worker 状态、三机完成覆盖及 JSONL 计数闭合。
# 修改时间：2026-09-18。
# 修改目的：让批量计划显式保留核心 runner 已确认的模型配置与设备参数。
# 修改内容：新增可选 --config 和默认设备 0，并把实际值写入计划与子命令。
# 修改时间：2026-09-18。
# 修改目的：避免父批次预创建单轮目录与 runner 的排他输出目录契约冲突。
# 修改内容：把子进程日志移到批次 logs 目录并让 runner 自行创建每轮输出。
# 修改时间：2026-09-18。
# 修改目的：为 YOLO 旁路提供六天气双随机种子的可审计串行批量入口。
# 修改内容：默认仅打印十二轮计划，显式指定 --execute 后才逐轮启动仿真并保存批次清单。
"""生成或执行 YOLO 旁路的六天气串行实验计划。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .paths import OUTPUT_ROOT, PROJECT_ROOT, RUNTIME_ROOT, SCENARIO_ROOT, SIM_ROOT


WEATHERS = (
    "Clear_Skies",
    "Partly_Cloudy",
    "Rain",
    "Foggy",
    "Snow_Light",
    "Sand_Dust_Calm",
)
DEFAULT_LAYOUT = SCENARIO_ROOT / "static-decoys.json"
RUNNER_MODULE = "yolo_sidecar_study"
RESULTS_NAME = "yolo_sidecar_results.jsonl"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_args(args, parser):
    args.layout = args.layout.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.output = args.output.resolve()
    if args.config is not None:
        args.config = args.config.resolve()
    allowed_output_root = OUTPUT_ROOT.resolve()
    if not args.layout.is_file():
        parser.error(f"--layout 不存在：{args.layout}")
    if args.config is not None and not args.config.is_file():
        parser.error(f"--config 不存在：{args.config}")
    if not is_relative_to(args.output, allowed_output_root):
        parser.error(f"--output 必须位于 {allowed_output_root} 下")
    if args.duration <= 0:
        parser.error("--duration 必须大于 0")
    if not 5 <= args.fov <= 50:
        parser.error("--fov 必须位于比赛允许的 [5, 50] 度范围")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不能重复")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds 不能为负数")


def count_jsonl_records(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return sum(bool(line.strip()) for line in stream)


def validate_run_output(run_output):
    results_path = run_output / RESULTS_NAME
    submissions_path = run_output / "yolo_sidecar_submissions.jsonl"
    summary_path = run_output / "summary.json"
    sidecar_summary_path = run_output / "yolo_sidecar_summary.json"
    required = (results_path, submissions_path, summary_path, sidecar_summary_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"ok": False, "reasons": ["missing_required_artifact"], "missing": missing}

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        sidecar = json.loads(sidecar_summary_path.read_text(encoding="utf-8-sig"))
        results_rows = count_jsonl_records(results_path)
        submission_rows = count_jsonl_records(submissions_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {"ok": False, "reasons": [f"artifact_read_failed:{type(error).__name__}:{error}"]}

    reasons = []
    worker = sidecar.get("worker", {})
    counts = sidecar.get("counts", {})
    expected_uids = [str(uid) for uid in counts.get("expected_uids", [])]
    completed_by_uid = {
        str(uid): value for uid, value in counts.get("completed_by_uid", {}).items()
    }
    completed = counts.get("completed")
    submitted = counts.get("submitted")
    final_status = counts.get("by_final_status", {})
    if summary.get("status") != "completed":
        reasons.append("summary_not_completed")
    if not summary.get("sidecar_ok"):
        reasons.append("summary_sidecar_not_ok")
    if not worker.get("ready"):
        reasons.append("worker_not_ready")
    if worker.get("errors"):
        reasons.append("worker_has_errors")
    if worker.get("exitcode") != 0:
        reasons.append("worker_nonzero_exit")
    if not isinstance(completed, int) or completed <= 0:
        reasons.append("no_completed_results")
    if len(expected_uids) != 3:
        reasons.append("expected_uids_not_three")
    if any(completed_by_uid.get(uid, 0) <= 0 for uid in expected_uids):
        reasons.append("not_all_uids_completed")
    if isinstance(completed, int) and sum(completed_by_uid.values()) != completed:
        reasons.append("completed_by_uid_mismatch")
    if isinstance(completed, int) and results_rows != completed:
        reasons.append("results_row_count_mismatch")
    if isinstance(submitted, int) and submission_rows != submitted:
        reasons.append("submissions_row_count_mismatch")
    if isinstance(submitted, int) and sum(final_status.values()) != submitted:
        reasons.append("final_status_count_mismatch")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "summary_status": summary.get("status"),
        "sidecar_ok": summary.get("sidecar_ok"),
        "worker": {
            "ready": worker.get("ready"),
            "errors": worker.get("errors"),
            "exitcode": worker.get("exitcode"),
        },
        "counts": {
            "expected_uids": expected_uids,
            "submitted": submitted,
            "submission_rows": submission_rows,
            "completed": completed,
            "result_rows": results_rows,
            "completed_by_uid": completed_by_uid,
            "by_final_status": final_status,
        },
    }


def child_command(item, args):
    command = [
        sys.executable,
        "-B",
        "-X",
        "utf8",
        str(PROJECT_ROOT / "tools/run_official.py"),
        "--sim-root",
        str(SIM_ROOT),
        "--runtime-root",
        str(args.runtime_root),
        RUNNER_MODULE,
        "--runtime-root",
        str(args.runtime_root),
        "--layout",
        str(args.layout),
        "--weather",
        item["weather"],
        "--duration",
        str(args.duration),
        "--seed",
        str(item["seed"]),
        "--fov",
        str(args.fov),
        "--output",
        item["output"],
        "--device",
        args.device,
    ]
    if args.config is not None:
        command.extend(["--config", str(args.config)])
    return command


def build_plan(args):
    runs = []
    for weather in args.weathers:
        for seed in args.seeds:
            run_id = f"{weather.lower()}-seed-{seed}"
            item = {
                "run_id": run_id,
                "weather": weather,
                "seed": seed,
                "duration_s": args.duration,
                "fov_deg": args.fov,
                "output": str(args.output / "runs" / run_id),
            }
            item["command"] = child_command(item, args)
            runs.append(item)
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "runner_module": RUNNER_MODULE,
        "results_name": RESULTS_NAME,
        "layout": str(args.layout),
        "runtime_root": str(args.runtime_root),
        "output": str(args.output),
        "duration_s": args.duration,
        "fov_deg": args.fov,
        "config": None if args.config is None else str(args.config),
        "device": args.device,
        "weathers": list(args.weathers),
        "seeds": list(args.seeds),
        "run_count": len(runs),
        "runs": runs,
    }


def execute(plan, args):
    args.output.mkdir(parents=True, exist_ok=False)
    logs = args.output / "logs"
    logs.mkdir()
    write_json(args.output / "batch_plan.json", plan)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "plan": str(args.output / "batch_plan.json"),
        "runs": [],
    }
    write_json(args.output / "batch_manifest.json", manifest)

    for index, item in enumerate(plan["runs"], start=1):
        run_output = Path(item["output"])
        record = {
            "run_id": item["run_id"],
            "weather": item["weather"],
            "seed": item["seed"],
            "output": item["output"],
            "command": item["command"],
            "status": "running",
            "started_at": utc_now(),
        }
        manifest["runs"].append(record)
        write_json(args.output / "batch_manifest.json", manifest)
        print(
            f"[{index}/{plan['run_count']}] {item['weather']} seed={item['seed']}",
            flush=True,
        )
        # 单轮输出目录由 runner 排他创建；父批次日志独立保存，避免预创建导致 runner 拒绝启动。
        with (logs / f"{item['run_id']}.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                item["command"],
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        results_path = run_output / RESULTS_NAME
        validation = validate_run_output(run_output)
        record.update(
            returncode=completed.returncode,
            results=str(results_path),
            results_present=results_path.is_file(),
            validation=validation,
            status=(
                "completed"
                if completed.returncode == 0 and validation["ok"]
                else "failed"
            ),
            finished_at=utc_now(),
        )
        write_json(args.output / "batch_manifest.json", manifest)
        if record["status"] != "completed":
            manifest.update(
                status="failed",
                stop_reason=f"run_failed:{item['run_id']}",
                finished_at=utc_now(),
            )
            write_json(args.output / "batch_manifest.json", manifest)
            return 1

    manifest.update(status="completed", finished_at=utc_now())
    write_json(args.output / "batch_manifest.json", manifest)
    return 0


def parser():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    result.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    result.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "personal_v1" / "yolo-sidecar" / f"batch-{stamp}",
    )
    result.add_argument("--weathers", nargs="+", choices=WEATHERS, default=WEATHERS)
    result.add_argument("--seeds", nargs="+", type=int, default=(1, 2))
    result.add_argument("--duration", type=float, default=120.0)
    result.add_argument("--fov", type=float, default=48.0)
    result.add_argument("--config", type=Path, help="传给核心 runner 的检测器配置")
    result.add_argument("--device", default="0", help="传给核心 runner 的推理设备")
    result.add_argument(
        "--execute",
        action="store_true",
        help="显式执行计划；省略时只向标准输出打印计划，不创建目录或启动仿真",
    )
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    validate_args(args, argument_parser)
    plan = build_plan(args)
    if not args.execute:
        print(
            json.dumps(
                {**plan, "mode": "plan_only", "starts_simulation": False},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    return execute(plan, args)


if __name__ == "__main__":
    raise SystemExit(main())
