# 修改时间：2026-09-19。
# 修改目的：生成四档 YOLO 的无歧义串行计划并分别指定两代 TensorRT 权重。
# 修改内容：规范旧别名并透传 V2 独立配置与 engine，保持默认仅计划。
# 修改时间：2026-09-19。
# 修改目的：让方向敏感性候选方案可在后续多天气批次中独立验证。
# 修改内容：透传默认关闭的输入旋转参数并在批次计划中记录。
# 修改时间：2026-09-19。
# 修改目的：让后续批量验证按需保留同帧离线重放所需的图像。
# 修改内容：批次命令透传已处理帧保存开关并记录在计划中。
# 修改时间：2026-09-18。
# 修改目的：让三档在线 YOLO 方案按相同天气、种子和时长独立计划并保留完整身份链路。
# 修改内容：新增 profile 与 TensorRT engine 传播，并校验每轮 metadata、summary 和结果行的方案一致性。
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
from .yolo_profiles import (
    DEFAULT_CONFIG, DEFAULT_TRT_ENGINE, DEFAULT_V2_CONFIG, DEFAULT_V2_ENGINE,
    YOLO_PROFILES, canonical_profile,
)


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
    if args.trt_engine is not None:
        args.trt_engine = args.trt_engine.resolve()
    if args.config is not None:
        args.config = args.config.resolve()
    args.v2_config = args.v2_config.resolve()
    args.v2_engine = args.v2_engine.resolve()
    allowed_output_root = OUTPUT_ROOT.resolve()
    if not args.layout.is_file():
        parser.error(f"--layout 不存在：{args.layout}")
    if args.config is not None and not args.config.is_file():
        parser.error(f"--config 不存在：{args.config}")
    if args.trt_engine is not None and not args.trt_engine.is_file():
        parser.error(f"--trt-engine 不存在：{args.trt_engine}")
    if args.execute and "V2" in args.yolo_profiles:
        if not args.v2_config.is_file() or not args.v2_engine.is_file():
            parser.error("执行 V2 时 --v2-config 和 --v2-engine 必须指向实际文件")
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
    if len(set(args.yolo_profiles)) != len(args.yolo_profiles):
        parser.error("--yolo-profiles 不能重复")


def read_profile_trace(metadata, summary):
    """提取核心 runner 冻结的实际方案；不在批次侧复制 profile 定义。"""
    resources = metadata.get("resources") or summary.get("resources") or {}
    return {
        "yolo_profile": metadata.get("yolo_profile", summary.get("yolo_profile")),
        "profile": resources.get("profile"),
        "config_path": resources.get("config_path"),
        "config_sha256": resources.get("config_sha256"),
        "weights_path": resources.get("weights_path"),
        "weights_sha256": resources.get("weights_sha256"),
        "model_format": resources.get("model_format"),
        "effective_options": resources.get("effective_options"),
        "tracker_high_multiplier": resources.get("tracker_high_multiplier"),
    }


def validate_run_output(run_output, expected_profile):
    results_path = run_output / RESULTS_NAME
    submissions_path = run_output / "yolo_sidecar_submissions.jsonl"
    metadata_path = run_output / "metadata.json"
    summary_path = run_output / "summary.json"
    sidecar_summary_path = run_output / "yolo_sidecar_summary.json"
    required = (
        results_path,
        submissions_path,
        metadata_path,
        summary_path,
        sidecar_summary_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"ok": False, "reasons": ["missing_required_artifact"], "missing": missing}

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        sidecar = json.loads(sidecar_summary_path.read_text(encoding="utf-8-sig"))
        result_profiles = []
        with results_path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    result_profiles.append(json.loads(line).get("yolo_profile"))
        results_rows = len(result_profiles)
        submission_profiles = []
        with submissions_path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    submission_profiles.append(json.loads(line).get("yolo_profile"))
        submission_rows = len(submission_profiles)
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
    profile_trace = read_profile_trace(metadata, summary)
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
    if metadata.get("yolo_profile") != expected_profile:
        reasons.append("metadata_profile_mismatch")
    if summary.get("yolo_profile") != expected_profile:
        reasons.append("summary_profile_mismatch")
    if profile_trace["profile"] != expected_profile:
        reasons.append("resource_profile_mismatch")
    if any(profile != expected_profile for profile in result_profiles):
        reasons.append("result_profile_mismatch")
    if any(profile != expected_profile for profile in submission_profiles):
        reasons.append("submission_profile_mismatch")
    if not profile_trace["effective_options"]:
        reasons.append("missing_effective_options")
    if not profile_trace["model_format"]:
        reasons.append("missing_model_format")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "summary_status": summary.get("status"),
        "sidecar_ok": summary.get("sidecar_ok"),
        "profile_trace": profile_trace,
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
        "--yolo-profile",
        item["yolo_profile"],
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
        "--image-rotation-deg",
        str(args.image_rotation_deg),
    ]
    if args.config is not None and item["yolo_profile"] != "V2":
        command.extend(["--config", str(args.config)])
    if args.trt_engine is not None and item["yolo_profile"] == "V1-v3":
        command.extend(["--trt-engine", str(args.trt_engine)])
    if item["yolo_profile"] == "V2":
        command.extend(["--v2-config", str(args.v2_config), "--v2-engine", str(args.v2_engine)])
    if args.save_processed_frames:
        command.append("--save-processed-frames")
    return command


def build_plan(args):
    runs = []
    for profile in args.yolo_profiles:
        for weather in args.weathers:
            for seed in args.seeds:
                run_id = f"{profile}-{weather.lower()}-seed-{seed}"
                item = {
                    "run_id": run_id,
                    "yolo_profile": profile,
                    "weather": weather,
                    "seed": seed,
                    "duration_s": args.duration,
                    "fov_deg": args.fov,
                    "config": str(args.v2_config if profile == "V2" else args.config or DEFAULT_CONFIG),
                    "engine": (str(args.v2_engine) if profile == "V2" else
                               str(args.trt_engine or DEFAULT_TRT_ENGINE) if profile == "V1-v3" else None),
                    "output": str(args.output / "runs" / run_id),
                }
                item["command"] = child_command(item, args)
                runs.append(item)
    return {
        "schema_version": 2,
        "created_at": utc_now(),
        "runner_module": RUNNER_MODULE,
        "results_name": RESULTS_NAME,
        "layout": str(args.layout),
        "runtime_root": str(args.runtime_root),
        "output": str(args.output),
        "duration_s": args.duration,
        "fov_deg": args.fov,
        "config": None if args.config is None else str(args.config),
        "trt_engine": None if args.trt_engine is None else str(args.trt_engine),
        "v2_config": str(args.v2_config),
        "v2_engine": str(args.v2_engine),
        "device": args.device,
        "save_processed_frames": args.save_processed_frames,
        "image_rotation_deg": args.image_rotation_deg,
        "yolo_profiles": list(args.yolo_profiles),
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
        "schema_version": 2,
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
            "yolo_profile": item["yolo_profile"],
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
            f"[{index}/{plan['run_count']}] {item['yolo_profile']} "
            f"{item['weather']} seed={item['seed']}",
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
        validation = validate_run_output(run_output, item["yolo_profile"])
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
    result.add_argument(
        "--yolo-profiles",
        nargs="+",
        type=canonical_profile,
        choices=YOLO_PROFILES,
        default=YOLO_PROFILES,
        help="默认 V1-v1 V1-v2 V1-v3 V2；小写 v1/v2/v3 仅为旧模型三档别名",
    )
    result.add_argument("--config", type=Path, help="传给核心 runner 的检测器配置")
    result.add_argument(
        "--trt-engine",
        type=Path,
        help="传给核心 runner 的 V1-v3 TensorRT engine 覆盖路径",
    )
    result.add_argument("--v2-config", type=Path, default=DEFAULT_V2_CONFIG, help="V2 独立配置")
    result.add_argument("--v2-engine", type=Path, default=DEFAULT_V2_ENGINE, help="V2 本机 raw two-class FP16 engine")
    result.add_argument("--device", default="0", help="传给核心 runner 的推理设备")
    result.add_argument("--image-rotation-deg", type=int, choices=(0, 90, 180, 270), default=0,
                        help="传给核心 runner 的单视图输入旋转，默认 0")
    result.add_argument("--save-processed-frames", action="store_true",
                        help="保存实际推理帧，供逐帧错分检查和离线重放")
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
