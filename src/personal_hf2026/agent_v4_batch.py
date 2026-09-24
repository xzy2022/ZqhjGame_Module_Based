# 修改时间：2026-09-24。
# 修改目的：提供 Agent4 多天气多随机种子的严格串行批量入口。
# 修改内容：沿用现有批处理外围，并将子运行接至 agent_v4_study。
"""PersonalV4 多天气、多 seed 串行测评计划。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from .agent_v4_study import DEFAULT_LAYOUT, WEATHERS, _vision_mode
from .paths import OUTPUT_ROOT, PROJECT_ROOT, RUNTIME_ROOT, SIM_ROOT


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _child_command(item, args):
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
        "agent_v4_study",
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
        "--device",
        str(args.device),
        "--output",
        item["output"],
        "--vision-diagnostic",
        args.vision_diagnostic,
    ]
    if args.detector_config is not None:
        command.extend(["--detector-config", str(args.detector_config)])
    if args.weights is not None:
        command.extend(["--weights", str(args.weights)])
    if args.save_images:
        command.append("--save-images")
    if args.detailed_log:
        command.append("--detailed-log")
    if args.visualize:
        command.append("--visualize")
    return command


def _build_plan(args):
    runs = []
    for weather in args.weathers:
        for seed in args.seeds:
            run_id = f"{weather.lower()}-seed-{seed}"
            item = {
                "run_id": run_id,
                "weather": weather,
                "seed": seed,
                "duration_s": args.duration,
                "output": str(args.output / "runs" / run_id),
            }
            item["command"] = _child_command(item, args)
            runs.append(item)
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "execution": "strictly_serial_one_ue_process_at_a_time",
        "duration_s": args.duration,
        "weathers": list(args.weathers),
        "seeds": list(args.seeds),
        "run_count": len(runs),
        "runs": runs,
    }


def _validate_output(path):
    required = ("run.json", "metadata.json", "prepared_scenario.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return {"ok": False, "missing": missing}
    try:
        run = json.loads((path / "run.json").read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": repr(exc)}
    evaluation = run.get("evaluation") or {}
    return {
        "ok": run.get("status") == "completed" and not run.get("error"),
        "status": run.get("status"),
        "score": evaluation.get("total_score"),
        "n_destroyed": evaluation.get("n_destroyed"),
        "n_reports": evaluation.get("n_reports"),
    }


def _execute(plan, args):
    args.output.mkdir(parents=True, exist_ok=False)
    logs = args.output / "logs"
    logs.mkdir()
    _write_json(args.output / "batch_plan.json", plan)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "runs": [],
    }
    _write_json(args.output / "batch_manifest.json", manifest)
    failures = 0
    for index, item in enumerate(plan["runs"], start=1):
        record = {
            "run_id": item["run_id"],
            "weather": item["weather"],
            "seed": item["seed"],
            "output": item["output"],
            "command": item["command"],
            "status": "running",
            "started_at": _utc_now(),
        }
        manifest["runs"].append(record)
        _write_json(args.output / "batch_manifest.json", manifest)
        print(
            f"[{index}/{plan['run_count']}] {item['weather']} seed={item['seed']}",
            flush=True,
        )
        with (logs / f"{item['run_id']}.log").open(
            "x", encoding="utf-8"
        ) as stream:
            completed = subprocess.run(
                item["command"],
                cwd=PROJECT_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        validation = _validate_output(Path(item["output"]))
        ok = completed.returncode == 0 and validation["ok"]
        failures += int(not ok)
        record.update(
            returncode=completed.returncode,
            validation=validation,
            status="completed" if ok else "failed",
            finished_at=_utc_now(),
        )
        _write_json(args.output / "batch_manifest.json", manifest)
    manifest.update(
        status="completed" if not failures else "completed_with_failures",
        failures=failures,
        finished_at=_utc_now(),
    )
    _write_json(args.output / "batch_manifest.json", manifest)
    return 1 if failures else 0


def _parser():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "agent-v4" / f"batch-{stamp}",
    )
    parser.add_argument("--weathers", nargs="+", choices=WEATHERS, default=WEATHERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1,))
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--detector-config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--detailed-log", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--vision-diagnostic", type=_vision_mode, default="000",
        help="开发诊断三位开关，默认 000；透传给每个 agent_v4_study 子运行",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    args.runtime_root = args.runtime_root.resolve()
    args.layout = args.layout.resolve()
    args.output = args.output.resolve()
    if not args.layout.is_file():
        parser.error(f"--layout 不存在：{args.layout}")
    if args.output.exists():
        parser.error(f"输出目录已存在，拒绝覆盖：{args.output}")
    if args.duration <= 0:
        parser.error("--duration 必须大于 0")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds 必须非负且不能重复")
    if args.detector_config is not None:
        args.detector_config = args.detector_config.resolve()
    if args.weights is not None:
        args.weights = args.weights.resolve()
    plan = _build_plan(args)
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
    return _execute(plan, args)


if __name__ == "__main__":
    raise SystemExit(main())
