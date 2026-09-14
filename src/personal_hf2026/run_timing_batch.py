# 修改时间：2026-09-14
# 修改目的：让个人实验脱离官方仓库的后续修改并支持独立运行。
# 修改内容：统一模块、SDK、运行资源及输出路径并保留实验行为。
"""顺序执行多轮实验，保持感知和天气不变并使用不同随机种子。"""

import argparse
from datetime import datetime
import json
from pathlib import Path

from .paths import PROJECT_ROOT, SIM_ROOT, RUNTIME_ROOT, OUTPUT_ROOT, SCENARIO_ROOT
import subprocess
import sys


ROOT = PROJECT_ROOT
DEFAULT_LAYOUT = SCENARIO_ROOT / "static-decoys.json"
DEFAULT_OUTPUT = OUTPUT_ROOT / "personal_v1" / "协同跟踪数据集"


def _read_result(output):
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    result["recorded_agents"] = [name for name in ("A", "B", "C") if name in result]
    evaluations = list(output.glob("*.evaluation.json"))
    if evaluations:
        evaluation = json.loads(evaluations[0].read_text(encoding="utf-8"))
        result["evaluation"] = {
            "total_score": evaluation.get("total_score"),
            "passed": evaluation.get("passed"),
            "n_destroyed": evaluation.get("n_destroyed"),
            "per_target": evaluation.get("per_target"),
            "undestroyed_decoy_misid_s": evaluation.get("undestroyed_decoy_misid_s"),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default=str(DEFAULT_LAYOUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--duration", type=float, default=120.0)
    args = parser.parse_args()

    layout = Path(args.layout).resolve()
    if not layout.is_file():
        raise FileNotFoundError(layout)
    output_root = Path(args.output).resolve()
    batch = output_root / f"batch-{datetime.now():%Y%m%d-%H%M%S}"
    batch.mkdir(parents=True)
    records = []
    print(f"数据集目录：{batch}")

    for index in range(args.runs):
        seed = args.start_seed + index
        run_output = batch / f"run-{index + 1:02d}-seed-{seed}"
        command = [
            sys.executable, "-B", "-X", "utf8", str(PROJECT_ROOT / "tools/run_official.py"),
            "timing_study",
            "--layout", str(layout),
            "--revision", "working",
            "--duration", str(args.duration),
            "--seed", str(seed),
            "--output", str(run_output),
        ]
        print(f"\n[{index + 1}/{args.runs}] seed={seed}")
        completed = subprocess.run(command, cwd=ROOT, check=False)
        record = {"run": index + 1, "seed": seed,
                  "output": str(run_output), "returncode": completed.returncode}
        if completed.returncode == 0:
            record["result"] = _read_result(run_output)
        records.append(record)
        (batch / "batch_summary.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = sum(record["returncode"] != 0 for record in records)
    print(f"\n完成：成功 {args.runs - failures} 轮，失败 {failures} 轮")
    print(f"汇总文件：{batch / 'batch_summary.json'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
