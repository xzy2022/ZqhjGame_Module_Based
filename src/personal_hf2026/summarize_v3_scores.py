# 修改时间：2026-09-22。
# 修改目的：让已有运行的三项裁判得分可直接在终端中快速阅读。
# 修改内容：默认输出中文汇总表，保留 --json 供后续机器读取。
# 修改时间：2026-09-22。
# 修改目的：快速汇总多天气或单次 PersonalV3 已完成运行的三项裁判得分。
# 修改内容：读取官方 evaluation 产物，输出消灭数、定位精度维度平均得分和 200m 内近距扣分次数。
"""汇总一个运行目录或批量运行目录中的官方裁判得分。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _number(value, field: str, path: Path):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} 缺少有效的 {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} 的 {field} 不是有限数")
    return result


def _evaluation_files(input_path: Path):
    if input_path.is_file():
        if input_path.name.endswith(".evaluation.json"):
            return [input_path]
        raise ValueError("输入文件必须是 *.evaluation.json；目录可直接传入运行目录。")
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    direct = sorted(input_path.glob("*.evaluation.json"))
    return direct or sorted(input_path.rglob("*.evaluation.json"))


def _record(evaluation_file: Path):
    evaluation = _read_json(evaluation_file)
    dimensions = evaluation.get("dimension_scores")
    penalties = evaluation.get("penalty_breakdown")
    proximity = penalties.get("proximity") if isinstance(penalties, dict) else None
    if not isinstance(dimensions, dict) or not isinstance(proximity, dict):
        raise ValueError(f"{evaluation_file} 缺少标准裁判维度或近距扣分明细。")
    return {
        "run_id": evaluation_file.parent.name,
        "evaluation_file": str(evaluation_file),
        "n_destroyed": int(_number(
            evaluation.get("n_destroyed"), "n_destroyed", evaluation_file
        )),
        "accuracy_score": _number(
            dimensions.get("accuracy"), "dimension_scores.accuracy", evaluation_file
        ),
        "proximity_penalty_count_under_200m": int(_number(
            proximity.get("count"), "penalty_breakdown.proximity.count", evaluation_file
        )),
    }


def summarize(input_path: Path):
    evaluation_files = _evaluation_files(input_path)
    if not evaluation_files:
        raise FileNotFoundError(f"{input_path} 下未找到 *.evaluation.json")
    runs = [_record(path) for path in evaluation_files]
    count = len(runs)
    return {
        "input": str(input_path),
        "runs": runs,
        "aggregate": {
            "n_runs": count,
            "n_destroyed_total": sum(item["n_destroyed"] for item in runs),
            "accuracy_score_mean": sum(item["accuracy_score"] for item in runs) / count,
            "proximity_penalty_count_under_200m_total": sum(
                item["proximity_penalty_count_under_200m"] for item in runs
            ),
        },
        "field_notes": {
            "accuracy_score": "官方 dimension_scores.accuracy 定位精度维度得分，不是 YOLO 检测准确率。",
            "proximity_penalty_count_under_200m": (
                "官方 penalty_breakdown.proximity.count：无人机间距离小于 200m 的扣分事件数。"
            ),
        },
    }


def _print_readable(summary):
    aggregate = summary["aggregate"]
    print(f"输入路径：{summary['input']}")
    print(f"已统计运行数：{aggregate['n_runs']}")
    print()
    print(f"{'运行':<30}{'消灭个数':>10}{'定位精度得分':>16}{'<200m 扣分次数':>18}")
    for item in summary["runs"]:
        print(
            f"{item['run_id']:<30}{item['n_destroyed']:>10}"
            f"{item['accuracy_score']:>16.2f}"
            f"{item['proximity_penalty_count_under_200m']:>18}"
        )
    print("-" * 74)
    print(f"消灭个数（总计）：{aggregate['n_destroyed_total']}")
    print(f"定位精度平均得分：{aggregate['accuracy_score_mean']:.2f}")
    print(
        "<200m 距离扣分次数（总计）："
        f"{aggregate['proximity_penalty_count_under_200m_total']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="单次运行目录、包含多个运行目录的批量目录，或 *.evaluation.json 文件",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args()
    summary = summarize(args.input.resolve())
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        _print_readable(summary)


if __name__ == "__main__":
    main()
