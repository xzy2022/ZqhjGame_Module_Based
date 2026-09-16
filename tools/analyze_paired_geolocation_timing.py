# 修改时间：2026-09-16。
# 修改目的：避免把门控前陈旧候选的等待时间误解为合格双机估计的处理延迟。
# 修改内容：用预测帧标识关联时间日志，并分别汇总全部候选与最终成功估计。
# 修改时间：2026-09-16。
# 修改目的：从真实双机定位产物离线汇总图像新鲜度、双机时间差和姿态取样错配。
# 修改内容：优先读取候选配对时间日志，并兼容旧产物中仅有成功预测记录的有限证据。
"""离线汇总 paired_geolocation_live_study 的时间可观测性。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "mean": fmean(ordered) if ordered else None,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1] if ordered else None,
    }


def _append(bucket: dict[str, list[float]], key: str, value: Any) -> None:
    number = _number(value)
    if number is not None:
        bucket.setdefault(key, []).append(number)


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    views = row.get("views", {})
    if not isinstance(views, dict) or len(views) != 2:
        return None
    try:
        frames = tuple(sorted(
            (str(view["uid"]), int(view["frame_no"]))
            for view in views.values()))
        return (*frames, str(row["target_id"]))
    except (KeyError, TypeError, ValueError):
        return None


def _collect(rows: Iterable[dict[str, Any]]) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    for row in rows:
        _append(metrics, "pair.source_time_delta_s", row.get("source_time_delta_s"))
        _append(metrics, "pair.redis_read_wall_delta_s", row.get("redis_read_wall_delta_s"))
        _append(metrics, "pair.evaluator_receive_wall_delta_s",
                row.get("evaluator_receive_wall_delta_s"))
        for view in row.get("views", {}).values():
            frame_age = view.get("frame_age_sim_s", view.get("pose_time_delta_s"))
            pose_mismatch = view.get("pose_time_mismatch_s", view.get("pose_time_delta_s"))
            _append(metrics, "view.frame_age_sim_s", frame_age)
            _append(metrics, "view.pose_time_mismatch_s", pose_mismatch)
            _append(metrics, "view.truth_time_mismatch_s",
                    view.get("truth_time_mismatch_s", view.get("world_truth_time_delta_s")))
            _append(metrics, "view.redis_read_to_evaluator_s",
                    view.get("redis_read_to_evaluator_s"))
            _append(metrics, "view.context_to_redis_read_wall_s",
                    view.get("context_to_redis_read_wall_s"))
            _append(metrics, "view.redis_read_to_pair_match_s",
                    view.get("redis_read_to_pair_match_s"))
    return metrics


def analyze(run_root: Path) -> dict[str, Any]:
    timing_path = run_root / "paired_geolocation_timing.jsonl"
    prediction_path = run_root / "paired_geolocation_predictions.jsonl"
    timing_rows = list(_rows(timing_path))
    prediction_rows = list(_rows(prediction_path))
    source = timing_path if timing_rows else prediction_path
    rows = timing_rows if timing_rows else prediction_rows
    accepted_keys = {_pair_key(row) for row in prediction_rows}
    accepted_timing_rows = [
        row for row in timing_rows if _pair_key(row) in accepted_keys]
    metrics = _collect(rows)
    accepted_metrics = _collect(
        accepted_timing_rows if timing_rows else prediction_rows)

    return {
        "schema_version": 1,
        "run_root": str(run_root),
        "source": str(source),
        "source_kind": "candidate_pair_timing" if timing_rows else "legacy_success_predictions",
        "rows": len(rows),
        "prediction_rows": len(prediction_rows),
        "accepted_rows": len(accepted_timing_rows) if timing_rows else len(prediction_rows),
        "accepted_timing_coverage": (
            len(accepted_timing_rows) / len(prediction_rows)
            if timing_rows and prediction_rows else None),
        "metrics": {key: _metric(values) for key, values in sorted(metrics.items())},
        "accepted_metrics": {
            key: _metric(values) for key, values in sorted(accepted_metrics.items())},
        "interpretation": {
            "metrics": "门控前候选配对，包含等待另一机新帧时形成的陈旧配对",
            "accepted_metrics": "按两侧 uid/frame_no/target_id 与最终预测关联的成功估计",
            "frame_age_sim_s": "状态样本仿真时刻减 Redis source_sim_time；后者尚未验证为曝光时刻",
            "pose_time_mismatch_s": "当前姿态样本仿真时刻减 Redis source_sim_time",
            "source_time_delta_s": "组成候选配对的两张图像 Redis source_sim_time 绝对差",
            "wall_clock_metrics": "仅覆盖 Python 首次读 Redis 到评估配对，不含未知的相机曝光到 Redis 传输段",
        },
        "limitations": ([
            "旧产物没有 paired_geolocation_timing.jsonl，只能从成功预测记录恢复仿真时差；本机接收和匹配时刻不可恢复。"
        ] if not timing_rows else [
            "Redis source_sim_time 未验证为相机曝光时刻，因此不能把 frame_age_sim_s 直接称为端到端图像延迟。",
            *( ["候选时间日志达到记录上限，accepted_metrics 只覆盖仍可与时间日志关联的成功预测子集。"]
               if len(accepted_timing_rows) < len(prediction_rows) else []),
        ]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.run_root.resolve())
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
