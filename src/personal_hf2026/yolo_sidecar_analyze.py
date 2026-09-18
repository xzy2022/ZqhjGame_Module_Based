# 修改时间：2026-09-18。
# 修改目的：让人读报告直接呈现两随机种子的分天气波动范围与去重判断阈值。
# 修改内容：在天气表加入 Precision、Recall 的轮次范围，并记录连续重复率百分之五的机械判断口径。
# 修改时间：2026-09-18。
# 修改目的：让 YOLO 旁路实验能事后回答检测效果、天气稳定性、延迟和精确哈希去重价值。
# 修改内容：聚合逐帧旁路日志并输出总体、逐天气、逐轮及哈希复用证据的 JSON 与 Markdown 报告。
"""分析 yolo_sidecar_results.jsonl，输出可审计的批次统计。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean


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


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def bbox_of(item):
    bbox = item.get("bbox_xyxy", item.get("bbox"))
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if not all(finite_number(value) for value in bbox):
        return None
    values = [float(value) for value in bbox]
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def bbox_iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(ground_truth, predictions, threshold):
    candidates = []
    for gt_index, gt_bbox in enumerate(ground_truth):
        for prediction_index, prediction_bbox in enumerate(predictions):
            overlap = bbox_iou(gt_bbox, prediction_bbox)
            if overlap >= threshold:
                candidates.append((overlap, gt_index, prediction_index))
    used_gt = set()
    used_predictions = set()
    matches = []
    for overlap, gt_index, prediction_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        matches.append(overlap)
    return matches


class Accumulator:
    def __init__(self):
        self.frames = 0
        self.gt = 0
        self.predictions = 0
        self.tp = 0
        self.false_positive_frames = 0
        self.wall_ms = []
        self.sim_latency_s = []
        self.rows_with_hash = 0
        self.missing_hash = 0
        self.unique_hash_keys = set()
        self.repeated_hashes = 0
        self.consecutive_repeated_hashes = 0
        self.consecutive_repeat_wall_ms = 0.0
        self.last_hash_by_stream = {}

    def add(self, row, run_id, iou_threshold):
        self.frames += 1
        ground_truth = [bbox for item in row.get("ground_truth", []) if (bbox := bbox_of(item))]
        predictions = [bbox for item in row.get("predictions", []) if (bbox := bbox_of(item))]
        matches = greedy_match(ground_truth, predictions, iou_threshold)
        self.gt += len(ground_truth)
        self.predictions += len(predictions)
        self.tp += len(matches)
        if len(predictions) > len(matches):
            self.false_positive_frames += 1

        wall_ms = row.get("inference_wall_ms")
        if finite_number(wall_ms) and wall_ms >= 0:
            self.wall_ms.append(float(wall_ms))
        received = row.get("image_received_sim_time")
        completed = row.get("inference_completed_sim_time")
        if finite_number(received) and finite_number(completed) and completed >= received:
            self.sim_latency_s.append(float(completed - received))

        uid = row.get("uid", row.get("uav_id"))
        image_hash = row.get("image_sha256", row.get("image_hash"))
        if uid is None or not image_hash:
            self.missing_hash += 1
            return
        self.rows_with_hash += 1
        stream = (run_id, str(uid))
        key = (*stream, str(image_hash))
        if key in self.unique_hash_keys:
            self.repeated_hashes += 1
        else:
            self.unique_hash_keys.add(key)
        if self.last_hash_by_stream.get(stream) == image_hash:
            self.consecutive_repeated_hashes += 1
            if finite_number(wall_ms) and wall_ms >= 0:
                self.consecutive_repeat_wall_ms += float(wall_ms)
        self.last_hash_by_stream[stream] = image_hash

    def merge(self, other):
        self.frames += other.frames
        self.gt += other.gt
        self.predictions += other.predictions
        self.tp += other.tp
        self.false_positive_frames += other.false_positive_frames
        self.wall_ms.extend(other.wall_ms)
        self.sim_latency_s.extend(other.sim_latency_s)
        self.rows_with_hash += other.rows_with_hash
        self.missing_hash += other.missing_hash
        self.unique_hash_keys.update(other.unique_hash_keys)
        self.repeated_hashes += other.repeated_hashes
        self.consecutive_repeated_hashes += other.consecutive_repeated_hashes
        self.consecutive_repeat_wall_ms += other.consecutive_repeat_wall_ms

    def result(self):
        fp = self.predictions - self.tp
        fn = self.gt - self.tp
        duplicate_rate = safe_ratio(self.repeated_hashes, self.rows_with_hash)
        consecutive_rate = safe_ratio(self.consecutive_repeated_hashes, self.rows_with_hash)
        wall_total = sum(self.wall_ms)
        repeat_wall_share = safe_ratio(self.consecutive_repeat_wall_ms, wall_total)
        if consecutive_rate is None:
            assessment = "insufficient_hash_evidence"
        elif consecutive_rate >= 0.05:
            assessment = "worth_considering_exact_hash_dedup"
        else:
            assessment = "not_supported_by_current_exact_hash_evidence"
        return {
            "frames": self.frames,
            "ground_truth_objects": self.gt,
            "predictions": self.predictions,
            "tp": self.tp,
            "fp": fp,
            "fn": fn,
            "precision": safe_ratio(self.tp, self.predictions),
            "recall": safe_ratio(self.tp, self.gt),
            "false_positives_per_frame": safe_ratio(fp, self.frames),
            "frames_with_false_positive": self.false_positive_frames,
            "inference_wall_ms": distribution(self.wall_ms),
            "simulation_time_completion_minus_receive_s": distribution(self.sim_latency_s),
            "field_coverage": {
                "wall_latency_rows": len(self.wall_ms),
                "simulation_latency_rows": len(self.sim_latency_s),
                "hash_rows": self.rows_with_hash,
                "missing_hash_rows": self.missing_hash,
            },
            "exact_hash_reuse": {
                "scope": "same_run_and_uav",
                "unique_hashes": len(self.unique_hash_keys),
                "repeated_hash_rows": self.repeated_hashes,
                "repeated_hash_rate": duplicate_rate,
                "consecutive_repeated_hash_rows": self.consecutive_repeated_hashes,
                "consecutive_repeated_hash_rate": consecutive_rate,
                "consecutive_repeat_inference_wall_ms": self.consecutive_repeat_wall_ms,
                "consecutive_repeat_wall_time_share": repeat_wall_share,
                "assessment_threshold": "consecutive_repeated_hash_rate >= 0.05",
                "assessment": assessment,
            },
        }


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metadata_for(run_output):
    path = run_output / "metadata.json"
    return load_json(path) if path.is_file() else {}


def discover_runs(input_path):
    plan_path = input_path / "batch_plan.json"
    if plan_path.is_file():
        plan = load_json(plan_path)
        return [
            {
                "run_id": str(item["run_id"]),
                "weather": item.get("weather"),
                "seed": item.get("seed"),
                "output": Path(item["output"]).resolve(),
            }
            for item in plan["runs"]
        ]
    direct = input_path / RESULTS_NAME
    paths = [direct] if direct.is_file() else sorted(input_path.glob(f"**/{RESULTS_NAME}"))
    runs = []
    for path in paths:
        metadata = metadata_for(path.parent)
        runs.append(
            {
                "run_id": path.parent.name,
                "weather": metadata.get("weather", metadata.get("requested_weather")),
                "seed": metadata.get("seed"),
                "output": path.parent.resolve(),
            }
        )
    return runs


def iter_rows(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"JSONL 解析失败：{path}:{line_number}: {error}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL 行不是对象：{path}:{line_number}")
                yield row


def metric_range(run_results, field):
    values = [item["metrics"].get(field) for item in run_results]
    values = [float(value) for value in values if finite_number(value)]
    return {
        "count": len(values),
        "mean": fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "range": max(values) - min(values) if values else None,
    }


def analyze(input_path, iou_threshold):
    discovered = discover_runs(input_path)
    if not discovered:
        raise FileNotFoundError(f"未找到 batch_plan.json 或 {RESULTS_NAME}：{input_path}")
    overall = Accumulator()
    by_weather = defaultdict(Accumulator)
    by_uav = defaultdict(Accumulator)
    run_results = []
    missing_runs = []
    for item in discovered:
        results_path = item["output"] / RESULTS_NAME
        if not results_path.is_file():
            missing_runs.append({**item, "output": str(item["output"])})
            continue
        accumulator = Accumulator()
        metadata = metadata_for(item["output"])
        weather = item["weather"] or metadata.get("weather") or metadata.get("requested_weather") or "unknown"
        seed = item["seed"] if item["seed"] is not None else metadata.get("seed")
        for row in iter_rows(results_path):
            accumulator.add(row, item["run_id"], iou_threshold)
            uav_id = str(row.get("uid", row.get("uav_id", "unknown")))
            by_uav[uav_id].add(row, item["run_id"], iou_threshold)
        overall.merge(accumulator)
        by_weather[str(weather)].merge(accumulator)
        run_results.append(
            {
                "run_id": item["run_id"],
                "weather": weather,
                "seed": seed,
                "results": str(results_path),
                "metrics": accumulator.result(),
            }
        )

    weather_results = {}
    for weather, accumulator in sorted(by_weather.items()):
        runs = [item for item in run_results if str(item["weather"]) == weather]
        weather_results[weather] = {
            "aggregate": accumulator.result(),
            "run_stability": {
                "run_count": len(runs),
                "precision": metric_range(runs, "precision"),
                "recall": metric_range(runs, "recall"),
                "false_positives_per_frame": metric_range(runs, "false_positives_per_frame"),
                "inference_wall_p95_ms": {
                    "count": len([
                        item for item in runs
                        if item["metrics"]["inference_wall_ms"]["p95"] is not None
                    ]),
                    "values_by_run": {
                        item["run_id"]: item["metrics"]["inference_wall_ms"]["p95"]
                        for item in runs
                    },
                },
            },
        }
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "input": str(input_path),
        "matching": {
            "algorithm": "greedy_descending_iou_one_to_one",
            "iou_threshold": iou_threshold,
            "class_aware": False,
            "bbox_format": "xyxy",
        },
        "evidence_boundary": {
            "ground_truth": "runner 记录的 UE 真值投影框，仅用于开发审计",
            "detection_metrics": "当前对象检测不区分真车与诱饵，按类别无关 IoU 匹配",
            "simulation_latency": "完成仿真时间减接收仿真时间，不等同于墙钟推理延迟",
            "hash": "仅统计同一轮同一无人机的精确哈希；建议只依据连续重复估算安全跳帧机会",
        },
        "planned_runs": len(discovered),
        "analyzed_runs": len(run_results),
        "missing_runs": missing_runs,
        "overall": overall.result(),
        "by_weather": weather_results,
        "by_uav": {key: value.result() for key, value in sorted(by_uav.items())},
        "runs": run_results,
    }


def percent(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def number(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def percent_span(value):
    if value["min"] is None:
        return "N/A"
    return f"{percent(value['min'])}～{percent(value['max'])}"


def markdown(report):
    overall = report["overall"]
    lines = [
        "# YOLO 旁路批次分析",
        "",
        f"- 已分析轮次：{report['analyzed_runs']} / {report['planned_runs']}",
        f"- IoU 阈值：{report['matching']['iou_threshold']}",
        f"- Precision：{percent(overall['precision'])}",
        f"- Recall：{percent(overall['recall'])}",
        f"- 误报：{overall['fp']}，每帧 {number(overall['false_positives_per_frame'])}",
        f"- 推理墙钟耗时 P50 / P95：{number(overall['inference_wall_ms']['p50'])} / {number(overall['inference_wall_ms']['p95'])} ms",
        "",
        "## 分天气结果",
        "",
        "| 天气 | 轮次 | 帧 | Precision | P 轮次范围 | Recall | R 轮次范围 | FP/帧 | 推理 P95(ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for weather, item in report["by_weather"].items():
        metric = item["aggregate"]
        stability = item["run_stability"]
        lines.append(
            f"| {weather} | {item['run_stability']['run_count']} | {metric['frames']} | "
            f"{percent(metric['precision'])} | {percent_span(stability['precision'])} | "
            f"{percent(metric['recall'])} | {percent_span(stability['recall'])} | "
            f"{number(metric['false_positives_per_frame'])} | "
            f"{number(metric['inference_wall_ms']['p95'])} |"
        )
    hash_result = overall["exact_hash_reuse"]
    lines.extend(
        [
            "",
            "## 精确哈希去重证据",
            "",
            f"- 有哈希的推理记录：{overall['field_coverage']['hash_rows']}",
            f"- 任意历史重复率：{percent(hash_result['repeated_hash_rate'])}",
            f"- 连续重复率：{percent(hash_result['consecutive_repeated_hash_rate'])}",
            f"- 连续重复推理墙钟占比：{percent(hash_result['consecutive_repeat_wall_time_share'])}",
            f"- 机械判断：`{hash_result['assessment']}`",
            "",
            "该判断只针对精确相同且连续的图像；是否启用去重还需同时确认跳过后不会破坏结果新鲜度与控制时序。",
            "",
            "## 证据边界",
            "",
            "UE 投影真值框是开发审计信息，不是正式像素感知证据。仿真时间差与墙钟推理耗时分别报告，不能互相替代。",
        ]
    )
    if report["missing_runs"]:
        lines.extend(["", "## 缺失轮次", ""])
        lines.extend(f"- {item['run_id']}：{item['output']}" for item in report["missing_runs"])
    return "\n".join(lines) + "\n"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", required=True, type=Path, help="批次根或单轮输出目录")
    result.add_argument("--output", type=Path, help="默认写入批次根 analysis 子目录")
    result.add_argument("--iou-threshold", type=float, default=0.5)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.input = args.input.resolve()
    if not args.input.is_dir():
        raise FileNotFoundError(args.input)
    if not 0 < args.iou_threshold <= 1:
        raise ValueError("--iou-threshold 必须在 (0, 1] 内")
    output = (args.output or (args.input / "analysis")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = analyze(args.input, args.iou_threshold)
    write_json(output / "metrics.json", report)
    (output / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "metrics": str(output / "metrics.json"),
                "report": str(output / "REPORT.md"),
                "analyzed_runs": report["analyzed_runs"],
                "missing_runs": len(report["missing_runs"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
