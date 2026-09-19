# 修改时间：2026-09-19。
# 修改目的：让没有TP的分组也与公共离线P/R/F1定义保持一致。
# 修改内容：类别无关与强制分类均复用公共计数指标函数，并保留旧输出键。
# 修改时间：2026-09-19。
# 修改目的：让V2低分恢复框和uncertain在批次默认报告中遵循同一离线评价口径。
# 修改内容：复用公共预测归一化与匹配，默认score阈值0.25，保留class_id强制分类和接受预测两套指标及原始输出计数。
# 修改时间：2026-09-19。
# 修改目的：避免把相同 seed 误解为三档使用完全相同的诱饵路线。
# 修改内容：在分析产物中记录官方 SDK 对诱饵路线使用未种子化随机数的比较边界。
# 修改时间：2026-09-18。
# 修改目的：让 v1/v2/v3 在线结果可按实际模型、配置、翻转和阈值公平比较。
# 修改内容：新增 profile 分组、类别感知指标与提交到结果观测的完整在线延迟，并审计每轮身份一致性。
# 修改时间：2026-09-18。
# 修改目的：把全部已提交图像的去重证据与仅完成推理的检测样本严格分开。
# 修改内容：新增 submissions 全量哈希统计及逐无人机口径，并将 results 哈希明确标为 completed-only。
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

from .yolo_offline_eval import (
    CLASS_NAMES, detection_metrics, greedy_match as common_greedy_match,
    normalize_predictions,
)


RESULTS_NAME = "yolo_sidecar_results.jsonl"
EXPECTED_CLASSES = {"TargetVehicle": "real_vehicle", "DecoyVehicle": "model_prop"}


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


def class_name(item, ground_truth=False):
    if ground_truth:
        return EXPECTED_CLASSES.get(item.get("class"))
    return item.get("class_name")


def greedy_match(ground_truth, predictions, threshold, class_aware=False):
    # 与离线使用同一排序及一对一匹配；uncertain仍按class_id参与强制二分类。
    names = {value: key for key, value in CLASS_NAMES.items()}
    truth = [{"bbox_xyxy": bbox_of(item), "class_id": names[class_name(item, True)]}
             for item in ground_truth]
    matches, _, _ = common_greedy_match(truth, predictions, threshold, same_class=class_aware)
    return matches


class Accumulator:
    def __init__(self):
        self.frames = 0
        self.gt = 0
        self.predictions = 0
        self.tp = 0
        self.class_aware_tp = 0
        self.raw_predictions = 0
        self.invalid_bbox_predictions = 0
        self.below_threshold_predictions = 0
        self.below_threshold_recovered_predictions = 0
        self.uncertain_predictions = 0
        self.accepted_predictions = 0
        self.accepted_class_aware_tp = 0
        self.false_positive_frames = 0
        self.class_aware_false_positive_frames = 0
        self.wall_ms = []
        self.decode_ms = []
        self.accepted_to_result_wall_ms = []
        self.source_to_receive_sim_s = []
        self.sim_latency_s = []
        self.source_to_result_sim_s = []
        self.rows_with_hash = 0
        self.missing_hash = 0
        self.unique_hash_keys = set()
        self.repeated_hashes = 0
        self.hash_comparisons = 0
        self.consecutive_repeated_hashes = 0
        self.consecutive_repeat_wall_ms = 0.0
        self.last_hash_by_stream = {}

    def add(self, row, run_id, iou_threshold, confidence_threshold=0.25):
        self.frames += 1
        ground_truth = [item for item in row.get("ground_truth", [])
                        if bbox_of(item) and class_name(item, True) in CLASS_NAMES.values()]
        raw_predictions = row.get("predictions", [])
        valid_predictions = [item for item in raw_predictions if bbox_of(item)]
        predictions = normalize_predictions(valid_predictions, row.get("image_width", 0),
                                            row.get("image_height", 0), confidence_threshold)
        accepted_predictions = [item for item in predictions if item["accepted"]]
        self.raw_predictions += len(raw_predictions)
        self.invalid_bbox_predictions += len(raw_predictions) - len(valid_predictions)
        self.below_threshold_predictions += len(valid_predictions) - len(predictions)
        self.below_threshold_recovered_predictions += sum(
            item["score"] < confidence_threshold and bool(item.get("recovered_low_score"))
            for item in valid_predictions)
        self.uncertain_predictions += len(predictions) - len(accepted_predictions)
        self.accepted_predictions += len(accepted_predictions)
        self.accepted_class_aware_tp += len(greedy_match(
            ground_truth, accepted_predictions, iou_threshold, class_aware=True))
        matches = greedy_match(ground_truth, predictions, iou_threshold)
        class_aware_matches = greedy_match(
            ground_truth, predictions, iou_threshold, class_aware=True
        )
        self.gt += len(ground_truth)
        self.predictions += len(predictions)
        self.tp += len(matches)
        self.class_aware_tp += len(class_aware_matches)
        if len(predictions) > len(matches):
            self.false_positive_frames += 1
        if len(predictions) > len(class_aware_matches):
            self.class_aware_false_positive_frames += 1

        wall_ms = row.get("inference_wall_ms")
        if finite_number(wall_ms) and wall_ms >= 0:
            self.wall_ms.append(float(wall_ms))
        decode_ms = row.get("decode_ms")
        if finite_number(decode_ms) and decode_ms >= 0:
            self.decode_ms.append(float(decode_ms))
        submitted_wall = row.get("submitted_perf_counter")
        observed_wall = row.get("result_observed_perf_counter")
        if (finite_number(submitted_wall) and finite_number(observed_wall)
                and observed_wall >= submitted_wall):
            self.accepted_to_result_wall_ms.append(
                float(observed_wall - submitted_wall) * 1000.0
            )
        source = row.get("source_sim_time")
        received = row.get("image_received_sim_time")
        completed = row.get(
            "result_observed_sim_time", row.get("inference_completed_sim_time")
        )
        if finite_number(source) and finite_number(received) and received >= source:
            self.source_to_receive_sim_s.append(float(received - source))
        if finite_number(received) and finite_number(completed) and completed >= received:
            self.sim_latency_s.append(float(completed - received))
        if finite_number(source) and finite_number(completed) and completed >= source:
            self.source_to_result_sim_s.append(float(completed - source))

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
        if stream in self.last_hash_by_stream:
            self.hash_comparisons += 1
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
        self.class_aware_tp += other.class_aware_tp
        self.raw_predictions += other.raw_predictions
        self.invalid_bbox_predictions += other.invalid_bbox_predictions
        self.below_threshold_predictions += other.below_threshold_predictions
        self.below_threshold_recovered_predictions += other.below_threshold_recovered_predictions
        self.uncertain_predictions += other.uncertain_predictions
        self.accepted_predictions += other.accepted_predictions
        self.accepted_class_aware_tp += other.accepted_class_aware_tp
        self.false_positive_frames += other.false_positive_frames
        self.class_aware_false_positive_frames += other.class_aware_false_positive_frames
        self.wall_ms.extend(other.wall_ms)
        self.decode_ms.extend(other.decode_ms)
        self.accepted_to_result_wall_ms.extend(other.accepted_to_result_wall_ms)
        self.source_to_receive_sim_s.extend(other.source_to_receive_sim_s)
        self.sim_latency_s.extend(other.sim_latency_s)
        self.source_to_result_sim_s.extend(other.source_to_result_sim_s)
        self.rows_with_hash += other.rows_with_hash
        self.missing_hash += other.missing_hash
        self.unique_hash_keys.update(other.unique_hash_keys)
        self.repeated_hashes += other.repeated_hashes
        self.hash_comparisons += other.hash_comparisons
        self.consecutive_repeated_hashes += other.consecutive_repeated_hashes
        self.consecutive_repeat_wall_ms += other.consecutive_repeat_wall_ms

    def result(self):
        fp = self.predictions - self.tp
        fn = self.gt - self.tp
        class_aware_fp = self.predictions - self.class_aware_tp
        class_aware_fn = self.gt - self.class_aware_tp
        duplicate_rate = safe_ratio(self.repeated_hashes, self.rows_with_hash)
        consecutive_rate = safe_ratio(self.consecutive_repeated_hashes, self.hash_comparisons)
        wall_total = sum(self.wall_ms)
        repeat_wall_share = safe_ratio(self.consecutive_repeat_wall_ms, wall_total)
        class_agnostic = {
            **detection_metrics(self.tp, fp, fn),
            "false_positives_per_frame": safe_ratio(fp, self.frames),
            "frames_with_false_positive": self.false_positive_frames,
        }
        class_aware = {
            **detection_metrics(self.class_aware_tp, class_aware_fp, class_aware_fn),
            "false_positives_per_frame": safe_ratio(class_aware_fp, self.frames),
            "frames_with_false_positive": self.class_aware_false_positive_frames,
        }
        online_latency = {
            "inference_wall_ms": distribution(self.wall_ms),
            "decode_ms": distribution(self.decode_ms),
            "accepted_to_result_observed_wall_ms": distribution(
                self.accepted_to_result_wall_ms
            ),
            "source_to_image_received_sim_s": distribution(
                self.source_to_receive_sim_s
            ),
            "image_received_to_result_observed_sim_s": distribution(
                self.sim_latency_s
            ),
            "source_to_result_observed_sim_s": distribution(
                self.source_to_result_sim_s
            ),
        }
        return {
            "frames": self.frames,
            "ground_truth_objects": self.gt,
            "predictions": self.predictions,
            "tp": self.tp,
            "fp": fp,
            "fn": fn,
            "precision": class_agnostic["precision"],
            "recall": class_agnostic["recall"],
            "false_positives_per_frame": safe_ratio(fp, self.frames),
            "frames_with_false_positive": self.false_positive_frames,
            "inference_wall_ms": distribution(self.wall_ms),
            "simulation_time_completion_minus_receive_s": distribution(self.sim_latency_s),
            "class_agnostic": class_agnostic,
            "class_aware": class_aware,
            "class_aware_forced_class_id": class_aware,
            "accepted_class_aware": detection_metrics(
                self.accepted_class_aware_tp,
                self.accepted_predictions - self.accepted_class_aware_tp,
                self.gt - self.accepted_class_aware_tp),
            "raw_prediction_counts": {
                "emitted": self.raw_predictions,
                "invalid_bbox": self.invalid_bbox_predictions,
                "below_confidence_threshold": self.below_threshold_predictions,
                "below_threshold_recovered_low_score": self.below_threshold_recovered_predictions,
                "evaluated": self.predictions,
                "uncertain_evaluated": self.uncertain_predictions,
                "accepted_evaluated": self.accepted_predictions,
            },
            "online_latency": online_latency,
            "field_coverage": {
                "wall_latency_rows": len(self.wall_ms),
                "simulation_latency_rows": len(self.sim_latency_s),
                "accepted_to_result_wall_rows": len(self.accepted_to_result_wall_ms),
                "source_to_result_sim_rows": len(self.source_to_result_sim_s),
                "hash_rows": self.rows_with_hash,
                "missing_hash_rows": self.missing_hash,
            },
            "completed_result_hashes": {
                "evidence_scope": "completed_inference_results_only_not_dedup_decision_basis",
                "scope": "same_run_and_uav",
                "unique_hashes": len(self.unique_hash_keys),
                "repeated_hash_rows": self.repeated_hashes,
                "repeated_hash_rate": duplicate_rate,
                "consecutive_comparisons": self.hash_comparisons,
                "consecutive_repeated_hash_rows": self.consecutive_repeated_hashes,
                "consecutive_repeated_hash_rate": consecutive_rate,
                "consecutive_repeat_inference_wall_ms": self.consecutive_repeat_wall_ms,
                "consecutive_repeat_wall_time_share": repeat_wall_share,
            },
        }

    @staticmethod
    def _f1(tp, fp, fn):
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2.0 * precision * recall / (precision + recall)


class SubmissionHashAccumulator:
    """按提交顺序统计全量输入；同一轮同一无人机之间才允许连续比较。"""

    def __init__(self):
        self.rows = 0
        self.rows_with_hash = 0
        self.missing_identity_or_hash = 0
        self.unique_hash_keys = set()
        self.repeated_hashes = 0
        self.comparisons = 0
        self.consecutive_same_hash = 0
        self.consecutive_same_frame_no = 0
        self.consecutive_sim_gap_s = []
        self.same_hash_sim_gap_s = []
        self.last_by_stream = {}

    def add(self, row, run_id):
        self.rows += 1
        uid = row.get("uid", row.get("uav_id"))
        image_hash = row.get("image_sha256", row.get("image_hash"))
        if uid is None or not image_hash:
            self.missing_identity_or_hash += 1
            return
        self.rows_with_hash += 1
        stream = (run_id, str(uid))
        key = (*stream, str(image_hash))
        if key in self.unique_hash_keys:
            self.repeated_hashes += 1
        else:
            self.unique_hash_keys.add(key)

        previous = self.last_by_stream.get(stream)
        if previous is not None:
            self.comparisons += 1
            same_hash = previous["image_hash"] == image_hash
            if same_hash:
                self.consecutive_same_hash += 1
            frame_no = row.get("frame_no")
            if frame_no is not None and frame_no == previous["frame_no"]:
                self.consecutive_same_frame_no += 1
            received = row.get("image_received_sim_time")
            if (finite_number(received)
                    and finite_number(previous["image_received_sim_time"])
                    and received >= previous["image_received_sim_time"]):
                gap = float(received - previous["image_received_sim_time"])
                self.consecutive_sim_gap_s.append(gap)
                if same_hash:
                    self.same_hash_sim_gap_s.append(gap)
        self.last_by_stream[stream] = {
            "image_hash": image_hash,
            "frame_no": row.get("frame_no"),
            "image_received_sim_time": row.get("image_received_sim_time"),
        }

    def merge(self, other):
        self.rows += other.rows
        self.rows_with_hash += other.rows_with_hash
        self.missing_identity_or_hash += other.missing_identity_or_hash
        self.unique_hash_keys.update(other.unique_hash_keys)
        self.repeated_hashes += other.repeated_hashes
        self.comparisons += other.comparisons
        self.consecutive_same_hash += other.consecutive_same_hash
        self.consecutive_same_frame_no += other.consecutive_same_frame_no
        self.consecutive_sim_gap_s.extend(other.consecutive_sim_gap_s)
        self.same_hash_sim_gap_s.extend(other.same_hash_sim_gap_s)

    def result(self):
        consecutive_rate = safe_ratio(self.consecutive_same_hash, self.comparisons)
        if consecutive_rate is None:
            assessment = "insufficient_submission_hash_evidence"
        elif consecutive_rate >= 0.05:
            assessment = "worth_considering_exact_hash_dedup"
        else:
            assessment = "not_supported_by_current_exact_hash_evidence"
        return {
            "evidence_scope": "all_accepted_submissions_before_newest_only_supersession",
            "scope": "same_run_and_uav",
            "submission_rows": self.rows,
            "hash_rows": self.rows_with_hash,
            "missing_identity_or_hash_rows": self.missing_identity_or_hash,
            "unique_hashes": len(self.unique_hash_keys),
            "repeated_hash_rows": self.repeated_hashes,
            "repeated_hash_rate": safe_ratio(self.repeated_hashes, self.rows_with_hash),
            "consecutive_comparisons": self.comparisons,
            "consecutive_same_hash": self.consecutive_same_hash,
            "consecutive_same_hash_rate": consecutive_rate,
            "consecutive_same_frame_no": self.consecutive_same_frame_no,
            "consecutive_sim_gap_s": distribution(self.consecutive_sim_gap_s),
            "same_hash_sim_gap_s": distribution(self.same_hash_sim_gap_s),
            "assessment_threshold": "consecutive_same_hash_rate >= 0.05",
            "assessment": assessment,
        }


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metadata_for(run_output):
    path = run_output / "metadata.json"
    return load_json(path) if path.is_file() else {}


def profile_trace_for(run_output, requested_profile=None):
    metadata = metadata_for(run_output)
    summary_path = run_output / "summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    resources = metadata.get("resources") or summary.get("resources") or {}
    trace = {
        "requested_profile": requested_profile,
        "metadata_profile": metadata.get("yolo_profile"),
        "summary_profile": summary.get("yolo_profile"),
        "resource_profile": resources.get("profile"),
        "config_path": resources.get("config_path"),
        "config_sha256": resources.get("config_sha256"),
        "weights_path": resources.get("weights_path"),
        "weights_sha256": resources.get("weights_sha256"),
        "model_format": resources.get("model_format"),
        "effective_options": resources.get("effective_options"),
        "tracker_high_multiplier": resources.get("tracker_high_multiplier"),
    }
    candidates = [
        trace["metadata_profile"],
        trace["summary_profile"],
        trace["resource_profile"],
    ]
    actual_profile = next((value for value in candidates if value), requested_profile)
    trace["actual_profile"] = actual_profile or "unknown"
    issues = []
    if requested_profile and any(
        value is not None and value != requested_profile for value in candidates
    ):
        issues.append("requested_profile_mismatch")
    if len({value for value in candidates if value is not None}) > 1:
        issues.append("artifact_profile_mismatch")
    for field in ("model_format", "config_sha256", "weights_sha256", "effective_options"):
        if not trace[field]:
            issues.append(f"missing_{field}")
    return trace, issues


def discover_runs(input_path):
    plan_path = input_path / "batch_plan.json"
    if plan_path.is_file():
        plan = load_json(plan_path)
        return [
            {
                "run_id": str(item["run_id"]),
                "yolo_profile": item.get("yolo_profile"),
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
                "yolo_profile": metadata.get("yolo_profile"),
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


def metric_range(run_results, field, metric_group=None):
    values = [
        (
            item["metrics"].get(metric_group, {}).get(field)
            if metric_group else item["metrics"].get(field)
        )
        for item in run_results
    ]
    values = [float(value) for value in values if finite_number(value)]
    return {
        "count": len(values),
        "mean": fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "range": max(values) - min(values) if values else None,
    }


def run_stability(run_results):
    return {
        "run_count": len(run_results),
        "class_aware_precision": metric_range(
            run_results, "precision", "class_aware"
        ),
        "class_aware_recall": metric_range(run_results, "recall", "class_aware"),
        "class_aware_f1": metric_range(run_results, "f1", "class_aware"),
        "class_aware_false_positives_per_frame": metric_range(
            run_results, "false_positives_per_frame", "class_aware"
        ),
        "inference_wall_p95_ms": {
            "count": len([
                item for item in run_results
                if item["metrics"]["online_latency"]["inference_wall_ms"]["p95"]
                is not None
            ]),
            "values_by_run": {
                item["run_id"]: item["metrics"]["online_latency"]
                ["inference_wall_ms"]["p95"]
                for item in run_results
            },
        },
        "accepted_to_result_wall_p95_ms": {
            "count": len([
                item for item in run_results
                if item["metrics"]["online_latency"]
                ["accepted_to_result_observed_wall_ms"]["p95"] is not None
            ]),
            "values_by_run": {
                item["run_id"]: item["metrics"]["online_latency"]
                ["accepted_to_result_observed_wall_ms"]["p95"]
                for item in run_results
            },
        },
    }


def unique_profile_traces(run_results):
    variants = {}
    for item in run_results:
        trace = item["profile_trace"]
        identity = {
            key: trace.get(key)
            for key in (
                "resource_profile",
                "config_path",
                "config_sha256",
                "weights_path",
                "weights_sha256",
                "model_format",
                "effective_options",
                "tracker_high_multiplier",
            )
        }
        variants[json.dumps(identity, ensure_ascii=False, sort_keys=True)] = identity
    return list(variants.values())


def analyze(input_path, iou_threshold, confidence_threshold=0.25):
    discovered = discover_runs(input_path)
    if not discovered:
        raise FileNotFoundError(f"未找到 batch_plan.json 或 {RESULTS_NAME}：{input_path}")
    overall = Accumulator()
    by_profile = defaultdict(Accumulator)
    by_weather_profile = defaultdict(Accumulator)
    by_profile_uav = defaultdict(lambda: defaultdict(Accumulator))
    overall_submissions = SubmissionHashAccumulator()
    by_profile_submissions = defaultdict(SubmissionHashAccumulator)
    by_weather_profile_submissions = defaultdict(SubmissionHashAccumulator)
    by_profile_uav_submissions = defaultdict(
        lambda: defaultdict(SubmissionHashAccumulator)
    )
    run_results = []
    missing_runs = []
    missing_submission_logs = []
    profile_consistency_issues = []
    for item in discovered:
        results_path = item["output"] / RESULTS_NAME
        if not results_path.is_file():
            missing_runs.append({**item, "output": str(item["output"])})
            continue
        accumulator = Accumulator()
        metadata = metadata_for(item["output"])
        profile_trace, trace_issues = profile_trace_for(
            item["output"], item.get("yolo_profile")
        )
        profile = str(profile_trace["actual_profile"])
        weather = item["weather"] or metadata.get("weather") or metadata.get("requested_weather") or "unknown"
        seed = item["seed"] if item["seed"] is not None else metadata.get("seed")
        submissions_path = item["output"] / "yolo_sidecar_submissions.jsonl"
        run_submissions = SubmissionHashAccumulator()
        if submissions_path.is_file():
            for row in iter_rows(submissions_path):
                run_submissions.add(row, item["run_id"])
                uav_id = str(row.get("uid", row.get("uav_id", "unknown")))
                by_profile_uav_submissions[profile][uav_id].add(
                    row, item["run_id"]
                )
            overall_submissions.merge(run_submissions)
            by_profile_submissions[profile].merge(run_submissions)
            by_weather_profile_submissions[(str(weather), profile)].merge(
                run_submissions
            )
        else:
            missing_submission_logs.append({
                "run_id": item["run_id"],
                "path": str(submissions_path),
            })
        result_profiles = set()
        for row in iter_rows(results_path):
            result_profiles.add(row.get("yolo_profile"))
            accumulator.add(row, item["run_id"], iou_threshold, confidence_threshold)
            uav_id = str(row.get("uid", row.get("uav_id", "unknown")))
            by_profile_uav[profile][uav_id].add(
                row, item["run_id"], iou_threshold, confidence_threshold
            )
        if result_profiles != {profile}:
            trace_issues.append("result_row_profile_mismatch")
        if trace_issues:
            profile_consistency_issues.append(
                {"run_id": item["run_id"], "issues": sorted(set(trace_issues))}
            )
        overall.merge(accumulator)
        by_profile[profile].merge(accumulator)
        by_weather_profile[(str(weather), profile)].merge(accumulator)
        run_results.append(
            {
                "run_id": item["run_id"],
                "yolo_profile": profile,
                "weather": weather,
                "seed": seed,
                "results": str(results_path),
                "profile_trace": profile_trace,
                "profile_consistency_issues": sorted(set(trace_issues)),
                "metrics": accumulator.result(),
                "submission_hashes": (
                    run_submissions.result() if submissions_path.is_file() else None
                ),
            }
        )

    profile_results = {}
    for profile, accumulator in sorted(by_profile.items()):
        runs = [item for item in run_results if item["yolo_profile"] == profile]
        profile_results[profile] = {
            "aggregate": accumulator.result(),
            "submission_hashes": by_profile_submissions[profile].result(),
            "run_stability": run_stability(runs),
            "resource_variants": unique_profile_traces(runs),
            "by_uav": {
                uid: value.result()
                for uid, value in sorted(by_profile_uav[profile].items())
            },
            "submission_hashes_by_uav": {
                uid: value.result()
                for uid, value in sorted(
                    by_profile_uav_submissions[profile].items()
                )
            },
        }
    weather_profile_results = defaultdict(dict)
    for (weather, profile), accumulator in sorted(by_weather_profile.items()):
        runs = [
            item for item in run_results
            if str(item["weather"]) == weather and item["yolo_profile"] == profile
        ]
        weather_profile_results[weather][profile] = {
            "aggregate": accumulator.result(),
            "submission_hashes": by_weather_profile_submissions[
                (weather, profile)
            ].result(),
            "run_stability": run_stability(runs),
        }
    return {
        "schema_version": 3,
        "created_at": utc_now(),
        "input": str(input_path),
        "matching": {
            "algorithm": "greedy_descending_iou_one_to_one",
            "iou_threshold": iou_threshold,
            "confidence_threshold": confidence_threshold,
            "metric_modes": ["class_agnostic", "class_aware_forced_class_id", "accepted_class_aware"],
            "class_aware_alias": "class_aware是class_aware_forced_class_id的兼容键；uncertain不剔除，按class_id匹配。",
            "previous_schema_boundary": "schema<=2未应用score阈值且按class_name匹配，不能和新默认值直接混合汇总。",
            "class_mapping": EXPECTED_CLASSES,
            "bbox_format": "xyxy",
        },
        "evidence_boundary": {
            "ground_truth": "runner 记录的 UE 真值投影框，仅用于开发审计",
            "profile_comparison": (
                "跨 profile overall 仅用于数据完整性；方案优劣必须看 by_profile "
                "和 by_weather_by_profile，并核对 resource_variants"
            ),
            "scenario_randomization": (
                "官方 coop_decoy runner 的 seed 固定真车路线，但诱饵路线按设计使用"
                "未种子化 RNG；同 weather/seed 的不同 profile 不是逐场景严格配对实验"
            ),
            "detection_metrics": (
                "先应用confidence_threshold；class_aware按class_id强制二分类，含uncertain；"
                "accepted_class_aware单独去除uncertain再匹配；与离线使用同一匹配实现"
            ),
            "online_latency": (
                "accepted_to_result_observed_wall_ms 是提交被接受至 runner 观测结果的墙钟总延迟；"
                "inference_wall_ms 仅为 detector.predict"
            ),
            "simulation_latency": (
                "source/receive/result 的 sim-time 差用于观察链路，不是已验证曝光时延"
            ),
            "hash": (
                "去重判断使用 yolo_sidecar_submissions.jsonl 的全部已接受提交；"
                "completed results 的哈希只描述实际推理子集"
            ),
        },
        "planned_runs": len(discovered),
        "analyzed_runs": len(run_results),
        "missing_runs": missing_runs,
        "profile_consistency_issues": profile_consistency_issues,
        "overall": overall.result(),
        "submission_hash_evidence": {
            "overall": overall_submissions.result(),
            "missing_submission_logs": missing_submission_logs,
        },
        "by_profile": profile_results,
        "by_weather_by_profile": dict(weather_profile_results),
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


def file_name(value):
    return "N/A" if not value else Path(value).name


def short_hash(value):
    return "N/A" if not value else str(value)[:12]


def markdown(report):
    overall = report["overall"]
    overall_aware = overall["class_aware"]
    overall_latency = overall["online_latency"]
    lines = [
        "# YOLO 旁路批次分析",
        "",
        f"- 已分析轮次：{report['analyzed_runs']} / {report['planned_runs']}",
        f"- IoU 阈值：{report['matching']['iou_threshold']}",
        f"- score 阈值：{report['matching']['confidence_threshold']}；class_aware按class_id强制二分类，uncertain仍保留。",
        "- accepted_class_aware单独报告剔除uncertain后的指标；schema≤2旧报告未筛score且按class_name计分，不能直接混用。",
        f"- 原始输出框 {overall['raw_prediction_counts']['emitted']}；阈值剔除 {overall['raw_prediction_counts']['below_confidence_threshold']}（其中低分恢复框 {overall['raw_prediction_counts']['below_threshold_recovered_low_score']}）；评价中uncertain {overall['raw_prediction_counts']['uncertain_evaluated']}。",
        "- 下列跨 profile 合计只用于核对数据完整性，不能用于判断某档优劣。",
        f"- 类别感知 Precision / Recall / F1：{percent(overall_aware['precision'])} / {percent(overall_aware['recall'])} / {percent(overall_aware['f1'])}",
        f"- 类别感知误报：{overall_aware['fp']}，每帧 {number(overall_aware['false_positives_per_frame'])}",
        f"- 提交接受→结果观测墙钟总延迟 P50 / P95：{number(overall_latency['accepted_to_result_observed_wall_ms']['p50'])} / {number(overall_latency['accepted_to_result_observed_wall_ms']['p95'])} ms",
        f"- detector.predict P50 / P95：{number(overall_latency['inference_wall_ms']['p50'])} / {number(overall_latency['inference_wall_ms']['p95'])} ms",
        "",
        "## Profile 身份",
        "",
        "| Profile | 变体数 | 格式 | 模型 | 模型 SHA | 配置 | 配置 SHA | flip | high / low / unknown |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    for profile, item in report["by_profile"].items():
        variants = item["resource_variants"]
        trace = variants[0] if len(variants) == 1 else {}
        options = trace.get("effective_options") or {}
        thresholds = " / ".join(
            number(options.get(field))
            for field in (
                "tracker_high_confidence_threshold",
                "tracker_low_confidence_threshold",
                "unknown_class_confidence_threshold",
            )
        )
        lines.append(
            f"| {profile} | {len(variants)} | {trace.get('model_format') or 'N/A'} | "
            f"{file_name(trace.get('weights_path'))} | {short_hash(trace.get('weights_sha256'))} | "
            f"{file_name(trace.get('config_path'))} | {short_hash(trace.get('config_sha256'))} | "
            f"{options.get('flip_enabled', 'N/A')} | {thresholds} |"
        )
    lines.extend(
        [
            "",
            "每个 Profile 应只有一个资源变体；若变体数大于 1，先检查路径、哈希和有效参数，再比较指标。",
            "",
            "## 分 Profile 结果",
            "",
            "| Profile | 轮次 | 帧 | P / R / F1（类别感知） | FP/帧 | 总延迟 P50 / P95(ms) | 推理 P50 / P95(ms) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for profile, item in report["by_profile"].items():
        metric = item["aggregate"]
        aware = metric["class_aware"]
        latency = metric["online_latency"]
        lines.append(
            f"| {profile} | {item['run_stability']['run_count']} | {metric['frames']} | "
            f"{percent(aware['precision'])} / {percent(aware['recall'])} / {percent(aware['f1'])} | "
            f"{number(aware['false_positives_per_frame'])} | "
            f"{number(latency['accepted_to_result_observed_wall_ms']['p50'])} / {number(latency['accepted_to_result_observed_wall_ms']['p95'])} | "
            f"{number(latency['inference_wall_ms']['p50'])} / {number(latency['inference_wall_ms']['p95'])} |"
        )
    lines.extend(
        [
            "",
            "## 分天气 × Profile 结果",
            "",
            "| 天气 | Profile | 轮次 | 帧 | P | P 轮次范围 | R | R 轮次范围 | F1 | FP/帧 | 总延迟 P95(ms) | 推理 P95(ms) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for weather, profiles in report["by_weather_by_profile"].items():
        for profile, item in profiles.items():
            metric = item["aggregate"]
            aware = metric["class_aware"]
            latency = metric["online_latency"]
            stability = item["run_stability"]
            lines.append(
                f"| {weather} | {profile} | {stability['run_count']} | {metric['frames']} | "
                f"{percent(aware['precision'])} | {percent_span(stability['class_aware_precision'])} | "
                f"{percent(aware['recall'])} | {percent_span(stability['class_aware_recall'])} | "
                f"{percent(aware['f1'])} | {number(aware['false_positives_per_frame'])} | "
                f"{number(latency['accepted_to_result_observed_wall_ms']['p95'])} | "
                f"{number(latency['inference_wall_ms']['p95'])} |"
            )
    hash_evidence = report["submission_hash_evidence"]
    hash_result = hash_evidence["overall"]
    lines.extend(
        [
            "",
            "## 精确哈希去重证据",
            "",
            f"- 全部已接受提交：{hash_result['submission_rows']}，其中有哈希 {hash_result['hash_rows']}",
            f"- 任意历史重复率：{percent(hash_result['repeated_hash_rate'])}",
            f"- 同一轮同一无人机连续比较：{hash_result['consecutive_comparisons']}",
            f"- 连续相同哈希率：{percent(hash_result['consecutive_same_hash_rate'])}",
            f"- 连续相同哈希的仿真时间间隔 P50 / P95：{number(hash_result['same_hash_sim_gap_s']['p50'])} / {number(hash_result['same_hash_sim_gap_s']['p95'])} s",
            f"- 机械判断：`{hash_result['assessment']}`",
            "",
            "该判断来自进入 newest-only 槽位前的全部 accepted submissions，而不是只看完成推理的子集。它只针对精确相同且连续的图像；是否启用去重还需确认跳过后不会破坏结果新鲜度与控制时序。",
            "",
            "## 证据边界",
            "",
            "UE 投影真值框是开发审计信息，不是正式像素感知证据。提交接受到结果观测的墙钟差覆盖在线排队、解码/推理、IPC 和 runner 轮询；它仍不含未进入提交日志之前的相机链路。source/receive/result 的仿真时间差不是已验证曝光时延，也不能替代墙钟延迟。",
            "",
            "官方 coop_decoy runner 的 seed 只固定真车路线；诱饵路线按设计使用未种子化 RNG。因此相同 weather/seed 的不同 profile 仍可能面对不同诱饵布局，不能当作逐场景严格配对，只能结合双种子聚合、轮次范围和实际场景文件解释。",
        ]
    )
    if report["profile_consistency_issues"]:
        lines.extend(["", "## Profile 一致性问题", ""])
        lines.extend(
            f"- {item['run_id']}：{', '.join(item['issues'])}"
            for item in report["profile_consistency_issues"]
        )
    if report["missing_runs"]:
        lines.extend(["", "## 缺失轮次", ""])
        lines.extend(f"- {item['run_id']}：{item['output']}" for item in report["missing_runs"])
    if hash_evidence["missing_submission_logs"]:
        lines.extend(["", "## 缺失 submissions 日志", ""])
        lines.extend(
            f"- {item['run_id']}：{item['path']}"
            for item in hash_evidence["missing_submission_logs"]
        )
    return "\n".join(lines) + "\n"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", required=True, type=Path, help="批次根或单轮输出目录")
    result.add_argument("--output", type=Path, help="默认写入批次根 analysis 子目录")
    result.add_argument("--iou-threshold", type=float, default=0.5)
    result.add_argument("--confidence-threshold", type=float, default=0.25)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.input = args.input.resolve()
    if not args.input.is_dir():
        raise FileNotFoundError(args.input)
    if not 0 < args.iou_threshold <= 1:
        raise ValueError("--iou-threshold 必须在 (0, 1] 内")
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("--confidence-threshold 必须在 [0, 1] 内")
    output = (args.output or (args.input / "analysis")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = analyze(args.input, args.iou_threshold, args.confidence_threshold)
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
                "profiles": sorted(report["by_profile"]),
                "profile_consistency_issues": len(
                    report["profile_consistency_issues"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
