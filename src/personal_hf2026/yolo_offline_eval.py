# 修改时间：2026-09-18。
# 修改目的：为双类别 YOLO 提供可替换、可复现的离线整帧评估入口。
# 修改内容：实现 manifest 校验、推理适配、逐帧预测保存及按天气汇总的检测与纯算法耗时指标。
"""在 eval_frames.jsonl 上评估真车/诱饵双类别检测器。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback


SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 1
CLASS_NAMES = {0: "real_vehicle", 1: "model_prop"}
REQUIRED_FRAME_FIELDS = {
    "schema_version",
    "dataset_id",
    "fov_deg",
    "weather",
    "sample_id",
    "sequence_id",
    "sequence_index",
    "timestamp_s",
    "image_path",
    "width",
    "height",
    "category",
    "gt_objects",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value, field):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} 不是有限数：{value!r}")
    return result


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def find_allowed_output_root(start):
    start = Path(start).resolve()
    for base in (start, *start.parents):
        candidate = base / "output"
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("从当前目录向上未找到既有 output 目录")


def prepare_output(path):
    output = Path(path).resolve()
    allowed = find_allowed_output_root(Path.cwd())
    if output == allowed or not is_relative_to(output, allowed):
        raise ValueError(f"输出必须位于 {allowed} 的新子目录内：{output}")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖既有输出：{output}")
    output.mkdir(parents=True)
    return output, allowed


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    return str(value)


class ManifestAudit:
    def __init__(self):
        self.frames = 0
        self.objects = 0
        self.datasets = Counter()
        self.fovs = Counter()
        self.weathers = Counter()
        self.categories = Counter()
        self.classes = Counter()
        self.edge_clipped_objects = 0
        self.exposure_time_verified_frames = 0
        self.sequences = set()
        self.sample_ids = set()
        self.last_sequence_id = None
        self.closed_sequences = set()
        self.sequence_next_index = {}
        self.sequence_last_timestamp = {}

    def add(self, row, line_number, *, verify_image_path=False):
        missing = sorted(REQUIRED_FRAME_FIELDS - set(row))
        if missing:
            raise ValueError(f"manifest 第 {line_number} 行缺少字段：{missing}")
        if int(row["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"manifest 第 {line_number} 行 schema_version 非 {SCHEMA_VERSION}")
        sample_id = str(row["sample_id"])
        if sample_id in self.sample_ids:
            raise ValueError(f"manifest 第 {line_number} 行 sample_id 重复：{sample_id}")
        self.sample_ids.add(sample_id)

        width, height = int(row["width"]), int(row["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"manifest 第 {line_number} 行图片尺寸非法：{width}x{height}")
        fov_deg = finite_float(row["fov_deg"], "fov_deg")
        timestamp_s = finite_float(row["timestamp_s"], "timestamp_s")
        sequence_id = str(row["sequence_id"])
        sequence_index = int(row["sequence_index"])
        expected = self.sequence_next_index.get(sequence_id, 0)
        if sequence_index != expected:
            raise ValueError(
                f"manifest 第 {line_number} 行序列索引不连续：{sequence_id}，"
                f"期望 {expected}，实际 {sequence_index}"
            )
        if sequence_id in self.sequence_last_timestamp:
            if timestamp_s <= self.sequence_last_timestamp[sequence_id]:
                raise ValueError(f"manifest 第 {line_number} 行序列时间未递增：{sequence_id}")
        if sequence_id != self.last_sequence_id:
            if sequence_id in self.closed_sequences:
                raise ValueError(f"manifest 第 {line_number} 行序列非连续重现：{sequence_id}")
            if self.last_sequence_id is not None:
                self.closed_sequences.add(self.last_sequence_id)
            self.last_sequence_id = sequence_id
        self.sequence_next_index[sequence_id] = expected + 1
        self.sequence_last_timestamp[sequence_id] = timestamp_s
        self.sequences.add(sequence_id)

        objects = row["gt_objects"]
        if not isinstance(objects, list):
            raise ValueError(f"manifest 第 {line_number} 行 gt_objects 不是列表")
        present_classes = set()
        for object_index, item in enumerate(objects):
            class_id = int(item["class_id"])
            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"manifest 第 {line_number} 行对象 {object_index} class_id 非法：{class_id}"
                )
            if item.get("class_name") != CLASS_NAMES[class_id]:
                raise ValueError(
                    f"manifest 第 {line_number} 行对象 {object_index} 类别名不一致"
                )
            bbox = item.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"manifest 第 {line_number} 行对象 {object_index} bbox 非法")
            x1, y1, x2, y2 = [finite_float(value, "bbox_xyxy") for value in bbox]
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                raise ValueError(
                    f"manifest 第 {line_number} 行对象 {object_index} bbox 越界或为空：{bbox}"
                )
            present_classes.add(class_id)
            self.classes[CLASS_NAMES[class_id]] += 1
            self.edge_clipped_objects += int(bool(item.get("edge_clipped")))
        expected_category = (
            "no_vehicle" if not present_classes else
            "target_only" if present_classes == {0} else
            "decoy_only" if present_classes == {1} else
            "mixed"
        )
        if row["category"] != expected_category:
            raise ValueError(
                f"manifest 第 {line_number} 行 category 不一致："
                f"{row['category']} != {expected_category}"
            )
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            raise ValueError(f"manifest 第 {line_number} 行 image_path 不是绝对路径")
        if verify_image_path and not image_path.is_file():
            raise FileNotFoundError(f"manifest 第 {line_number} 行图片不存在：{image_path}")

        self.frames += 1
        self.objects += len(objects)
        self.datasets[str(row["dataset_id"])] += 1
        self.fovs[str(fov_deg)] += 1
        self.weathers[str(row["weather"])] += 1
        self.categories[str(row["category"])] += 1
        self.exposure_time_verified_frames += int(bool(row.get("exposure_time_verified")))

    def result(self, *, truncated):
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "completed",
            "truncated_by_max_frames": bool(truncated),
            "frames": self.frames,
            "objects": self.objects,
            "sequences": len(self.sequences),
            "datasets": dict(sorted(self.datasets.items())),
            "fov_deg_frames": dict(sorted(self.fovs.items())),
            "weather_frames": dict(sorted(self.weathers.items())),
            "categories": dict(sorted(self.categories.items())),
            "classes": dict(sorted(self.classes.items())),
            "edge_clipped_objects": self.edge_clipped_objects,
            "exposure_time_verified_frames": self.exposure_time_verified_frames,
        }


def iter_manifest(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def bbox_iou(left, right):
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(gt_objects, predictions, threshold, *, same_class):
    candidates = []
    for gt_index, gt in enumerate(gt_objects):
        for prediction_index, prediction in enumerate(predictions):
            if same_class and gt["class_id"] != prediction["class_id"]:
                continue
            iou = bbox_iou(gt["bbox_xyxy"], prediction["bbox_xyxy"])
            if iou >= threshold:
                candidates.append((iou, gt_index, prediction_index))
    candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
    matched_gt = set()
    matched_predictions = set()
    matches = []
    for iou, gt_index, prediction_index in candidates:
        if gt_index in matched_gt or prediction_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(prediction_index)
        matches.append({
            "gt_index": gt_index,
            "prediction_index": prediction_index,
            "iou": iou,
        })
    return matches, matched_gt, matched_predictions


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def detection_metrics(tp, fp, fn):
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_ratio(2 * precision * recall, precision + recall),
    }


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def timing_metrics(values):
    if not values:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
            "fps": None,
        }
    total = sum(values)
    return {
        "count": len(values),
        "total_ms": total,
        "mean_ms": total / len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "fps": len(values) * 1000.0 / total if total > 0 else None,
    }


class MetricAccumulator:
    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold
        self.frames = 0
        self.gt_objects = 0
        self.predictions = 0
        self.accepted_predictions = 0
        self.uncertain_predictions = 0
        self.localization_tp = 0
        self.class_aware_tp = 0
        self.accepted_class_aware_tp = 0
        self.class_gt = Counter()
        self.class_predictions = Counter()
        self.accepted_class_predictions = Counter()
        self.class_tp = Counter()
        self.accepted_class_tp = Counter()
        self.confusion = Counter()
        self.decision_confusion = Counter()
        self.false_positive_frames = 0
        self.accepted_false_positive_frames = 0
        self.empty_gt_false_positives = 0
        self.accepted_empty_gt_false_positives = 0
        self.step_ms = []
        self.decode_ms = []

    def add(self, gt_objects, predictions, step_ms, decode_ms):
        localization, _, _ = greedy_match(
            gt_objects, predictions, self.iou_threshold, same_class=False
        )
        class_aware, _, class_matched_predictions = greedy_match(
            gt_objects, predictions, self.iou_threshold, same_class=True
        )
        accepted_source_indices = [
            index for index, item in enumerate(predictions) if item["accepted"]
        ]
        accepted_predictions = [predictions[index] for index in accepted_source_indices]
        accepted_class_aware, _, accepted_matched_predictions = greedy_match(
            gt_objects, accepted_predictions, self.iou_threshold, same_class=True
        )
        false_positives = len(predictions) - len(class_matched_predictions)
        accepted_false_positives = (
            len(accepted_predictions) - len(accepted_matched_predictions)
        )
        self.frames += 1
        self.gt_objects += len(gt_objects)
        self.predictions += len(predictions)
        self.accepted_predictions += len(accepted_predictions)
        self.uncertain_predictions += len(predictions) - len(accepted_predictions)
        self.localization_tp += len(localization)
        self.class_aware_tp += len(class_aware)
        self.accepted_class_aware_tp += len(accepted_class_aware)
        self.false_positive_frames += int(false_positives > 0)
        self.accepted_false_positive_frames += int(accepted_false_positives > 0)
        if not gt_objects:
            self.empty_gt_false_positives += false_positives
            self.accepted_empty_gt_false_positives += accepted_false_positives
        self.step_ms.append(step_ms)
        self.decode_ms.append(decode_ms)
        for item in gt_objects:
            self.class_gt[item["class_id"]] += 1
        for item in predictions:
            self.class_predictions[item["class_id"]] += 1
        for item in accepted_predictions:
            self.accepted_class_predictions[item["class_id"]] += 1
        for match in class_aware:
            class_id = gt_objects[match["gt_index"]]["class_id"]
            self.class_tp[class_id] += 1
        for match in accepted_class_aware:
            class_id = gt_objects[match["gt_index"]]["class_id"]
            self.accepted_class_tp[class_id] += 1
        for match in localization:
            gt_class = gt_objects[match["gt_index"]]["class_id"]
            prediction = predictions[match["prediction_index"]]
            prediction_class = prediction["class_id"]
            self.confusion[(gt_class, prediction_class)] += 1
            self.decision_confusion[(gt_class, prediction["decision_name"])] += 1
        accepted_matches_with_source_index = [
            {
                **match,
                "prediction_index": accepted_source_indices[match["prediction_index"]],
            }
            for match in accepted_class_aware
        ]
        return localization, class_aware, accepted_matches_with_source_index

    def result(self):
        localization = detection_metrics(
            self.localization_tp,
            self.predictions - self.localization_tp,
            self.gt_objects - self.localization_tp,
        )
        class_aware = detection_metrics(
            self.class_aware_tp,
            self.predictions - self.class_aware_tp,
            self.gt_objects - self.class_aware_tp,
        )
        accepted_class_aware = detection_metrics(
            self.accepted_class_aware_tp,
            self.accepted_predictions - self.accepted_class_aware_tp,
            self.gt_objects - self.accepted_class_aware_tp,
        )
        by_class = {}
        accepted_by_class = {}
        for class_id, class_name in CLASS_NAMES.items():
            tp = self.class_tp[class_id]
            by_class[class_name] = detection_metrics(
                tp,
                self.class_predictions[class_id] - tp,
                self.class_gt[class_id] - tp,
            )
            accepted_tp = self.accepted_class_tp[class_id]
            accepted_by_class[class_name] = detection_metrics(
                accepted_tp,
                self.accepted_class_predictions[class_id] - accepted_tp,
                self.class_gt[class_id] - accepted_tp,
            )
        confusion = {
            CLASS_NAMES[gt_class]: {
                CLASS_NAMES[prediction_class]: self.confusion[(gt_class, prediction_class)]
                for prediction_class in CLASS_NAMES
            }
            for gt_class in CLASS_NAMES
        }
        decision_confusion = {
            CLASS_NAMES[gt_class]: {
                decision_name: self.decision_confusion[(gt_class, decision_name)]
                for decision_name in (*CLASS_NAMES.values(), "uncertain")
            }
            for gt_class in CLASS_NAMES
        }
        return {
            "frames": self.frames,
            "ground_truth_objects": self.gt_objects,
            "predictions": self.predictions,
            "accepted_predictions": self.accepted_predictions,
            "uncertain_predictions": self.uncertain_predictions,
            "uncertain_prediction_rate": safe_ratio(
                self.uncertain_predictions, self.predictions
            ),
            "localization_class_agnostic": localization,
            "detection_class_aware_forced_class_id": class_aware,
            "by_class_forced_class_id": by_class,
            "accepted_detection_class_aware": accepted_class_aware,
            "accepted_by_class": accepted_by_class,
            "localized_pair_forced_class_confusion": confusion,
            "localized_pair_decision_confusion": decision_confusion,
            "false_positive_analysis": {
                "forced_class_id": {
                    "false_positives": class_aware["fp"],
                    "false_positives_per_frame": safe_ratio(
                        class_aware["fp"], self.frames
                    ),
                    "frames_with_false_positive": self.false_positive_frames,
                    "empty_gt_false_positives": self.empty_gt_false_positives,
                },
                "accepted_only": {
                    "false_positives": accepted_class_aware["fp"],
                    "false_positives_per_frame": safe_ratio(
                        accepted_class_aware["fp"], self.frames
                    ),
                    "frames_with_false_positive": self.accepted_false_positive_frames,
                    "empty_gt_false_positives": self.accepted_empty_gt_false_positives,
                },
            },
            "timing": {
                "algorithm_step": timing_metrics(self.step_ms),
                "image_decode_io": timing_metrics(self.decode_ms),
            },
        }


def load_detector(spec, config, device):
    if ":" not in spec:
        raise ValueError("--detector 必须是 module:factory")
    module_name, factory_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    started = time.perf_counter()
    detector = factory(config=None if config is None else str(config), device=device)
    initialization_ms = (time.perf_counter() - started) * 1000.0
    if not callable(getattr(detector, "predict", None)):
        raise TypeError("detector 必须提供 predict(image_bgr, timestamp, sequence_id)")
    return detector, initialization_ms


def detector_runtime_metadata(detector):
    value = getattr(detector, "runtime_metadata", None)
    if callable(value):
        value = value()
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("detector.runtime_metadata 必须是字典或返回字典的 callable")
    return jsonable(value)


def synchronize_cuda(device):
    if str(device).lower() == "cpu":
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def normalize_predictions(result, width, height, confidence_threshold):
    if isinstance(result, dict):
        result = result.get("detections")
    if not isinstance(result, (list, tuple)):
        raise TypeError("predict 返回值必须是 detection 列表或包含 detections 的字典")
    predictions = []
    for index, raw in enumerate(result):
        if not isinstance(raw, dict):
            raise TypeError(f"第 {index} 个 detection 不是字典")
        class_id = int(raw["class_id"])
        if class_id not in CLASS_NAMES:
            raise ValueError(f"第 {index} 个 detection class_id 非法：{class_id}")
        bbox = raw.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"第 {index} 个 detection bbox_xyxy 非法")
        bbox = [finite_float(value, "prediction.bbox_xyxy") for value in bbox]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"第 {index} 个 detection bbox 为空：{bbox}")
        score = finite_float(raw["score"], "prediction.score")
        if not 0 <= score <= 1:
            raise ValueError(f"第 {index} 个 detection score 不在 [0, 1]：{score}")
        if score < confidence_threshold:
            continue
        decision_name = str(raw.get("class_name", CLASS_NAMES[class_id]))
        if decision_name not in (*CLASS_NAMES.values(), "uncertain"):
            raise ValueError(
                f"第 {index} 个 detection class_name 非法：{decision_name!r}"
            )
        prediction = jsonable(raw)
        prediction.update({
            "bbox_xyxy": bbox,
            "score": score,
            "class_id": class_id,
            "class_name": decision_name,
            "class_id_name": CLASS_NAMES[class_id],
            "decision_name": decision_name,
            "accepted": decision_name != "uncertain",
            "bbox_outside_image": bool(
                bbox[0] < 0 or bbox[1] < 0 or bbox[2] > width or bbox[3] > height
            ),
        })
        predictions.append(prediction)
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions


def evaluate(args, output, manifest_sha256):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("实际推理需要安装 opencv-python") from error

    detector, initialization_ms = load_detector(args.detector, args.detector_config, args.device)
    runtime_metadata = detector_runtime_metadata(detector)
    audit = ManifestAudit()
    overall = MetricAccumulator(args.iou_threshold)
    by_weather = defaultdict(lambda: MetricAccumulator(args.iou_threshold))
    predictions_path = output / "predictions.jsonl"
    temporary_predictions = output / "predictions.jsonl.tmp"
    current_sequence_id = None
    truncated = False

    with temporary_predictions.open("w", encoding="utf-8", newline="\n") as prediction_stream:
        for line_number, row in iter_manifest(args.manifest):
            if args.max_frames and audit.frames >= args.max_frames:
                truncated = True
                break
            audit.add(row, line_number, verify_image_path=True)
            sequence_id = str(row["sequence_id"])
            if sequence_id != current_sequence_id:
                reset = getattr(detector, "reset", None)
                if callable(reset):
                    reset()
                current_sequence_id = sequence_id

            decode_started = time.perf_counter()
            image_bgr = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if image_bgr is None:
                raise ValueError(f"图片解码失败：{row['image_path']}")
            actual_height, actual_width = image_bgr.shape[:2]
            if actual_width != int(row["width"]) or actual_height != int(row["height"]):
                raise ValueError(
                    f"图片尺寸与 manifest 不一致：{row['image_path']}，"
                    f"实际 {actual_width}x{actual_height}，"
                    f"记录 {row['width']}x{row['height']}"
                )
            if args.verify_image_sha256:
                actual_sha256 = file_sha256(row["image_path"])
                if actual_sha256 != row.get("image_sha256"):
                    raise ValueError(f"图片 SHA-256 不一致：{row['image_path']}")

            synchronize_cuda(args.device)
            step_started = time.perf_counter()
            result = detector.predict(
                image_bgr,
                timestamp=float(row["timestamp_s"]),
                sequence_id=sequence_id,
            )
            synchronize_cuda(args.device)
            step_ms = (time.perf_counter() - step_started) * 1000.0
            predictions = normalize_predictions(
                result,
                int(row["width"]),
                int(row["height"]),
                args.confidence_threshold,
            )
            gt_objects = [
                {
                    "object_id": item.get("object_id"),
                    "class_id": int(item["class_id"]),
                    "class_name": item["class_name"],
                    "bbox_xyxy": [float(value) for value in item["bbox_xyxy"]],
                    "edge_clipped": bool(item.get("edge_clipped")),
                }
                for item in row["gt_objects"]
            ]
            localization, class_aware, accepted_class_aware = overall.add(
                gt_objects, predictions, step_ms, decode_ms
            )
            by_weather[str(row["weather"])].add(
                gt_objects, predictions, step_ms, decode_ms
            )
            prediction_row = {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "source_manifest_sha256": manifest_sha256,
                "source_line": line_number,
                "dataset_id": row["dataset_id"],
                "fov_deg": row["fov_deg"],
                "weather": row["weather"],
                "sample_id": row["sample_id"],
                "sequence_id": sequence_id,
                "sequence_index": row["sequence_index"],
                "timestamp_s": row["timestamp_s"],
                "image_path": row["image_path"],
                "gt_objects": gt_objects,
                "predictions": predictions,
                "localization_matches": localization,
                "class_aware_matches": class_aware,
                "accepted_class_aware_matches": accepted_class_aware,
                "step_ms": step_ms,
                "decode_ms": decode_ms,
            }
            prediction_stream.write(
                json.dumps(prediction_row, ensure_ascii=False, allow_nan=False) + "\n"
            )
    temporary_predictions.replace(predictions_path)
    metrics = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "completed",
        "created_at": utc_now(),
        "matching": {
            "algorithm": "greedy_descending_iou_one_to_one",
            "iou_threshold": args.iou_threshold,
            "confidence_threshold": args.confidence_threshold,
            "localization_ignores_class": True,
            "class_aware_requires_same_class": True,
        },
        "metric_semantics": {
            "localization_class_agnostic": "所有预测均参与，不区分类别与 uncertain 决策",
            "forced_class_id": "所有预测按 class_id 强制二分类，含 class_name=uncertain",
            "accepted": "只保留 class_name 非 uncertain 的最终接受预测",
        },
        "timing_boundary": {
            "algorithm_step_ms": "仅 detector.predict；图片读取、解码、manifest 解析、计分、序列 reset 与模型初始化均不计入",
            "cuda_synchronize": "device 非 cpu 且 torch CUDA 可用时，在计时前后同步",
            "model_initialization_ms": initialization_ms,
        },
        "detector_runtime_metadata": runtime_metadata,
        "overall": overall.result(),
        "by_weather": {
            weather: accumulator.result()
            for weather, accumulator in sorted(by_weather.items())
        },
    }
    return audit.result(truncated=truncated), metrics


def audit_manifest(args):
    audit = ManifestAudit()
    truncated = False
    for line_number, row in iter_manifest(args.manifest):
        if args.max_frames and audit.frames >= args.max_frames:
            truncated = True
            break
        audit.add(row, line_number, verify_image_path=args.verify_image_paths)
    return audit.result(truncated=truncated)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument(
        "--detector",
        default="personal_hf2026.vehicle_prop:create_detector",
        help="推理工厂 module:factory；仅 --manifest-only 时不导入",
    )
    result.add_argument("--detector-config", type=Path)
    result.add_argument("--device", default="0")
    result.add_argument("--iou-threshold", type=float, default=0.5)
    result.add_argument("--confidence-threshold", type=float, default=0.25)
    result.add_argument("--max-frames", type=int, default=0, help="0 表示处理完整 manifest")
    result.add_argument("--manifest-only", action="store_true", help="只校验并汇总 manifest")
    result.add_argument(
        "--verify-image-paths",
        action="store_true",
        help="manifest-only 时额外检查每张图片是否存在",
    )
    result.add_argument(
        "--verify-image-sha256",
        action="store_true",
        help="实际推理时逐张复核图片哈希；哈希时间不计入算法耗时",
    )
    return result


def validate_args(args):
    args.manifest = args.manifest.resolve()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest 不存在：{args.manifest}")
    if args.detector_config is not None:
        args.detector_config = args.detector_config.resolve()
        if not args.detector_config.is_file():
            raise FileNotFoundError(f"detector config 不存在：{args.detector_config}")
    if not 0 < args.iou_threshold <= 1:
        raise ValueError("--iou-threshold 必须在 (0, 1] 内")
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("--confidence-threshold 必须在 [0, 1] 内")
    if args.max_frames < 0:
        raise ValueError("--max-frames 不能为负数")


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    output, allowed_output_root = prepare_output(args.output)
    manifest_sha256 = file_sha256(args.manifest)
    run = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "running",
        "started_at": utc_now(),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "evaluator_source": str(Path(__file__).resolve()),
        "evaluator_source_sha256": file_sha256(__file__),
        "output": str(output),
        "allowed_output_root": str(allowed_output_root),
        "mode": "manifest_only" if args.manifest_only else "evaluation",
        "detector": None if args.manifest_only else args.detector,
        "detector_config": None if args.detector_config is None else str(args.detector_config),
        "detector_config_sha256": (
            None if args.detector_config is None else file_sha256(args.detector_config)
        ),
        "device": None if args.manifest_only else args.device,
        "iou_threshold": args.iou_threshold,
        "confidence_threshold": args.confidence_threshold,
        "max_frames": args.max_frames,
        "verify_image_paths": args.verify_image_paths,
        "verify_image_sha256": args.verify_image_sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv if argv is None else [sys.argv[0], *argv],
    }
    write_json(output / "run.json", run)
    try:
        if args.manifest_only:
            manifest_summary = audit_manifest(args)
            metrics = None
        else:
            manifest_summary, metrics = evaluate(args, output, manifest_sha256)
            write_json(output / "metrics.json", metrics)
        write_json(output / "manifest_summary.json", manifest_summary)
        run.update({
            "status": "completed",
            "completed_at": utc_now(),
            "processed_frames": manifest_summary["frames"],
            "predictions": None if args.manifest_only else str(output / "predictions.jsonl"),
            "metrics": None if args.manifest_only else str(output / "metrics.json"),
        })
        write_json(output / "run.json", run)
        print(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as error:
        failure = {
            "status": "failed",
            "failed_at": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "pipeline_error.json", failure)
        run.update(failure)
        write_json(output / "run.json", run)
        raise


if __name__ == "__main__":
    main()
