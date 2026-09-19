# 修改时间：2026-09-19。
# 修改目的：用真实标注图片核验本机 V2 TensorRT 与源 PT 的数值及指标一致性。
# 修改内容：支持外置验证清单与图片根目录，保存逐帧对照、每种 FOV 指标和资源哈希。
"""以真实验证图片比较 V2 FP32 PT 与本机 raw 两类 TensorRT。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from personal_hf2026.paths import PROJECT_ROOT
from personal_hf2026.yolo_offline_eval import MetricAccumulator, normalize_predictions
from . import CONFIG_PATH
from .realtime_backend import RealtimeDetector, sha256
from .temporal_tracker import overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".predictions.jsonl").exists():
        raise FileExistsError("请使用新的验证报告路径")
    config = json.loads(args.config.read_text(encoding="utf8"))
    weights = args.weights or PROJECT_ROOT / config["weights"]
    metadata_path = args.engine.with_suffix(".engine.json")
    engine_metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    if sha256(weights) != engine_metadata["weights_sha256"]:
        raise ValueError("engine 与所选 PT 的来源哈希不一致")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf8").splitlines() if line.strip()]
    if args.max_frames:
        indices = np.linspace(0, len(rows) - 1, min(args.max_frames, len(rows)), dtype=int)
        rows = [rows[index] for index in indices]
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    shape = config["detector"]["shape"]
    threshold = float(config["tracker"]["high"])
    unknown = float(config["unknown_threshold"])
    trt = RealtimeDetector(args.engine, shape=shape, device=args.device, confidence=.001)
    pt = RealtimeDetector(weights, shape=shape, device=args.device, half=False, confidence=.001)
    trt.warmup()
    pt.warmup()
    accumulators = defaultdict(lambda: {"fp32": MetricAccumulator(.5), "fp16": MetricAccumulator(.5)})
    ious, score_differences, class_differences = [], [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def output(array):
        result = []
        for item in array[array[:, 4] >= threshold]:
            class_id = int(item[5:7].argmax())
            class_name = ("real_vehicle", "model_prop")[class_id] if item[5:7].max() >= unknown else "uncertain"
            result.append({"bbox_xyxy": item[:4].tolist(), "score": float(item[4]),
                           "class_id": class_id, "class_name": class_name,
                           "class_probabilities": item[5:7].tolist()})
        return result

    with args.output.with_suffix(".predictions.jsonl").open("w", encoding="utf8") as log:
        for row in rows:
            image_path = args.image_root / row["image"]
            if row.get("image_sha256") and sha256(image_path) != row["image_sha256"]:
                raise ValueError(f"验证图片哈希不一致：{image_path}")
            image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法解码验证图片：{image_path}")
            started = time.perf_counter()
            before = pt.predict(image)
            pt_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            after = trt.predict(image)
            trt_ms = (time.perf_counter() - started) * 1000
            raw_outputs = {"fp32": output(before), "fp16": output(after)}
            truth = [{"bbox_xyxy": item["xyxy"], "class_id": item["class_id"]} for item in row["objects"]]
            for backend, step_ms in (("fp32", pt_ms), ("fp16", trt_ms)):
                predictions = normalize_predictions(raw_outputs[backend], image.shape[1], image.shape[0], threshold)
                accumulators[row["domain"]][backend].add(truth, predictions, step_ms, 0.)
            for box in before[before[:, 4] >= threshold]:
                if not len(after):
                    ious.append(0.)
                    continue
                pair_ious = overlaps(box[None, :4], after[:, :4])[0]
                matched = int(pair_ious.argmax())
                ious.append(float(pair_ious[matched]))
                score_differences.append(abs(float(box[4] - after[matched, 4])))
                class_differences.append(float(np.max(np.abs(box[5:7] - after[matched, 5:7]))))
            log.write(json.dumps({"key": row.get("key"), "domain": row["domain"],
                                  "image": str(image_path.resolve()), "fp32": raw_outputs["fp32"],
                                  "fp16": raw_outputs["fp16"], "fp32_ms": pt_ms, "fp16_ms": trt_ms}) + "\n")
    domains = {}
    for domain, accum in accumulators.items():
        metrics = {backend: value.result() for backend, value in accum.items()}
        compact = {
            backend: {
                "detection_f1": value["localization_class_agnostic"]["f1"],
                "macro_class_f1": sum(item["f1"] for item in value["accepted_by_class"].values()) / 2,
            } for backend, value in metrics.items()
        }
        domains[domain] = {**compact, "delta": {name: compact["fp16"][name] - compact["fp32"][name]
                                                for name in compact["fp32"]}, "metrics": metrics}
    fraction = float(np.mean(np.asarray(ious) >= .9)) if ious else 0.
    passed = (bool(ious) and fraction >= .99 and {"fov30", "fov48"}.issubset(domains)
              and all(delta >= -.005 for domain in domains.values() for delta in domain["delta"].values()))
    report = {
        "status": "passed" if passed else "failed", "frames": len(rows), "domains": domains,
        "confidence": threshold, "unknown_threshold": unknown, "high_score_boxes": len(ious),
        "fraction_iou_ge_09": fraction, "minimum_matched_iou": min(ious, default=None),
        "maximum_score_difference": max(score_differences, default=None),
        "maximum_class_probability_difference": max(class_differences, default=None),
        "weights_sha256": sha256(weights), "engine_sha256": sha256(args.engine),
        "engine_metadata_sha256": sha256(metadata_path), "manifest_sha256": sha256(args.manifest),
        "config_sha256": sha256(args.config), "input_shape": [1, 3, *shape],
        "scorer": "personal_hf2026.yolo_offline_eval.MetricAccumulator; IoU 0.5; accepted class decisions",
        "acceptance": "FOV30/FOV48 检测及宏平均类别 F1 降幅各不超过 0.5 个百分点，且至少 99% 的高分 PT 框可匹配 IoU>=0.9 的 engine 框。",
        "scope": "真实验证子集的量化回归检查；不能替代在线 UE 或完整测试集结果。",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": report["status"], "frames": len(rows), "fraction_iou_ge_09": fraction,
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
