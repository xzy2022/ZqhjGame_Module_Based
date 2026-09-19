# 修改时间：2026-09-19。
# 修改目的：用统一离线口径复核在线二分类偏置并保存可回放的错分证据。
# 修改内容：新增按时间窗口的混淆统计、错分图和 HTML 索引以及同帧离线评估清单导出。
"""复核一个旁路实验，不启动仿真或模型推理。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path

from personal_hf2026.yolo_offline_eval import (
    CLASS_NAMES, ManifestAudit, MetricAccumulator, normalize_predictions,
    prepare_output, write_json,
)


GT_CLASSES = {"TargetVehicle": 0, "DecoyVehicle": 1}


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def load_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def truth_objects(row, counts):
    result = []
    for item in row.get("ground_truth", []):
        bbox = item.get("bbox", item.get("bbox_xyxy"))
        if item.get("class") not in GT_CLASSES:
            counts["unknown_gt_classes"] += 1
            continue
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(finite(value) for value in bbox)
                or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            counts["invalid_gt_boxes"] += 1
            continue
        class_id = GT_CLASSES[item["class"]]
        result.append({
            "object_id": str(item.get("target_id", "")),
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "bbox_xyxy": [float(value) for value in bbox],
            "edge_clipped": False,
        })
    return result


def classification_rates(metric):
    confusion = metric["localized_pair_forced_class_confusion"]
    return {
        name: {
            "localized_pairs": sum(confusion[name].values()),
            "correct": confusion[name][name],
            "accuracy": (confusion[name][name] / sum(confusion[name].values())
                         if sum(confusion[name].values()) else None),
        }
        for name in CLASS_NAMES.values()
    }


def finish(accumulator):
    result = accumulator.result()
    result["localized_pair_forced_class_accuracy"] = classification_rates(result)
    return result


def pct(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def image_path_for(row, run):
    if not row.get("image_path"):
        return None
    path = Path(row["image_path"])
    return path if path.is_absolute() else run / path


def draw_error(row, truth, predictions, mismatches, source, destination):
    # 使用实际处理帧作诊断标注，顶部文字单独留出区域以免遮挡图像。
    from PIL import Image, ImageDraw, ImageFont
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 78), (22, 27, 35))
    canvas.paste(image, (0, 78))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/consola.ttf")
    font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    title = (f"uid={row.get('uid')} frame={row.get('frame_no')} "
             f"received={row.get('image_received_sim_time', 0):.3f}s")
    draw.text((10, 8), title, fill="white", font=font)
    draw.text((10, 30), "GT green | Prediction blue | Wrong class red | 0=real 1=decoy", fill="white", font=font)
    draw.text((10, 52), "UE projected audit boxes; exposure alignment is not verified", fill=(190, 190, 190), font=font)
    wrong_predictions = {match["prediction_index"] for match in mismatches}
    for index, item in enumerate(truth):
        x1, y1, x2, y2 = item["bbox_xyxy"]
        y1 += 78
        y2 += 78
        draw.rectangle((x1, y1, x2, y2), outline=(50, 230, 80), width=2)
        label = f"G{index}:{item['class_id']} id={item['object_id']}"
        draw.text((max(0, x1), max(78, y1 - 17)), label, fill=(50, 230, 80), font=font, stroke_width=1, stroke_fill="black")
    for index, item in enumerate(predictions):
        x1, y1, x2, y2 = item["bbox_xyxy"]
        y1 += 78
        y2 += 78
        color = (255, 75, 70) if index in wrong_predictions else (65, 165, 255)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        label = f"P{index}:{item['class_id']} {item['decision_name']} s={item['score']:.2f} h={item.get('track_hits', '?')}"
        draw.text((max(0, x1), min(canvas.height - 19, y2 + 2)), label, fill=color, font=font, stroke_width=1, stroke_fill="black")
    canvas.save(destination, quality=91)


def export_manifest(records, run, output, weather, counts):
    # 逐无人机保持实际处理顺序，并在在线已重置的边界切段，避免重放跨机共享跟踪器。
    streams = defaultdict(list)
    for row, truth in records:
        source = image_path_for(row, run)
        if source is not None and source.is_file():
            streams[str(row.get("uid", row.get("uav_id")))].append((row, truth, source))
    audit = ManifestAudit()
    with (output / "eval_frames.jsonl").open("w", encoding="utf-8") as stream:
        for uid, items in sorted(streams.items()):
            segment, sequence_index, previous_timestamp = 0, 0, None
            for row, truth, source in items:
                timestamp = row.get("source_sim_time")
                if not finite(timestamp):
                    timestamp = row["image_received_sim_time"]
                if previous_timestamp is not None and (
                    timestamp <= previous_timestamp or row.get("tracker_reset_reason")
                ):
                    segment += 1
                    sequence_index = 0
                previous_timestamp = timestamp
                width, height = int(row["image_width"]), int(row["image_height"])
                exported_truth = []
                for item in truth:
                    bbox = item["bbox_xyxy"]
                    clipped = [max(0., bbox[0]), max(0., bbox[1]), min(float(width), bbox[2]), min(float(height), bbox[3])]
                    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                        counts["manifest_empty_clipped_gt"] += 1
                        continue
                    exported_truth.append(dict(item, bbox_xyxy=clipped, edge_clipped=clipped != bbox))
                    counts["manifest_clipped_gt"] += clipped != bbox
                classes = {item["class_id"] for item in exported_truth}
                category = ("no_vehicle" if not classes else "target_only" if classes == {0}
                            else "decoy_only" if classes == {1} else "mixed")
                exported = {
                    "schema_version": 1, "dataset_id": run.name,
                    "fov_deg": row.get("observed_gimbal_fov_deg", 48.0), "weather": weather,
                    "sample_id": f"{run.name}-{uid}-{row.get('job_id', row.get('submission_id'))}",
                    "sequence_id": f"{run.name}-{uid}-{segment}", "sequence_index": sequence_index,
                    "timestamp_s": timestamp, "source_sim_time": row.get("source_sim_time"),
                    "image_received_sim_time": row.get("image_received_sim_time"),
                    "image_path": str(source.resolve()), "image_sha256": row.get("image_sha256"),
                    "width": width, "height": height, "category": category,
                    "gt_objects": exported_truth, "exposure_time_verified": False,
                    "ground_truth_source": row.get("ground_truth_source"),
                    "source_submission_id": row.get("submission_id"),
                }
                audit.add(exported, audit.frames + 1, verify_image_path=True)
                stream.write(json.dumps(exported, ensure_ascii=False) + "\n")
                sequence_index += 1
    return audit.result(truncated=False)


def analyze(args, output):
    input_path = args.input.resolve()
    result_path = input_path / "yolo_sidecar_results.jsonl" if input_path.is_dir() else input_path
    run = result_path.parent
    rows = load_rows(result_path)
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    weather = args.weather or metadata.get("weather") or "unknown"
    if isinstance(weather, dict):
        weather = weather.get("preset", "unknown")
    accumulator = MetricAccumulator(args.iou_threshold)
    windows = {end: MetricAccumulator(args.iou_threshold) for end in args.windows}
    normalized_windows = {end: MetricAccumulator(args.iou_threshold) for end in args.windows}
    counts = Counter()
    records, error_cards = [], []
    source_start = min((r["source_sim_time"] for r in rows if finite(r.get("source_sim_time"))), default=None)
    pair_fields = ("source_sim_time", "image_received_sim_time", "frame_no", "uid", "job_id", "image_sha256")
    visual_dir = output / "misclassified"
    visual_dir.mkdir()
    with (output / "localized_pairs.jsonl").open("w", encoding="utf-8") as pair_stream:
        for row in rows:
            truth = truth_objects(row, counts)
            predictions = normalize_predictions(row.get("predictions", []), int(row["image_width"]), int(row["image_height"]), args.confidence_threshold)
            counts["predictions_below_score_threshold"] += len(row.get("predictions", [])) - len(predictions)
            matches, _, _ = accumulator.add(truth, predictions, float(row.get("inference_wall_ms", 0)), float(row.get("decode_ms", 0)))
            received = row.get("image_received_sim_time")
            source_time = row.get("source_sim_time")
            for end in args.windows:
                if finite(received) and 0 <= received <= end:
                    windows[end].add(truth, predictions, 0, 0)
                if finite(source_time) and source_start is not None and 0 <= source_time - source_start <= end:
                    normalized_windows[end].add(truth, predictions, 0, 0)
            mismatches = []
            for match in matches:
                gt, prediction = truth[match["gt_index"]], predictions[match["prediction_index"]]
                wrong = gt["class_id"] != prediction["class_id"]
                if wrong:
                    mismatches.append(match)
                pair = {field: row.get(field) for field in pair_fields}
                pair.update(match)
                pair.update({"gt": gt, "prediction": prediction, "wrong_forced_class": wrong,
                             "image_path": row.get("image_path")})
                pair_stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
            records.append((row, truth))
            counts["tracker_reset_frames"] += bool(row.get("tracker_reset_reason"))
            counts["wrong_class_frames"] += bool(mismatches)
            counts["wrong_class_pairs"] += len(mismatches)
            image_path = image_path_for(row, run)
            if image_path is None or not image_path.is_file():
                counts["missing_image_frames"] += 1
                continue
            if args.verify_image_sha256:
                if hashlib.sha256(image_path.read_bytes()).hexdigest() != row.get("image_sha256"):
                    raise ValueError(f"图片哈希与实际处理帧不一致：{image_path}")
                counts["verified_image_sha256_frames"] += 1
            if mismatches and (args.max_visualizations == 0 or counts["visualized_frames"] < args.max_visualizations):
                name = f"{row.get('uid')}-{row.get('job_id', row.get('submission_id'))}.jpg"
                draw_error(row, truth, predictions, mismatches, image_path, visual_dir / name)
                counts["visualized_frames"] += 1
                error_cards.append((name, row, mismatches))
    manifest = export_manifest(records, run, output, str(weather), counts)
    report = {
        "status": "completed", "input": str(result_path), "weather": weather,
        "iou_threshold": args.iou_threshold, "confidence_threshold": args.confidence_threshold,
        "metric_semantics": {
            "forced": "与离线 MetricAccumulator 完全相同，uncertain 仍按 class_id 强制二分类。",
            "accepted": "去除 uncertain 后重配对，报告类别感知 precision/recall/f1。",
            "online_native": "旧在线 class_aware 按 class_name 比较，uncertain 不能成为TP却保留在预测分母；勿与 forced 混用。",
            "gt": "Redis 同帧 UE 投影框是审计元数据，不等于人工像素真值。",
            "windows": "主窗口按 image_received_sim_time；另报 source 减首个完成帧 source 的窗口，二者均非曝光时间。",
        },
        "overall": finish(accumulator),
        "received_time_windows": {str(end): finish(acc) for end, acc in windows.items()},
        "normalized_source_time_windows": {str(end): finish(acc) for end, acc in normalized_windows.items()},
        "source_time_start": source_start, "counts": dict(counts), "replay_manifest": manifest,
        "visualization_selection": "按完成日志顺序保存前 max_visualizations 个错分帧；0 表示全部。",
    }
    write_json(output / "metrics.json", report)
    lines = [f"# {run.name} 二分类复核", "", "类别口径与 yolo_offline_eval.MetricAccumulator 共用；IoU >= 0.5。", "",
             "| 接收时间窗口 | 完成帧 | 真车配对 | 真车判对 | 诱饵配对 | 诱饵判对 | forced F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    metrics_rows = [(f"0–{end}s", report["received_time_windows"][str(end)]) for end in args.windows]
    metrics_rows.append(("全部", report["overall"]))
    for name, metric in metrics_rows:
        rate = metric["localized_pair_forced_class_accuracy"]
        lines.append(f"| {name} | {metric['frames']} | {rate['real_vehicle']['localized_pairs']} | {pct(rate['real_vehicle']['accuracy'])} | {rate['model_prop']['localized_pairs']} | {pct(rate['model_prop']['accuracy'])} | {pct(metric['detection_class_aware_forced_class_id']['f1'])} |")
    lines.extend(["", f"错分帧 {counts['wrong_class_frames']}，已标注 {counts['visualized_frames']}，无图像路径/文件 {counts['missing_image_frames']}。",
                  f"重放清单 {manifest['frames']} 帧，保留各无人机 source 时间间隔和原始图像哈希。", "",
                  "GT 为 UE 审计投影框；不宣称严格曝光对齐。HTML 中展示前若干错分帧；全部配对见 localized_pairs.jsonl。"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cards = []
    for name, row, wrong in error_cards:
        label = f"uid={row.get('uid')} frame={row.get('frame_no')} received={row.get('image_received_sim_time'):.3f}s，错分 {len(wrong)} 个"
        cards.append(f'<article><p>{html.escape(label)}</p><a href="misclassified/{name}"><img loading="lazy" src="misclassified/{name}"></a></article>')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>YOLO 错分证据</title><style>body{font:16px sans-serif;background:#111923;color:#edf2fa;margin:28px}article{margin:24px 0}img{max-width:1100px;width:100%}pre{white-space:pre-wrap}</style><h1>YOLO 错分证据</h1><pre>' + html.escape("\n".join(lines)) + "</pre>" + "".join(cards) + "</html>"
    (output / "index.html").write_text(page, encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="实验目录或 yolo_sidecar_results.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="既有 output 根下的新目录")
    parser.add_argument("--weather", help="覆盖天气名称；用于导出的清单")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--windows", type=float, nargs="+", default=[20, 40, 60, 120, 200])
    parser.add_argument("--max-visualizations", type=int, default=120, help="最多保存错分图数量；0 为全部")
    parser.add_argument("--verify-image-sha256", action="store_true")
    args = parser.parse_args(argv)
    output, _ = prepare_output(args.output)
    report = analyze(args, output)
    print(json.dumps({"output": str(output), "counts": report["counts"],
                      "forced_accuracy": report["overall"]["localized_pair_forced_class_accuracy"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
