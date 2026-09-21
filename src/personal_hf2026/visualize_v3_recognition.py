# 修改时间：2026-09-21。
# 修改目的：让识别叠图只保留有 UE 或有效 YOLO 证据的帧，并清晰区分两者类别。
# 修改内容：UE 改为浅色实线、YOLO 改为深色虚线，并按预测类别显示正确概率。
# 修改时间：2026-09-21。
# 修改目的：让单局 V3 日志可逐帧核对 YOLO 输出与 UE 投影真值。
# 修改内容：按无人机分目录叠加原图、YOLO 框和仅供审计的 UE 真实 ID 框。
"""把 --save-images --detailed-log 的 V3 运行渲染成逐帧识别对比图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _rows(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{number} 不是合法 JSONL") from exc


def _box(item):
    raw = next((item.get(key) for key in ("bbox", "bbox_xyxy", "xyxy", "box")
                if isinstance(item.get(key), (list, tuple))), None)
    if raw is None or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _ue_label(item):
    identity = next((item.get(key) for key in ("target_id", "uid", "id", "object_id", "name")
                     if item.get(key) is not None), "?")
    category = str(next((item.get(key) for key in ("class", "category", "type", "label")
                         if item.get(key) is not None), "unknown"))
    return str(identity), category


def _ue_color(category: str):
    normalized = category.lower()
    return "#ff8a8a" if ("target" in normalized or "ground" in normalized) and "decoy" not in normalized else "#ffc27a"


def _ue_category(category: str):
    normalized = category.lower()
    return "target" if ("target" in normalized or "ground" in normalized) and "decoy" not in normalized else "decoy"


def _yolo_category(item):
    return "target" if str(item.get("class_name")) == "real_vehicle" else "decoy"


def _yolo_color(category: str):
    return "#8b0000" if category == "target" else "#c45100"


def _draw_label(draw, xy, text, color, font):
    bounds = draw.textbbox(xy, text, font=font, stroke_width=1)
    draw.rectangle(bounds, fill=color)
    draw.text(xy, text, fill="black", font=font, stroke_width=1, stroke_fill="white")


def _dashed_line(draw, start, end, color, width, dash=5, gap=3):
    """Pillow 没有虚线矩形接口，逐边绘制短实线段。"""
    x1, y1 = start
    x2, y2 = end
    length = max(abs(x2 - x1), abs(y2 - y1))
    if length <= 0:
        return
    position = 0.0
    while position < length:
        finish = min(length, position + dash)
        ratio_a, ratio_b = position / length, finish / length
        draw.line((x1 + (x2 - x1) * ratio_a, y1 + (y2 - y1) * ratio_a,
                   x1 + (x2 - x1) * ratio_b, y1 + (y2 - y1) * ratio_b),
                  fill=color, width=width)
        position += dash + gap


def _dashed_rectangle(draw, box, color, width=2):
    x1, y1, x2, y2 = box
    _dashed_line(draw, (x1, y1), (x2, y1), color, width)
    _dashed_line(draw, (x2, y1), (x2, y2), color, width)
    _dashed_line(draw, (x2, y2), (x1, y2), color, width)
    _dashed_line(draw, (x1, y2), (x1, y1), color, width)


def render(run: Path) -> dict:
    frames_path, predictions_path = run / "visual_frames.jsonl", run / "visual_predictions.jsonl"
    if not frames_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("需要 --save-images --detailed-log 产生 visual_frames.jsonl 和 visual_predictions.jsonl")
    predictions = {(str(row.get("uid")), str(row.get("frame_id"))): row
                   for row in _rows(predictions_path)}
    font = ImageFont.load_default()
    counts = {"frames": 0, "rendered": 0, "skipped_without_ue_or_yolo": 0, "missing_images": 0,
              "ue_boxes": 0, "yolo_boxes": 0, "frames_without_prediction": 0}
    for frame in _rows(frames_path):
        counts["frames"] += 1
        ue_items = [(item, _box(item)) for item in frame.get("ue_projected_objects", [])
                    if isinstance(item, dict) and _box(item) is not None]
        prediction = predictions.get((str(frame.get("uid")), str(frame.get("frame_id"))))
        yolo_items = [(item, _box(item)) for item in (prediction or {}).get("objects", [])
                      if isinstance(item, dict) and _box(item) is not None]
        relative = frame.get("image_path")
        destination = (run / "visual" / str(frame.get("uid", "unknown"))
                       / Path(relative).name) if relative else None
        if not ue_items and not yolo_items:
            if destination is not None and destination.is_file():
                destination.unlink()
            counts["skipped_without_ue_or_yolo"] += 1
            if prediction is None:
                counts["frames_without_prediction"] += 1
            continue
        if not relative:
            counts["missing_images"] += 1
            continue
        source = run / relative
        if not source.is_file():
            counts["missing_images"] += 1
            continue
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        for item, box in ue_items:
            identity, category = _ue_label(item)
            color = _ue_color(category)
            draw.rectangle(box, outline=color, width=3)
            _draw_label(draw, (box[0], max(0, box[1] - 13)),
                        f"UE {_ue_category(category)} {identity}", color, font)
            counts["ue_boxes"] += 1
        if prediction is None:
            counts["frames_without_prediction"] += 1
        else:
            for item, box in yolo_items:
                category = _yolo_category(item)
                color = _yolo_color(category)
                probability = item.get(
                    "real_probability" if category == "target" else "decoy_probability",
                    item.get("detector_confidence", 0.0),
                )
                _dashed_rectangle(draw, box, color, width=2)
                _draw_label(draw, (box[0], min(image.height - 12, box[3] + 1)),
                            f"YOLO {category} p={float(probability):.2f}", color, font)
                counts["yolo_boxes"] += 1
        destination = run / "visual" / str(frame.get("uid", "unknown")) / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, quality=92)
        counts["rendered"] += 1
    (run / "visual" / "recognition_summary.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="单局输出目录")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    if not run.is_dir():
        parser.error(f"--run 不存在：{run}")
    print(json.dumps(render(run), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
