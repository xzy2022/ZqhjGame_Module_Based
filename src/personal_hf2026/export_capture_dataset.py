# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：把采集结果按真目标、假目标和混合画面分目录，便于人工检查。
# 修改内容：导出原图硬链接、带框整图、放大目标预览和 CSV 索引。
"""离线整理 PersonalV2 采集结果；不运行仿真或视觉模型。"""
import argparse
import csv
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {"TargetVehicle": (255, 48, 48), "DecoyVehicle": (0, 220, 255)}
LABELS = {"TargetVehicle": "TRUE", "DecoyVehicle": "DECOY"}


def _rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _category(objects):
    kinds = {item["target_type"] for item in objects}
    if not kinds:
        return "00_no_projected_bbox"
    if kinds == {"TargetVehicle"}:
        return "01_true_only"
    if kinds == {"DecoyVehicle"}:
        return "02_decoy_only"
    return "03_true_and_decoy"


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        # 跨磁盘或文件系统不支持硬链接时才复制。
        destination.write_bytes(source.read_bytes())
        return "copy"


def _label(draw, xy, text, color):
    font = ImageFont.load_default()
    box = draw.textbbox(xy, text, font=font, stroke_width=1)
    draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(0, 0, 0))
    draw.text(xy, text, fill=color, font=font, stroke_width=1, stroke_fill=(0, 0, 0))


def _annotated(image, objects):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for item in objects:
        bbox = item["ue_projected_bbox"]
        color = COLORS[item["target_type"]]
        draw.rectangle(bbox, outline=color, width=3)
        _label(draw, (max(1, bbox[0]), max(1, bbox[1] - 13)),
               f'{LABELS[item["target_type"]]} {item["target_id"]}', color)
    return result


def _preview(image, item, size=256):
    width, height = image.size
    x1, y1, x2, y2 = item["ue_projected_bbox"]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    side = max(80, max(bw, bh) * 6)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = max(0, min(width - side, cx - side / 2))
    top = max(0, min(height - side, cy - side / 2))
    right, bottom = min(width, left + side), min(height, top + side)
    crop = image.crop((round(left), round(top), round(right), round(bottom)))
    scale_x, scale_y = size / crop.width, size / crop.height
    crop = crop.resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(crop)
    local = ((x1 - left) * scale_x, (y1 - top) * scale_y,
             (x2 - left) * scale_x, (y2 - top) * scale_y)
    color = COLORS[item["target_type"]]
    draw.rectangle(local, outline=color, width=3)
    _label(draw, (4, 4), f'{LABELS[item["target_type"]]} {item["target_id"]}', color)
    return crop


def export(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    counts = {"frames": {}, "objects": {"true": 0, "decoy": 0},
              "invalid_bbox": 0, "edge_clipped": 0, "link_mode": {}}
    frame_fields = ["category", "uid", "frame_no", "source_t", "image_path",
                    "object_count", "target_ids", "target_types"]
    object_fields = ["target_type", "target_id", "uid", "frame_no", "source_t",
                     "source_image", "preview_image", "bbox_x1", "bbox_y1",
                     "bbox_x2", "bbox_y2", "edge_clipped"]
    with (output / "frames.csv").open("w", newline="", encoding="utf-8-sig") as frame_stream, \
         (output / "objects.csv").open("w", newline="", encoding="utf-8-sig") as object_stream:
        frame_writer, object_writer = csv.DictWriter(frame_stream, frame_fields), csv.DictWriter(object_stream, object_fields)
        frame_writer.writeheader()
        object_writer.writeheader()
        for row in _rows(source / "dataset/samples.jsonl"):
            objects = row["ue_projected_objects"]
            category = _category(objects)
            counts["frames"][category] = counts["frames"].get(category, 0) + 1
            original = source / row["image_path"]
            name = f'{row["uid"]}_f{row["frame_no"]}_{round(row["source_sim_time"] * 1000000)}{original.suffix.lower()}'
            link_mode = _link(original, output / "frames" / category / name)
            counts["link_mode"][link_mode] = counts["link_mode"].get(link_mode, 0) + 1
            frame_writer.writerow(dict(category=category, uid=row["uid"], frame_no=row["frame_no"],
                source_t=row["source_t"], image_path=f"frames/{category}/{name}", object_count=len(objects),
                target_ids=";".join(str(item["target_id"]) for item in objects),
                target_types=";".join(item["target_type"] for item in objects)))
            if not objects:
                continue
            with Image.open(original) as opened:
                image = opened.convert("RGB")
            annotated = output / "annotated_frames" / category / name
            annotated.parent.mkdir(parents=True, exist_ok=True)
            _annotated(image, objects).save(annotated, "JPEG", quality=90, optimize=True)
            for index, item in enumerate(objects):
                x1, y1, x2, y2 = item["ue_projected_bbox"]
                if x2 <= x1 or y2 <= y1:
                    counts["invalid_bbox"] += 1
                    continue
                edge = x1 <= 0 or y1 <= 0 or x2 >= image.width or y2 >= image.height
                counts["edge_clipped"] += int(edge)
                kind = "true" if item["target_type"] == "TargetVehicle" else "decoy"
                counts["objects"][kind] += 1
                preview_name = f'{Path(name).stem}_o{index}_{item["target_id"]}{"_edge" if edge else ""}.jpg'
                preview_path = output / "object_previews" / kind / preview_name
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                preview = _preview(image, item)
                preview.save(preview_path, "JPEG", quality=92, optimize=True)
                object_writer.writerow(dict(target_type=item["target_type"], target_id=item["target_id"],
                    uid=row["uid"], frame_no=row["frame_no"], source_t=row["source_t"],
                    source_image=f"frames/{category}/{name}",
                    preview_image=f"object_previews/{kind}/{preview_name}",
                    bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2, edge_clipped=edge))
    counts["source"] = str(source)
    counts["notes"] = [
        "红框 TRUE，蓝框 DECOY；类别和框均来自 UE 离线投影数据。",
        "投影框没有验证遮挡；00_no_projected_bbox 不能直接当作可靠背景。",
        "frames 目录优先使用硬链接，不额外占用一份原图空间。",
        "object_previews 是带上下文的放大预览，不是模型输入裁剪。",
    ]
    (output / "summary.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.txt").write_text("\n".join(counts["notes"]) + "\n", encoding="utf-8")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="包含 dataset/samples.jsonl 的采集输出目录")
    parser.add_argument("output", type=Path, help="必须是尚不存在的新目录")
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), ensure_ascii=False, indent=2))
