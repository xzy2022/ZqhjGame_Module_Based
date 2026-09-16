# 修改时间：2026-09-16。
# 修改目的：把 FOV30 采集结果整理为目标与诱饵分类数据集。
# 修改内容：流式校验 UE 投影框，导出无标注裁剪、匿名训练索引、审计索引及 copy/hardlink 源图入口。
"""从已有采集 run 导出目标/诱饵图像分类数据集。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from PIL import Image


CLASSES = {
    "TargetVehicle": (0, "target"),
    "DecoyVehicle": (1, "decoy"),
}


def jsonl_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def anonymous_id(*parts, length=24):
    value = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:length]


def link_or_copy(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if mode == "hardlink" and os.path.samefile(source, destination):
            return False
        raise FileExistsError(f"目标图像已存在且不是同一硬链接：{destination}")
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)
    return True


def inspect_object(sample, item, object_index, drop_edge):
    target_type = item.get("target_type")
    if target_type not in CLASSES:
        raise ValueError("unknown_target_type")
    if item.get("target_id") is None:
        raise ValueError("target_id_missing")
    bbox = item.get("ue_projected_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("invalid_bbox")
    try:
        original_bbox = [float(value) for value in bbox]
    except (TypeError, ValueError) as error:
        raise ValueError("invalid_bbox") from error
    if not all(math.isfinite(value) for value in original_bbox):
        raise ValueError("invalid_bbox")
    width, height = int(sample["width"]), int(sample["height"])
    x1, y1, x2, y2 = original_bbox
    clipped_bbox = [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]
    if clipped_bbox[2] <= clipped_bbox[0] or clipped_bbox[3] <= clipped_bbox[1]:
        raise ValueError("empty_bbox")
    edge_clipped = x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height
    if edge_clipped and drop_edge:
        raise ValueError("edge_bbox")
    crop_bbox = [
        math.floor(clipped_bbox[0]),
        math.floor(clipped_bbox[1]),
        math.ceil(clipped_bbox[2]),
        math.ceil(clipped_bbox[3]),
    ]
    if crop_bbox[2] <= crop_bbox[0] or crop_bbox[3] <= crop_bbox[1]:
        raise ValueError("empty_integer_crop")
    class_id, class_name = CLASSES[target_type]
    return {
        "object_index": object_index,
        "target_id": str(item["target_id"]),
        "target_type": target_type,
        "class_id": class_id,
        "class_name": class_name,
        "original_bbox_xyxy": original_bbox,
        "bbox_xyxy": clipped_bbox,
        "crop_bbox_xyxy": crop_bbox,
        "edge_clipped": edge_clipped,
        "projection_visibility": item.get("visibility"),
        "projection_visibility_annotation": item.get("visibility_annotation"),
    }


def matching_reference(sample, inspected):
    reference = sample.get("reference") or {}
    matches = [item for item in reference.get("objects") or []
               if str(item.get("target_id")) == inspected["target_id"]]
    if not matches:
        return "missing", None
    if len(matches) != 1:
        return "not_unique", None
    if matches[0].get("target_type") != inspected["target_type"]:
        return "type_mismatch", matches[0]
    return "matched", matches[0]


def rejection_row(run_id, source_line, sample, object_index, reason, item):
    return {
        "run": run_id,
        "source_line": source_line,
        "uid": str(sample.get("uid")),
        "frame_no": sample.get("frame_no"),
        "source_sim_time": sample.get("source_sim_time"),
        "object_index": object_index,
        "target_id": item.get("target_id") if isinstance(item, dict) else None,
        "target_type": item.get("target_type") if isinstance(item, dict) else None,
        "bbox_xyxy": item.get("ue_projected_bbox") if isinstance(item, dict) else None,
        "reason": reason,
    }


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_run(run_root, run_output, *, image_mode, drop_edge,
               limit_records=0, dry_run=False):
    run_root = Path(run_root).resolve()
    run_output = Path(run_output).resolve()
    run_id = run_root.name
    samples_path = run_root / "dataset" / "samples.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError(f"缺少采集产物：{samples_path}")
    if run_output == run_root or run_output.is_relative_to(run_root):
        raise ValueError("输出目录不能放进原始 run")
    if image_mode == "hardlink" and run_root.drive.lower() != run_output.drive.lower():
        raise ValueError("硬链接要求源 run 与输出目录位于同一磁盘卷")
    if run_output.exists() and not dry_run:
        raise FileExistsError(f"输出目录已存在：{run_output}")

    summary = {
        "status": "running",
        "run": run_id,
        "source_run": str(run_root),
        "source_samples": str(samples_path),
        "source_samples_sha256": sha256_file(samples_path),
        "image_mode": image_mode,
        "drop_edge": drop_edge,
        "limit_records": limit_records,
        "label_source": "ue_projected_objects",
        "visibility_annotation": "unannotated",
        "exposure_time_verified": False,
    }
    if not dry_run:
        run_output.mkdir(parents=False, exist_ok=False)
        for directory in ("images", "crops/target", "crops/decoy"):
            (run_output / directory).mkdir(parents=True, exist_ok=True)
        write_json(run_output / "summary.json", summary)

    counts = Counter()
    rejection_reasons = Counter()
    try:
        with ExitStack() as stack:
            if dry_run:
                training_stream = audit_stream = rejected_stream = None
            else:
                training_stream = stack.enter_context(
                    (run_output / "samples.jsonl").open("w", encoding="utf-8"))
                audit_stream = stack.enter_context(
                    (run_output / "audit.jsonl").open("w", encoding="utf-8"))
                rejected_stream = stack.enter_context(
                    (run_output / "rejected_objects.jsonl").open("w", encoding="utf-8"))

            seen_samples = set()
            for source_line, sample in jsonl_rows(samples_path):
                if limit_records and counts["source_records"] >= limit_records:
                    break
                counts["source_records"] += 1
                objects = sample.get("ue_projected_objects") or []
                counts["projected_objects"] += len(objects)
                if objects:
                    counts["frames_with_projected_objects"] += 1

                accepted = []
                for object_index, item in enumerate(objects):
                    try:
                        accepted.append(inspect_object(sample, item, object_index, drop_edge))
                    except (KeyError, TypeError, ValueError) as error:
                        reason = str(error)
                        counts["rejected_objects"] += 1
                        rejection_reasons[reason] += 1
                        if rejected_stream is not None:
                            rejected_stream.write(json.dumps(
                                rejection_row(run_id, source_line, sample, object_index, reason, item),
                                ensure_ascii=False, allow_nan=False) + "\n")
                if not accepted:
                    continue

                source_image = (run_root / str(sample.get("image_path", ""))).resolve()
                if not source_image.is_relative_to(run_root):
                    raise ValueError(f"图像路径超出 run：{source_image}")
                if not source_image.is_file():
                    raise FileNotFoundError(f"源图不存在：{source_image}")
                with Image.open(source_image) as opened:
                    image = opened.convert("RGB")
                expected_size = (int(sample["width"]), int(sample["height"]))
                if image.size != expected_size:
                    raise ValueError(f"图像尺寸与标签不一致：{source_image}")
                expected_hash = sample.get("image_sha256")
                actual_hash = sha256_file(source_image)
                if expected_hash and actual_hash != expected_hash:
                    raise ValueError(f"图像校验和不一致：{source_image}")

                frame_id = anonymous_id(run_id, source_line, sample.get("uid"),
                                        sample.get("frame_no"), sample.get("source_sim_time"))
                image_relative = Path("images") / str(sample["uid"]) / (frame_id + source_image.suffix.lower())
                if not dry_run:
                    link_or_copy(source_image, run_output / image_relative, image_mode)
                counts["exported_frames"] += 1
                counts["materialized_original_bytes"] += source_image.stat().st_size

                for inspected in accepted:
                    sample_id = anonymous_id(frame_id, inspected["object_index"])
                    if sample_id in seen_samples:
                        raise ValueError(f"重复分类样本：{sample_id}")
                    seen_samples.add(sample_id)
                    crop_relative = Path("crops") / inspected["class_name"] / (sample_id + ".png")
                    crop = image.crop(inspected["crop_bbox_xyxy"])
                    if not dry_run:
                        crop.save(run_output / crop_relative, "PNG")
                        counts["crop_bytes"] += (run_output / crop_relative).stat().st_size

                    group_id = anonymous_id("physical-object", run_id, inspected["target_id"])
                    training_row = {
                        "sample_id": sample_id,
                        "image": crop_relative.as_posix(),
                        "class_id": inspected["class_id"],
                        "class_name": inspected["class_name"],
                        "group_id": group_id,
                        "width": crop.width,
                        "height": crop.height,
                    }
                    reference_status, truth = matching_reference(sample, inspected)
                    audit_row = {
                        **training_row,
                        "run": run_id,
                        "source_line": source_line,
                        "uid": str(sample["uid"]),
                        "frame_no": int(sample["frame_no"]),
                        "source_sim_time": sample.get("source_sim_time"),
                        "source_image": str(source_image),
                        "exported_image": image_relative.as_posix(),
                        "source_image_sha256": actual_hash,
                        "target_id": inspected["target_id"],
                        "target_type": inspected["target_type"],
                        "object_index": inspected["object_index"],
                        "original_bbox_xyxy": inspected["original_bbox_xyxy"],
                        "bbox_xyxy": inspected["bbox_xyxy"],
                        "crop_bbox_xyxy": inspected["crop_bbox_xyxy"],
                        "edge_clipped": inspected["edge_clipped"],
                        "projection_visibility": inspected["projection_visibility"],
                        "projection_visibility_annotation": inspected["projection_visibility_annotation"],
                        "frame_visibility_annotation": sample.get("visibility_annotation"),
                        "exposure_time_verified": sample.get("exposure_time_verified", False),
                        "reference_status": reference_status,
                        "reference_object": truth,
                    }
                    if training_stream is not None:
                        training_stream.write(json.dumps(
                            training_row, ensure_ascii=False, allow_nan=False) + "\n")
                        audit_stream.write(json.dumps(
                            audit_row, ensure_ascii=False, allow_nan=False) + "\n")
                    counts["accepted_objects"] += 1
                    counts[inspected["class_name"]] += 1
                    counts["edge_clipped_objects"] += int(inspected["edge_clipped"])
                    counts[f"reference_{reference_status}"] += 1

                if counts["exported_frames"] % 500 == 0:
                    print(json.dumps({
                        "run": run_id,
                        "exported_frames": counts["exported_frames"],
                        "accepted_objects": counts["accepted_objects"],
                    }, ensure_ascii=False), flush=True)

        summary.update(
            status="completed",
            counts=dict(counts),
            rejection_reasons=dict(sorted(rejection_reasons.items())),
            split_group="opaque hash of run+physical target_id",
            training_index="samples.jsonl",
            audit_index="audit.jsonl",
        )
        if not dry_run:
            write_json(run_output / "label_schema.json", {
                "classes": {"0": "target", "1": "decoy"},
                "training_index": {
                    "path": "samples.jsonl",
                    "model_input": "image 指向的 RGB crop；其余字段仅用于监督和分组",
                    "privacy": "不含 run、天气、无人机、target_id、bbox、地理或位姿字段",
                },
                "audit_index": {
                    "path": "audit.jsonl",
                    "purpose": "标签溯源和人工核验，不得作为分类模型输入",
                },
                "crop": "将 UE bbox 裁到图像范围后 floor(left/top), ceil(right/bottom)，原始 RGB PNG，无绘制框",
                "visibility": "UE 投影框未经过人工可见性确认",
            })
            write_json(run_output / "summary.json", summary)
        return summary
    except BaseException as error:
        summary.update(
            status="failed",
            error=str(error),
            counts=dict(counts),
            rejection_reasons=dict(sorted(rejection_reasons.items())),
        )
        if not dry_run and run_output.is_dir():
            write_json(run_output / "summary.json", summary)
        raise


def selected_runs(source_root, run_names):
    source_root = Path(source_root).resolve()
    if run_names:
        runs = [source_root / name for name in run_names]
    else:
        runs = sorted(path for path in source_root.iterdir()
                      if path.is_dir() and (path / "dataset" / "samples.jsonl").is_file())
    missing = [str(path) for path in runs
               if not (path / "dataset" / "samples.jsonl").is_file()]
    if missing:
        raise FileNotFoundError(f"run 缺少 dataset/samples.jsonl：{missing}")
    if not runs:
        raise FileNotFoundError(f"没有找到可导出的 run：{source_root}")
    return runs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="包含各采集 run 子目录的 runs 目录")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-names", nargs="*", help="只导出指定 run；省略则处理全部 run")
    parser.add_argument("--image-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--drop-edge", action="store_true",
                        help="拒绝触边框；默认沿用旧整理器行为，裁到图像范围后保留")
    parser.add_argument("--limit-per-run", type=int, default=0,
                        help="每个 run 最多读取的非空 samples 记录数，0 表示全量")
    parser.add_argument("--dry-run", action="store_true", help="完整读取和校验，但不写文件")
    args = parser.parse_args(argv)
    if args.limit_per_run < 0:
        parser.error("limit-per-run 必须非负")
    return args


def main(argv=None):
    args = parse_args(argv)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"源 runs 目录不存在：{source_root}")
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("输出根不能放进源 runs 目录")
    if output_root.exists() and not args.dry_run:
        raise FileExistsError(f"输出目录已存在；请使用新的目录：{output_root}")
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=False)

    summaries = []
    for run_root in selected_runs(source_root, args.run_names):
        summary = export_run(
            run_root,
            output_root / run_root.name,
            image_mode=args.image_mode,
            drop_edge=args.drop_edge,
            limit_records=args.limit_per_run,
            dry_run=args.dry_run,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)

    total_counts = Counter()
    for summary in summaries:
        total_counts.update(summary["counts"])
    result = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "image_mode": args.image_mode,
        "drop_edge": args.drop_edge,
        "runs": summaries,
        "totals": dict(total_counts),
    }
    if not args.dry_run:
        write_json(output_root / "manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
