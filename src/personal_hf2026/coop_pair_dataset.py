# 修改时间：2026-09-16。
# 修改目的：把已有采集 run 整理为可直接研究双机几何定位的成对数据集。
# 修改内容：流式关联协同 pair 与逐帧样本，严格筛选双侧标签，并以硬链接或复制方式导出原图。
"""从已有 FOV30 采集结果导出严格的双机协同定位数据集。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil


POSE_FIELDS = ("lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt", "gimbal_fov_deg")
ATTITUDE_FIELDS = ("roll", "pitch", "yaw")


def jsonl_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def frame_key(value):
    return str(value["uid"]), int(value["frame_no"])


def complete_fields(value, fields):
    return isinstance(value, dict) and all(value.get(field) is not None for field in fields)


def matching_object(sample, target_id, field):
    matches = [item for item in sample.get(field) or []
               if str(item.get("target_id")) == str(target_id)]
    return matches[0] if len(matches) == 1 else None, len(matches)


def inspect_side(run_root, sample, pair_ref, target_id):
    reasons = []
    if abs(float(sample.get("source_sim_time")) - float(pair_ref["source_sim_time"])) > 1e-6:
        reasons.append("source_time_mismatch")

    projected, projected_count = matching_object(sample, target_id, "ue_projected_objects")
    if projected_count == 0:
        reasons.append("target_bbox_missing")
    elif projected_count > 1:
        reasons.append("target_bbox_not_unique")
    bbox = projected.get("ue_projected_bbox") if projected else None
    valid_bbox = isinstance(bbox, list) and len(bbox) == 4
    if valid_bbox:
        try:
            bbox = [float(value) for value in bbox]
            valid_bbox = bbox[2] > bbox[0] and bbox[3] > bbox[1]
        except (TypeError, ValueError):
            valid_bbox = False
    if projected is not None and not valid_bbox:
        reasons.append("target_bbox_nonpositive_or_invalid")
    width, height = sample.get("width"), sample.get("height")
    edge_touching = bool(valid_bbox and width and height and (
        bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= width or bbox[3] >= height))

    if not complete_fields(sample.get("source_pose"), POSE_FIELDS):
        reasons.append("source_pose_incomplete")
    if not complete_fields(sample.get("aircraft_attitude"), ATTITUDE_FIELDS):
        reasons.append("aircraft_attitude_incomplete")

    reference = sample.get("reference") or {}
    truth, truth_count = matching_object(reference, target_id, "objects")
    if truth_count == 0:
        reasons.append("ground_truth_missing")
    elif truth_count > 1:
        reasons.append("ground_truth_not_unique")
    if truth is not None and not complete_fields(truth, ("lat", "lon", "alt")):
        reasons.append("ground_truth_incomplete")

    source_image = run_root / str(sample.get("image_path", ""))
    if not source_image.is_file():
        reasons.append("image_missing")
    return {
        "sample": sample,
        "bbox": bbox if valid_bbox else None,
        "projected": projected,
        "truth": truth,
        "reference": reference,
        "source_image": source_image,
        "edge_touching": edge_touching,
        "reasons": reasons,
    }


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


def exported_side(run_output, inspected, pair_ref, mode, dry_run):
    sample = inspected["sample"]
    uid = str(sample["uid"])
    destination = run_output / "images" / uid / inspected["source_image"].name
    if not dry_run:
        link_or_copy(inspected["source_image"], destination, mode)
    projected = inspected["projected"]
    return {
        "uid": uid,
        "role": sample["cooperation"]["role"],
        "partner_uid": str(sample["cooperation"]["partner_uid"]),
        "frame_no": int(sample["frame_no"]),
        "source_sim_time": sample["source_sim_time"],
        "coop_seconds": pair_ref.get("coop_seconds"),
        "image": destination.relative_to(run_output).as_posix(),
        "image_sha256": sample.get("image_sha256"),
        "image_bytes": sample.get("image_bytes"),
        "width": sample.get("width"),
        "height": sample.get("height"),
        "bbox_xyxy": inspected["bbox"],
        "bbox_edge_touching": inspected["edge_touching"],
        "bbox_visibility": projected.get("visibility"),
        "bbox_visibility_annotation": projected.get("visibility_annotation"),
        "source_pose": sample["source_pose"],
        "aircraft_attitude": sample["aircraft_attitude"],
        "aircraft_attitude_alignment": sample.get("aircraft_attitude_alignment"),
        "aircraft_attitude_time_delta_s": sample.get("aircraft_attitude_time_delta_s"),
        "cooperation_alignment": sample["cooperation"].get("alignment"),
        "exposure_time_verified": sample.get("exposure_time_verified", False),
    }


def exported_truth(inspected):
    truth = inspected["truth"]
    reference = inspected["reference"]
    return {
        "lat": truth["lat"],
        "lon": truth["lon"],
        "alt": truth["alt"],
        "reference_sim_time": reference.get("sim_time"),
        "reference_time_delta_s": reference.get("time_delta_s"),
        "label_source": reference.get("label_source"),
    }


def session_text(session_id):
    return "-".join(str(value) for value in session_id)


def load_needed_samples(run_root, pairs):
    needed = {}
    for pair in pairs:
        for side in ("master", "follower"):
            reference = pair[side]
            needed[frame_key(reference)] = reference
    found = {}
    samples_path = run_root / "dataset" / "samples.jsonl"
    for _, sample in jsonl_rows(samples_path):
        key = frame_key(sample)
        if key in needed:
            found[key] = sample
            if len(found) == len(needed):
                break
    return found


def run_id_from_path(run_root):
    return run_root.name


def export_run(run_root, run_output, *, image_mode, keep_edge,
               max_time_delta_s, dry_run=False):
    run_root = Path(run_root).resolve()
    run_output = Path(run_output).resolve()
    run_id = run_id_from_path(run_root)
    pairs_path = run_root / "dataset" / "coop_pairs.jsonl"
    samples_path = run_root / "dataset" / "samples.jsonl"
    calibration_path = run_root / "camera_calibration.json"
    for required in (pairs_path, samples_path, calibration_path):
        if not required.is_file():
            raise FileNotFoundError(f"缺少采集产物：{required}")
    if image_mode == "hardlink" and run_root.drive.lower() != run_output.drive.lower():
        raise ValueError("硬链接要求源 run 与输出目录位于同一磁盘卷")
    if not dry_run:
        run_output.mkdir(parents=False, exist_ok=False)

    pairs = [row for _, row in jsonl_rows(pairs_path)]
    samples = load_needed_samples(run_root, pairs)
    reason_counts = Counter()
    accepted_rows, rejected_rows = [], []
    linked_images = 0

    for pair_index, pair in enumerate(pairs, 1):
        reasons = []
        delta = abs(float(pair.get("source_time_delta_s", float("inf"))))
        if delta > max_time_delta_s:
            reasons.append("pair_time_delta_exceeded")
        sides = {}
        for side in ("master", "follower"):
            reference = pair[side]
            sample = samples.get(frame_key(reference))
            if sample is None:
                reasons.append(f"{side}_sample_missing")
                continue
            sides[side] = inspect_side(run_root, sample, reference,
                                       pair["accepted_target_id"])
            reasons.extend(f"{side}_{reason}" for reason in sides[side]["reasons"])
        edge_touching = bool(sides and any(value["edge_touching"] for value in sides.values()))
        if edge_touching and not keep_edge:
            reasons.append("edge_bbox")

        pair_stub = {
            "run": run_id,
            "source_pair_line": pair_index,
            "session_id": pair.get("session_id"),
            "target_id": str(pair.get("accepted_target_id")),
        }
        if reasons:
            unique_reasons = list(dict.fromkeys(reasons))
            rejected_rows.append({**pair_stub, "reasons": unique_reasons})
            reason_counts.update(unique_reasons)
            continue

        pair_id = (f"{run_id}__{session_text(pair['session_id'])}__{pair['accepted_target_id']}__"
                   f"{pair['master']['uid']}-{pair['master']['frame_no']}__"
                   f"{pair['follower']['uid']}-{pair['follower']['frame_no']}")
        master = exported_side(run_output, sides["master"], pair["master"], image_mode, dry_run)
        follower = exported_side(run_output, sides["follower"], pair["follower"], image_mode, dry_run)
        linked_images += 2
        accepted_rows.append({
            "pair_id": pair_id,
            **pair_stub,
            "master": master,
            "follower": follower,
            "ground_truth": {
                "canonical_side": "master",
                "master": exported_truth(sides["master"]),
                "follower": exported_truth(sides["follower"]),
            },
            "quality": {
                "source_time_delta_s": pair["source_time_delta_s"],
                "max_time_delta_s": max_time_delta_s,
                "edge_touching": edge_touching,
                "fields_complete": True,
                "visibility_annotation": "unannotated",
                "exposure_time_verified": bool(
                    master["exposure_time_verified"] and follower["exposure_time_verified"]),
            },
        })

    summary = {
        "status": "completed",
        "run": run_id,
        "source_run": str(run_root),
        "source_pairs": len(pairs),
        "accepted_pairs": len(accepted_rows),
        "rejected_pairs": len(rejected_rows),
        "linked_images": linked_images,
        "image_mode": image_mode,
        "keep_edge": keep_edge,
        "max_time_delta_s": max_time_delta_s,
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "split_group": "run+session_id",
        "visibility_annotation": "unannotated",
        "exposure_time_verified": False,
        "camera_calibration_status": "derived_under_unverified_assumptions",
    }
    if not dry_run:
        with (run_output / "pairs.jsonl").open("w", encoding="utf-8") as stream:
            for row in accepted_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        with (run_output / "rejected_pairs.jsonl").open("w", encoding="utf-8") as stream:
            for row in rejected_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        shutil.copy2(calibration_path, run_output / "camera_calibration.json")
        (run_output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
    return summary


def selected_runs(source_root, run_names):
    source_root = Path(source_root).resolve()
    if run_names:
        runs = [source_root / name for name in run_names]
    else:
        runs = sorted(path for path in source_root.iterdir() if path.is_dir())
    missing = [str(path) for path in runs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"run 目录不存在：{missing}")
    return runs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="包含各采集 run 子目录的 runs 目录")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-names", nargs="*", help="只导出指定 run；省略则处理全部子目录")
    parser.add_argument("--image-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--keep-edge", action="store_true", help="保留任一侧目标框触边的 pair")
    parser.add_argument("--max-time-delta-s", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true", help="完整读取和筛选，但不写文件")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"源 runs 目录不存在：{source_root}")
    if output_root.exists() and not args.dry_run:
        raise FileExistsError(f"输出目录已存在；请使用新的目录：{output_root}")
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=False)
    summaries = []
    for run_root in selected_runs(source_root, args.run_names):
        summary = export_run(
            run_root, output_root / run_root.name,
            image_mode=args.image_mode, keep_edge=args.keep_edge,
            max_time_delta_s=args.max_time_delta_s, dry_run=args.dry_run)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    result = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "runs": summaries,
        "totals": {
            "source_pairs": sum(row["source_pairs"] for row in summaries),
            "accepted_pairs": sum(row["accepted_pairs"] for row in summaries),
            "rejected_pairs": sum(row["rejected_pairs"] for row in summaries),
            "linked_images": sum(row["linked_images"] for row in summaries),
        },
    }
    if not args.dry_run:
        (output_root / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
