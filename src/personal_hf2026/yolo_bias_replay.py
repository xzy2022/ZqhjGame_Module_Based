# 修改时间：2026-09-19。
# 修改目的：在完全相同的在线原图上比较 V1-v3 与 V2，并控制旋转探针的缩放混杂。
# 修改内容：新增显式档位与权重选择、V2 独立多流状态及正方形补边旋转诊断。
# 修改时间：2026-09-19。
# 修改目的：用离线回放直接验证将在线部署的检测器旋转实现。
# 修改内容：新增 image-rotation-deg 覆盖入口，并与额外诊断旋转分开记录。
# 修改时间：2026-09-19。
# 修改目的：复现在线显式图像旋转并避免离线消融结果携带原在线时序含义。
# 修改内容：读取在线检测器旋转参数，重放结果分离原在线时序和实际离线间隔与耗时。
# 修改时间：2026-09-19。
# 修改目的：在冻结的在线原图上检查观察方向与亮度域差异。
# 修改内容：增加确定性间隔取样、直角旋转及全图亮度变换，并把检测框逆变换回原图计分。
# 修改时间：2026-09-19。
# 修改目的：用在线旁路实际处理的原图和时间顺序区分输入差异、在线调度和模型分类问题。
# 修改内容：新增 v3 同帧离线重放、逐框一致性对照、纯 CPU 解码核验及有限消融参数。
"""重放旁路已处理图像；每架无人机保留独立的时序状态。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from .paths import PROJECT_ROOT
from .yolo_offline_eval import MetricAccumulator, greedy_match, normalize_predictions
from .yolo_profiles import (
    DEFAULT_CONFIG, DEFAULT_V2_CONFIG, YOLO_PROFILES, canonical_profile,
    detector_kwargs, resolve_resources,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path, limit, sample_every=1):
    with path.open(encoding="utf-8") as stream:
        selected = 0
        for index, line in enumerate(stream):
            if index % sample_every:
                continue
            if limit and selected >= limit:
                break
            selected += 1
            yield json.loads(line)


def restore_boxes(result, width, height, rotation):
    """只逆变换模型输出框，分类过程不读取真值框。"""
    for item in result:
        a, b, c, d = item["bbox_xyxy"]
        if rotation == 90:
            bbox = [width-d, a, width-b, c]
        elif rotation == 180:
            bbox = [width-c, height-d, width-a, height-b]
        elif rotation == 270:
            bbox = [b, height-c, d, height-a]
        else:
            continue
        item["bbox_xyxy"] = bbox
        item["xyxy"] = bbox


def truth_objects(row):
    names = {"TargetVehicle": (0, "real_vehicle"), "DecoyVehicle": (1, "model_prop")}
    result = []
    for item in row.get("ground_truth", []):
        bbox = item["bbox"]
        if item.get("class") not in names or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        class_id, name = names[item["class"]]
        result.append({"object_id": item.get("target_id"), "class_id": class_id,
                       "class_name": name, "bbox_xyxy": bbox})
    return result


def compare_predictions(online, replay, bbox_tolerance, score_tolerance):
    """先按框几何一一匹配，再核对类别、跟踪状态和浮点字段。"""
    matches, _, _ = greedy_match(online, replay, .5, same_class=False)
    detail = []
    max_bbox, max_score = 0., 0.
    for match in matches:
        left, right = online[match["gt_index"]], replay[match["prediction_index"]]
        bbox_error = max(abs(float(a) - float(b)) for a, b in zip(left["bbox_xyxy"], right["bbox_xyxy"]))
        numeric_errors = {}
        for field in ("score", "class_confidence", "detector_confidence", "single_frame_probabilities", "class_probabilities"):
            if field not in left or field not in right:
                continue
            a, b = left[field], right[field]
            numeric_errors[field] = max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.) if isinstance(a, list) else abs(float(a) - float(b))
        score_error = max(numeric_errors.values(), default=0.)
        discrete = {field: [left.get(field), right.get(field)] for field in
                    ("class_id", "class_name", "track_id", "track_hits", "recovered_low_score")
                    if field in left and field in right and left[field] != right[field]}
        same = bbox_error <= bbox_tolerance and score_error <= score_tolerance and not discrete
        max_bbox, max_score = max(max_bbox, bbox_error), max(max_score, score_error)
        detail.append({"online_index": match["gt_index"], "replay_index": match["prediction_index"],
                       "bbox_max_abs_error": bbox_error, "numeric_abs_errors": numeric_errors,
                       "discrete_differences": discrete, "within_tolerance": same})
    return {"online_count": len(online), "replay_count": len(replay), "matched_count": len(matches),
            "within_tolerance": len(online) == len(replay) == len(matches) and all(item["within_tolerance"] for item in detail),
            "bbox_max_abs_error": max_bbox, "score_max_abs_error": max_score, "matches": detail}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", required=True, type=Path, help="含 metadata.json 和 yolo_sidecar_results.jsonl 的在线目录")
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--max-frames", type=int, default=0)
    result.add_argument("--sample-every", type=int, default=1, help="仅每 N 个已处理帧取一个，最大帧数在取样后生效")
    result.add_argument("--audit-only", action="store_true", help="仅 CPU 核对原图哈希、尺寸和两种 OpenCV 解码")
    result.add_argument("--device", default="0")
    result.add_argument("--detector-config", type=Path, help="消融配置；未指定时要求配置哈希与在线一致")
    result.add_argument("--yolo-profile", type=canonical_profile, choices=YOLO_PROFILES,
                        help="重放使用的模型档位；默认复现源在线档位，旧小写v2仍为V1-v2")
    result.add_argument("--weights", type=Path, help="重放权重覆盖；换模型时使用相应模型权重")
    result.add_argument("--tracker-high", type=float, help="消融高置信阈值；默认取在线实际值")
    result.add_argument("--reset-each-frame", action="store_true", help="消融时序融合；默认保持每机原处理顺序")
    result.add_argument("--channel-order", choices=("bgr", "rgb"), default="bgr", help="默认原始 BGR；RGB 仅作通道交换消融")
    result.add_argument("--image-rotation-deg", type=int, choices=(0, 90, 180, 270), help="检测器旋转；默认复现在线资源记录")
    result.add_argument("--rotation-deg", type=int, choices=(0, 90, 180, 270), default=0)
    result.add_argument("--rotation-square-pad", action="store_true",
                        help="先用114在底/右补成正方形再旋转，控制不同方向的letterbox缩放尺度")
    result.add_argument("--contrast-scale", type=float, default=1.)
    result.add_argument("--brightness-offset", type=float, default=0.)
    result.add_argument("--gamma", type=float, default=1., help="全图查表 gamma；小于 1 提亮")
    result.add_argument("--bbox-tolerance", type=float, default=.001)
    result.add_argument("--score-tolerance", type=float, default=.000001)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.sample_every < 1 or args.gamma <= 0:
        raise ValueError("sample-every 和 gamma 必须大于零")
    import cv2
    import numpy as np

    run = args.run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    resources = metadata["resources"]
    source_profile = canonical_profile(resources["profile"])
    profile = args.yolo_profile or source_profile
    same_profile = profile == source_profile
    options = resources["effective_options"]
    default_config = DEFAULT_V2_CONFIG if profile == "V2" else DEFAULT_CONFIG
    config = args.detector_config or (Path(resources["config_path"]) if same_profile else default_config)
    if not config.is_file() and args.detector_config is None:
        config = default_config
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    if same_profile and args.detector_config is None and config_hash != resources["config_sha256"]:
        raise ValueError("重放基础配置 SHA256 与在线不一致")
    high = args.tracker_high if args.tracker_high is not None else (options["tracker_high_confidence_threshold"] if same_profile else None)
    detector_rotation = args.image_rotation_deg if args.image_rotation_deg is not None else (options.get("image_rotation_deg", 0) if same_profile else 0)
    replay_resources = None
    if not args.audit_only:
        replay_resources = resolve_resources(
            profile, config=config, v2_config=config,
            weights=args.weights or (resources["weights_path"] if same_profile else None),
            tracker_high=high, image_rotation_deg=detector_rotation,
        )
        if same_profile and args.weights is None and replay_resources["weights_sha256"] != resources["weights_sha256"]:
            raise ValueError("重放基础权重 SHA256 与在线不一致")
        high = replay_resources["effective_options"]["tracker_high_confidence_threshold"]
    identity = (same_profile and args.weights is None and args.detector_config is None and args.tracker_high is None and not args.reset_each_frame
                and args.channel_order == "bgr" and args.rotation_deg == 0 and args.contrast_scale == 1.
                and args.brightness_offset == 0. and args.gamma == 1. and args.sample_every == 1
                and not args.rotation_square_pad
                and detector_rotation == options.get("image_rotation_deg", 0))
    summary = {"run": str(run), "source_results_sha256": hashlib.sha256((run / "yolo_sidecar_results.jsonl").read_bytes()).hexdigest(),
               "mode": "cpu_input_audit" if args.audit_only else "yolo_same_frame_replay",
               "yolo_profile": profile, "source_yolo_profile": source_profile,
               "resources": replay_resources,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "requested_fov_deg": metadata.get("requested_fov_deg", metadata.get("effective_fov_deg")),
               "identity_settings": identity, "config_path": str(config), "config_sha256": config_hash,
               "source_resources": resources, "effective_tracker_high": high,
               "channel_order": args.channel_order, "reset_each_frame": args.reset_each_frame,
               "sample_every": args.sample_every, "rotation_deg": args.rotation_deg,
               "rotation_square_pad": args.rotation_square_pad,
               "rotation_square_pad_placement": "image_top_left_bottom_right_padding_114" if args.rotation_square_pad else None,
               "contrast_scale": args.contrast_scale, "brightness_offset": args.brightness_offset, "gamma": args.gamma,
               "source_detector_image_rotation_deg": options.get("image_rotation_deg", 0),
               "detector_image_rotation_deg": detector_rotation,
               "bbox_tolerance": args.bbox_tolerance, "score_tolerance": args.score_tolerance,
               "evidence_boundary": "同帧 UE 投影框仅用于审计；不代表人工可见性标注或真实曝光时间。"}
    detector, states = None, {}
    if not args.audit_only:
        from .vehicle_prop import create_detector
        from .vehicle_prop.temporal_tracker import CameraMotion, TemporalTracker
        detector = create_detector(device=args.device, **detector_kwargs(replay_resources))
        manages_streams = bool(getattr(detector, "manages_streams", False))
        max_source_gap_s = float(detector.pipeline.config.get("max_source_gap_s", .25))
        summary["runtime"] = detector.runtime_metadata()
    counts = Counter()
    dimensions, by_uid, previous = Counter(), Counter(), {}
    online_metric, replay_metric = MetricAccumulator(.5), MetricAccumulator(.5)
    max_bbox, max_score = 0., 0.
    started = time.perf_counter()
    with (output / "audit.jsonl").open("w", encoding="utf-8") as audit, (output / "replay_results.jsonl").open("w", encoding="utf-8") as replay_stream:
        for row in read_rows(run / "yolo_sidecar_results.jsonl", args.max_frames, args.sample_every):
            path = Path(row["image_path"])
            if not path.is_absolute():
                path = run / path
            raw = path.read_bytes()
            decode_started = time.perf_counter()
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            replay_decode_ms = (time.perf_counter() - decode_started) * 1000.
            disk_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            decoded_equal = image is not None and disk_image is not None and np.array_equal(image, disk_image)
            hash_equal = hashlib.sha256(raw).hexdigest() == row["image_sha256"]
            size_equal = image is not None and tuple(image.shape[:2]) == (row["image_height"], row["image_width"])
            if not hash_equal or not size_equal or not decoded_equal:
                raise ValueError(f"原图核验失败：submission={row['submission_id']}, hash={hash_equal}, size={size_equal}, decode={decoded_equal}")
            uid, timestamp = str(row["uid"]), float(row["source_sim_time"])
            gap = timestamp - previous[uid] if uid in previous else None
            counts["frames"] += 1
            counts["hash_equal"] += int(hash_equal)
            counts["decode_pixel_equal"] += int(decoded_equal)
            counts["size_equal"] += int(size_equal)
            counts["source_non_increasing"] += int(gap is not None and gap <= 0)
            dimensions[f"{image.shape[1]}x{image.shape[0]}"] += 1
            by_uid[uid] += 1
            previous[uid] = timestamp
            record = {"submission_id": row["submission_id"], "uid": uid, "frame_no": row["frame_no"], "image_path": str(path),
                      "hash_equal": hash_equal, "decode_pixel_equal": decoded_equal, "size_equal": size_equal, "source_gap_s": gap}
            if detector is not None:
                if not manages_streams:
                    if uid not in states:
                        states[uid] = {"camera": CameraMotion(), "tracker": TemporalTracker(**detector.pipeline.config["tracker"]),
                                       "previous_t": None, "sequence": uid}
                    for name, value in states[uid].items():
                        setattr(detector.pipeline, name, value)
                    if args.reset_each_frame:
                        detector.reset()
                elif args.reset_each_frame:
                    detector.reset(stream_id=uid)
                if args.channel_order == "rgb":
                    image = np.ascontiguousarray(image[:, :, ::-1])
                if args.contrast_scale != 1. or args.brightness_offset != 0.:
                    image = np.clip(image.astype(np.float32) * args.contrast_scale + args.brightness_offset, 0, 255).astype(np.uint8)
                if args.gamma != 1.:
                    lookup = np.rint(255. * (np.arange(256) / 255.) ** args.gamma).astype(np.uint8)
                    image = cv2.LUT(image, lookup)
                restore_width, restore_height = image.shape[1], image.shape[0]
                if args.rotation_square_pad:
                    side = max(image.shape[:2])
                    square = np.full((side, side, 3), 114, dtype=np.uint8)
                    square[:image.shape[0], :image.shape[1]] = image
                    image = square
                    restore_width = restore_height = side
                if args.rotation_deg:
                    image = np.ascontiguousarray(np.rot90(image, args.rotation_deg // 90))
                step = time.perf_counter()
                if manages_streams:
                    result = detector.predict(image, timestamp=timestamp, sequence_id=uid, stream_id=uid)
                else:
                    result = detector.predict(image, timestamp=timestamp, sequence_id=uid)
                step_ms = (time.perf_counter() - step) * 1000.
                restore_boxes(result, restore_width, restore_height, args.rotation_deg)
                if args.rotation_square_pad:
                    # 逆旋转后只保留原图区域内的有效框，补边区域没有对应真值。
                    clipped = []
                    for prediction in result:
                        x1, y1, x2, y2 = prediction["bbox_xyxy"]
                        bbox = [max(0., x1), max(0., y1), min(float(row["image_width"]), x2), min(float(row["image_height"]), y2)]
                        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            prediction["bbox_xyxy"] = prediction["xyxy"] = bbox
                            clipped.append(prediction)
                    result = clipped
                if not manages_streams:
                    states[uid] = {name: getattr(detector.pipeline, name) for name in ("camera", "tracker", "previous_t", "sequence")}
                # 保留所有流水线结果核对；评估时沿用原离线最低检测分数 0.25。
                replay = normalize_predictions(result, row["image_width"], row["image_height"], 0.)
                online = normalize_predictions(row["predictions"], row["image_width"], row["image_height"], 0.)
                parity = compare_predictions(online, replay, args.bbox_tolerance, args.score_tolerance)
                record["parity"] = parity
                counts["parity_frames"] += int(parity["within_tolerance"])
                counts["online_predictions"] += len(online)
                counts["replay_predictions"] += len(replay)
                max_bbox = max(max_bbox, parity["bbox_max_abs_error"])
                max_score = max(max_score, parity["score_max_abs_error"])
                truth = truth_objects(row)
                online_metric.add(truth, [p for p in online if p["score"] >= .25], row["inference_wall_ms"], row["decode_ms"])
                replay_metric.add(truth, [p for p in replay if p["score"] >= .25], step_ms, replay_decode_ms)
                reset_reason = None
                if manages_streams:
                    actual_reason = detector.pipeline.last_metadata.get("reset_reason")
                    reset_reason = {"source_gap": "source_gap_exceeds_tracker_max_gap",
                                    "nonincreasing_source_time": "non_increasing_source_time_skipped"}.get(actual_reason, actual_reason)
                if args.reset_each_frame:
                    reset_reason = "explicit_reset_each_frame"
                elif not manages_streams and gap is not None and gap <= 0:
                    reset_reason = "non_increasing_source_time"
                elif not manages_streams and gap is not None and gap > detector.pipeline.tracker.max_gap_s:
                    reset_reason = "source_gap_exceeds_tracker_max_gap"
                timing_fields = ("inference_wall_ms", "decode_ms", "previous_processed_source_sim_time", "processed_source_gap_s",
                                 "tracker_reset_reason", "inference_completed_sim_time", "result_observed_sim_time",
                                 "result_observed_unix_s", "result_observed_perf_counter", "observation_latency_sim_s", "frame_save_ms",
                                 "worker_inference_started_perf_counter", "worker_completed_perf_counter", "detector_frame_metadata")
                replay_row = {**row, "predictions": replay, "replay_step_ms": step_ms,
                              "yolo_profile": profile, "source_yolo_profile": source_profile,
                              "replay_settings_identity": identity, "same_frame_parity": parity["within_tolerance"],
                              "source_online_timing": {key: row[key] for key in timing_fields if key in row}}
                for key in timing_fields:
                    replay_row.pop(key, None)
                replay_row.update({"inference_wall_ms": step_ms, "decode_ms": replay_decode_ms,
                                   "previous_processed_source_sim_time": timestamp-gap if gap is not None else None,
                                   "processed_source_gap_s": gap, "tracker_reset_reason": reset_reason})
                if manages_streams:
                    replay_row["detector_frame_metadata"] = dict(detector.pipeline.last_metadata)
                replay_stream.write(json.dumps(replay_row, ensure_ascii=False, allow_nan=False) + "\n")
            audit.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if counts["frames"] % 200 == 0:
                print(json.dumps({"frames": counts["frames"], "parity_frames": counts["parity_frames"]}), flush=True)
    summary.update({"counts": dict(counts), "dimensions": dict(dimensions), "frames_by_uid": dict(by_uid),
                    "elapsed_s": time.perf_counter() - started, "max_bbox_abs_error": max_bbox, "max_score_abs_error": max_score,
                    "max_frames": args.max_frames})
    if detector is not None:
        summary["online_metrics"] = online_metric.result()
        summary["replay_metrics"] = replay_metric.result()
    write_json(output / "summary.json", summary)
    print(json.dumps({"summary": str(output / "summary.json"), "counts": dict(counts),
                      "max_bbox_abs_error": max_bbox, "max_score_abs_error": max_score}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
