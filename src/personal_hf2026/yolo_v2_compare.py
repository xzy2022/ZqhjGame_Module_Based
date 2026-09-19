# 修改时间：2026-09-19。
# 修改目的：区分V2重复源时间导致的空结果回调与真正完成的模型推理。
# 修改内容：增加跳过原因、实际推理计数和独立耗时与吞吐，精度仍评价所有回调输入。
# 修改时间：2026-09-19。
# 修改目的：避免直接汇总方向清单时把原在线抽样行当成完整在线吞吐。
# 修改内容：识别诊断清单为sampled_online，只计算精度及所选耗时，不报告FPS和场景每分钟误报。
# 修改时间：2026-09-19。
# 修改目的：不把旧回放缺失的解码计时写成实测零毫秒。
# 修改内容：仅将确实存在的decode_ms纳入公共累加器的解码耗时分布。
# 修改时间：2026-09-19。
# 修改目的：避免稀疏方向探针误报率分母及FOV标签造成错误比较。
# 修改内容：方向探针不报FP每分钟，记录实际FOV并核对指定FOV标签。
# 修改时间：2026-09-19。
# 修改目的：让同帧精度归因同时验证输入次数和整帧真值一致性。
# 修改内容：新增重复帧键及共同帧GT审计，共同目标必须同类别、同框且所在帧真值一致。
# 修改时间：2026-09-19。
# 修改目的：避免旧版回放日志继承的在线耗时被误当成离线速度。
# 修改内容：优先读取replay_step_ms，忽略无法溯源的旧回放decode和保存耗时，并按实际输入重算源间隔。
# 修改时间：2026-09-19。
# 修改目的：透明披露V2输出的低于共同评价阈值的恢复框。
# 修改内容：记录原始框数、score阈值剔除数及其中低分恢复框数。
# 修改时间：2026-09-19。
# 修改目的：使重复图像时序身份和离线模型来源可审计。
# 修改内容：帧键加入源时间，补记离线实际资源及每个GT身份的条件判对计数。
# 修改时间：2026-09-19。
# 修改目的：保留实际回放资源及其输入来源，避免复用原在线元数据造成误归因。
# 修改内容：支持 replay_results 目录发现、回放原天气与来源恢复和配置参数留档。
# 修改时间：2026-09-19。
# 修改目的：统一两代检测器的同图精度、实际在线吞吐和方向探针评价口径。
# 修改内容：新增实际日志汇总、共同目标对照以及保留来源的离线连续片段和方向探针清单。
"""只读模型日志并准备诊断清单；不启动模型推理或 UE。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

from .yolo_bias_audit import classification_rates, truth_objects
from .yolo_offline_eval import (
    CLASS_NAMES, ManifestAudit, MetricAccumulator, bbox_iou, file_sha256,
    normalize_predictions, percentile, prepare_output, write_json,
)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def distribution(values):
    values = [float(value) for value in values if value is not None]
    return {"count": len(values), "mean": ratio(sum(values), len(values)),
            "p50": percentile(values, .5), "p95": percentile(values, .95),
            "max": max(values) if values else None}


def frame_key(row):
    return str(row.get("uid", "")) + ":" + str(
        row.get("image_sha256") or row.get("sample_id") or row.get("image_path")) + ":" + str(source_time(row))


def source_time(row):
    return row.get("timestamp_s", row.get("source_sim_time"))


def source_stream(row):
    return str(row.get("sequence_id", row.get("uid", "unknown")))


def finish(accumulator):
    result = accumulator.result()
    result["localized_pair_forced_class_accuracy"] = classification_rates(result)
    result["class_macro_f1_forced_class_id"] = sum(
        item["f1"] for item in result["by_class_forced_class_id"].values()) / 2
    return result


def inspect_run(entry):
    source = Path(entry["input"]).resolve()
    if source.is_dir():
        source = next(source / name for name in ("yolo_sidecar_results.jsonl", "replay_results.jsonl", "predictions.jsonl")
                      if (source / name).is_file())
    run = source.parent
    rows = read_rows(source)
    metadata = read_json(run / "metadata.json")
    summary = read_json(run / "yolo_sidecar_summary.json")
    replay_summary = read_json(run / "summary.json")
    offline_metrics = read_json(run / "metrics.json")
    mode = entry.get("mode", "auto")
    if mode == "auto":
        mode = ("replay" if source.name == "replay_results.jsonl" or (rows and "replay_step_ms" in rows[0]) else
                "offline" if rows and "gt_objects" in rows[0] else "online")
    if mode == "replay" and not metadata and replay_summary.get("run"):
        metadata = read_json(Path(replay_summary["run"]) / "metadata.json")
    if mode == "online" and metadata.get("diagnostic_subset"):
        mode = "sampled_online"
    accumulators = defaultdict(lambda: MetricAccumulator(.5))
    counters, per_frame, matched_objects = Counter(), {}, {}
    identities = defaultdict(Counter)
    source_times, observed_times, gaps, by_uid = defaultdict(list), [], [], defaultdict(list)
    inference_values, decode_values, previous_source = [], [], {}
    active_inference_values, active_rows, active_gaps, skip_reasons = [], [], [], Counter()
    previous_active_source = {}
    for row in rows:
        truth = row.get("gt_objects")
        if truth is None:
            truth = truth_objects(row, counters)
        predictions = normalize_predictions(
            row.get("predictions", []), row.get("image_width", row.get("width", 0)),
            row.get("image_height", row.get("height", 0)), .25)
        counters["raw_emitted_predictions"] += len(row.get("predictions", []))
        counters["predictions_below_score_0_25"] += len(row.get("predictions", [])) - len(predictions)
        counters["recovered_predictions_below_score_0_25"] += sum(
            item["score"] < .25 and bool(item.get("recovered_low_score")) for item in row.get("predictions", []))
        weather = row.get("weather", metadata.get("weather", "unknown"))
        if isinstance(weather, dict):
            weather = weather.get("preset", "unknown")
        timing = (row.get("replay_step_ms", row.get("inference_wall_ms", 0)) if mode == "replay"
                  else row.get("step_ms", row.get("inference_wall_ms", 0)))
        decode = (row.get("decode_ms") if mode != "replay" or "source_online_timing" in row
                  else row.get("replay_decode_ms"))
        inference_values.append(timing)
        decode_values.append(decode)
        frame_metadata = row.get("detector_frame_metadata") or {}
        skipped = bool(frame_metadata.get("skipped"))
        if skipped:
            skip_reasons[str(frame_metadata.get("reset_reason", "unspecified"))] += 1
        else:
            active_rows.append(row)
            active_inference_values.append(timing)
            active_stream, active_timestamp = source_stream(row), source_time(row)
            if active_timestamp is not None:
                if active_stream in previous_active_source:
                    active_gaps.append(active_timestamp - previous_active_source[active_stream])
                previous_active_source[active_stream] = active_timestamp
        uid = str(row.get("uid", "unknown"))
        for group_key in ("overall", f"weather/{weather}", f"uid/{uid}"):
            frame_matches, _, _ = accumulators[group_key].add(truth, predictions, timing, decode or 0)
            if group_key == "overall":
                matches = frame_matches
            if decode is None:
                accumulators[group_key].decode_ms.pop()
        counters["empty_gt_frames"] += not truth
        counters["empty_gt_frames_with_prediction"] += not truth and bool(predictions)
        counters["empty_gt_predictions"] += len(predictions) if not truth else 0
        for item in truth:
            identities[f"{CLASS_NAMES[item['class_id']]}/{item.get('object_id')}"]["gt_instances"] += 1
        matched_pred = {match["prediction_index"] for match in matches}
        for index, prediction in enumerate(predictions):
            if index in matched_pred:
                continue
            max_iou = max((bbox_iou(item["bbox_xyxy"], prediction["bbox_xyxy"]) for item in truth), default=0)
            counters["unmatched_prediction_max_iou_lt_0_1"] += max_iou < .1
            counters["unmatched_prediction_max_iou_0_1_to_0_5"] += .1 <= max_iou < .5
            counters["unmatched_prediction_max_iou_ge_0_5"] += max_iou >= .5
        key = frame_key(row)
        per_frame[key] = sorted((int(item["class_id"]), str(item.get("object_id")),
                                 tuple(float(value) for value in item["bbox_xyxy"])) for item in truth)
        for match in matches:
            gt, pred = truth[match["gt_index"]], predictions[match["prediction_index"]]
            object_key = key + ":" + str(gt.get("object_id", match["gt_index"]))
            matched_objects[object_key] = {"class_id": gt["class_id"], "correct": gt["class_id"] == pred["class_id"],
                                            "bbox_xyxy": gt["bbox_xyxy"], "frame_key": key}
            identity = identities[f"{CLASS_NAMES[gt['class_id']]}/{gt.get('object_id')}"]
            identity["localized_pairs"] += 1
            identity["correct_class"] += gt["class_id"] == pred["class_id"]
        timestamp = source_time(row)
        if timestamp is not None:
            stream = source_stream(row)
            source_times[stream].append(timestamp)
            if stream in previous_source:
                gaps.append(timestamp - previous_source[stream])
            previous_source[stream] = timestamp
        by_uid[uid].append(row)
        if mode == "online" and row.get("result_observed_perf_counter") is not None:
            observed_times.append(row["result_observed_perf_counter"])
    metrics = finish(accumulators["overall"])
    source_seconds = sum(max(values) - min(values) for values in source_times.values() if len(values) > 1)
    sparse_probe = bool(metadata.get("diagnostic_subset")) or replay_summary.get("sample_every", 1) > 1
    rate_seconds = None if sparse_probe else source_seconds
    fov_values = sorted({float(value) for row in rows
                         if (value := row.get("fov_deg", row.get("observed_gimbal_fov_deg", metadata.get("effective_fov_deg")))) is not None})
    duration = metadata.get("duration") if mode == "online" else None
    result = {"label": entry["label"], "mode": mode, "input": str(source),
              "input_sha256": file_sha256(source), "fov_deg": entry.get("fov_deg", metadata.get("effective_fov_deg")),
              "observed_fov_values": fov_values,
              "observed_fov_matches_requested": all(abs(value - float(entry["fov_deg"])) < 1e-6 for value in fov_values) if fov_values and entry.get("fov_deg") is not None else None,
              "metrics": metrics, "by_weather": {k[8:]: finish(v) for k, v in accumulators.items() if k.startswith("weather/")},
              "by_uid": {k[4:]: finish(v) for k, v in accumulators.items() if k.startswith("uid/")},
              "unique_frame_keys": len(per_frame), "counts": dict(counters),
              "inference_completion_counts": {"completed_callback_rows": len(rows), "active_inference_rows": len(active_rows),
                                               "skipped_callback_rows": len(rows) - len(active_rows), "skip_reasons": dict(skip_reasons)},
              "duplicate_frame_key_count": len(rows) - len(per_frame),
              "by_gt_identity": {key: {**value, "conditional_accuracy": ratio(value["correct_class"], value["localized_pairs"])} for key, value in sorted(identities.items())},
              "source_uav_seconds": source_seconds,
              "fp_rate_scope": "sparse_probe_per_minute_suppressed" if sparse_probe else "evaluated_events_per_selected_stream_source_span",
              "empty_background_frame_false_positive_rate": ratio(counters["empty_gt_frames_with_prediction"], counters["empty_gt_frames"]),
              "class_agnostic_fp_per_uav_source_minute": ratio(metrics["localization_class_agnostic"]["fp"] * 60, rate_seconds),
              "non_vehicle_fp_proxy_per_uav_source_minute": ratio(counters["unmatched_prediction_max_iou_lt_0_1"] * 60, rate_seconds),
              "class_agnostic_fp_per_scene_sim_minute": ratio(metrics["localization_class_agnostic"]["fp"] * 60, duration),
              "source_gap_s": distribution(gaps), "source_gap_over_0_25_count": sum(value > .25 for value in gaps),
              "source_gap_over_0_25_ratio": ratio(sum(value > .25 for value in gaps), len(gaps)),
              "active_inference_source_gap_s": distribution(active_gaps),
              "active_inference_source_gap_over_0_25_ratio": ratio(sum(value > .25 for value in active_gaps), len(active_gaps)),
              "resources": metadata.get("resources", {}) if mode == "online" else replay_summary.get("runtime", offline_metrics.get("detector_runtime_metadata", metadata.get("resources", {}))),
              "replay_settings": {key: replay_summary[key] for key in ("reset_each_frame", "rotation_deg", "rotation_square_pad", "detector_image_rotation_deg", "sample_every") if key in replay_summary},
              "latency": {"inference_ms": distribution(inference_values),
                          "active_inference_ms": distribution(active_inference_values),
                          "decode_ms": distribution(decode_values),
                          "frame_save_ms": distribution([r.get("frame_save_ms") for r in rows] if mode == "online" else [])},
              "legacy_replay_inherited_online_timing_ignored": mode == "replay" and any("source_online_timing" not in r for r in rows)}
    if mode == "online":
        submissions_path = run / "yolo_sidecar_submissions.jsonl"
        submissions = read_rows(submissions_path) if submissions_path.is_file() else rows
        starts = [r["submitted_perf_counter"] for r in submissions if r.get("submitted_perf_counter") is not None]
        wall_seconds = max(observed_times) - min(starts) if starts and observed_times else None
        completion_seconds = max(observed_times) - min(observed_times) if len(observed_times) > 1 else None
        uid_timing = {}
        for uid, uid_rows in by_uid.items():
            times = sorted(r["result_observed_perf_counter"] for r in uid_rows if r.get("result_observed_perf_counter") is not None)
            intervals = [right - left for left, right in zip(times, times[1:])]
            uid_gaps = [r["processed_source_gap_s"] for r in uid_rows if r.get("processed_source_gap_s") is not None]
            uid_active = [r for r in uid_rows if not (r.get("detector_frame_metadata") or {}).get("skipped")]
            uid_timing[uid] = {"completed": len(uid_rows), "wall_window_fps": ratio(len(uid_rows), wall_seconds),
                               "active_inference_count": len(uid_active), "skipped_count": len(uid_rows) - len(uid_active),
                               "active_inference_wall_window_fps": ratio(len(uid_active), wall_seconds),
                               "completion_fps": ratio(len(times) - 1, max(times) - min(times)) if len(times) > 1 else None,
                               "result_observation_interval_wall_s": distribution(intervals),
                               "source_gap_s": distribution(uid_gaps),
                               "source_gap_over_0_25_ratio": ratio(sum(value > .25 for value in uid_gaps), len(uid_gaps)),
                               "wall_interval_over_0_25_ratio": ratio(sum(value > .25 for value in intervals), len(intervals))}
        worker_times = [r["worker_completed_perf_counter"] for r in rows if r.get("worker_completed_perf_counter") is not None]
        active_observed = [r["result_observed_perf_counter"] for r in active_rows if r.get("result_observed_perf_counter") is not None]
        active_worker = [r["worker_completed_perf_counter"] for r in active_rows if r.get("worker_completed_perf_counter") is not None]
        result["online"] = {"window_semantics": "首个 accepted submission 到末个 Runner 结果观察；包含启动等待、IPC和保存，不含UE启动前开销。",
                            "wall_window_s": wall_seconds, "wall_window_total_fps": ratio(len(rows), wall_seconds),
                            "active_inference_wall_window_fps": ratio(len(active_rows), wall_seconds),
                            "completion_wall_span_s": completion_seconds,
                            "result_observation_completion_fps": ratio(len(observed_times) - 1, completion_seconds),
                            "worker_completion_fps": ratio(len(worker_times) - 1, max(worker_times) - min(worker_times)) if len(worker_times) > 1 else None,
                            "active_result_observation_completion_fps": ratio(len(active_observed) - 1, max(active_observed) - min(active_observed)) if len(active_observed) > 1 else None,
                            "active_worker_completion_fps": ratio(len(active_worker) - 1, max(active_worker) - min(active_worker)) if len(active_worker) > 1 else None,
                            "requested_sim_duration_s": duration, "frames_per_requested_sim_second": ratio(len(rows), duration),
                            "requested_sim_seconds_per_wall_second": ratio(duration, wall_seconds),
                            "counts": summary.get("counts", {"submitted": len(submissions), "completed": len(rows)}),
                            "per_uid": uid_timing}
        result["latency"]["submitted_to_result_observed_wall_ms"] = distribution([
            (r["result_observed_perf_counter"] - r["submitted_perf_counter"]) * 1000 for r in rows
            if r.get("result_observed_perf_counter") is not None and r.get("submitted_perf_counter") is not None])
        result["latency"]["result_observation_sim_s"] = distribution([r.get("observation_latency_sim_s") for r in rows])
    elif mode != "sampled_online":
        result["offline_algorithm_fps"] = ratio(len(active_inference_values) * 1000, sum(active_inference_values))
        result["offline_all_callback_fps"] = metrics["timing"]["algorithm_step"]["fps"]
    else:
        result["timing_boundary"] = "仅为源在线运行的稀疏抽样行，不能计算完整在线或离线吞吐。"
    return result, per_frame, matched_objects


def compare(args):
    output, _ = prepare_output(args.output)
    spec = read_json(args.spec)
    entries = spec if isinstance(spec, list) else spec["runs"]
    results, frame_sets, frame_truths, pairs = {}, {}, {}, {}
    for entry in entries:
        result, frames, matches = inspect_run(entry)
        results[entry["label"]], frame_sets[entry["label"]], pairs[entry["label"]] = result, set(frames), matches
        frame_truths[entry["label"]] = frames
    comparisons = []
    for left, right in itertools.combinations(results, 2):
        common_frames = frame_sets[left] & frame_sets[right]
        if not common_frames:
            continue
        matching_truth_frames = {key for key in common_frames if frame_truths[left][key] == frame_truths[right][key]}
        common_objects = pairs[left].keys() & pairs[right].keys()
        by_class = {}
        for class_id, name in CLASS_NAMES.items():
            keys = [key for key in common_objects if pairs[left][key]["class_id"] == class_id
                    and pairs[right][key]["class_id"] == class_id
                    and pairs[left][key]["bbox_xyxy"] == pairs[right][key]["bbox_xyxy"]
                    and pairs[left][key]["frame_key"] in matching_truth_frames]
            by_class[name] = {"common_localized_objects": len(keys),
                              "left_accuracy": ratio(sum(pairs[left][k]["correct"] for k in keys), len(keys)),
                              "right_accuracy": ratio(sum(pairs[right][k]["correct"] for k in keys), len(keys))}
        comparisons.append({"left": left, "right": right, "common_frames": len(common_frames),
                            "identical_frame_set": frame_sets[left] == frame_sets[right],
                            "unique_same_frame_inputs": frame_sets[left] == frame_sets[right] and not results[left]["duplicate_frame_key_count"] and not results[right]["duplicate_frame_key_count"],
                            "common_frame_gt_mismatch_count": len(common_frames) - len(matching_truth_frames),
                            "common_localized_gt": by_class})
    report = {"runs": results, "same_frame_comparisons": comparisons, "semantics": {
        "thresholds": "IoU>=0.5；score>=0.25；uncertain保留并按class_id强制二分类。另列accepted指标。",
        "macro": "类别宏F1为两类F1均值；micro F1来自总体TP/FP/FN，与队友把uncertain计错的宏F1不能直接相减。",
        "false_positive": "非车辆FP代理指未匹配且与任意GT最大IoU<0.1的预测；投影标签未人工可见性核验，不等于已证实非车辆。定位偏差/重复框另计。",
        "time": "实际wall FPS来自perf_counter。source、observation sim时间不是曝光时间；稀疏片段FP/min只以已取样序列跨度为分母。",
        "online_comparison": "不同UE运行的场景/处理帧不完全相同；同图回放用于精度归因，真实在线用于吞吐观察。",
        "offline": "离线算法FPS不含UE渲染、IPC及最新帧覆盖；不替代在线FPS。",
        "skipped_inputs": "精度包含所有回调输入，原生skipped返回空仍计入系统漏检；active_inference耗时和FPS排除这些未推理回调，completed默认计回调。",
        "labels": "UE投影GT属于审计元数据；空GT不保证人工可见性意义的空背景。"}}
    write_json(output / "comparison.json", report)
    lines = ["# V1-v3 / V2 统一日志评价", "", "IoU≥0.5，score≥0.25；分类按 class_id。GT 为未验证可见性的 UE 投影。", "",
             "| 项目 | 帧 | 定位P/R/F1 | 分类P/R/F1 | 真车条件判对 | 诱饵条件判对 | 空GT帧误报率 |", "|---|---:|---|---|---:|---:|---:|"]
    fmt = lambda value: "n/a" if value is None else f"{value:.2%}"
    for label, result in results.items():
        metric = result["metrics"]
        loc, cls = metric["localization_class_agnostic"], metric["detection_class_aware_forced_class_id"]
        rates = metric["localized_pair_forced_class_accuracy"]
        lines.append(f"| {label} | {metric['frames']} | {'/'.join(fmt(loc[k]) for k in ('precision','recall','f1'))} | {'/'.join(fmt(cls[k]) for k in ('precision','recall','f1'))} | {fmt(rates['real_vehicle']['accuracy'])} | {fmt(rates['model_prop']['accuracy'])} | {fmt(result['empty_background_frame_false_positive_rate'])} |")
    lines += ["", "指标定义、真实墙钟吞吐、每机间隔/覆盖计数、误报代理及同图共同GT比较见 comparison.json。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": list(results)}, ensure_ascii=False))


def rank_key(seed, key):
    return hashlib.sha256((seed + str(key)).encode()).hexdigest()


def select_offline(args):
    output, _ = prepare_output(args.output)
    provenance, exported = [], []
    for manifest in args.manifest:
        rows = read_rows(manifest)
        groups = defaultdict(list)
        # 在原序列和类别边界切片，片段内部保持连续；不读取预测结果。
        for (weather, sequence, category), group in itertools.groupby(
                rows, lambda row: (row["weather"], row["sequence_id"], row["category"])):
            group = list(group)
            for offset in range(0, len(group), args.chunk_size):
                chunk = group[offset:offset + args.chunk_size]
                key = f"{sequence}:{category}:{chunk[0]['sequence_index']}"
                groups[weather].append((rank_key(args.seed, key), chunk))
        for weather, chunks in sorted(groups.items()):
            chunks.sort(key=lambda item: item[0])
            selected, used = [], set()
            # 每类至少四分之一配额，剩余四分之一按固定哈希选片，确保稀有诱饵不会被背景淹没。
            for category in ("no_vehicle", "target_only", "decoy_only", None):
                remaining = args.per_weather // 4 if category is not None else args.per_weather - len(selected)
                for index, (rank, chunk) in enumerate(chunks):
                    if not remaining:
                        break
                    if index in used or (category is not None and chunk[0]["category"] != category):
                        continue
                    taken = chunk[:remaining]
                    selected.extend((rank, item) for item in taken)
                    used.add(index)
                    remaining -= len(taken)
            for rank, item in selected:
                exported.append({**item, "source_manifest": str(manifest.resolve()),
                                 "source_sequence_id": item["sequence_id"], "source_sequence_index": item["sequence_index"],
                                 "sequence_id": item["sequence_id"] + "-diagnostic-" + rank[:12]})
        provenance.append({"path": str(manifest.resolve()), "sha256": file_sha256(manifest), "frames": len(rows)})
    exported.sort(key=lambda row: (row["fov_deg"], row["weather"], row["sequence_id"], row["source_sequence_index"]))
    summary = {"source_manifests": provenance, "selection_seed": args.seed, "per_weather": args.per_weather,
               "policy": f"每FOV/天气{args.per_weather}帧；背景/真车单类/诱饵单类各{args.per_weather // 4}帧，余量固定SHA排序补足。连续片段最多{args.chunk_size}帧；不读取预测。属于标签分层诊断，不能估计全分布平均。",
               "training_overlap": "未拿到队友实际train/val/test逐图清单，不能核实本清单重叠；FOV48旧权重已见部分图，两域同地图/身份，不是盲测。",
               "image_copied_bytes": 0, "by_fov": {}}
    total_bytes = 0
    for fov, group in itertools.groupby(exported, lambda row: row["fov_deg"]):
        group = list(group)
        audit, previous_sequence, index = ManifestAudit(), None, 0
        for item in group:
            if item["sequence_id"] != previous_sequence:
                index = 0
                previous_sequence = item["sequence_id"]
            item["sequence_index"] = index
            index += 1
            audit.add(item, audit.frames + 1, verify_image_path=True)
            total_bytes += Path(item["image_path"]).stat().st_size
        name = f"fov{int(fov)}"
        folder = output / name
        folder.mkdir()
        write_rows(folder / "eval_frames.jsonl", group)
        summary["by_fov"][name] = {"audit": audit.result(truncated=False),
                                   "manifest_sha256": file_sha256(folder / "eval_frames.jsonl"),
                                   "weather_classes": {weather: dict(Counter(CLASS_NAMES[obj['class_id']] for item in group if item['weather'] == weather for obj in item['gt_objects'])) for weather in sorted({item['weather'] for item in group})}}
    summary["referenced_image_bytes"] = total_bytes
    summary["frames"] = len(exported)
    write_json(output / "selection.json", summary)
    print(json.dumps({"output": str(output), "frames": len(exported), "referenced_image_gib": total_bytes / 2**30, "image_copied_bytes": 0}, ensure_ascii=False))


def spread(items, count):
    if count >= len(items):
        return items
    return [items[index * len(items) // count] for index in range(count)] if count else []


def select_probe(args):
    output, _ = prepare_output(args.output)
    run = args.run.resolve()
    source = run / "yolo_sidecar_results.jsonl"
    rows = read_rows(source)
    selected, groups = set(), defaultdict(list)
    for index, row in enumerate(rows):
        for item in truth_objects(row, Counter()):
            groups[(item["class_id"], str(item["object_id"]))].append(index)
    # 每类先按身份轮询，再均匀覆盖该身份的时间；GT只用于诊断抽样，不输入模型。
    for class_id in CLASS_NAMES:
        pools = [spread(indices, min(len(indices), args.frames)) for key, indices in sorted(groups.items()) if key[0] == class_id]
        quota = args.frames // 3
        added = 0
        for candidates in itertools.zip_longest(*(spread(pool, max(1, quota // max(1, len(pools)))) for pool in pools)):
            for index in candidates:
                if index is not None and index not in selected and added < quota:
                    selected.add(index)
                    added += 1
    empty = [index for index, row in enumerate(rows) if not row.get("ground_truth")]
    selected.update(spread(empty, min(args.frames - len(selected), args.frames // 3)))
    remaining = [index for index in range(len(rows)) if index not in selected]
    selected.update(spread(remaining, args.frames - len(selected)))
    probe = []
    classes, identities = Counter(), Counter()
    for index in sorted(selected):
        row = dict(rows[index], probe_source_row_index=index)
        image = Path(row["image_path"])
        if not image.is_absolute():
            image = run / image
        if not image.is_file():
            raise FileNotFoundError(image)
        row["image_path"] = str(image.resolve())
        probe.append(row)
        for item in truth_objects(row, Counter()):
            classes[CLASS_NAMES[item["class_id"]]] += 1
            identities[f"{CLASS_NAMES[item['class_id']]}/{item['object_id']}"] += 1
    write_rows(output / "yolo_sidecar_results.jsonl", probe)
    metadata = read_json(run / "metadata.json")
    metadata["diagnostic_subset"] = {"source_run": str(run), "original_results_sha256": file_sha256(source)}
    write_json(output / "metadata.json", metadata)
    report = {"source_run": str(run), "source_results_sha256": file_sha256(source), "frames": len(probe),
              "selected_submission_ids": [row.get("submission_id") for row in probe], "classes": dict(classes),
              "objects_by_identity": dict(identities), "empty_gt_frames": sum(not row["ground_truth"] for row in probe),
              "selection": "各类约1/3帧按GT身份均匀时间取样，背景约1/3，余量全时域补足；不读取预测，非全分布估计。",
              "probe_protocol": "两模型同一清单，0/90/180/270，reset-each-frame。V2矩形输入使raw旋转有尺度混杂；主方向诊断四角均启用rotation-square-pad，另列raw四角；主全序列对照不pad。",
              "copied_image_bytes": 0}
    write_json(output / "selection.json", report)
    print(json.dumps({"output": str(output), "frames": len(probe), "classes": dict(classes), "identities": len(identities)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--spec", type=Path, required=True, help="JSON列表，每项label/input/mode以及可选fov_deg。")
    comparison.add_argument("--output", type=Path, required=True)
    comparison.set_defaults(action=compare)
    selection = sub.add_parser("select-offline")
    selection.add_argument("--manifest", action="append", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--per-weather", type=int, default=400)
    selection.add_argument("--chunk-size", type=int, default=50)
    selection.add_argument("--seed", default="v2-six-weather-diagnostic-v1")
    selection.set_defaults(action=select_offline)
    probe = sub.add_parser("select-probe")
    probe.add_argument("--run", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--frames", type=int, default=240)
    probe.set_defaults(action=select_probe)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
