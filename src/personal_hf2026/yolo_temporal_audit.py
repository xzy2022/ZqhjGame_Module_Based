# 修改时间：2026-09-19。
# 修改目的：用同一短测的前段和全程独立核对历史分类偏置是否复现。
# 修改内容：增加按已记录图像接收仿真时间截取窗口的参数和产物口径字段。
# 修改时间：2026-09-19。
# 修改目的：区分分类决策阈值与置信拒识对系统偏置的影响。
# 修改内容：按预先指定的六个真车概率阈值报告两类判对率和均衡准确率。
# 修改时间：2026-09-19。
# 修改目的：定位小目标与具体诱饵的单帧偏置并评估概率拒识的取舍。
# 修改内容：增加短边面积身份分组、固定置信阈值表和新日志对 EMA 反解的数值核验。
# 修改时间：2026-09-19。
# 修改目的：用已保存的在线结果区分跳帧重置、时序融合和目标关联导致的分类偏置。
# 修改内容：审计逐机时间与轨迹身份并反解可验证的单帧概率，输出匹配明细和分组统计。
"""读取真实旁路日志进行离线诊断，不加载模型、不启动仿真。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import fmean


NAMES = {"TargetVehicle": "real_vehicle", "DecoyVehicle": "model_prop"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def distribution(values):
    return dict(count=len(values), mean=fmean(values) if values else None,
                p05=percentile(values, .05), p50=percentile(values, .5), p95=percentile(values, .95),
                max=max(values) if values else None)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def matches(truth, predictions, threshold):
    candidates = [(iou(gt["bbox"], prediction["bbox_xyxy"]), i, j)
                  for i, gt in enumerate(truth) for j, prediction in enumerate(predictions)]
    used_gt, used_prediction = set(), set()
    output = []
    for overlap, i, j in sorted(candidates, reverse=True):
        if overlap < threshold or i in used_gt or j in used_prediction:
            continue
        used_gt.add(i)
        used_prediction.add(j)
        output.append((i, j, overlap))
    return output


def predicted_name(real_probability, unknown_threshold):
    if real_probability is None:
        return None
    if max(real_probability, 1 - real_probability) < unknown_threshold:
        return "uncertain"
    return "real_vehicle" if real_probability >= .5 else "model_prop"


def classification(rows):
    output = {}
    for truth in NAMES.values():
        selected = [row for row in rows if row["gt_class"] == truth]
        raw = [row for row in selected if row["single_name"] is not None]
        confusion = Counter(row["fused_name"] for row in selected)
        single = Counter(row["single_name"] for row in raw)
        output[truth] = {
            "matched": len(selected), "fused_confusion": dict(confusion),
            "fused_accuracy": confusion[truth] / len(selected) if selected else None,
            "single_available": len(raw), "single_confusion": dict(single),
            "single_accuracy": single[truth] / len(raw) if raw else None,
            "single_argmax_accuracy": sum((row["single_real_probability"] >= .5) == (truth == "real_vehicle") for row in raw) / len(raw) if raw else None,
            "fused_argmax_accuracy": sum((row["fused_real_probability"] >= .5) == (truth == "real_vehicle") for row in selected) / len(selected) if selected else None,
            "single_correct_fused_wrong": sum(row["single_name"] == truth and row["fused_name"] != truth for row in raw),
            "single_wrong_fused_correct": sum(row["single_name"] != truth and row["fused_name"] == truth for row in raw),
            "both_wrong": sum(row["single_name"] != truth and row["fused_name"] != truth for row in raw),
            "single_real_probability": distribution([row["single_real_probability"] for row in raw]),
            "fused_real_probability": distribution([row["fused_real_probability"] for row in selected]),
            "single_class_confidence": distribution([max(row["single_real_probability"], 1 - row["single_real_probability"]) for row in raw]),
            "fused_class_confidence": distribution([max(row["fused_real_probability"], 1 - row["fused_real_probability"]) for row in selected]),
        }
    return output


def grouped_classification(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {label: classification(group) for label, group in sorted(groups.items())}


def size_bucket(value, boundaries):
    for lower, upper in zip([0, *boundaries], [*boundaries, float("inf")]):
        if value < upper:
            return f"{lower}-{upper}"


def rejection_tradeoff(rows):
    output = {}
    for stage in ("single", "fused"):
        stage_output = {}
        for threshold in (.5, .7, .9, .99):
            by_class = {}
            for truth in NAMES.values():
                selected = [row for row in rows if row["gt_class"] == truth and row[f"{stage}_real_probability"] is not None]
                predictions = [predicted_name(row[f"{stage}_real_probability"], threshold) for row in selected]
                accepted = sum(name != "uncertain" for name in predictions)
                correct = sum(name == truth for name in predictions)
                wrong = accepted - correct
                by_class[truth] = dict(matched=len(selected), accepted=accepted, correct=correct, wrong=wrong,
                                       rejected=len(selected) - accepted,
                                       coverage=accepted / len(selected) if selected else None,
                                       correct_fraction=correct / len(selected) if selected else None,
                                       wrong_fraction=wrong / len(selected) if selected else None,
                                       accepted_accuracy=correct / accepted if accepted else None)
            stage_output[str(threshold)] = by_class
        output[stage] = stage_output
    return output


def decision_threshold_tradeoff(rows):
    output = {}
    for stage in ("single", "fused"):
        stage_output = {}
        for threshold in (.5, .9, .99, .999, .9999, .99999):
            by_class = {}
            for truth in NAMES.values():
                selected = [row for row in rows if row["gt_class"] == truth and row[f"{stage}_real_probability"] is not None]
                correct = sum((row[f"{stage}_real_probability"] > threshold) == (truth == "real_vehicle") for row in selected)
                by_class[truth] = dict(matched=len(selected), correct=correct,
                                       accuracy=correct / len(selected) if selected else None)
            accuracies = [value["accuracy"] for value in by_class.values() if value["accuracy"] is not None]
            stage_output[str(threshold)] = dict(by_class=by_class,
                                                balanced_accuracy=fmean(accuracies) if len(accuracies) == 2 else None)
        output[stage] = stage_output
    return output


def audit(args):
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    smoothing = float(config["tracker"]["smoothing"])
    unknown_threshold = float(config["unknown_threshold"])
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    files = sorted(args.input.rglob("yolo_sidecar_results.jsonl"))
    rows, tracks, run_stats = [], {}, {}
    reset_examples, reconstruction, inverse_validation = [], Counter(), []
    for source in files:
        summary_path = source.with_name("yolo_sidecar_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        resource = summary.get("resources", {})
        if resource.get("config_sha256", config_sha) != config_sha:
            raise ValueError(f"配置哈希不一致，不能假设原 EMA 参数：{source}")
        run = source.parent.name
        previous_t, previous_track = {}, {}
        gaps, reset_reasons, counts = [], Counter(), Counter()
        for line in source.open(encoding="utf-8"):
            frame = json.loads(line)
            if args.max_received_sim_time is not None and frame["image_received_sim_time"] > args.max_received_sim_time:
                continue
            uid = str(frame["uid"])
            timestamp = float(frame["source_sim_time"])
            dt = frame.get("processed_source_gap_s")
            counts["frames"] += 1
            if dt is not None:
                gaps.append(dt)
                counts["gaps_over_0_25"] += int(dt > .25)
                counts["gaps_non_increasing"] += int(dt <= 0)
            if uid in previous_t:
                counts["gap_field_mismatches"] += int(dt is None or abs(timestamp - previous_t[uid] - dt) > 1e-6)
                counts["previous_timestamp_mismatches"] += int(frame.get("previous_processed_source_sim_time") is None or abs(frame["previous_processed_source_sim_time"] - previous_t[uid]) > 1e-6)
            previous_t[uid] = timestamp
            if frame.get("tracker_reset_reason"):
                reset_reasons[frame["tracker_reset_reason"]] += 1
                reset_examples.append(dict(run=run, uid=uid, frame_no=frame["frame_no"],
                                           source_sim_time=timestamp, dt=dt,
                                           predictions=len(frame["predictions"]), ground_truth=len(frame["ground_truth"])))
            probabilities = []
            for prediction in frame["predictions"]:
                key = (uid, prediction["track_id"])
                hits = prediction["track_hits"]
                fused = float(prediction["class_confidence"])
                if prediction["class_id"] == 1:
                    fused = 1 - fused
                direct = prediction.get("single_frame_probabilities")
                if direct is not None:
                    single, method = float(direct[0]), "saved_directly"
                    if hits == 1:
                        inverse_validation.append(abs(single - fused))
                    elif key in previous_track and previous_track[key][0] + 1 == hits:
                        reconstructed = (fused - smoothing * previous_track[key][1]) / (1 - smoothing)
                        inverse_validation.append(abs(single - reconstructed))
                elif hits == 1:
                    single, method = fused, "track_birth"
                elif key in previous_track and previous_track[key][0] + 1 == hits:
                    # 当前配置无背景概率，二分类归一化 EMA 可精确反解。
                    single = (fused - smoothing * previous_track[key][1]) / (1 - smoothing)
                    method = "inverse_ema_contiguous_hits"
                    if not -1e-5 <= single <= 1 + 1e-5:
                        single, method = None, "inverse_outside_probability_range"
                    elif single is not None:
                        single = max(0, min(1, single))
                else:
                    single, method = None, "missing_predecessor_or_hit_gap"
                previous_track[key] = (hits, fused)
                probabilities.append((fused, single, method))
                reconstruction[method] += 1
            paired = matches(frame["ground_truth"], frame["predictions"], args.iou)
            counts["matches"] += len(paired)
            counts["ground_truth"] += len(frame["ground_truth"])
            counts["predictions"] += len(frame["predictions"])
            for gi, pi, overlap in paired:
                truth, prediction = frame["ground_truth"][gi], frame["predictions"][pi]
                fused, single, method = probabilities[pi]
                key = f"{run}/{uid}/{prediction['track_id']}"
                state = tracks.setdefault(key, dict(run=run, uid=uid, track_id=prediction["track_id"],
                                                     ids=set(), classes=set(), previous_id=None,
                                                     previous_class=None, id_switches=0, class_switches=0, matched=0))
                target = str(truth["target_id"])
                gt_class = NAMES.get(truth["class"], truth["class"])
                state["id_switches"] += int(state["previous_id"] is not None and state["previous_id"] != target)
                state["class_switches"] += int(state["previous_class"] is not None and state["previous_class"] != gt_class)
                state["previous_id"], state["previous_class"] = target, gt_class
                state["ids"].add(target)
                state["classes"].add(gt_class)
                state["matched"] += 1
                box = truth["bbox"]
                rows.append(dict(run=run, uid=uid, frame_no=frame["frame_no"], source_sim_time=timestamp,
                                 image_received_sim_time=frame["image_received_sim_time"],
                                 source_gap_s=dt, track_key=key, target_id=target, gt_class=gt_class,
                                 track_hits=prediction["track_hits"], iou=overlap,
                                 detector_confidence=prediction["detector_confidence"],
                                 fused_name=prediction["class_name"], single_name=predicted_name(single, unknown_threshold),
                                 fused_real_probability=fused, single_real_probability=single,
                                 single_probability_source=method, gt_width=box[2] - box[0], gt_height=box[3] - box[1],
                                 gt_area=(box[2] - box[0]) * (box[3] - box[1])))
        run_stats[run] = dict(source=str(source), counts=dict(counts), source_gaps=distribution(gaps),
                              reset_reasons=dict(reset_reasons), submission_counts=summary.get("counts", {}),
                              weights_sha256=resource.get("weights_sha256"))
    for row in rows:
        row["track_ever_switches_gt_identity"] = len(tracks[row["track_key"]]["ids"]) > 1
        row["track_ever_switches_gt_class"] = len(tracks[row["track_key"]]["classes"]) > 1
    serial_tracks = [{**value, "key": key, "ids": sorted(value["ids"]), "classes": sorted(value["classes"])}
                     for key, value in sorted(tracks.items())]
    stats = {
        "input": str(args.input), "config": str(args.config), "config_sha256": config_sha,
        "time_window": {"field": "image_received_sim_time", "inclusive_upper_bound": args.max_received_sim_time,
                        "semantics": "Runner observation score_view.sim_time when image was submitted; not verified exposure time."},
        "iou_threshold": args.iou, "smoothing": smoothing, "unknown_threshold": unknown_threshold,
        "single_probability_method": "Prefer saved single_frame_probabilities; otherwise inverse EMA q_t = alpha*q_prev + (1-alpha)*p_t only for contiguous track hits; first-hit q_t=p_t. Requires normalized two-class probabilities and zero background as implemented in arrays_for_tracker.",
        "ground_truth_caveat": "UE frame-projected boxes and target IDs are oracle audit metadata; occlusion and exposure alignment are not independently verified.",
        "selection_caveat": "Classification is conditional on IoU-matched emitted detections; raw probability inversion does not rerun candidate selection, recovery, or association.",
        "runs": run_stats, "reconstruction_counts": dict(reconstruction),
        "inverse_vs_direct_absolute_error": distribution(inverse_validation),
        "total_frames": sum(run["counts"]["frames"] for run in run_stats.values()),
        "source_gap_comparisons": sum(run["source_gaps"]["count"] for run in run_stats.values()),
        "source_gap_resets": sum(run["counts"].get("gaps_over_0_25", 0) for run in run_stats.values()),
        "non_increasing_resets": sum(run["counts"].get("gaps_non_increasing", 0) for run in run_stats.values()),
        "classification": classification(rows), "by_run": grouped_classification(rows, lambda row: row["run"]),
        "by_track_hits": grouped_classification(rows, lambda row: "1" if row["track_hits"] == 1 else "2-5" if row["track_hits"] < 6 else "6+"),
        "by_gt_identity_switch": grouped_classification(rows, lambda row: row["track_ever_switches_gt_identity"]),
        "by_gt_class_switch": grouped_classification(rows, lambda row: row["track_ever_switches_gt_class"]),
        "by_object_size": grouped_classification(rows, lambda row: "min_lt_16" if min(row["gt_width"], row["gt_height"]) < 16 else "min_16_to_31" if min(row["gt_width"], row["gt_height"]) < 32 else "min_ge_32"),
        "by_gt_short_side_px": grouped_classification(rows, lambda row: size_bucket(min(row["gt_width"], row["gt_height"]), [12, 20, 32, 48])),
        "by_gt_area_px2": grouped_classification(rows, lambda row: size_bucket(row["gt_area"], [144, 400, 1024, 2304])),
        "by_target_id": grouped_classification(rows, lambda row: row["target_id"]),
        "by_run_target_id": grouped_classification(rows, lambda row: f"{row['run']}/{row['target_id']}"),
        "confidence_rejection_tradeoff": rejection_tradeoff(rows),
        "real_class_decision_threshold_tradeoff": decision_threshold_tradeoff(rows),
        "track_count": len(tracks), "tracks_switching_gt_identity": sum(len(track["ids"]) > 1 for track in serial_tracks),
        "tracks_switching_gt_class": sum(len(track["classes"]) > 1 for track in serial_tracks),
        "identity_switch_events": sum(track["id_switches"] for track in serial_tracks),
        "class_switch_events": sum(track["class_switches"] for track in serial_tracks),
        "reset_examples": reset_examples,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "temporal_audit.json", stats)
    write_json(args.output / "tracks.json", serial_tracks)
    with (args.output / "matched_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = ["# 在线时序审计", "", f"来源：`{args.input}`。", "",
             "仅分析已输出且 IoU≥0.5 定位匹配的预测；GT 是 UE 投影审计值。单帧结果由原始日志直接读取或由连续 track_hits 的 EMA 反解。", "",
             "| 类别 | 匹配数 | 融合 argmax 判对率 | 单帧 argmax 判对率 | 融合 class_name 判对率 |", "|---|---:|---:|---:|---:|"]
    for name, value in stats["classification"].items():
        lines.append(f"| {name} | {value['matched']} | {value['fused_argmax_accuracy']:.4%} | {value['single_argmax_accuracy']:.4%} | {value['fused_accuracy']:.4%} |")
    lines += ["", f"GT 匹配轨迹 {len(tracks)} 条；混入不同 GT ID 的轨迹 {stats['tracks_switching_gt_identity']} 条；混入不同 GT 类别的轨迹 {stats['tracks_switching_gt_class']} 条。",
              "", "详见 temporal_audit.json 的 by_run、by_track_hits、by_gt_class_switch、reset_examples，以及 matched_predictions.jsonl 的逐匹配证据。", ""]
    (args.output / "temporal_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"runs": len(files), "matched": len(rows), "classification": stats["classification"],
                      "reconstruction_counts": stats["reconstruction_counts"],
                      "tracks_switching_gt_class": stats["tracks_switching_gt_class"],
                      "output": str(args.output)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/detectors/vehicle_prop/vehicle_frontier.json")
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--max-received-sim-time", type=float)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
