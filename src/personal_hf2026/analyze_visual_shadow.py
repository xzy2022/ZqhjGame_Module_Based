# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13（单 UE 复测）
# 修改目的：直接汇总相机唯一新帧的积压程度。
# 修改内容：增加逐机新帧数量、时效通过率及帧龄分位数。
# 修改时间：2026-09-13
# 修改目的：分开评价视觉分类、无高度关联和控制不变性。
# 修改内容：仅离线读取帧框审计与裁判坐标，输出分层统计和错误实例。
"""旁路实验离线分析；本文件绝不作为 Agent 在线输入。"""
import argparse
from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), "median": values[len(values)//2],
            "p95": values[min(len(values)-1, math.ceil(.95*len(values))-1)], "max": values[-1]}


def iou(box, raw):
    a = [box[k] for k in ("x1", "y1", "x2", "y2")]
    overlap = max(0, min(a[2],raw[2])-max(a[0],raw[0])) * max(0,min(a[3],raw[3])-max(a[1],raw[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (raw[2]-raw[0])*(raw[3]-raw[1]) - overlap
    return overlap/max(1,union)


def distance(a, b):
    return math.hypot((a[0]-b[0])*111320, (a[1]-b[1])*111320*math.cos(math.radians(a[0])))


def analyze(output):
    audit = {(r["uid"], r["frame_no"], r["source_sim_time"]): r for r in rows(output/"frame_audit.jsonl") if "frame_no" in r}
    judge = sorted(rows(output/"judge.jsonl"), key=lambda r:r["t"])
    times = [r["t"] for r in judge]
    counts, confusion, statuses = Counter(), Counter(), Counter()
    frames_by_uid, classes_by_uid = {}, {}
    frame_ages, result_ages, inference_times, decide_times = [], [], [], []
    unique_frame_ages, unique_frames_by_uid = [], {}
    wrong, examples, confirmed_targets = [], [], set()
    bindings_checked, binding_correct = 0, 0
    for path in sorted(output.glob("frame_age_*.jsonl")):
        for row in rows(path):
            uid = row["uid"]
            age = row["frame_age_s"]
            unique_frame_ages.append(age)
            item = unique_frames_by_uid.setdefault(uid, {"seen": 0, "fresh": 0, "stale": 0})
            item["seen"] += 1
            item["fresh" if row["fresh"] else "stale"] += 1
    for path in sorted(output.glob("visual_*.jsonl")):
        for row in rows(path):
            if "error" in row:
                counts["visual_errors"] += 1
                continue
            uid = row["uid"]
            frames_by_uid[uid] = frames_by_uid.get(uid,0)+1
            counts["frames"] += 1
            if row.get("frame_age_s") is not None:
                frame_ages.append(row["frame_age_s"])
            if row.get("result_age_s") is not None:
                result_ages.append(row["result_age_s"])
            inference_times.append(row["inference_wall_ms"])
            statuses["alignment_"+row["alignment"]] += 1
            reference = audit.get((uid,row["frame_no"],row["source_sim_time"]),{})
            raw_boxes = reference.get("ue_detections",[])
            if reference:
                counts["frames_with_audit"] += 1
            else:
                counts["frames_missing_audit"] += 1
            valid_raw = [b for b in raw_boxes if len(b.get("bbox",[]))==4]
            if valid_raw:
                counts["frames_with_projected_vehicles"] += 1
            if row["boxes"]:
                counts["frames_with_model_boxes"] += 1
            index = bisect_left(times,row["source_t"])
            near = min((i for i in (index-1,index) if 0<=i<len(times)),
                       key=lambda i:abs(times[i]-row["source_t"]), default=None)
            world = {}
            if near is not None and abs(times[near]-row["source_t"])<=.25:
                world = dict(judge[near].get("world_targets",{}),**judge[near].get("world_decoys",{}))
            matches = {}
            for j,box in enumerate(row["boxes"]):
                counts["model_boxes"] += 1
                predicted = box["category"]
                classes_by_uid.setdefault(uid,Counter())[predicted] += 1
                ranked = sorted(((iou(box,b["bbox"]),b) for b in valid_raw), key=lambda item:-item[0])
                if not ranked or ranked[0][0]<.15 or (len(ranked)>1 and ranked[1][0]>=.15):
                    counts["boxes_unresolved_by_audit"] += 1
                    continue
                overlap, target = ranked[0]
                actual = "true_vehicle" if target["class"]=="TargetVehicle" else "decoy_vehicle"
                confusion[actual+" -> "+predicted] += 1
                matches[j] = target
            for binding in row["bindings"]:
                statuses[binding["status"]] += 1
                if binding["status"]!="bound":
                    continue
                box_index = binding["box_index"]
                target = matches.get(box_index)
                position = row["sample"]["tracks"][str(binding["track_id"])]
                closest = min(world, key=lambda k:distance(position,(world[k]["lat"],world[k]["lon"])), default=None)
                if closest is not None and distance(position,(world[closest]["lat"],world[closest]["lon"]))>5:
                    closest = None
                if target is None or closest is None:
                    counts["bindings_unresolved_by_audit"] += 1
                    continue
                bindings_checked += 1
                correct = str(target["target_id"])==closest
                binding_correct += int(correct)
                detail = {"uid":uid,"frame_no":row["frame_no"],"t":row["source_t"],
                    "image_path":row["image_path"],"box_index":box_index,
                    "pixel_target":str(target["target_id"]),"coordinate_target":closest,
                    "predicted":row["boxes"][box_index]["category"],
                    "identity":binding.get("identity"),"track_id":binding["track_id"],
                    "bearing_error_deg":binding["candidates"][0]["error_deg"]}
                if not correct:
                    wrong.append(detail)
                elif len(examples)<12:
                    examples.append(detail)
                if binding.get("identity")=="true_vehicle":
                    counts["audited_confirmed_true_bindings"] += 1
                    if correct and target["class"]=="TargetVehicle":
                        confirmed_targets.add(closest)
                    else:
                        counts["audited_wrong_confirmed_true_bindings"] += 1
    for row in rows(output/"observations.jsonl"):
        decide_times.append(row["decide_wall_ms"])
    evaluations = list(output.glob("*.evaluation.json"))
    evaluation = json.loads(evaluations[0].read_text(encoding="utf-8")) if evaluations else {}
    result = {"counts":dict(counts),"frames_by_uid":frames_by_uid,
        "predictions_by_uid":{k:dict(v) for k,v in classes_by_uid.items()},
        "classification_confusion_on_unique_bbox_matches":dict(confusion),"statuses":dict(statuses),
        "binding_audit":{"checked":bindings_checked,"correct":binding_correct,
                         "wrong":bindings_checked-binding_correct},
        "confirmed_true_target_ids":sorted(confirmed_targets),
        "unique_frames_by_uid":unique_frames_by_uid,
        "unique_frame_age_s":quantiles(unique_frame_ages),
        "frame_age_s":quantiles(frame_ages),"result_age_s":quantiles(result_ages),
        "inference_wall_ms":quantiles(inference_times),"decide_wall_ms":quantiles(decide_times),
        "control_equivalence":json.loads((output/"control_equivalence.json").read_text()) if (output/"control_equivalence.json").exists() else None,
        "simulated_duration_s":times[-1] if times else 0,
        "evaluation":{k:evaluation.get(k) for k in ("total_score","n_destroyed","n_reports","targeting_rmse_m","passed")},
        "wrong_binding_examples":wrong[:30],"correct_binding_examples":examples,
        "limits":["旁路分数来自 V1 控制，不能归因于视觉。",
                  "分类准确性仅在与 UE 审计框唯一重叠的子集评估，不能代表全量召回率。",
                  "UE 审计框不保证实际无遮挡可见；所有真实身份仅用于离线分析。",
                  "相同输入的动作一致性不等于加入推理后的闭环轨迹与无推理实验完全一致。",
                  "帧源时间不是已经校准的曝光时刻，水平 FOV 和零 roll 仍为待验证模型假设。"]}
    (output/"visual_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    print(json.dumps(analyze(args.output),ensure_ascii=False,indent=2))
