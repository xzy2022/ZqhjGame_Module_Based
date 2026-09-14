# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：使用同一次仿真保存的旧画面单独验证视觉和方向关联。
# 修改内容：按帧源时间回放观测历史，不运行引擎，不把回放当成在线成功。
"""对同一仿真保存的画面做离线几何和分类回放。"""
import argparse
from collections import deque, Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

from .analyze_visual_shadow import rows, analyze
from .tracking import CandidateTrackSet
from .paths import PROJECT_ROOT
from .visual_geometry import aligned_sample, bind_boxes, angle_delta


def replay(source, output):
    import torch
    import cv2
    from .visual_appearance import VehicleAppearance, PatchPhotoDetector
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    model = VehicleAppearance()
    model.load_state_dict(torch.load(PROJECT_ROOT / "assets/personal_v2/vision.pt", map_location="cpu", weights_only=True))
    detector = PatchPhotoDetector(model, .55)
    detector.warmup()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("frame_audit.jsonl","judge.jsonl","observations.jsonl","control_equivalence.json"):
        if (source/name).exists():
            shutil.copyfile(source/name,output/name)
    histories, banks, offsets = {}, {}, {}
    for row in rows(source/"observations.jsonl"):
        uid, now, own = row["uid"], row["t"], row["self"]
        if row["sim_time"] is None:
            continue
        offsets[uid] = row["sim_time"]-now
        bank = banks.setdefault(uid,CandidateTrackSet())
        detections = own["detections"] or [own["detection"]]
        positions = sorted({(d["target_lat"],d["target_lon"]) for d in detections
            if d["detected"] and d["target_lat"] is not None and d["target_lon"] is not None})
        tracks = {str(k):list(s.position) for k,s in bank.update(now,positions)
                  if s.position is not None and now-s.last_seen<=.25}
        sample = {"t":now,"own":{k:own[k] for k in
            ("lat","lon","alt","heading_deg","gimbal_pan","gimbal_tilt","gimbal_fov_deg")},"tracks":tracks}
        history = histories.setdefault(uid,[])
        if history and now>history[-1]["t"]:
            sample["angular_rate_dps"] = max(abs(angle_delta(sample["own"][k],history[-1]["own"][k]))
                /(now-history[-1]["t"]) for k in ("heading_deg","gimbal_pan","gimbal_tilt"))
        history.append(sample)
    logs = {uid:(output/f"visual_{uid}.jsonl").open("w",encoding="utf-8",buffering=1) for uid in histories}
    evidence = {}
    recovered = list(rows(source/"recovery_frames.jsonl"))
    # 补采可能比在线审计线程更早读到新帧；同帧审计随图一起补齐。
    with (output/"frame_audit.jsonl").open("a",encoding="utf-8") as audit:
        for frame in recovered:
            audit.write(json.dumps(frame)+"\n")
    try:
        for frame in recovered:
            uid = frame["uid"]
            t = frame["source_sim_time"]-offsets[uid]
            sample, reason = aligned_sample(histories[uid],t)
            image_path = source/frame["image_path"]
            started = time.perf_counter()
            boxes = detector(image_path.read_bytes())
            elapsed = (time.perf_counter()-started)*1000
            bindings = bind_boxes(boxes,sample) if sample else []
            for binding in bindings:
                binding["identity"] = "unknown"
                if binding["status"]!="bound":
                    continue
                box = boxes[binding["box_index"]]
                votes = evidence.setdefault((uid,binding["track_id"]),deque(maxlen=5))
                while votes and t-votes[0][0]>3:
                    votes.popleft()
                if box.confidence>=.65 and box.class_margin>=.35:
                    votes.append((t,box.category))
                labels = Counter(x[1] for x in votes)
                if len(labels)==1 and max(labels.values())>=2:
                    binding["identity"] = next(iter(labels))
            row = dict(uid=uid,frame_no=frame["frame_no"],source_sim_time=frame["source_sim_time"],
                source_t=t,alignment=reason,sample=sample,boxes=[asdict(b) for b in boxes],bindings=bindings,
                inference_wall_ms=elapsed,frame_age_s=None,result_age_s=None,
                mode="offline_source_time_replay",image_path=str(image_path))
            logs[uid].write(json.dumps(row)+"\n")
    finally:
        for stream in logs.values():
            stream.close()
    result = analyze(output)
    result["mode"] = "offline_replay_not_online_success"
    result["replayed_frame_count"] = len(recovered)
    result["source_run"] = str(source)
    (output/"visual_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    print(json.dumps(replay(args.source,args.output),ensure_ascii=False,indent=2))
