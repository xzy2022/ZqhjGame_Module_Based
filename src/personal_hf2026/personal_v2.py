# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13（单 UE 复测）
# 修改目的：量化相机新帧从源时间到 Agent 读取时的积压。
# 修改内容：逐机记录每个唯一新帧的帧龄及是否满足 1.5 秒时效门槛。
# 修改时间：2026-09-13（200 秒实验后）
# 修改目的：正确跳过首拍尚未初始化的公开计时。
# 修改内容：未收到 score_view 时等待下一拍，避免记录非视觉故障。
# 修改时间：2026-09-13
# 修改目的：验证视觉分类和轨迹绑定，同时保持 V1 的全部控制决策。
# 修改内容：增加只写旁路结果的 PersonalV2Agent 和每机独立异步视觉观察器。
"""PersonalV2 首版：视觉旁路，不用视觉结果替换 V1 的速度判别。"""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from .personal_v1 import PersonalV1Agent
from .tracking import CandidateTrackSet
from .visual_geometry import aligned_sample, bind_boxes, angle_delta


class VisualShadow:
    def __init__(self, uid, output, weights):
        import torch
        from .visual_appearance import VehicleAppearance, PatchPhotoDetector
        model = VehicleAppearance()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.detector = PatchPhotoDetector(model, confidence=0.55)
        self.detector.warmup()
        self.uid = uid
        self.output = Path(output)
        self.frames = self.output / "visual_frames" / uid
        self.frames.mkdir(parents=True)
        self.log = (self.output / f"visual_{uid}.jsonl").open("w", encoding="utf-8", buffering=1)
        self.age_log = (self.output / f"frame_age_{uid}.jsonl").open("w", encoding="utf-8", buffering=1)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visual-" + uid)
        self.pending = None
        self.stats = Counter()
        self.reset()

    def reset(self):
        self.tracks = CandidateTrackSet()
        self.history = deque(maxlen=120)
        self.identities = {}
        self.last_frame = None
        self.last_seen_frame = None
        self.last_submit = -math.inf

    def _infer(self, image, record):
        started = time.perf_counter()
        boxes = self.detector(image)
        record["inference_wall_ms"] = (time.perf_counter() - started) * 1000
        record["boxes"] = [asdict(box) for box in boxes]
        record["bindings"] = bind_boxes(boxes, record["sample"]) if record["sample"] else []
        suffix = ".jpg" if image[:2] == b"\xff\xd8" else ".png"
        path = self.frames / (str(record["frame_no"]) + suffix)
        path.write_bytes(image)
        record["image_path"] = str(path.relative_to(self.output))
        return record

    def _collect(self, now, wait=False):
        if self.pending is None or (not wait and not self.pending.done()):
            return
        try:
            row = self.pending.result()
            row["result_t"] = now
            row["result_age_s"] = now - row["source_t"]
            self.stats["processed_frames"] += 1
            self.stats["boxes"] += len(row["boxes"])
            current = {str(c.track_id) for c in self.tracks._tracks}
            for binding in row["bindings"]:
                self.stats[binding["status"]] += 1
                if binding["status"] != "bound":
                    continue
                key = binding["track_id"]
                box = row["boxes"][binding["box_index"]]
                binding["identity"] = "unknown"
                if key not in current or row["result_age_s"] > 1.5:
                    binding["identity_reason"] = "stale_result_or_track"
                    continue
                evidence = self.identities.setdefault(key, deque(maxlen=5))
                while evidence and row["source_t"] - evidence[0][0] > 3:
                    evidence.popleft()
                if box["confidence"] >= 0.65 and box["class_margin"] >= 0.35:
                    evidence.append((row["source_t"], row["frame_no"], box["category"]))
                labels = Counter(item[2] for item in evidence)
                label, count = labels.most_common(1)[0] if labels else ("unknown", 0)
                # 两张不同新帧一致且没有相反类别，才输出旁路身份。
                if count >= 2 and len(labels) == 1:
                    binding["identity"] = label
                    self.stats["confirmed_" + label] += 1
                binding["evidence_count"] = count
            self.identities = {k: v for k, v in self.identities.items() if k in current}
            self.log.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:
            self.stats["failures"] += 1
            self.log.write(json.dumps({"uid": self.uid, "error": repr(exc), "t": now}) + "\n")
        finally:
            self.pending = None

    def observe(self, obs, packet, agent):
        if obs.briefing.score_view is None:
            self.stats["waiting_for_clock_ticks"] += 1
            return
        now = float(obs.briefing.score_view.sim_time)
        own = obs.self
        positions = {(d.target_lat, d.target_lon) for d in
                     (own.detections or (own.detection,))
                     if d.detected and d.target_lat is not None and d.target_lon is not None}
        snapshots = self.tracks.update(now, sorted(positions))
        candidates = {str(key): list(snap.position) for key, snap in snapshots
                      if snap.position is not None and now - snap.last_seen <= 0.25}
        sample = {"t": now, "own": {k: getattr(own, k) for k in
                  ("lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt", "gimbal_fov_deg")},
                  "tracks": candidates}
        if self.history and now > self.history[-1]["t"]:
            previous = self.history[-1]
            sample["angular_rate_dps"] = max(abs(angle_delta(sample["own"][k], previous["own"][k]))
                / (now - previous["t"]) for k in ("heading_deg", "gimbal_pan", "gimbal_tilt"))
        self.history.append(sample)
        # JSON 的轨迹编号使用字符串，与历史候选字典保持一致。
        self._collect(now)
        if not packet or not own.photo:
            self.stats["missing_photo_ticks"] += 1
            return
        source_t = packet["source_t"]
        age = now - source_t
        if packet["frame_no"] != self.last_seen_frame:
            fresh = 0 <= age <= 1.5
            self.age_log.write(json.dumps({"uid": self.uid, "frame_no": packet["frame_no"],
                "source_t": source_t, "source_sim_time": packet["source_sim_time"],
                "observe_t": now, "frame_age_s": age, "fresh": fresh}, ensure_ascii=False) + "\n")
            self.stats["unique_frames_seen"] += 1
            self.stats["fresh_unique_frames" if fresh else "stale_unique_frames"] += 1
            self.last_seen_frame = packet["frame_no"]
        if packet["frame_no"] == self.last_frame or now - self.last_submit < 0.5:
            return
        if self.pending is not None:
            self.stats["worker_busy_ticks"] += 1
            return
        if not 0 <= age <= 1.5:
            self.stats["invalid_frame_age_ticks"] += 1
            return
        aligned, reason = aligned_sample(list(self.history), source_t)
        self.last_frame, self.last_submit = packet["frame_no"], now
        record = {"uid": self.uid, "frame_no": packet["frame_no"], "source_t": source_t,
                  "source_sim_time": packet["source_sim_time"], "submit_t": now,
                  "frame_age_s": age, "alignment": reason, "sample": aligned,
                  "image_sha256": hashlib.sha256(own.photo).hexdigest(),
                  "v1_active_epoch": agent._track.epoch, "v1_position": agent._track.position,
                  "v1_phase": agent._state, "v1_session": agent._coordinator.current_session}
        self.stats["submitted_frames"] += 1
        self.stats["alignment_" + reason] += 1
        self.pending = self.executor.submit(self._infer, own.photo, record)

    def close(self):
        now = self.history[-1]["t"] if self.history else 0
        self._collect(now, wait=True)
        self.executor.shutdown(wait=True)
        self.log.close()
        self.age_log.close()
        (self.output / f"visual_{self.uid}_summary.json").write_text(
            json.dumps(dict(self.stats), indent=2), encoding="utf-8")


class PersonalV2Agent(PersonalV1Agent):
    """先执行原版 V1，再观察同一拍输入；旁路不回写任何控制状态。"""
    def reset(self):
        super().reset()
        self.photo_packet = None
        if getattr(self, "shadow", None) is not None:
            self.shadow.reset()

    def decide(self, obs, dt):
        commands = super().decide(obs, dt)
        if getattr(self, "shadow", None) is not None:
            try:
                self.shadow.observe(obs, self.photo_packet, self)
            except Exception as exc:
                # 旁路异常只记入自身日志，不能吞掉已经计算出的 V1 动作。
                self.shadow.stats["observe_failures"] += 1
                self.shadow.log.write(json.dumps({"uid": self.my_uid, "error": repr(exc), "t": self._t}) + "\n")
        return commands
