# 修改时间：2026-09-24。
# 修改目的：兼容官方 Runner 启动时尚未提供 score_view 的首拍观测。
# 修改内容：首拍使用本机累计 dt 并在传感回调等待正式仿真时间。
# 修改时间：2026-09-24。
# 修改目的：提供不继承 V1/V3 控制状态机的官方 Agent4 生命周期入口。
# 修改内容：由 sensor 保存完整像素快照，由 decide 去重消费并返回五状态控制命令。
"""正式 PersonalV4Agent。"""
from __future__ import annotations

from competition.sdk.core.agent import Agent
from competition.sdk.core.observation import Detection

from .v4_control import V4Control
from .v4_perception import submit_observation
from .v3_perception import _ground_projection


class PersonalV4Agent(Agent):
    SEARCH_FOV_DEG = 48.0

    def __init__(self, my_uid):
        super().__init__(my_uid)
        self._perception_provider = None
        self.reset()

    def set_perception_provider(self, provider):
        self._perception_provider = provider

    def reset(self):
        self.control = V4Control(self.my_uid)
        self._last_snapshot = None
        self._consumed_frame_id = None
        self._t = 0.0

    def sensor(self, obs, dt):
        if self._perception_provider is None:
            return []
        if getattr(getattr(obs, "briefing", None), "score_view", None) is None:
            return []
        snapshot = self._perception_provider(obs, dt)
        if snapshot is None:
            return []
        self._last_snapshot = snapshot
        if snapshot.error:
            return []
        real = [item for item in snapshot.effective_yolo_objects
                if item.class_name == "real_vehicle"]
        if not real:
            return []
        width, height = snapshot.image_size
        item = min(real, key=lambda obj: ((obj.bbox_xyxy[0] + obj.bbox_xyxy[2] - width) ** 2
                                         + (obj.bbox_xyxy[1] + obj.bbox_xyxy[3] - height) ** 2))
        box = item.bbox_xyxy
        point, _ = _ground_projection(((box[0] + box[2]) * 0.5,
                                       (box[1] + box[3]) * 0.5),
                                      snapshot.image_size, snapshot.source_pose)
        if point is None:
            return []
        return [Detection(detected=True,
                          confidence=item.detector_confidence * item.real_probability,
                          target_lat=point[0], target_lon=point[1],
                          target_type="ground_vehicle")]

    def decide(self, obs, dt):
        score = getattr(obs.briefing, "score_view", None)
        now = (float(score.sim_time) if score is not None else
               self._t + max(0.0, float(dt)))
        self._t = now
        snapshot = self._last_snapshot
        if snapshot is not None and snapshot.frame_id != self._consumed_frame_id:
            self._consumed_frame_id = snapshot.frame_id
            self.control.consume_visual(snapshot)
        return self.control.step(obs, now)

    @property
    def completion_summary(self):
        return {"state": self.control.state,
                "entity_id": (self.control.entity.current.entity_id
                              if self.control.entity.current else None),
                "completed_sessions": self.control.completed_sessions,
                "comm_sent": self.control.coord.sent_events,
                "comm_received": self.control.coord.received_events}
