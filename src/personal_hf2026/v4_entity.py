# 修改时间：2026-09-24。
# 修改目的：用本机唯一实体容忍短暂分类抖动且避免切换到旁边目标。
# 修改内容：仅匹配 real_vehicle 包围框并按最近中心及时间门管理实体生命周期。
"""每机唯一视觉 Entity。"""
from __future__ import annotations

from dataclasses import dataclass
import math


ENTITY_LOST_S = 5.0


def _center(box):
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


@dataclass
class Entity:
    entity_id: str
    bbox_xyxy: tuple[float, float, float, float] | None
    visible: bool
    missing_s: float
    lost: bool
    observed_frames: int
    last_box: tuple[float, float, float, float]
    last_seen_s: float
    previous_box: tuple[float, float, float, float] | None = None
    previous_seen_s: float | None = None


class EntityManager:
    def __init__(self, uid, lost_s=ENTITY_LOST_S):
        self.uid = str(uid)
        self.lost_s = float(lost_s)
        self.counter = 0
        self.current: Entity | None = None

    def clear(self):
        self.current = None

    def update(self, objects, image_size, now_s):
        """返回 (entity, event)；空观测保留已有实体直至超时。"""
        now_s = float(now_s)
        real = [item for item in objects if item.class_name == "real_vehicle"]
        entity = self.current
        if entity is None:
            if not real:
                return None, None
            width, height = image_size
            item = min(real, key=lambda obj: math.dist(_center(obj.bbox_xyxy),
                                                       (width * 0.5, height * 0.5)))
            self.counter += 1
            box = item.bbox_xyxy
            entity = Entity(f"uav_{self.uid}_entity_{self.counter}", box, True, 0.0,
                            False, 1, box, now_s)
            self.current = entity
            return entity, "entity_created"
        prediction = _center(entity.last_box)
        if entity.previous_box is not None and entity.previous_seen_s is not None:
            delta = entity.last_seen_s - entity.previous_seen_s
            if delta > 1e-6:
                speed = ((_center(entity.last_box)[0] - _center(entity.previous_box)[0]) / delta,
                         (_center(entity.last_box)[1] - _center(entity.previous_box)[1]) / delta)
                ahead = min(1.0, max(0.0, now_s - entity.last_seen_s))
                prediction = (prediction[0] + speed[0] * ahead,
                              prediction[1] + speed[1] * ahead)
        diagonal = math.hypot(entity.last_box[2] - entity.last_box[0],
                              entity.last_box[3] - entity.last_box[1])
        gate = max(80.0, 1.5 * diagonal)
        nearest = min(real, key=lambda obj: math.dist(_center(obj.bbox_xyxy), prediction),
                      default=None)
        if nearest is not None and math.dist(_center(nearest.bbox_xyxy), prediction) <= gate:
            entity.previous_box = entity.last_box
            entity.previous_seen_s = entity.last_seen_s
            entity.last_box = nearest.bbox_xyxy
            entity.last_seen_s = now_s
            entity.bbox_xyxy = nearest.bbox_xyxy
            entity.visible = True
            entity.missing_s = 0.0
            entity.observed_frames += 1
            return entity, "entity_matched"
        entity.bbox_xyxy = None
        entity.visible = False
        entity.missing_s = max(0.0, now_s - entity.last_seen_s)
        if entity.missing_s > self.lost_s:
            entity.lost = True
            self.current = None
            return entity, "entity_lost"
        return entity, "entity_missing"
