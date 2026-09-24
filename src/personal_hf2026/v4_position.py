# 修改时间：2026-09-24。
# 修改目的：只为召唤和协同几何提供宽松的 H=0 粗位置。
# 修改内容：按新视觉帧投影并对最近五个合法经纬度取中位数。
"""独立于实体身份和动静判定的粗坐标。"""
from __future__ import annotations

from collections import deque
from statistics import median

from .v3_perception import _ground_projection


class RoughPosition:
    def __init__(self):
        self.points = deque(maxlen=5)

    def reset(self):
        self.points.clear()

    @property
    def position(self):
        if not self.points:
            return None
        return median(point[0] for point in self.points), median(point[1] for point in self.points)

    def update(self, box, image_size, pose):
        center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
        point, _ = _ground_projection(center, image_size, pose)
        if point is not None:
            self.points.append((point[0], point[1]))
        return self.position
