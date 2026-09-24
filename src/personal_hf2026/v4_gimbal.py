# 修改时间：2026-09-24。
# 修改目的：等待图像反馈后再做小步云台纠偏，减少连续命令造成的超调。
# 修改内容：每 0.30 秒最多积分并请求发送一次，以 0.5 倍角度误差和 2.5 度步长控制 pan 与 tilt。
# 修改时间：2026-09-24。
# 修改目的：使像素到角度换算与项目相机水平 FOV 的针孔模型一致。
# 修改内容：水平和垂直误差共用按图像宽度定义的焦距。
# 修改时间：2026-09-24。
# 修改目的：让主机使用单死区与完整像素误差稳定地追踪当前实体。
# 修改内容：仅在新帧积分一次并对单帧 pan/tilt 增量限幅。
"""单死区 bbox 云台控制。"""
from __future__ import annotations

import math


DEAD_X = 45.0
DEAD_Y = 35.0
CONTROL_INTERVAL_S = 0.30
MAX_STEP_DEG = 2.5
GAIN = 0.5


class VisualGimbal:
    def __init__(self):
        self.pan = None
        self.tilt = None
        self.last_error = (None, None)
        self.deadzone_hit = False
        self.last_control_time = None
        self.command_pending = False

    def reset(self):
        self.__init__()

    def update(self, box, image_size, pose, now):
        width, height = image_size
        dx = (box[0] + box[2]) * 0.5 - width * 0.5
        dy = (box[1] + box[3]) * 0.5 - height * 0.5
        self.last_error = (dx, dy)
        self.deadzone_hit = abs(dx) <= DEAD_X and abs(dy) <= DEAD_Y
        if self.pan is None:
            self.pan = float(pose["gimbal_pan"])
            self.tilt = float(pose["gimbal_tilt"])
        if self.deadzone_hit:
            return self.pan, self.tilt
        if (self.last_control_time is not None
                and now - self.last_control_time < CONTROL_INTERVAL_S):
            return self.pan, self.tilt
        focal = width / (2.0 * math.tan(math.radians(float(pose["gimbal_fov_deg"])) / 2.0))
        horizontal = math.degrees(math.atan(dx / focal))
        vertical = math.degrees(math.atan(dy / focal))
        self.pan += max(-MAX_STEP_DEG, min(MAX_STEP_DEG, GAIN * horizontal))
        self.tilt -= max(-MAX_STEP_DEG, min(MAX_STEP_DEG, GAIN * vertical))
        self.pan = max(-180.0, min(180.0, self.pan))
        self.tilt = max(-90.0, min(20.0, self.tilt))
        self.last_control_time = now
        self.command_pending = True
        return self.pan, self.tilt
