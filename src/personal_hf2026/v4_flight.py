# 修改时间：2026-09-24。
# 修改目的：避免云台近正下方时仅用水平像素误差生成错误飞行方位。
# 修改内容：用合法本机姿态和完整 bbox 中心像素射线求水平目标方位。
# 修改时间：2026-09-24。
# 修改目的：统一视觉飞行方向与本项目相机的水平 FOV 定义。
# 修改内容：按图像宽度计算焦距，再由完整水平像素误差求方位角。
# 修改时间：2026-09-24。
# 修改目的：让搜索、视觉追踪、从机接近和协同盘旋独立组合。
# 修改内容：复用纯航线与几何函数并提供五状态控制可直接使用的航点。
"""Agent4 的纯飞行几何。"""
from __future__ import annotations

import math

from .survey_search import SurveySearchRoute
from .visual_geometry import pixel_ray
from .v3_simple_control import SimpleCoopControl, ground_distance_m, offset_position, solve_ground_aim


def visual_waypoint(pose, box, image_size, distance_m=180.0, origin_position=None):
    center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
    ray, _ = pixel_ray(center[0], center[1], image_size[0], image_size[1], pose)
    bearing = (math.atan2(ray[0], ray[1]) if math.hypot(ray[0], ray[1]) > 1e-6
               else math.radians(float(pose["heading_deg"]) + float(pose["gimbal_pan"])))
    origin = origin_position or (pose["lat"], pose["lon"])
    return offset_position(origin,
                           distance_m * math.sin(bearing), distance_m * math.cos(bearing))


def follower_ready(own, master, target):
    return (master is not None and target is not None
            and ground_distance_m(own, master) <= 220.0
            and ground_distance_m(own, target) <= 200.0)


__all__ = ["SurveySearchRoute", "SimpleCoopControl", "ground_distance_m",
           "solve_ground_aim", "visual_waypoint", "follower_ready"]
