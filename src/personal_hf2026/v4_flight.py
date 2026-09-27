# 修改时间：2026-09-27。
# 修改目的：让 V4 双机绕目标飞行由实际机位闭环导引。
# 修改内容：新增固定半径短期航点、主从三档速度导引及半径相位就位判断。
# 修改时间：2026-09-26。
# 修改目的：让 V4 控制器能直接使用三机协同搜索规划器。
# 修改内容：导出新规划器并保留旧搜索航线接口供比较。
# 修改时间：2026-09-24。
# 修改目的：配合从机先飞向半径 180 米的对侧入口点。
# 修改内容：从机到位时要求距入口小于 30 米且距主机大于 280 米。
# 修改时间：2026-09-24。
# 修改目的：让从机从目标圆心外侧进入半径 130 米的对置轨道。
# 修改内容：提供按主机方位生成双机槽位的航点，并按槽位距离和两机间距判断从机到位。
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

from .coordinated_search import CoordinatedSweepRoute
from .survey_search import SurveySearchRoute
from .visual_geometry import pixel_ray
from .v3_simple_control import (SimpleCoopControl, bearing_deg, ground_distance_m,
                                offset_position, solve_ground_aim)


def visual_waypoint(pose, box, image_size, distance_m=180.0, origin_position=None):
    center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
    ray, _ = pixel_ray(center[0], center[1], image_size[0], image_size[1], pose)
    bearing = (math.atan2(ray[0], ray[1]) if math.hypot(ray[0], ray[1]) > 1e-6
               else math.radians(float(pose["heading_deg"]) + float(pose["gimbal_pan"])))
    origin = origin_position or (pose["lat"], pose["lon"])
    return offset_position(origin,
                           distance_m * math.sin(bearing), distance_m * math.cos(bearing))


def wrap180(angle_deg):
    """将角度约束到 [-180, 180) 度。"""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def orbit_point(target, phase_deg, radius_m=130.0):
    """按目标中心、相位与半径求圆上的几何点。"""
    phase = math.radians(float(phase_deg))
    return offset_position(target, radius_m * math.sin(phase),
                           radius_m * math.cos(phase))


def orbit_short_waypoint(target, own_position, speed_mps, *, radius_m=130.0,
                         lookahead_s=1.5, direction=1):
    """按飞机实际相位生成前方短期航点，不依赖仿真时间。"""
    theta = bearing_deg(target, own_position)
    lookahead_deg = math.degrees(float(speed_mps) / radius_m * lookahead_s)
    waypoint = orbit_point(target, theta + direction * lookahead_deg, radius_m)
    return waypoint, theta, lookahead_deg


def master_orbit_guidance(target, own, *, radius_m=130.0, speed_mps=25.0,
                          lookahead_s=1.5, direction=1):
    """主机持续沿自身实际相位前进。"""
    destination, theta, lookahead_deg = orbit_short_waypoint(
        target, own, speed_mps, radius_m=radius_m,
        lookahead_s=lookahead_s, direction=direction)
    debug = {
        "own_phase_deg": theta,
        "radius_m_actual": ground_distance_m(target, own),
        "lookahead_deg": lookahead_deg,
        "short_waypoint": destination,
    }
    return destination, speed_mps, debug


def follower_orbit_guidance(
        target, master, own, *, radius_m=130.0, base_speed_mps=25.0,
        slow_speed_mps=15.0, fast_speed_mps=35.0,
        phase_threshold_deg=20.0, lookahead_s=1.5, direction=1):
    """从机按主机实际对置相位选择速度，按自身实际相位生成航点。"""
    master_phase = bearing_deg(target, master)
    follower_phase = bearing_deg(target, own)
    desired_phase = (master_phase + 180.0) % 360.0
    # 固定绕圈方向为正方向；误差为正表示从机落后，应加速追赶。
    phase_error = wrap180(desired_phase - follower_phase)
    if phase_error > phase_threshold_deg:
        speed = fast_speed_mps
    elif phase_error < -phase_threshold_deg:
        speed = slow_speed_mps
    else:
        speed = base_speed_mps
    long_slot = orbit_point(target, desired_phase, radius_m)
    destination, _, lookahead_deg = orbit_short_waypoint(
        target, own, speed, radius_m=radius_m,
        lookahead_s=lookahead_s, direction=direction)
    debug = {
        "master_phase_deg": master_phase,
        "follower_phase_deg": follower_phase,
        "desired_follower_phase_deg": desired_phase,
        "phase_error_deg": phase_error,
        "master_radius_m": ground_distance_m(target, master),
        "follower_radius_m": ground_distance_m(target, own),
        "long_slot": long_slot,
        "short_waypoint": destination,
        "commanded_speed_mps": speed,
        "lookahead_deg": lookahead_deg,
    }
    return destination, speed, debug


def formation_ready(target, master, own, *, radius_m=130.0,
                    radius_tol_m=40.0, phase_tol_deg=30.0):
    """双机都在指定圆环内且实际相位大体对置时就位。"""
    if master is None or own is None or target is None:
        return False
    master_radius = ground_distance_m(target, master)
    follower_radius = ground_distance_m(target, own)
    desired_phase = (bearing_deg(target, master) + 180.0) % 360.0
    phase_error = wrap180(desired_phase - bearing_deg(target, own))
    return (abs(master_radius - radius_m) <= radius_tol_m
            and abs(follower_radius - radius_m) <= radius_tol_m
            and abs(phase_error) <= phase_tol_deg)


__all__ = ["CoordinatedSweepRoute", "SurveySearchRoute", "SimpleCoopControl", "ground_distance_m",
           "solve_ground_aim", "visual_waypoint", "wrap180", "orbit_point",
           "orbit_short_waypoint", "master_orbit_guidance", "follower_orbit_guidance",
           "formation_ready"]
