# 修改时间：2026-09-28。
# 修改目的：解除 READY 相位死锁并对照实际仿真检验相位控制方向。
# 修改内容：就位仅检查双机半径，保留反向双边前视角偏置作为待验证候选。
# 修改时间：2026-09-28。
# 修改目的：用等速圆周航点前视角修正正式协同的对置相位。
# 修改内容：增加主机相位模式及双机前视角偏置，保持从机槽位捕获逻辑不变。
# 修改时间：2026-09-27。
# 修改目的：让从机捕获真实对置槽位，并让双机共同修正入圆后的相位误差。
# 修改内容：增加限幅相位捕获航点、统一相位几何及双机三档速度函数。
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

COOP_SYNC_SPEED_MPS = 25.0
COOP_PHASE_LOOKAHEAD_BIAS_DEG = 10.0
COOP_PHASE_THRESHOLD_DEG = 20.0


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
                         lookahead_s=1.5, direction=1, lookahead_bias_deg=0.0):
    """按飞机实际相位生成前方短期航点，不依赖仿真时间。"""
    theta = bearing_deg(target, own_position)
    base_lookahead_deg = math.degrees(float(speed_mps) / radius_m * lookahead_s)
    effective_lookahead_deg = max(1.0, base_lookahead_deg + lookahead_bias_deg)
    waypoint = orbit_point(target, theta + direction * effective_lookahead_deg, radius_m)
    return waypoint, theta, base_lookahead_deg, effective_lookahead_deg


def phase_sync_mode(phase_error_deg):
    """正误差表示从机落后，模式仅由主机选择。"""
    if phase_error_deg > COOP_PHASE_THRESHOLD_DEG:
        return "F"
    if phase_error_deg < -COOP_PHASE_THRESHOLD_DEG:
        return "M"
    return "N"


def phase_lookahead_bias(sync_mode, role):
    if sync_mode == "F":
        return (-COOP_PHASE_LOOKAHEAD_BIAS_DEG if role == "MASTER"
                else COOP_PHASE_LOOKAHEAD_BIAS_DEG)
    if sync_mode == "M":
        return (COOP_PHASE_LOOKAHEAD_BIAS_DEG if role == "MASTER"
                else -COOP_PHASE_LOOKAHEAD_BIAS_DEG)
    return 0.0


def master_orbit_guidance(target, own, *, radius_m=130.0, speed_mps=25.0,
                          lookahead_s=1.5, direction=1, sync_mode="N"):
    """主机持续沿自身实际相位前进。"""
    destination, theta, base_lookahead_deg, effective_lookahead_deg = orbit_short_waypoint(
        target, own, speed_mps, radius_m=radius_m,
        lookahead_s=lookahead_s, direction=direction,
        lookahead_bias_deg=phase_lookahead_bias(sync_mode, "MASTER"))
    debug = {
        "own_phase_deg": theta,
        "radius_m_actual": ground_distance_m(target, own),
        "lookahead_deg": effective_lookahead_deg,
        "base_lookahead_deg": base_lookahead_deg,
        "effective_lookahead_deg": effective_lookahead_deg,
        "short_waypoint": destination,
    }
    return destination, speed_mps, debug


def pair_phase_geometry(target, master, follower):
    """从双方实际位置计算同一对置相位误差。"""
    master_phase = bearing_deg(target, master)
    follower_phase = bearing_deg(target, follower)
    desired_phase = (master_phase + 180.0) % 360.0
    return {
        "master_phase_deg": master_phase,
        "follower_phase_deg": follower_phase,
        "desired_phase_deg": desired_phase,
        "phase_error_deg": wrap180(desired_phase - follower_phase),
        "master_radius_m": ground_distance_m(target, master),
        "follower_radius_m": ground_distance_m(target, follower),
        "pair_distance_m": ground_distance_m(master, follower),
    }


def follower_capture_guidance(target, master, own, *, radius_m=130.0,
                              speed_mps=35.0, max_phase_step_deg=30.0):
    """每拍朝主机对侧槽位最多修正指定角度，航点始终在固定圆上。"""
    geometry = pair_phase_geometry(target, master, own)
    phase_error = geometry["phase_error_deg"]
    phase_step = max(-max_phase_step_deg, min(max_phase_step_deg, phase_error))
    capture_phase = (geometry["follower_phase_deg"] + phase_step) % 360.0
    long_slot = orbit_point(target, geometry["desired_phase_deg"], radius_m)
    destination = orbit_point(target, capture_phase, radius_m)
    debug = {
        **geometry,
        "own_phase_deg": geometry["follower_phase_deg"],
        "capture_phase_step_deg": phase_step,
        "capture_phase_deg": capture_phase,
        "long_slot": long_slot,
        "short_waypoint": destination,
        "commanded_speed_mps": speed_mps,
    }
    return destination, speed_mps, debug


def follower_orbit_guidance(
        target, master, own, *, radius_m=130.0, speed_mps=COOP_SYNC_SPEED_MPS,
        sync_mode="N", lookahead_s=1.5, direction=1):
    """从机执行主机下发的模式，按自身实际相位生成等速航点。"""
    geometry = pair_phase_geometry(target, master, own)
    destination, _, base_lookahead_deg, effective_lookahead_deg = orbit_short_waypoint(
        target, own, speed_mps, radius_m=radius_m,
        lookahead_s=lookahead_s, direction=direction,
        lookahead_bias_deg=phase_lookahead_bias(sync_mode, "FOLLOWER"))
    debug = {
        **geometry,
        "desired_follower_phase_deg": geometry["desired_phase_deg"],
        "sync_mode": sync_mode,
        "local_phase_error_deg": geometry["phase_error_deg"],
        "master_command_speed_mps": speed_mps,
        "follower_command_speed_mps": speed_mps,
        "short_waypoint": destination,
        "commanded_speed_mps": speed_mps,
        "lookahead_deg": effective_lookahead_deg,
        "base_lookahead_deg": base_lookahead_deg,
        "effective_lookahead_deg": effective_lookahead_deg,
    }
    return destination, speed_mps, debug


def formation_ready(target, master, own, *, radius_m=130.0,
                    radius_tol_m=40.0):
    """双机均已进入正式协同轨道允许的径向范围。"""
    if master is None or own is None or target is None:
        return False
    geometry = pair_phase_geometry(target, master, own)
    return (abs(geometry["master_radius_m"] - radius_m) <= radius_tol_m
            and abs(geometry["follower_radius_m"] - radius_m) <= radius_tol_m)


__all__ = ["CoordinatedSweepRoute", "SurveySearchRoute", "SimpleCoopControl", "ground_distance_m",
           "solve_ground_aim", "visual_waypoint", "wrap180", "orbit_point",
           "orbit_short_waypoint", "master_orbit_guidance", "follower_capture_guidance",
           "pair_phase_geometry", "phase_sync_mode", "phase_lookahead_bias",
           "follower_orbit_guidance",
           "formation_ready"]
