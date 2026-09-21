# 修改时间：2026-09-21。
# 修改目的：为 V3 简化协同提供不依赖双机轨迹匹配的从机接近和定点瞄准几何。
# 修改内容：新增 H=0 目标的实时云台反解、220/200 米双门槛及可直接转成 fly_to/point_gimbal 的导引结果。
"""V3 简化协同的飞行与云台几何。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


EARTH_RADIUS_M = 6_371_000.0


def _wrap_signed_deg(angle_deg):
    """将相对方位约束到云台 pan 使用的 [-180, 180) 度。"""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def ground_distance_m(first_position, second_position):
    """按球面海佛距离计算两个经纬度点的水平距离。"""
    lat1, lon1 = map(float, first_position)
    lat2, lon2 = map(float, second_position)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lon = math.radians(lon2 - lon1)
    value = (math.sin(delta_phi / 2.0) ** 2
             + math.cos(phi1) * math.cos(phi2)
             * math.sin(delta_lon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(
        max(0.0, min(1.0, value))))


def bearing_deg(first_position, second_position):
    """返回从第一个经纬度点指向第二个点的真方位角。"""
    lat1, lon1 = map(float, first_position)
    lat2, lon2 = map(float, second_position)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    east = math.sin(delta_lon) * math.cos(phi2)
    north = (math.cos(phi1) * math.sin(phi2)
             - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon))
    return (math.degrees(math.atan2(east, north)) + 360.0) % 360.0


@dataclass(frozen=True)
class GroundAim:
    """指向一个已知经纬高坐标的云台指令解。"""

    target_position: tuple[float, float]
    target_alt_m: float
    ground_distance_m: float
    bearing_deg: float
    pan_deg: float
    tilt_deg: float


def solve_ground_aim(self_position, self_alt_m, self_heading_deg,
                     target_position, *, target_alt_m=0.0):
    """用本拍飞机位姿反解机体相对 pan 与世界俯仰 tilt。

    heading 和方位都以正北为零度、顺时针为正；pan 是相对机头
    的有符号夹角；tilt 向上为正。H=0 时目标在 500 米无人机下方，
    因而 tilt 为负值。
    """
    own = tuple(map(float, self_position))
    target = tuple(map(float, target_position))
    horizontal = ground_distance_m(own, target)
    target_bearing = bearing_deg(own, target)
    height_delta = float(target_alt_m) - float(self_alt_m)
    if horizontal <= 1e-6:
        # 正上方/正下方时方位无定义，pan 回中避免每拍跳变。
        pan = 0.0
        tilt = 90.0 if height_delta > 0.0 else -90.0
    else:
        pan = _wrap_signed_deg(target_bearing - float(self_heading_deg))
        tilt = math.degrees(math.atan2(height_delta, horizontal))
    return GroundAim(
        target_position=target,
        target_alt_m=float(target_alt_m),
        ground_distance_m=horizontal,
        bearing_deg=target_bearing,
        pan_deg=pan,
        tilt_deg=tilt,
    )


@dataclass(frozen=True)
class SimpleCoopControlConfig:
    """005 开发记录中的简化从机控制门槛。"""

    master_gate_m: float = 220.0
    target_gate_m: float = 200.0
    target_alt_m: float = 0.0
    follower_speed_mps: float = 22.0
    follower_loiter_radius_m: float = 100.0


@dataclass(frozen=True)
class FollowerGuidance:
    """FOLLOWER 每个控制 tick 的不可变导引快照。"""

    fly_to_position: tuple[float, float] | None
    fly_to_speed_mps: float | None
    fly_to_loiter_radius_m: float | None
    uav_distance_to_master_m: float | None
    target_distance_m: float | None
    master_gate_m: float
    target_gate_m: float
    within_master_gate: bool
    within_target_gate: bool
    rendezvous_ready: bool
    guidance_enabled: bool
    aiming_enabled: bool
    aim_target_lat: float | None
    aim_target_lon: float | None
    gimbal_pan_cmd_deg: float | None
    gimbal_tilt_cmd_deg: float | None

    def as_evidence(self):
        """返回稳定字段名，供旁路记录逐拍审计门槛与瞄准。"""
        return asdict(self)

    @property
    def ready(self):
        """为接入层提供简短的双门槛别名。"""
        return self.rendezvous_ready


class SimpleCoopControl:
    """FOLLOWER 赶赴 MASTER 目标坐标、到位后直指 H=0 坐标。"""

    def __init__(self, config=None):
        self.config = config or SimpleCoopControlConfig()
        self.reset()

    def reset(self):
        """离开协同会话时清除从机已到位锁存。"""
        self._follower_session_key = None
        self._follower_aim_latched = False

    def master_aim(self, *, self_position, self_alt_m, self_heading_deg,
                   target_position):
        """MASTER 每拍用本机实时位姿重算对自己目标的指向。"""
        if target_position is None:
            return None
        return solve_ground_aim(
            self_position,
            self_alt_m,
            self_heading_deg,
            target_position,
            target_alt_m=self.config.target_alt_m,
        )

    def follower(self, *, self_position, self_alt_m, self_heading_deg,
                 master_position, follow_position, session_key=None):
        """按 MASTER 机位与广播目标坐标生成从机导引。

        follow_position 一旦可用就持续作为 fly_to 目的地；仅当本机
        到 MASTER 不超过 220 米且到该目标估计不超过 200 米时，
        才锁存并返回 point_gimbal 所需的 pan/tilt。同一会话锁存后即使
        短时离开门槛也仍持续指向；会话改变或 reset 后重新判定。
        """
        if session_key != self._follower_session_key:
            self._follower_session_key = session_key
            self._follower_aim_latched = False
        if follow_position is None:
            return FollowerGuidance(
                fly_to_position=None,
                fly_to_speed_mps=None,
                fly_to_loiter_radius_m=None,
                uav_distance_to_master_m=(
                    None if master_position is None
                    else ground_distance_m(self_position, master_position)
                ),
                target_distance_m=None,
                master_gate_m=self.config.master_gate_m,
                target_gate_m=self.config.target_gate_m,
                within_master_gate=False,
                within_target_gate=False,
                rendezvous_ready=False,
                guidance_enabled=False,
                aiming_enabled=False,
                aim_target_lat=None,
                aim_target_lon=None,
                gimbal_pan_cmd_deg=None,
                gimbal_tilt_cmd_deg=None,
            )

        target = tuple(map(float, follow_position))
        distance_to_target = ground_distance_m(self_position, target)
        distance_to_master = (
            None if master_position is None
            else ground_distance_m(self_position, master_position)
        )
        within_master = (distance_to_master is not None
                         and distance_to_master <= self.config.master_gate_m)
        within_target = distance_to_target <= self.config.target_gate_m
        ready = within_master and within_target
        self._follower_aim_latched = self._follower_aim_latched or ready
        aim = (solve_ground_aim(
            self_position,
            self_alt_m,
            self_heading_deg,
            target,
            target_alt_m=self.config.target_alt_m,
        ) if self._follower_aim_latched else None)
        return FollowerGuidance(
            fly_to_position=target,
            fly_to_speed_mps=self.config.follower_speed_mps,
            fly_to_loiter_radius_m=self.config.follower_loiter_radius_m,
            uav_distance_to_master_m=distance_to_master,
            target_distance_m=distance_to_target,
            master_gate_m=self.config.master_gate_m,
            target_gate_m=self.config.target_gate_m,
            within_master_gate=within_master,
            within_target_gate=within_target,
            rendezvous_ready=ready,
            guidance_enabled=True,
            aiming_enabled=self._follower_aim_latched,
            aim_target_lat=target[0] if self._follower_aim_latched else None,
            aim_target_lon=target[1] if self._follower_aim_latched else None,
            gimbal_pan_cmd_deg=aim.pan_deg if aim is not None else None,
            gimbal_tilt_cmd_deg=aim.tilt_deg if aim is not None else None,
        )


__all__ = [
    "FollowerGuidance",
    "GroundAim",
    "SimpleCoopControl",
    "SimpleCoopControlConfig",
    "bearing_deg",
    "ground_distance_m",
    "solve_ground_aim",
]
