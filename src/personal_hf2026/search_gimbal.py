# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：为两条普查航线增加横向搜索覆盖余量。
# 修改内容：将搜索云台侧视角由偏离竖直三十度调整为三十五度。
# 修改时间：2026-09-13
# 修改目的：避免搜索进入协同时斜视姿态被朝下指令覆盖而丢失目标。
# 修改内容：新增按会话与轨迹隔离、检测中断暂停的限速云台交接控制器。
# 修改时间：2026-09-13
# 修改目的：扩大条带搜索的横向视野并保留候选确认所需的连续观察。
# 修改内容：新增按实际云台到位推进的横向巡视和无需目标高程的短时观察保持。
"""独立的搜索云台控制，不读取目标真值或地形高度。"""

from dataclasses import dataclass

from competition.baselines.coop_distributed import _bearing_deg, _haversine_m


def _wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class SearchGimbalConfig:
    side_angle_deg: float = 35.0
    side_dwell_s: float = 0.8
    center_dwell_s: float = 0.4
    tolerance_deg: float = 3.0
    near_nadir_m: float = 80.0


class SearchGimbalController:
    def __init__(self, config=None):
        self.config = config or SearchGimbalConfig()
        self.mode = "IDLE"
        self.step_index = 0
        self.arrived_since = None
        self.cycles = 0
        self.holds = 0
        self.hold_tilt = None
        self.hold_position = None
        self.command = (0.0, -90.0)

    def suspend(self):
        """离开搜索后停止巡视，协同云台仍由原控制流程负责。"""
        self.mode = "IDLE"
        self.arrived_since = None
        self.hold_tilt = None
        self.hold_position = None

    def scan(self, now, route_heading, heading, pan, tilt):
        c = self.config
        # 左右换边前先回正下方，再转水平轴，避免把换边误当成瞬时横扫。
        steps = ((-90, -90, 0), (-90, -90 + c.side_angle_deg, c.side_dwell_s),
                 (-90, -90, c.center_dwell_s), (90, -90, 0),
                 (90, -90 + c.side_angle_deg, c.side_dwell_s),
                 (90, -90, c.center_dwell_s))
        if self.mode != "SCAN":
            self.step_index = 0
            self.arrived_since = None
            self.hold_tilt = None
            self.hold_position = None
        self.mode = "SCAN"
        offset, target_tilt, dwell = steps[self.step_index]
        target_pan = _wrap(route_heading + offset - heading)
        arrived = (abs(_wrap(pan - target_pan)) <= c.tolerance_deg
                   and abs(tilt - target_tilt) <= c.tolerance_deg)
        if not arrived:
            self.arrived_since = None
        else:
            if self.arrived_since is None:
                self.arrived_since = now
            if now - self.arrived_since >= dwell:
                self.step_index = (self.step_index + 1) % len(steps)
                self.cycles += int(self.step_index == 0)
                self.arrived_since = None
                offset, target_tilt, _ = steps[self.step_index]
                target_pan = _wrap(route_heading + offset - heading)
        self.command = (target_pan, target_tilt)
        return self.command

    def observe_candidate(self, position, candidate, heading, actual_tilt):
        """保持发现时俯仰并修正方位；靠近后回正下方，不反算目标高度。"""
        switched = (self.hold_position is not None
                    and _haversine_m(*self.hold_position, *candidate) > 80.0)
        if self.mode != "OBSERVE" or switched:
            self.hold_tilt = actual_tilt
            self.holds += 1
        self.mode = "OBSERVE"
        self.arrived_since = None
        self.hold_position = candidate
        pan = _wrap(_bearing_deg(*position, *candidate) - heading)
        tilt = self.hold_tilt
        if _haversine_m(*position, *candidate) < self.config.near_nadir_m:
            pan, tilt = 0.0, -90.0
        self.command = (pan, tilt)
        return self.command

    @property
    def summary(self):
        return dict(mode=self.mode, step=self.step_index, cycles=self.cycles,
                    holds=self.holds, arrived_since_s=self.arrived_since,
                    side_angle_deg=self.config.side_angle_deg,
                    command_pan_deg=self.command[0], command_tilt_deg=self.command[1],
                    hold_tilt_deg=self.hold_tilt)


class GimbalHandoffController:
    """将已观察到的目标交给协同控制，不估计目标高程或替代有效锁定判定。"""

    def __init__(self, near_nadir_m=80.0, lower_rate_dps=6.0):
        self.near_nadir_m = near_nadir_m
        self.lower_rate_dps = lower_rate_dps
        self.reset()

    def reset(self):
        self.key = None
        self.mode = "IDLE"
        self.tilt = None

    def update(self, key, dt, position, target, heading, actual_pan, actual_tilt,
               target_visible):
        if key != self.key:
            self.reset()
            self.key = key
        if self.mode == "IDLE":
            # 从机仅收到召唤坐标时，不继承搜索中另一辆车的云台姿态。
            if target is None or not target_visible:
                return 0.0, -90.0
            self.tilt = actual_tilt
            self.mode = "HOLD"
        if self.mode == "NADIR":
            return 0.0, -90.0
        if target is None:
            return actual_pan, actual_tilt
        pan = _wrap(_bearing_deg(*position, *target) - heading)
        distance = _haversine_m(*position, *target)
        if not target_visible:
            # 缺少新观测时保持实际姿态，不继续把目标推向视野边缘。
            self.tilt = actual_tilt
            self.mode = "HOLD"
        elif distance <= self.near_nadir_m:
            self.mode = "LOWERING"
            # 限制单次步长，避免长决策间隔再次形成瞬时朝下跳变。
            step = self.lower_rate_dps * min(max(dt, 0.0), 0.25)
            self.tilt = max(-90.0, actual_tilt - step)
            if actual_tilt <= -89.9:
                self.mode = "NADIR"
                self.tilt = -90.0
                return 0.0, -90.0
        else:
            self.mode = "HOLD"
            self.tilt = actual_tilt
        return pan, self.tilt

    @property
    def summary(self):
        return dict(mode=self.mode, key=self.key, tilt_deg=self.tilt,
                    near_nadir_m=self.near_nadir_m, lower_rate_dps=self.lower_rate_dps)
