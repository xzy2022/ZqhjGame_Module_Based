# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：避免短暂空检测反复清空竞争方向并打断纠偏。
# 修改内容：在期望目标可见时沿用一秒内的方向历史更新偏移，不伪造新方向采样。
# 修改时间：2026-09-13
# 修改目的：在接近期望位置和大角度转向时降低速度以减少过冲。
# 修改内容：新增仅依赖位置误差和航向误差的可复用速度调节函数。
# 修改时间：2026-09-13
# 修改目的：避免正确锁定时保持绝对航向导致无人机飞离移动目标。
# 修改内容：持续跟随最新目标位置，将三帧加权竞争方向积分为有界相对偏移。
# 修改时间：2026-09-12
# 修改目的：让主从根据原生主锁定竞争独立调整飞行方向。
# 修改内容：按零点六、零点三、零点一平滑最近三次改善方向并输出直接航向控制。
"""不读取目标高度的协同竞争方向控制。"""

from collections import deque
import math

from competition.baselines.coop_distributed import _bearing_deg
from .tracking import _move, _offset_m


def adjust_flight_speed(distance_m, heading_error_deg, *, min_speed_mps=15.0,
                        max_speed_mps=40.0, slow_distance_m=40.0,
                        fast_distance_m=200.0, slow_turn_deg=60.0):
    """远且对准时加速，近或大幅转向时降速；距离指向期望位置而非目标本身。"""
    distance_weight = max(0.0, min(1.0, (distance_m - slow_distance_m)
                                   / (fast_distance_m - slow_distance_m)))
    turn = abs((heading_error_deg + 180.0) % 360.0 - 180.0)
    heading_weight = max(0.0, 1.0 - turn / slow_turn_deg)
    return min_speed_mps + (max_speed_mps - min_speed_mps) * distance_weight * heading_weight


class CompetitionDirectionController:
    WEIGHTS = (0.6, 0.3, 0.1)

    def __init__(self, update_period_s=0.5, offset_step_mps=5.0, max_offset_m=50.0,
                 history_timeout_s=1.0):
        self.update_period_s = update_period_s
        self.offset_step_mps = offset_step_mps
        self.max_offset_m = max_offset_m
        self.history_timeout_s = history_timeout_s
        self.reset()

    def reset(self):
        self.key = None
        self.history = deque(maxlen=3)
        self.last_update = None
        self.mode = "IDLE"
        self.competitor = None
        self.direction = (0.0, 0.0)
        self.heading_deg = None
        self.last_step = None
        self.offset = (0.0, 0.0)
        self.goal_position = None
        self.goal_distance_m = None

    @property
    def summary(self):
        return {"mode": self.mode, "competitor": self.competitor,
                "history": list(self.history), "direction_en": self.direction,
                "strength": math.hypot(*self.direction),
                "heading_deg": self.heading_deg, "offset_en_m": self.offset,
                "goal_position": self.goal_position,
                "goal_distance_m": self.goal_distance_m}

    def heading(self, now, key, own_position, current_heading, desired_position,
                desired_visible, primary_matches, primary_position):
        """根据最新目标位置加相对偏移求航向，不估计目标速度。"""
        if key != self.key:
            self.reset()
            self.key = key
        self.heading_deg = current_heading
        dt = max(0.0, now - self.last_step) if self.last_step is not None else 0.0
        self.last_step = now
        self.competitor = None
        self.goal_position = None
        self.goal_distance_m = None
        if desired_position is None:
            self.mode = "NO_TARGET"
        elif primary_matches:
            self.mode = "LOCK_MATCHED"
        elif not desired_visible:
            self.mode = "APPROACH"
            # 覆盖丢失时先撤销竞争偏移，靠近最后已知的目标位置。
            self.offset = (0.0, 0.0)
        elif primary_position is None:
            # 空检测不是竞争者消失的证据；只沿用近期方向，不补写历史采样。
            self.mode = ("IMPROVE_MEMORY" if self.history and self.last_update is not None
                         and now - self.last_update <= self.history_timeout_s
                         else "NO_COMPETITOR")
        else:
            self.mode = "IMPROVE"
            self.competitor = primary_position
            if self.last_update is None or now - self.last_update >= self.update_period_s:
                # 中间长时间没有竞争证据时，不沿用几秒前的方向。
                if self.last_update is not None and now - self.last_update > self.history_timeout_s:
                    self.history.clear()
                east, north = _offset_m(primary_position, desired_position)
                norm = math.hypot(east, north)
                if norm > 1e-6:
                    self.history.appendleft((now, east / norm, north / norm))
                    self.last_update = now
        if self.mode in ("IMPROVE", "IMPROVE_MEMORY"):
            self.direction = tuple(sum(w * item[axis] for w, item in
                                       zip(self.WEIGHTS, self.history))
                                   for axis in (1, 2))
            # 按实际时间积分位移，缺少的历史项不补权重，避免调用频率改变偏移速度。
            self.offset = tuple(p + self.offset_step_mps * dt * d
                                for p, d in zip(self.offset, self.direction))
            radius = math.hypot(*self.offset)
            if radius > self.max_offset_m:
                self.offset = tuple(p * self.max_offset_m / radius for p in self.offset)

        else:
            # 正确锁定时冻结相对偏移，而非冻结绝对航向；旧竞争方向不继续积分。
            self.history.clear()
            self.last_update = None
            self.direction = (0.0, 0.0)
        if desired_position is not None:
            self.goal_position = _move(desired_position, *self.offset)
            self.goal_distance_m = math.hypot(*_offset_m(own_position, self.goal_position))
            if self.goal_distance_m > 1.0:
                self.heading_deg = _bearing_deg(*own_position, *self.goal_position)
        return self.heading_deg
