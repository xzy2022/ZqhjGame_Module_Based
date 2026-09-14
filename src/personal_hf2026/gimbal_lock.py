# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：避免邻近诱饵被十五米距离门限误判为期望目标的主锁定。
# 修改内容：增加基于当前已关联观测的排他归属判断，区分匹配、其他对象、歧义和空检测。
# 修改时间：2026-09-12
# 修改目的：允许宽视野内存在多个目标时确认正确的原生主锁定。
# 修改内容：增加多目标同屏锁定开关，启用后锁定稳定性不再依赖候选数量。
# 修改时间：2026-09-12
# 修改目的：为协同跟踪增加独立的短时主锁定恢复窗口。
# 修改内容：输出主锁定连续缺失时长，并以零点五秒参数判断协同锁定是否超时。
# 修改时间：2026-09-12
# 修改目的：让搜索目标进入锁定阶段后立即获得较小视野并按目标数量继续收缩。
# 修改内容：绑定时直接设置十五度视野，多目标时以每秒两度缩到五度，单目标时保持视野。
# 修改时间：2026-09-12
# 修改目的：将期望目标、引擎主锁定校验和动态 FOV 控制从协同流程中独立出来。
# 修改内容：新增可复用的云台锁定控制器，按轨迹预测位置收缩或扩大 FOV。
"""按期望轨迹管理云台主锁定和动态视场角。"""

from dataclasses import dataclass

from competition.baselines.coop_distributed import _haversine_m


@dataclass(frozen=True)
class GimbalLockConfig:
    min_fov_deg: float = 5.0
    max_fov_deg: float = 50.0
    preferred_fov_deg: float = 15.0
    shrink_rate_dps: float = 2.0
    expand_rate_dps: float = 10.0
    primary_gate_m: float = 15.0
    stable_after_s: float = 0.5
    reacquire_timeout_s: float = 2.0
    coop_reacquire_timeout_s: float = 5.0
    allow_multiple_targets: bool = False
    primary_observation_gate_m: float = 2.0
    primary_exclusion_margin_m: float = 1.0


@dataclass(frozen=True)
class PrimaryLockResult:
    status: str
    competitor_position: tuple | None = None
    target_distance_m: float | None = None
    other_distance_m: float | None = None


def classify_primary_lock(primary_position, target_observation, other_observations,
                          config=None):
    """主检测必须唯一对应本帧已关联的目标观测；预测点不能代替实际观测。"""
    cfg = config or GimbalLockConfig()
    if primary_position is None:
        return PrimaryLockResult("EMPTY")
    if target_observation is None:
        return PrimaryLockResult("AMBIGUOUS")
    target_distance = _haversine_m(*primary_position, *target_observation)
    # 只合并重复坐标，不把一米内的不同候选提前合并，保留近邻歧义证据。
    others = tuple(dict.fromkeys(tuple(point) for point in other_observations
                                if _haversine_m(*point, *target_observation) > 1e-6))
    ranked = sorted((_haversine_m(*primary_position, *point), point) for point in others)
    other_distance = ranked[0][0] if ranked else None
    if (target_distance <= cfg.primary_observation_gate_m
            and (other_distance is None
                 or other_distance - target_distance >= cfg.primary_exclusion_margin_m)):
        return PrimaryLockResult("MATCH", None, target_distance, other_distance)
    if (ranked and other_distance <= cfg.primary_observation_gate_m
            and target_distance - other_distance >= cfg.primary_exclusion_margin_m
            and (len(ranked) == 1
                 or ranked[1][0] - other_distance >= cfg.primary_exclusion_margin_m)):
        return PrimaryLockResult("OTHER", ranked[0][1], target_distance, other_distance)
    return PrimaryLockResult("AMBIGUOUS", None, target_distance, other_distance)


@dataclass(frozen=True)
class GimbalLockUpdate:
    state: str
    fov_deg: float
    primary_matches: bool
    desired_visible: bool
    visible_count: int
    expired: bool
    missing_s: float
    coop_expired: bool


class GimbalLockController:
    """绑定一条本地轨迹，并确认引擎主检测是否仍指向它。"""

    IDLE = "IDLE"
    ACQUIRING = "ACQUIRING"
    LOCKED = "LOCKED"
    REACQUIRING = "REACQUIRING"

    def __init__(self, config=None):
        self.config = config or GimbalLockConfig()
        self.reset()

    def reset(self):
        self.state = self.IDLE
        self.track_epoch = None
        self.desired_position = None
        self.fov_deg = self.config.max_fov_deg
        self.primary_matches = False
        self.desired_visible = False
        self.visible_count = 0
        self.stable_since = None
        self.missing_since = None
        self.primary_association = PrimaryLockResult("EMPTY")

    @property
    def active(self):
        return self.track_epoch is not None

    @property
    def locked(self):
        return self.state == self.LOCKED

    def bind(self, track_epoch, position, current_fov_deg=None):
        """绑定新的期望轨迹；轨迹编号变化时重新开始确认。"""
        if track_epoch == self.track_epoch:
            self.desired_position = position
            return
        self.state = self.ACQUIRING
        self.track_epoch = track_epoch
        self.desired_position = position
        self.fov_deg = self._clamp(self.config.preferred_fov_deg)
        self.primary_matches = False
        self.desired_visible = False
        self.visible_count = 0
        self.stable_since = None
        self.missing_since = None

    def _clamp(self, value):
        return max(self.config.min_fov_deg,
                   min(self.config.max_fov_deg, float(value)))

    def _move_fov(self, current, target, rate, dt):
        current = self._clamp(current)
        step = max(0.0, float(rate) * max(0.0, float(dt)))
        if current < target:
            return self._clamp(min(target, current + step))
        return self._clamp(max(target, current - step))

    @staticmethod
    def _detection_position(detection):
        if (detection is None or not detection.detected
                or detection.target_lat is None or detection.target_lon is None):
            return None
        return detection.target_lat, detection.target_lon

    @staticmethod
    def _unique_positions(positions):
        unique = []
        for position in positions:
            if not any(_haversine_m(*position, *old) < 1.0 for old in unique):
                unique.append(position)
        return tuple(unique)

    def update(self, now, dt, track_epoch, desired_position,
               primary_detection, visible_positions, current_fov_deg, *,
               target_observation=None):
        """更新主锁定状态，并返回下一步应该设置的 FOV。"""
        if track_epoch is None or desired_position is None:
            self.reset()
            return GimbalLockUpdate(
                self.state, self.fov_deg, False, False, 0, False, 0.0, False)
        new_binding = track_epoch != self.track_epoch
        self.bind(track_epoch, desired_position, current_fov_deg)
        self.desired_position = desired_position

        positions = self._unique_positions(visible_positions)
        primary_position = self._detection_position(primary_detection)
        self.primary_association = classify_primary_lock(
            primary_position, target_observation, visible_positions, self.config)
        self.primary_matches = self.primary_association.status == "MATCH"
        self.desired_visible = any(
            _haversine_m(*desired_position, *position)
            <= self.config.primary_gate_m for position in positions)
        self.visible_count = len(positions)

        if new_binding:
            if not self.primary_matches:
                self.missing_since = now
            return GimbalLockUpdate(
                self.state, self.fov_deg, self.primary_matches,
                self.desired_visible, self.visible_count, False, 0.0, False)

        current_fov = self._clamp(current_fov_deg)
        if self.primary_matches and (self.config.allow_multiple_targets or self.visible_count <= 1):
            if self.stable_since is None:
                self.stable_since = now
            self.missing_since = None
            self.state = (self.LOCKED
                          if now - self.stable_since >= self.config.stable_after_s
                          else self.ACQUIRING)
            self.fov_deg = current_fov
        else:
            self.stable_since = None
            if not self.primary_matches and self.missing_since is None:
                self.missing_since = now
            elif self.primary_matches:
                self.missing_since = None

            if self.desired_visible and self.visible_count > 1:
                self.state = self.ACQUIRING
                self.fov_deg = self._move_fov(
                    current_fov, self.config.min_fov_deg,
                    self.config.shrink_rate_dps, dt)
            elif self.desired_visible:
                self.state = self.ACQUIRING
                self.fov_deg = current_fov
            else:
                self.state = self.REACQUIRING
                self.fov_deg = self._move_fov(
                    current_fov, self.config.max_fov_deg,
                    self.config.expand_rate_dps, dt)

        missing_s = (max(0.0, now - self.missing_since)
                     if self.missing_since is not None else 0.0)
        expired = (self.missing_since is not None
                   and missing_s >= self.config.reacquire_timeout_s)
        coop_expired = (self.missing_since is not None
                        and missing_s >= self.config.coop_reacquire_timeout_s)
        return GimbalLockUpdate(
            self.state, self.fov_deg, self.primary_matches,
            self.desired_visible, self.visible_count, expired,
            missing_s, coop_expired)
