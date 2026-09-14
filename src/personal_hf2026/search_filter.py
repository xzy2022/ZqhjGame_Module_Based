# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：复用通用候选轨迹集合而不改变搜索阶段的速度分类规则。
# 修改内容：将候选关联实现迁入 tracking，搜索类继承同一关联实现。
# 修改时间：2026-09-12
# 修改目的：缩短静止诱饵和活动目标在搜索阶段的速度判别等待时间。
# 修改内容：将活动目标确认时间和静止目标拒绝时间统一调整为两秒。
# 修改时间：2026-09-12
# 修改目的：让搜索阶段稳定区分多辆车辆并用全部轨迹共同控制搜索光轴。
# 修改内容：为候选轨迹增加稳定编号、短时保留和预测位置中心计算。
"""仅在搜索阶段使用的速度候选过滤器。"""

from dataclasses import dataclass
import math

from competition.baselines.coop_distributed import _haversine_m

from .tracking import CandidateTrackSet, LocalTrackManager, TrackConfig, estimate_track_velocity


@dataclass(frozen=True)
class SearchFilterConfig:
    confirm_after_s: float = 2.0
    moving_speed_mps: float = 3.0
    reject_after_s: float = 2.0
    stationary_speed_mps: float = 2.5
    timeout_s: float = 8.0
    rejected_radius_m: float = 15.0
    rejected_memory_s: float = 60.0
    track_retention_s: float = 2.0


@dataclass(frozen=True)
class SearchFilterResult:
    status: str
    speed_mps: float = 0.0
    reason: str = "not_started"


class SearchSpeedFilter:
    """搜索时区分静止诱饵；协同阶段不调用。"""

    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"

    def __init__(self, config=None):
        self.config = config or SearchFilterConfig()
        self.epoch = 0
        self.started_at = None
        self.status = self.PENDING
        self.rejected_positions = []
        self.accepted_count = 0
        self.rejected_count = 0
        self.timeout_count = 0
        self.last_result = SearchFilterResult(self.PENDING)

    def reset_current(self):
        self.epoch = 0
        self.started_at = None
        self.status = self.PENDING

    def _result(self, speed, reason):
        self.last_result = SearchFilterResult(self.status, speed, reason)
        return self.last_result

    def should_ignore(self, position, now):
        self.rejected_positions = [
            (point, until) for point, until in self.rejected_positions if until > now
        ]
        return any(_haversine_m(*position, *point) < self.config.rejected_radius_m
                   for point, _ in self.rejected_positions)

    def evaluate(self, now, snapshot):
        if snapshot.epoch != self.epoch:
            self.epoch = snapshot.epoch
            self.started_at = snapshot.points[0].t if snapshot.points else now
            self.status = self.PENDING
        if self.status != self.PENDING:
            velocity = estimate_track_velocity(snapshot.points)
            return self._result(math.hypot(*velocity), "decision_kept")
        if snapshot.state not in (LocalTrackManager.CONFIRMED, LocalTrackManager.COASTING):
            return self._result(0.0, "track_not_confirmed")
        if len(snapshot.points) < 2:
            return self._result(0.0, "insufficient_points")
        span = snapshot.points[-1].t - snapshot.points[0].t
        velocity = estimate_track_velocity(snapshot.points)
        speed = math.hypot(*velocity)
        if span >= self.config.confirm_after_s and speed >= self.config.moving_speed_mps:
            self.status = self.ACCEPTED
            self.accepted_count += 1
            reason = "moving"
        elif span >= self.config.reject_after_s and speed < self.config.stationary_speed_mps:
            self.status = self.REJECTED
            self.rejected_count += 1
            reason = "stationary"
            if snapshot.position is not None:
                self.rejected_positions.append(
                    (snapshot.position, now + self.config.rejected_memory_s))
        elif self.started_at is not None and now - self.started_at >= self.config.timeout_s:
            self.status = self.REJECTED
            self.timeout_count += 1
            reason = "timeout"
        else:
            reason = "collecting"
        return self._result(speed, reason)


class MultiCandidateSearch(CandidateTrackSet):
    """搜索阶段同时维护视野内多辆车的临时轨迹。"""

    PENDING = SearchSpeedFilter.PENDING
    ACCEPTED = SearchSpeedFilter.ACCEPTED
    REJECTED = SearchSpeedFilter.REJECTED

    def __init__(self, config=None):
        self.config = config or SearchFilterConfig()
        self.status = self.PENDING
        self.epoch = 0
        self.started_at = None
        self.rejected_positions = []
        self.accepted_count = 0
        self.rejected_count = 0
        self.timeout_count = 0
        self.last_result = SearchFilterResult(self.PENDING)
        self.focus_position = None
        self.view_center_position = None
        self.accepted_track_id = None
        super().__init__(TrackConfig(lost_after_s=self.config.track_retention_s))

    def reset_current(self):
        self.status = self.PENDING
        self.epoch = 0
        self.started_at = None
        self.focus_position = None
        self.view_center_position = None
        self.accepted_track_id = None
        self._tracks.clear()

    def should_ignore(self, position, now):
        self.rejected_positions = [
            (point, until) for point, until in self.rejected_positions if until > now
        ]
        return any(_haversine_m(*position, *point) < self.config.rejected_radius_m
                   for point, _ in self.rejected_positions)

    def _update_view_center(self, now):
        positions = [candidate.manager.predict_position(now)
                     for candidate in self._tracks]
        positions = [position for position in positions if position is not None]
        if not positions:
            self.view_center_position = None
            return
        self.view_center_position = (
            sum(position[0] for position in positions) / len(positions),
            sum(position[1] for position in positions) / len(positions),
        )

    @property
    def track_count(self):
        return len(self._tracks)

    def update(self, now, positions):
        """更新所有候选；返回首条确认移动的轨迹快照。"""
        positions = tuple(position for position in positions
                          if not self.should_ignore(position, now))
        self._associate(now, positions)
        self._update_view_center(now)
        accepted = []
        rejected = []
        timed_out = []
        pending = []
        for candidate in self._tracks:
            snapshot = candidate.manager.snapshot(now, window_s=6.0)
            if not snapshot.points:
                continue
            span = snapshot.points[-1].t - snapshot.points[0].t
            speed = math.hypot(*estimate_track_velocity(snapshot.points))
            if (snapshot.state in (LocalTrackManager.CONFIRMED,
                                   LocalTrackManager.COASTING)
                    and span >= self.config.confirm_after_s
                    and speed >= self.config.moving_speed_mps):
                accepted.append((span, speed, candidate.track_id, snapshot))
            elif (snapshot.state in (LocalTrackManager.CONFIRMED,
                                     LocalTrackManager.COASTING)
                  and span >= self.config.reject_after_s
                  and speed < self.config.stationary_speed_mps):
                rejected.append((candidate.track_id, snapshot))
            elif now - snapshot.points[0].t >= self.config.timeout_s:
                timed_out.append((candidate.track_id, snapshot))
            else:
                pending.append((span, speed, candidate.track_id, snapshot))

        for _, snapshot in rejected:
            if snapshot.position is not None:
                self.rejected_positions.append(
                    (snapshot.position, now + self.config.rejected_memory_s))
        if rejected:
            self.rejected_count += len(rejected)
        if timed_out:
            self.timeout_count += len(timed_out)
        if rejected or timed_out:
            removed_ids = {track_id for track_id, _ in rejected + timed_out}
            self._tracks = [candidate for candidate in self._tracks
                            if candidate.track_id not in removed_ids]
            self._update_view_center(now)

        if accepted:
            _, speed, track_id, snapshot = max(
                accepted, key=lambda item: (item[0], item[1]))
            self.status = self.ACCEPTED
            self.epoch = snapshot.epoch
            self.started_at = snapshot.points[0].t
            self.focus_position = snapshot.position
            self.accepted_track_id = track_id
            self.accepted_count += 1
            self.last_result = SearchFilterResult(self.ACCEPTED, speed, "moving")
            return snapshot

        if pending:
            span, speed, _, snapshot = max(
                pending, key=lambda item: (item[0], item[1]))
            self.status = self.PENDING
            self.epoch = snapshot.epoch
            self.started_at = snapshot.points[0].t
            self.focus_position = snapshot.position
            self.last_result = SearchFilterResult(self.PENDING, speed, "collecting")
        else:
            self.status = self.PENDING
            self.epoch = 0
            self.started_at = None
            self.focus_position = None
            self.view_center_position = None
            self.last_result = SearchFilterResult(self.PENDING, 0.0, "no_candidate")
        return None
