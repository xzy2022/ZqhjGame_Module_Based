# 修改时间：2026-09-21（静止证据连续性修复）。
# 修改目的：避免视觉轨迹编号切换清空同一目标历史，并区分拟合点数与 ACTIVE 连续确认帧数。
# 修改内容：以 H=0 时空门控续接轨迹编号、暴露拟合最小点数，并允许 HOLD 只更新运动历史。
# 修改时间：2026-09-21。
# 修改目的：让 V3 能用不同真实视觉帧的零高程位置鲁棒判断目标是否停止。
# 修改内容：为停止判定器增加局部 ENU 的 Theil-Sen 速度窗口与有界审计证据，同时保留原默认路径。
# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13 14:25
# 修改目的：避免短间隔真实急转弯仅因历史拟合方向滞后而丢失轨迹。
# 修改内容：小位移且预测误差小的连续点可绕过方向门限，其余运动约束保持不变。
# 修改时间：2026-09-13
# 修改目的：识别正常跟踪中运动目标连续三次新观测停止的完成线索。
# 修改内容：新增可复用停止判定器，检查身份、观测新鲜度、历史运动和预测容差。
# 修改时间：2026-09-13
# 修改目的：区分轨迹证据不足与已经观测到的轨迹不一致。
# 修改内容：为一致性结果增加统一的证据不足分类属性。
# 修改时间：2026-09-13
# 修改目的：让搜索与从机初始捕获共用候选轨迹，并按主机轨迹确认目标身份。
# 修改内容：提取候选轨迹集合，增加仅接受唯一一致候选的轨迹关联函数。
# 修改时间：2026-09-13
# 修改目的：减少历史运动方向滞后导致的真实转弯误拒绝。
# 修改内容：增加一秒运动估计窗口，供本地关联判断和位置预测共用。
# 修改时间：2026-09-12
# 修改目的：让三秒轨迹窗口适配实际通信能够提供的有效轨迹点数量。
# 修改内容：将双机轨迹一致性判断的最小配对点数从五个降为三个。
"""可复用的单机目标轨迹管理和双机轨迹一致性判断。"""

from collections import deque
from dataclasses import dataclass
import math
from statistics import median

from competition.baselines.coop_distributed import _haversine_m


@dataclass(frozen=True)
class TrackPoint:
    t: float
    lat: float
    lon: float


@dataclass(frozen=True)
class TrackSnapshot:
    epoch: int
    state: str
    points: tuple
    last_seen: float

    @property
    def position(self):
        return (self.points[-1].lat, self.points[-1].lon) if self.points else None


@dataclass(frozen=True)
class TrackMatchResult:
    passed: bool
    reason: str
    overlap_s: float = 0.0
    point_count: int = 0
    position_median_m: float = float("inf")
    position_p95_m: float = float("inf")
    latest_distance_m: float = float("inf")
    velocity_difference_mps: float = float("inf")
    heading_difference_deg: float = float("inf")

    @property
    def insufficient_evidence(self):
        """缺少比较依据不能直接证明目标不同；身份失效由协同状态机先检查。"""
        return not self.passed and self.reason in (
            "first_not_confirmed", "second_not_confirmed",
            "first_stale", "second_stale", "insufficient_points", "insufficient_overlap",
        )


@dataclass(frozen=True)
class TrackConfig:
    history_s: float = 6.0
    confirm_span_s: float = 1.0
    confirm_points: int = 5
    lost_after_s: float = 0.75
    acquire_radius_m: float = 80.0
    base_distance_gate_m: float = 5.0
    max_speed_mps: float = 25.0
    prediction_gate_m: float = 10.0
    max_acceleration_mps2: float = 12.0
    max_velocity_change_mps: float = 20.0
    max_heading_change_deg: float = 100.0
    turn_continuity_s: float = 0.25
    turn_displacement_m: float = 2.0
    turn_prediction_error_m: float = 3.0
    motion_window_s: float = 1.0


@dataclass(frozen=True)
class MatchConfig:
    window_s: float = 3.0
    fresh_s: float = 0.8
    min_overlap_s: float = 2.0
    min_points: int = 3
    interpolation_gap_s: float = 0.75
    position_median_m: float = 5.0
    position_p95_m: float = 10.0
    latest_distance_m: float = 10.0
    velocity_difference_mps: float = 5.0
    heading_difference_deg: float = 45.0


def _offset_m(origin, point):
    """将小范围经纬度差转换为东、北方向米制偏移。"""
    lat0, lon0 = origin
    lat, lon = point
    north = (lat - lat0) * 111320.0
    east = (lon - lon0) * 111320.0 * math.cos(math.radians((lat + lat0) * 0.5))
    return east, north


def _move(origin, east, north):
    lat, lon = origin
    new_lat = lat + north / 111320.0
    scale = 111320.0 * math.cos(math.radians(lat))
    return new_lat, lon + east / max(scale, 1.0)


def _fit_velocity(points):
    if len(points) < 2 or points[-1].t <= points[0].t:
        return 0.0, 0.0
    origin = (points[0].lat, points[0].lon)
    times = [p.t for p in points]
    mean_t = sum(times) / len(times)
    offsets = [_offset_m(origin, (p.lat, p.lon)) for p in points]
    denominator = sum((t - mean_t) ** 2 for t in times)
    if denominator <= 1e-9:
        return 0.0, 0.0
    mean_x = sum(p[0] for p in offsets) / len(offsets)
    mean_y = sum(p[1] for p in offsets) / len(offsets)
    vx = sum((t - mean_t) * (p[0] - mean_x) for t, p in zip(times, offsets)) / denominator
    vy = sum((t - mean_t) * (p[1] - mean_y) for t, p in zip(times, offsets)) / denominator
    return vx, vy


def _fit_robust_velocity(points):
    """分别取东、北方向两两斜率中位数，降低少量零高程投影离群点影响。"""
    if len(points) < 2 or points[-1].t <= points[0].t:
        return 0.0, 0.0
    origin = (points[0].lat, points[0].lon)
    rows = [(point.t, *_offset_m(origin, (point.lat, point.lon)))
            for point in points]

    def median_slope(component):
        slopes = [
            (second[component] - first[component]) / (second[0] - first[0])
            for index, first in enumerate(rows)
            for second in rows[index + 1:]
            if second[0] - first[0] > 1e-9
        ]
        return median(slopes) if slopes else 0.0

    return median_slope(1), median_slope(2)


def estimate_track_velocity(points):
    """对外提供统一的轨迹速度估计，搜索过滤和轨迹比较共用。"""
    return _fit_velocity(tuple(points))


class StoppedTargetDetector:
    """接收已关联轨迹或 V3 原始视觉点；静止线索是本地估计，不是裁判确认。"""

    def __init__(self, required_points=3, stationary_distance_m=0.05,
                 moving_speed_mps=3.0, *, robust_window_s=None,
                 robust_min_span_s=2.0, robust_min_points=5,
                 stationary_speed_mps=5.0, max_frame_gap_s=1.5,
                 track_switch_base_gate_m=8.0,
                 track_switch_speed_gate_mps=25.0):
        self.required_points = required_points
        self.stationary_distance_m = stationary_distance_m
        self.moving_speed_mps = moving_speed_mps
        self.robust_window_s = robust_window_s
        self.robust_min_span_s = robust_min_span_s
        self.robust_min_points = robust_min_points
        self.stationary_speed_mps = stationary_speed_mps
        self.max_frame_gap_s = max_frame_gap_s
        self.track_switch_base_gate_m = track_switch_base_gate_m
        self.track_switch_speed_gate_mps = track_switch_speed_gate_mps
        self.config = TrackConfig()
        self.reset()

    def reset(self):
        self.identity = None
        self.was_moving = False
        self.previous = None
        self.velocity = (0.0, 0.0)
        self.anchor = None
        self.stationary_points = 0
        self._observations = deque()
        self._last_frame_key = None
        self._last_source_time = None
        self.visual_track_id = None
        self.frame_id = None
        self.source_sim_time = None
        self.position_h0 = None
        self.window_span_s = 0.0
        self.distinct_frame_count = 0
        self.speed_mps = None
        self.near_zero = False
        self.ready = False
        self.duplicate_frames_ignored = 0
        self.out_of_order_frames_ignored = 0
        self.track_switch_count = 0
        self.track_switch_reset_count = 0
        self.last_track_switch_continuous = None
        self.last_track_switch_distance_m = None
        self.last_track_switch_gate_m = None
        self.confirmation_enabled = False

    @property
    def evidence(self):
        """返回 V3 运行记录所需的有界静止判定快照。"""
        return {
            "frame_id": self.frame_id,
            "source_sim_time": self.source_sim_time,
            "position_h0": self.position_h0,
            "window_span_s": self.window_span_s,
            "distinct_frame_count": self.distinct_frame_count,
            "fit_min_points": self.robust_min_points,
            "east_velocity_mps": self.velocity[0] if self.speed_mps is not None else None,
            "north_velocity_mps": self.velocity[1] if self.speed_mps is not None else None,
            "speed_mps": self.speed_mps,
            "speed_threshold_mps": self.stationary_speed_mps,
            "moving_speed_threshold_mps": self.moving_speed_mps,
            "was_moving": self.was_moving,
            "near_zero": self.near_zero,
            "stationary_consecutive_frames": self.stationary_points,
            "required_stationary_frames": self.required_points,
            "confirmation_enabled": self.confirmation_enabled,
            "ready": self.ready,
            "duplicate_frames_ignored": self.duplicate_frames_ignored,
            "out_of_order_frames_ignored": self.out_of_order_frames_ignored,
            "visual_track_id": self.visual_track_id,
            "track_switch_count": self.track_switch_count,
            "track_switch_reset_count": self.track_switch_reset_count,
            "last_track_switch_continuous": self.last_track_switch_continuous,
            "last_track_switch_distance_m": self.last_track_switch_distance_m,
            "last_track_switch_gate_m": self.last_track_switch_gate_m,
        }

    def update_observation(self, identity, position, *, frame_key,
                           frame_id, source_sim_time, valid, track_id=None,
                           confirm=True):
        """消费一张 V3 新视觉帧；同一帧在当前身份内最多消费一次。"""
        if not valid or frame_key is None or source_sim_time is None:
            return False
        if frame_key == self._last_frame_key:
            self.duplicate_frames_ignored += 1
            return False
        source_sim_time = float(source_sim_time)
        if (self._last_source_time is not None
                and source_sim_time <= self._last_source_time + 1e-9):
            self.out_of_order_frames_ignored += 1
            return False

        self._last_frame_key = frame_key
        self.frame_id = frame_id
        self.source_sim_time = source_sim_time
        self.confirmation_enabled = bool(confirm)
        if (self._last_source_time is not None
                and source_sim_time - self._last_source_time > self.max_frame_gap_s):
            self._observations.clear()
            self.stationary_points = 0
            self.velocity = (0.0, 0.0)
            self.window_span_s = 0.0
            self.distinct_frame_count = 0
            self.speed_mps = None
            self.near_zero = False
            self.ready = False
        self._last_source_time = source_sim_time

        if position is None:
            # 新帧未给出同一真目标的 H=0 点，不能延续连续静止确认。
            self.position_h0 = None
            self.stationary_points = 0
            self.near_zero = False
            self.ready = False
            return False
        if identity != self.identity:
            seen_key = frame_key
            ignored_duplicates = self.duplicate_frames_ignored
            ignored_out_of_order = self.out_of_order_frames_ignored
            self.reset()
            self.identity = identity
            self.visual_track_id = track_id
            self.confirmation_enabled = bool(confirm)
            self._last_frame_key = seen_key
            self.duplicate_frames_ignored = ignored_duplicates
            self.out_of_order_frames_ignored = ignored_out_of_order
            self.frame_id = frame_id
            self.source_sim_time = source_sim_time
            self._last_source_time = source_sim_time

        if (track_id is not None and self.visual_track_id is not None
                and track_id != self.visual_track_id):
            self.track_switch_count += 1
            previous = self._observations[-1] if self._observations else None
            distance = None
            gate = None
            continuous = False
            if previous is not None:
                dt = source_sim_time - previous.t
                if 0.0 < dt <= self.max_frame_gap_s:
                    offset = _offset_m(
                        (previous.lat, previous.lon),
                        (float(position[0]), float(position[1])),
                    )
                    distance = math.hypot(*offset)
                    gate = (self.track_switch_base_gate_m
                            + self.track_switch_speed_gate_mps * dt)
                    continuous = distance <= gate
            self.last_track_switch_continuous = continuous
            self.last_track_switch_distance_m = distance
            self.last_track_switch_gate_m = gate
            if not continuous:
                self.track_switch_reset_count += 1
                self._observations.clear()
                self.was_moving = False
                self.stationary_points = 0
                self.velocity = (0.0, 0.0)
                self.window_span_s = 0.0
                self.distinct_frame_count = 0
                self.speed_mps = None
                self.near_zero = False
                self.ready = False
            self.visual_track_id = track_id
        elif track_id is not None:
            self.visual_track_id = track_id

        point = TrackPoint(source_sim_time, float(position[0]), float(position[1]))
        self.position_h0 = (point.lat, point.lon)
        self._observations.append(point)
        window_s = (self.robust_window_s if self.robust_window_s is not None
                    else self.config.motion_window_s)
        while (self._observations
               and point.t - self._observations[0].t > window_s):
            self._observations.popleft()
        self.distinct_frame_count = len(self._observations)
        self.window_span_s = (0.0 if len(self._observations) < 2 else
                              self._observations[-1].t - self._observations[0].t)
        enough = (self.distinct_frame_count >= self.robust_min_points
                  and self.window_span_s >= self.robust_min_span_s)
        if enough:
            self.velocity = _fit_robust_velocity(tuple(self._observations))
            self.speed_mps = math.hypot(*self.velocity)
            if self.speed_mps >= self.moving_speed_mps:
                self.was_moving = True
        else:
            self.velocity = (0.0, 0.0)
            self.speed_mps = None
        self.near_zero = bool(
            enough and self.was_moving and self.speed_mps <= self.stationary_speed_mps)
        self.stationary_points = (
            self.stationary_points + 1 if self.near_zero and confirm else 0)
        self.ready = bool(
            self.was_moving and self.stationary_points >= self.required_points)
        return self.ready

    def update(self, now, identity, track, valid):
        if identity != self.identity:
            self.reset()
            self.identity = identity
        if (not valid or track.state != LocalTrackManager.CONFIRMED
                or not track.points or abs(track.last_seen - now) > 1e-6
                or abs(track.points[-1].t - now) > 1e-6):
            self.previous = self.anchor = None
            self.stationary_points = 0
            return False
        point = track.points[-1]
        if self.previous is not None and point.t <= self.previous.t:
            # 同一观测重复调用不能贡献第二个静止点。
            return False
        history = tuple(p for p in track.points
                        if 0.0 <= now - p.t <= self.config.motion_window_s)
        velocity = estimate_track_velocity(history)
        if (len(history) >= 3 and history[-1].t - history[0].t >= 0.5
                and math.hypot(*velocity) >= self.moving_speed_mps):
            self.was_moving = True
        compatible = False
        if self.previous is not None:
            dt = point.t - self.previous.t
            predicted = _move((self.previous.lat, self.previous.lon),
                              self.velocity[0] * dt, self.velocity[1] * dt)
            gate = self.config.prediction_gate_m + 0.5 * self.config.max_acceleration_mps2 * dt * dt
            compatible = (dt <= self.config.lost_after_s
                          and _haversine_m(*predicted, point.lat, point.lon) <= gate)
        if not compatible:
            self.anchor = None
            self.stationary_points = 0
        if (self.anchor is not None and _haversine_m(
                self.anchor.lat, self.anchor.lon, point.lat, point.lon) <= self.stationary_distance_m):
            self.stationary_points += 1
        else:
            self.anchor = point
            self.stationary_points = 1
        self.previous, self.velocity = point, velocity
        return self.was_moving and self.stationary_points >= self.required_points


def _heading_difference(v0, v1):
    speed0, speed1 = math.hypot(*v0), math.hypot(*v1)
    if speed0 < 1.0 or speed1 < 1.0:
        return 0.0
    dot = max(-1.0, min(1.0, (v0[0] * v1[0] + v0[1] * v1[1]) / (speed0 * speed1)))
    return math.degrees(math.acos(dot))


class LocalTrackManager:
    """每架无人机独立使用，防止当前检测跳到另一辆车。"""

    LOST = "LOST"
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    COASTING = "COASTING"

    def __init__(self, config=None):
        self.config = config or TrackConfig()
        self.epoch = 0
        self.state = self.LOST
        self.points = deque()
        self.last_seen = -1e9
        self.seed = None
        self._confirmed = False

    @property
    def position(self):
        return (self.points[-1].lat, self.points[-1].lon) if self.points else None

    def reset_for_acquisition(self, seed=None):
        self.state = self.LOST
        self.points.clear()
        self.last_seen = -1e9
        self.seed = seed
        self._confirmed = False

    def set_acquisition_seed(self, seed):
        self.seed = seed

    def clear_acquisition_seed(self):
        self.seed = None

    def _start(self, now, position):
        self.epoch += 1
        self.state = self.TENTATIVE
        self.points.clear()
        self.points.append(TrackPoint(now, *position))
        self.last_seen = now
        self._confirmed = False

    def _mark_lost(self):
        self.state = self.LOST
        self.points.clear()
        self._confirmed = False

    def _trim(self, now):
        while self.points and now - self.points[0].t > self.config.history_s:
            self.points.popleft()

    def _motion_velocity(self):
        """按最后接受点之前的短窗口估计运动，完整历史仍供轨迹统计使用。"""
        if not self.points:
            return 0.0, 0.0
        cutoff = self.points[-1].t - self.config.motion_window_s
        return _fit_velocity([point for point in self.points if point.t >= cutoff])

    def _compatible(self, now, position):
        last = self.points[-1]
        dt = now - last.t
        if dt <= 0.0:
            return False
        distance = _haversine_m(last.lat, last.lon, *position)
        if distance > self.config.base_distance_gate_m + self.config.max_speed_mps * dt:
            return False
        segment = _offset_m((last.lat, last.lon), position)
        segment_velocity = (segment[0] / dt, segment[1] / dt)
        if math.hypot(*segment_velocity) > self.config.max_speed_mps:
            return False
        history = list(self.points)
        if len(history) < 3 or history[-1].t - history[0].t < 0.5:
            return True
        velocity = self._motion_velocity()
        predicted = _move((last.lat, last.lon), velocity[0] * dt, velocity[1] * dt)
        prediction_error = _haversine_m(*predicted, *position)
        prediction_gate = (self.config.prediction_gate_m
                           + 0.5 * self.config.max_acceleration_mps2 * dt * dt)
        if prediction_error > prediction_gate:
            return False
        if math.hypot(segment_velocity[0] - velocity[0],
                      segment_velocity[1] - velocity[1]) > self.config.max_velocity_change_mps:
            return False
        # 高频小位移方向易受真实转弯和位置误差影响，必须同时满足紧预测门限。
        continuous_turn = (dt <= self.config.turn_continuity_s
                           and distance <= self.config.turn_displacement_m
                           and prediction_error <= self.config.turn_prediction_error_m)
        return (continuous_turn
                or _heading_difference(velocity, segment_velocity) <= self.config.max_heading_change_deg)

    def predict_position(self, now):
        """按已有轨迹预测 ``now`` 时刻的位置。"""
        if not self.points:
            return None
        last = self.points[-1]
        dt = max(0.0, now - last.t)
        velocity = self._motion_velocity()
        return _move((last.lat, last.lon), velocity[0] * dt, velocity[1] * dt)

    def association_distance(self, now, position):
        """返回候选点到预测位置的距离；不兼容时返回无穷大。"""
        if self.state == self.LOST or not self.points:
            return float("inf")
        if not self._compatible(now, position):
            return float("inf")
        predicted = self.predict_position(now)
        return _haversine_m(*predicted, *position)

    def select_position(self, now, positions):
        """从多目标检测中选择与当前轨迹最一致的坐标。"""
        positions = tuple(positions)
        if not positions:
            return None
        if self.state == self.LOST:
            if self.seed is None:
                return None
            ranked = sorted((_haversine_m(*self.seed, *position), position)
                            for position in positions)
            return (ranked[0][1]
                    if ranked[0][0] <= self.config.acquire_radius_m else None)
        ranked = sorted((self.association_distance(now, position), position)
                        for position in positions)
        return ranked[0][1] if math.isfinite(ranked[0][0]) else None

    def adopt(self, snapshot):
        """把搜索阶段确认的候选轨迹升级为本机正式轨迹。"""
        self.epoch += 1
        self.points = deque(snapshot.points)
        self.last_seen = snapshot.last_seen
        self._confirmed = snapshot.state in (self.CONFIRMED, self.COASTING)
        self.state = self.CONFIRMED if self._confirmed else self.TENTATIVE
        self.seed = None

    def update(self, now, position):
        """输入本帧检测坐标；返回更新后的轨迹快照。"""
        if self.state != self.LOST and now - self.last_seen > self.config.lost_after_s:
            self._mark_lost()
        if position is None:
            if self.state == self.CONFIRMED:
                self.state = self.COASTING
            return self.snapshot(now)
        if self.state == self.LOST:
            if self.seed is not None and _haversine_m(*position, *self.seed) > self.config.acquire_radius_m:
                return self.snapshot(now)
            self._start(now, position)
            return self.snapshot(now)
        if not self._compatible(now, position):
            self.state = self.COASTING if self._confirmed else self.TENTATIVE
            return self.snapshot(now)
        self.points.append(TrackPoint(now, *position))
        self.last_seen = now
        self._trim(now)
        span = self.points[-1].t - self.points[0].t
        if len(self.points) >= self.config.confirm_points and span >= self.config.confirm_span_s:
            self._confirmed = True
            self.state = self.CONFIRMED
        else:
            self.state = self.TENTATIVE
        return self.snapshot(now)

    def snapshot(self, now, window_s=3.0):
        points = tuple(p for p in self.points if now - p.t <= window_s)
        return TrackSnapshot(self.epoch, self.state, points, self.last_seen)


@dataclass
class CandidateTrack:
    track_id: int
    manager: LocalTrackManager


class CandidateTrackSet:
    """只维护一对一候选轨迹，不包含真假目标分类。"""

    def __init__(self, config=None):
        self.track_config = config or TrackConfig(lost_after_s=2.0)
        self._tracks = []
        self._next_track_id = 1

    def reset(self):
        self._tracks.clear()

    def _associate(self, now, positions):
        for candidate in self._tracks:
            candidate.manager.update(now, None)
        self._tracks = [c for c in self._tracks if c.manager.state != LocalTrackManager.LOST]
        pairs = []
        for track_index, candidate in enumerate(self._tracks):
            for position_index, position in enumerate(positions):
                distance = candidate.manager.association_distance(now, position)
                if math.isfinite(distance):
                    pairs.append((distance, track_index, position_index))
        used_tracks, used_positions = set(), set()
        for _, track_index, position_index in sorted(pairs):
            if track_index in used_tracks or position_index in used_positions:
                continue
            self._tracks[track_index].manager.update(now, positions[position_index])
            used_tracks.add(track_index)
            used_positions.add(position_index)
        for position_index, position in enumerate(positions):
            if position_index not in used_positions:
                manager = LocalTrackManager(self.track_config)
                manager.update(now, position)
                self._tracks.append(CandidateTrack(self._next_track_id, manager))
                self._next_track_id += 1

    def update(self, now, positions):
        self._associate(now, tuple(positions))
        return tuple((c.track_id, c.manager.snapshot(now)) for c in self._tracks)


@dataclass(frozen=True)
class TargetAssociationResult:
    status: str
    track_id: int | None = None
    track: TrackSnapshot | None = None


def associate_target_track(reference_track, candidate_tracks, now, config=None):
    """仅接受与有效主机轨迹唯一一致的候选，不因最近或静止而直接绑定。"""
    cfg = config or MatchConfig()
    if (reference_track.state not in (LocalTrackManager.CONFIRMED, LocalTrackManager.COASTING)
            or not reference_track.points or now - reference_track.last_seen > cfg.fresh_s):
        return TargetAssociationResult("NO_REFERENCE")
    matched = [(track_id, track) for track_id, track in candidate_tracks
               if compare_track_windows(reference_track, track, now, cfg).passed]
    if len(matched) == 1:
        return TargetAssociationResult("MATCH", *matched[0])
    return TargetAssociationResult("AMBIGUOUS" if matched else "NO_MATCH")


def _interpolate(points, t, max_gap):
    nearest = min(points, key=lambda p: abs(p.t - t), default=None)
    if nearest is not None and abs(nearest.t - t) <= 0.15:
        return nearest
    for left, right in zip(points, points[1:]):
        if left.t <= t <= right.t and right.t - left.t <= max_gap:
            ratio = (t - left.t) / max(right.t - left.t, 1e-9)
            return TrackPoint(t,
                              left.lat + (right.lat - left.lat) * ratio,
                              left.lon + (right.lon - left.lon) * ratio)
    return None


def compare_track_windows(first, second, now, config=None):
    """比较两条最近轨迹；不关心具体无人机或主从角色。"""
    cfg = config or MatchConfig()
    alive_states = (LocalTrackManager.CONFIRMED, LocalTrackManager.COASTING)
    if first.state not in alive_states:
        return TrackMatchResult(False, "first_not_confirmed")
    if second.state not in alive_states:
        return TrackMatchResult(False, "second_not_confirmed")
    if now - first.last_seen > cfg.fresh_s:
        return TrackMatchResult(False, "first_stale")
    if now - second.last_seen > cfg.fresh_s:
        return TrackMatchResult(False, "second_stale")
    first_points = tuple(p for p in first.points if now - p.t <= cfg.window_s)
    second_points = tuple(p for p in second.points if now - p.t <= cfg.window_s)
    if min(len(first_points), len(second_points)) < 2:
        return TrackMatchResult(False, "insufficient_points")
    samples, other, swap = ((first_points, second_points, False)
                            if len(first_points) <= len(second_points)
                            else (second_points, first_points, True))
    pairs = []
    for point in samples:
        matched = _interpolate(other, point.t, cfg.interpolation_gap_s)
        if matched is not None:
            pairs.append((matched, point) if swap else (point, matched))
    if len(pairs) < cfg.min_points:
        return TrackMatchResult(False, "insufficient_points", point_count=len(pairs))
    overlap = pairs[-1][0].t - pairs[0][0].t
    if overlap < cfg.min_overlap_s:
        return TrackMatchResult(False, "insufficient_overlap", overlap_s=overlap,
                                point_count=len(pairs))
    distances = [_haversine_m(a.lat, a.lon, b.lat, b.lon) for a, b in pairs]
    ordered = sorted(distances)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    first_aligned = [a for a, _ in pairs]
    second_aligned = [b for _, b in pairs]
    velocity_a, velocity_b = _fit_velocity(first_aligned), _fit_velocity(second_aligned)
    velocity_difference = math.hypot(velocity_a[0] - velocity_b[0],
                                     velocity_a[1] - velocity_b[1])
    heading_difference = _heading_difference(velocity_a, velocity_b)
    metrics = dict(overlap_s=overlap, point_count=len(pairs),
                   position_median_m=median(distances), position_p95_m=p95,
                   latest_distance_m=distances[-1],
                   velocity_difference_mps=velocity_difference,
                   heading_difference_deg=heading_difference)
    if distances[-1] > cfg.latest_distance_m:
        return TrackMatchResult(False, "latest_position", **metrics)
    if metrics["position_median_m"] > cfg.position_median_m or p95 > cfg.position_p95_m:
        return TrackMatchResult(False, "position", **metrics)
    if velocity_difference > cfg.velocity_difference_mps:
        return TrackMatchResult(False, "velocity", **metrics)
    if heading_difference > cfg.heading_difference_deg:
        return TrackMatchResult(False, "heading", **metrics)
    return TrackMatchResult(True, "matched", **metrics)
