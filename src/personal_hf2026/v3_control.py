# 修改时间：2026-09-20。
# 修改目的：为真实感知 V3 提供可独立集成的航线、协同门控和跟踪纠偏控制层。
# 修改内容：新增鸭子类型感知输入、连续五个新帧门控、固定四十八度视场及目标竞争方向控制。
"""V3 控制层：把真实感知结果适配到 PersonalV1 的协同状态机。"""

from dataclasses import dataclass, replace
from typing import Any

from competition.sdk.core.commands import Command
from competition.sdk.core.observation import Detection

from .competition_flight import CompetitionDirectionController
from .coordination import CoopCoordinator
from .personal_v1 import PersonalV1Agent


_MISSING = object()


def _field(value, name, default=None):
    """同时读取字典和对象字段，避免控制层绑定具体感知实现。"""
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _position(value):
    """提取已有地理坐标；像素观察本身不会被误当作经纬度。"""
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    lat = _field(value, "target_lat", _MISSING)
    lon = _field(value, "target_lon", _MISSING)
    if lat is _MISSING or lon is _MISSING:
        lat = _field(value, "lat", _MISSING)
        lon = _field(value, "lon", _MISSING)
    if lat in (_MISSING, None) or lon in (_MISSING, None):
        return None
    return float(lat), float(lon)


def _confidence(value):
    for name in ("real_score", "real_probability", "detector_confidence", "confidence"):
        confidence = _field(value, name, None)
        if confidence is not None:
            return max(0.0, min(1.0, float(confidence)))
    return 0.0


def _is_target(value):
    """仅接受真实模型的明确真车类别；SDK 的 ground_vehicle 不证明真假。"""
    name = _field(value, "class_name", None)
    if name is None:
        name = _field(value, "target_type", None)
    return str(name or "").strip().lower() in {"real_vehicle", "target_vehicle"}


@dataclass(frozen=True)
class V3PerceptionInput:
    """控制层输入；前三项可直接保存感知模块的 ``PixelObservation``。"""

    detection: Any = None
    track_predict: Any = None
    closest_others: Any = None
    target_position: tuple[float, float] | None = None
    competitor_position: tuple[float, float] | None = None
    frame_id: Any = None
    source_sim_time: float | None = None
    observed_sim_time: float | None = None
    primary_is_target: bool | None = None

    @property
    def is_target(self):
        return (_is_target(self.detection) if self.primary_is_target is None
                else bool(self.primary_is_target))

    @property
    def new_frame_key(self):
        if self.frame_id is not None or self.source_sim_time is not None:
            return self.frame_id, self.source_sim_time
        return None

    @property
    def primary_key(self):
        key = _field(self.detection, "track_id", None)
        return key if key is not None else _field(self.track_predict, "track_id", None)


class ConsecutiveTargetGate:
    """只让不同视觉帧贡献计数，重复 decide 不会累计协同资格。"""

    def __init__(self, required_frames=5):
        self.required_frames = int(required_frames)
        self.reset()

    def reset(self):
        self.count = 0
        self.last_key = None
        self.last_primary_key = None
        self.accepted_frames = 0

    def observe(self, frame, fallback_key):
        key = frame.new_frame_key
        if key is None:
            # 无帧标识时，每次显式 submit 视作一个新结果；集成层不应重复提交旧快照。
            key = ("submission", fallback_key)
        if key == self.last_key:
            return False
        self.last_key = key
        self.accepted_frames += 1
        if (self.last_primary_key is not None and frame.primary_key is not None
                and frame.primary_key != self.last_primary_key):
            self.count = 0
        self.last_primary_key = frame.primary_key
        self.count = self.count + 1 if frame.is_target else 0
        return True

    @property
    def ready(self):
        return self.count >= self.required_frames


class _GatedCoordinator(CoopCoordinator):
    """仅收紧 SEARCH 发起条件，不改变 V1 的通信协议和协同状态机。"""

    def __init__(self, *args, proposal_gate, **kwargs):
        self._proposal_gate = proposal_gate
        super().__init__(*args, **kwargs)

    def step(self, now, local, inbox, can_propose=True, **kwargs):
        return super().step(
            now, local, inbox,
            can_propose=bool(can_propose and self._proposal_gate()),
            **kwargs,
        )


class _V3CompetitionDirectionController(CompetitionDirectionController):
    """用融合后的目标/竞争对象坐标覆盖 V1 的原生主锁定推断。"""

    def reset(self):
        super().reset()
        self.target_position = None
        self.competitor_position = None

    def set_perception(self, target_position, competitor_position):
        self.target_position = target_position
        self.competitor_position = competitor_position

    def heading(self, now, key, own_position, current_heading, desired_position,
                desired_visible, primary_matches, primary_position):
        target = self.target_position or desired_position
        competitor = self.competitor_position
        return super().heading(
            now, key, own_position, current_heading, target,
            desired_visible=bool(target is not None),
            primary_matches=competitor is None,
            primary_position=competitor,
        )


class PersonalV3ControlAgent(PersonalV1Agent):
    """保持正式 Agent 接口的 V3 控制基类。

    感知实现应在同一 Agent 实例的 ``sensor()`` 中调用 ``submit_perception``；
    Runner 随后调用的 ``decide(obs, dt)`` 会消费该结果。像素对象不要求含经纬度，
    投影/融合模块可通过 ``target_position`` 和 ``competitor_position`` 单独提交。
    """

    SEARCH_FOV_DEG = 48.0
    FLIGHT_ALT_M = 500.0
    TARGET_ALT_M = 0.0
    TARGET_CONFIRM_FRAMES = 5
    PERCEPTION_STALE_S = 1.0

    def reset(self):
        super().reset()
        self._target_gate = ConsecutiveTargetGate(self.TARGET_CONFIRM_FRAMES)
        self._coordinator = _GatedCoordinator(
            self.my_uid, (self.A, self.B, self.C), self.COOP_DURATION_S,
            self.COOP_REACQUIRE_TIMEOUT_S,
            proposal_gate=lambda: self._target_gate.ready,
        )
        self._competition_flight = _V3CompetitionDirectionController(
            self.COMPETITION_UPDATE_PERIOD_S, self.COMPETITION_OFFSET_STEP_MPS,
            self.COMPETITION_MAX_OFFSET_M,
        )
        self._perception = None
        self._submission_serial = 0
        self._consumed_serial = 0
        self._active_target_position = None
        self._active_competitor_position = None

    def submit_perception(self, snapshot=None, *, detection=None,
                          track_predict=None, closest_others=None,
                          target_position=None, competitor_position=None,
                          frame_id=None, source_sim_time=None,
                          observed_sim_time=None, primary_is_target=None):
        """提交一帧感知/投影结果；参数均支持字典或带同名字段的对象。

        ``detection``、``track_predict``、``closest_others`` 可直接是视觉层对象；
        控制所需经纬度优先读取独立的位置参数，仅在对象确实含地理字段时回退提取。
        """
        if snapshot is not None:
            detection = (detection if detection is not None
                         else _field(snapshot, "detection", None))
            track_predict = (track_predict if track_predict is not None
                             else _field(snapshot, "track_predict", None))
            closest_others = (closest_others if closest_others is not None
                              else _field(snapshot, "closest_others", None))
            target_position = (target_position if target_position is not None
                               else _field(snapshot, "target_position", None))
            competitor_position = (
                competitor_position if competitor_position is not None
                else _field(snapshot, "competitor_position", None))
            frame_id = (frame_id if frame_id is not None
                        else _field(snapshot, "frame_id", None))
            source_sim_time = (
                source_sim_time if source_sim_time is not None
                else _field(snapshot, "source_sim_time", None))
            observed_sim_time = (
                observed_sim_time if observed_sim_time is not None
                else _field(snapshot, "observed_sim_time", None))
            primary_is_target = (
                primary_is_target if primary_is_target is not None
                else _field(snapshot, "primary_is_target", None))

        target_position = (_position(target_position) or _position(track_predict)
                           or _position(detection))
        competitor_position = (_position(competitor_position)
                               or _position(closest_others))
        frame = V3PerceptionInput(
            detection=detection,
            track_predict=track_predict,
            closest_others=closest_others,
            target_position=target_position,
            competitor_position=competitor_position,
            frame_id=frame_id,
            source_sim_time=(None if source_sim_time is None else float(source_sim_time)),
            observed_sim_time=(None if observed_sim_time is None else float(observed_sim_time)),
            primary_is_target=primary_is_target,
        )
        self._submission_serial += 1
        self._perception = frame
        self._target_gate.observe(frame, self._submission_serial)
        return frame

    def _frame_is_fresh(self, frame, now):
        observed = frame.observed_sim_time
        return observed is None or 0.0 <= now - observed <= self.PERCEPTION_STALE_S

    @staticmethod
    def _sdk_detection(frame, position):
        if position is None:
            return Detection(detected=False, confidence=0.0)
        return Detection(
            detected=True,
            confidence=_confidence(frame.detection),
            target_lat=position[0],
            target_lon=position[1],
            target_type="real_vehicle" if frame.is_target else "model_prop",
        )

    def _eligible_positions(self, detections, positions):
        # 轨迹管理器只接收融合后的目标预测，不允许竞争对象串入主轨迹。
        return ((self._active_target_position,)
                if self._active_target_position is not None else ())

    def _competition_positions(self, eligible_positions, all_positions):
        return tuple(position for position in (
            self._active_target_position, self._active_competitor_position)
            if position is not None)

    @property
    def completion_summary(self):
        summary = dict(super().completion_summary)
        summary.update({
            "v3_target_confirm_count": self._target_gate.count,
            "v3_target_confirm_required": self._target_gate.required_frames,
            "v3_target_gate_ready": self._target_gate.ready,
            "v3_perception_frame_id": (None if self._perception is None
                                        else self._perception.frame_id),
            "v3_target_position": self._active_target_position,
            "v3_competitor_position": self._active_competitor_position,
            "v3_fov_deg": self.SEARCH_FOV_DEG,
            "v3_flight_alt_m": self.FLIGHT_ALT_M,
            "v3_target_alt_m": self.TARGET_ALT_M,
        })
        return summary

    def decide(self, obs, dt):
        score = getattr(getattr(obs, "briefing", None), "score_view", None)
        now = float(score.sim_time) if score is not None else self._t + max(0.0, dt)
        frame = self._perception
        is_new = frame is not None and self._submission_serial != self._consumed_serial
        fresh = frame is not None and self._frame_is_fresh(frame, now)
        if fresh:
            self._active_target_position = frame.target_position
            self._active_competitor_position = frame.competitor_position
        else:
            self._active_target_position = None
            self._active_competitor_position = None
        self._competition_flight.set_perception(
            self._active_target_position, self._active_competitor_position)

        if is_new and fresh:
            primary = self._sdk_detection(frame, self._active_target_position)
            multiple = tuple(
                self._sdk_detection(frame, position)
                for position in (self._active_target_position,
                                 self._active_competitor_position)
                if position is not None)
            own = replace(obs.self, detection=primary,
                          detections=multiple if len(multiple) > 1 else ())
            control_obs = replace(obs, self=own)
        else:
            # 同一视觉帧不能在约十赫兹控制循环中伪造成连续新观测。
            own = replace(obs.self, detection=Detection(False, 0.0), detections=())
            control_obs = replace(obs, self=own)
        if is_new:
            self._consumed_serial = self._submission_serial

        commands = super().decide(control_obs, dt)
        fixed = []
        for command in commands:
            if command.verb == "set_destination":
                params = dict(command.params)
                params["altitude"] = self.FLIGHT_ALT_M
                command = Command(command.verb, params)
            elif command.verb == "set_fov":
                command = Command(command.verb, {"angle": self.SEARCH_FOV_DEG})
            fixed.append(command)
        return fixed
