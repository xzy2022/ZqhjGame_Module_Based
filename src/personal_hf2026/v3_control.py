# 修改时间：2026-09-21（MASTER 等待阶段连续性）。
# 修改目的：避免 MASTER 在等待从机确认时先按零点七五秒丢失并带着过期目标进入 ACTIVE。
# 修改内容：V3 MASTER 的 HOLD 与 ACTIVE 统一使用五秒丢失门限，其余角色和 SEARCH 保持原门限。
# 修改时间：2026-09-21（静止地面接触点接线）。
# 修改目的：把静止判定与瞄准中心投影分开，消除物体高度导致的绕飞投影漂移。
# 修改内容：感知输入新增 stationary_position，并只把该 H=0 接触点送入停止判定器。
# 修改时间：2026-09-21（MASTER 跟踪连续性）。
# 修改目的：让 V3 主机在 ACTIVE 中容忍约五秒缺测，并在滑行期持续瞄准运动预测点。
# 修改内容：仅为 ACTIVE MASTER 动态延长轨迹丢失门限，并用 predict_position(now) 更新 COASTING 云台与广播目标。
# 修改时间：2026-09-21（V3 静止帧接线）。
# 修改目的：让静止判定直接使用每张新视觉帧的原始 H=0 位置而不受本地轨迹拒绝影响。
# 修改内容：向简化协调器传递唯一帧键、源时间、视觉轨迹编号和真车零高程位置。
# 修改时间：2026-09-21（结束证据保留）。
# 修改目的：避免第五个仅诱饵帧触发完成后因回到搜索而把审计计数立即清零。
# 修改内容：在完成边沿的运行证据中保留本次达到阈值的连续诱饵帧数。
# 修改时间：2026-09-21（简化协同控制接线）。
# 修改目的：让主机持续盯住目标并让从机在两级距离门槛后一直指向主机广播坐标。
# 修改内容：接入简化飞行与云台几何、连续五个仅诱饵新帧结束及有界运行证据。
# 修改时间：2026-09-21。
# 修改目的：让 V3 使用不依赖双机轨迹匹配和协同计时的专用简化协调器。
# 修改内容：接入最近从机邀请、零高程目标广播、显式结束信号和结构化协调摘要。
# 修改时间：2026-09-20（异步锁定门槛适配）。
# 修改目的：避免不同真实视觉帧之间的空控制 tick 反复清零 V1 的半秒锁定计时。
# 修改内容：V3 保留五个不同真车帧和运动轨迹确认，并把额外主锁定稳定时间设为零。
# 修改时间：2026-09-20（预测主锁定接线）。
# 修改目的：避免把意图目标回写成预测主锁定而形成自证闭环。
# 修改内容：单独保存 track_predict 地理位置，并用它驱动 V1 的 primary_matches 与锁定暂停语义。
# 修改时间：2026-09-20（新帧消费与门控复位）。
# 修改目的：避免重复快照刷新轨迹，并保证每次协同发起都重新满足连续五帧。
# 修改内容：只为真正的新 frame key 递增提交序号，在感知陈旧及 SEARCH 离开时复位门控。
# 修改时间：2026-09-20。
# 修改目的：为真实感知 V3 提供可独立集成的航线、协同门控和跟踪纠偏控制层。
# 修改内容：新增鸭子类型感知输入、连续五个新帧门控、固定四十八度视场及目标竞争方向控制。
"""V3 控制层：把真实感知结果适配到 PersonalV1 的协同状态机。"""

from dataclasses import dataclass, replace
from typing import Any

from competition.sdk.core.commands import (
    Command, fly_to, point_gimbal,
)
from competition.sdk.core.observation import Detection

from .competition_flight import CompetitionDirectionController
from .coordination import CoopCoordinator
from .gimbal_lock import GimbalLockConfig, GimbalLockController
from .personal_v1 import PersonalV1Agent
from .v3_simple_control import SimpleCoopControl
from .v3_simple_coordination import V3SimpleCoordinator


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


def _is_decoy(value):
    """仅接受真实模型的明确诱饵类别。"""
    name = _field(value, "class_name", None)
    if name is None:
        name = _field(value, "target_type", None)
    return str(name or "").strip().lower() in {"model_prop", "decoy_vehicle"}


@dataclass(frozen=True)
class V3PerceptionInput:
    """控制层输入；前三项可直接保存感知模块的 ``PixelObservation``。"""

    detection: Any = None
    track_predict: Any = None
    closest_others: Any = None
    objects: tuple[Any, ...] = ()
    target_position: tuple[float, float] | None = None
    stationary_position: tuple[float, float] | None = None
    track_predict_position: tuple[float, float] | None = None
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
    def only_decoys(self):
        """要求本帧至少有一个对象，且所有对象都被模型判为诱饵。"""
        return (not self.is_target and bool(self.objects)
                and all(_is_decoy(item) for item in self.objects))

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
    COOP_DURATION_S = 0.0
    TARGET_CONFIRM_FRAMES = 5
    DECOY_ONLY_END_FRAMES = 5
    PERCEPTION_STALE_S = 1.0
    MASTER_ACTIVE_LOST_AFTER_S = 5.0

    def reset(self):
        super().reset()
        self._default_track_lost_after_s = self._track.config.lost_after_s
        # V3 的连续性证据来自五个不同的真实像素帧；异步推理帧之间会有空控制
        # tick，不能再沿用 V1 对每个约十赫兹 tick 连续锁定半秒的假设。
        self._gimbal_lock = GimbalLockController(GimbalLockConfig(
            min_fov_deg=self.SEARCH_FOV_DEG,
            max_fov_deg=self.SEARCH_FOV_DEG,
            preferred_fov_deg=self.SEARCH_FOV_DEG,
            stable_after_s=0.0,
            allow_multiple_targets=True,
            coop_reacquire_timeout_s=self.COOP_REACQUIRE_TIMEOUT_S,
        ))
        self._target_gate = ConsecutiveTargetGate(self.TARGET_CONFIRM_FRAMES)
        self._coordinator = V3SimpleCoordinator(
            self.my_uid, (self.A, self.B, self.C), self.COOP_DURATION_S,
            self.COOP_REACQUIRE_TIMEOUT_S,
            proposal_gate=lambda: self._target_gate.ready,
            master_prediction=lambda now: self._track.predict_position(now),
        )
        self._competition_flight = _V3CompetitionDirectionController(
            self.COMPETITION_UPDATE_PERIOD_S, self.COMPETITION_OFFSET_STEP_MPS,
            self.COMPETITION_MAX_OFFSET_M,
        )
        self._simple_control = SimpleCoopControl()
        self._perception = None
        self._submission_serial = 0
        self._consumed_serial = 0
        self._active_target_position = None
        self._active_track_predict_position = None
        self._active_competitor_position = None
        self._decoy_only_count = 0
        self._runtime_evidence = {}

    def submit_perception(self, snapshot=None, *, detection=None,
                          track_predict=None, closest_others=None, objects=None,
                          target_position=None, stationary_position=None,
                          track_predict_position=None,
                          competitor_position=None,
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
            objects = (objects if objects is not None
                       else _field(snapshot, "objects", ()))
            target_position = (target_position if target_position is not None
                               else _field(snapshot, "target_position", None))
            stationary_position = (
                stationary_position if stationary_position is not None
                else _field(snapshot, "stationary_position", None))
            track_predict_position = (
                track_predict_position if track_predict_position is not None
                else _field(snapshot, "track_predict_position", None))
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

        target_position = _position(target_position) or _position(detection)
        stationary_position = (
            _position(stationary_position)
            or _position(_field(detection, "ground_contact_h0", None))
            or target_position
        )
        track_predict_position = (
            _position(track_predict_position) or _position(track_predict)
        )
        competitor_position = (_position(competitor_position)
                               or _position(closest_others))
        if objects is None:
            objects = ()
        frame = V3PerceptionInput(
            detection=detection,
            track_predict=track_predict,
            closest_others=closest_others,
            objects=tuple(objects),
            target_position=target_position,
            stationary_position=stationary_position,
            track_predict_position=track_predict_position,
            competitor_position=competitor_position,
            frame_id=frame_id,
            source_sim_time=(None if source_sim_time is None else float(source_sim_time)),
            observed_sim_time=(None if observed_sim_time is None else float(observed_sim_time)),
            primary_is_target=primary_is_target,
        )
        next_serial = self._submission_serial + 1
        if not self._target_gate.observe(frame, next_serial):
            return self._perception
        self._submission_serial = next_serial
        self._perception = frame
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

    def signal_coordination_end(self, reason, position=None):
        """把连续诱饵等真实感知结论显式交给简化协调器。"""
        return self._coordinator.signal_end(reason, position)

    @property
    def runtime_evidence(self):
        """返回 Runner 可以低频采样的合法 Agent 内部证据。"""
        return dict(self._runtime_evidence)

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
            "v3_track_predict_position": self._active_track_predict_position,
            "v3_competitor_position": self._active_competitor_position,
            "v3_fov_deg": self.SEARCH_FOV_DEG,
            "v3_flight_alt_m": self.FLIGHT_ALT_M,
            "v3_target_alt_m": self.TARGET_ALT_M,
            "v3_decoy_only_count": self._decoy_only_count,
            "v3_decoy_only_required": self.DECOY_ONLY_END_FRAMES,
            "v3_runtime_evidence": self.runtime_evidence,
            "v3_simple_coordination": self._coordinator.event_summary,
        })
        return summary

    def decide(self, obs, dt):
        score = getattr(getattr(obs, "briefing", None), "score_view", None)
        now = float(score.sim_time) if score is not None else self._t + max(0.0, dt)
        master_tracking = (
            self._coordinator.role == self._coordinator.MASTER
            and self._coordinator.phase in (
                self._coordinator.HOLD, self._coordinator.ACTIVE)
        )
        desired_lost_after_s = (
            self.MASTER_ACTIVE_LOST_AFTER_S
            if master_tracking else self._default_track_lost_after_s
        )
        if self._track.config.lost_after_s != desired_lost_after_s:
            self._track.config = replace(
                self._track.config, lost_after_s=desired_lost_after_s)
        frame = self._perception
        is_new = frame is not None and self._submission_serial != self._consumed_serial
        fresh = frame is not None and self._frame_is_fresh(frame, now)
        finished_decoy_only_count = None
        if (is_new and fresh
                and self._coordinator.role == self._coordinator.MASTER
                and self._coordinator.phase == self._coordinator.ACTIVE):
            self._decoy_only_count = (
                self._decoy_only_count + 1 if frame.only_decoys else 0
            )
            if self._decoy_only_count >= self.DECOY_ONLY_END_FRAMES:
                accepted = self.signal_coordination_end(
                    self._coordinator.FINISH_MASTER_DECOY_ONLY,
                    self._coordinator.follow_position,
                )
                if accepted:
                    finished_decoy_only_count = self._decoy_only_count
        elif self._coordinator.role != self._coordinator.MASTER:
            self._decoy_only_count = 0
        if fresh:
            self._active_target_position = frame.target_position
            self._active_track_predict_position = frame.track_predict_position
            self._active_competitor_position = frame.competitor_position
        else:
            self._active_target_position = None
            self._active_track_predict_position = None
            self._active_competitor_position = None
            if self._coordinator.phase == CoopCoordinator.SEARCH:
                self._target_gate.reset()
        self._competition_flight.set_perception(
            self._active_target_position, self._active_competitor_position)

        if is_new and fresh:
            # 单数 detection 表示本机预测的原生最近对象，而非意图真目标。
            primary = self._sdk_detection(frame, self._active_track_predict_position)
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

        self._coordinator.submit_stationary_observation()
        if is_new and fresh:
            source_time = (frame.source_sim_time if frame.source_sim_time is not None
                           else frame.observed_sim_time)
            source_time = now if source_time is None else source_time
            frame_key = (frame.new_frame_key if frame.new_frame_key is not None
                         else ("submission", self._submission_serial))
            self._coordinator.submit_stationary_observation(
                frame_key=frame_key,
                frame_id=frame.frame_id,
                source_sim_time=source_time,
                track_id=frame.primary_key if frame.is_target else None,
                position_h0=(frame.stationary_position if frame.is_target else None),
            )

        phase_before = self._coordinator.phase
        commands = super().decide(control_obs, dt)
        if ((phase_before == CoopCoordinator.SEARCH)
                != (self._coordinator.phase == CoopCoordinator.SEARCH)):
            # 发起时消费资格，结束回到 SEARCH 时也清掉协同期间积累的旧帧。
            self._target_gate.reset()
            self._decoy_only_count = 0
        coordinator = self._coordinator
        role = coordinator.role
        phase = coordinator.phase
        session = coordinator.current_session
        control_evidence = {
            "guidance_enabled": False,
            "aiming_enabled": False,
            "master_gate_m": self._simple_control.config.master_gate_m,
            "target_gate_m": self._simple_control.config.target_gate_m,
            "rendezvous_ready": False,
        }
        if (role == coordinator.FOLLOWER
                and phase in (coordinator.INIT, coordinator.ACTIVE)):
            guidance = self._simple_control.follower(
                self_position=(obs.self.lat, obs.self.lon),
                self_alt_m=obs.self.alt,
                self_heading_deg=obs.self.heading_deg,
                master_position=coordinator.master_position,
                follow_position=coordinator.follow_position,
                session_key=session,
            )
            control_evidence = guidance.as_evidence()
            if guidance.guidance_enabled:
                commands = [command for command in commands
                            if command.verb != "set_destination"]
                commands.append(fly_to(
                    *guidance.fly_to_position,
                    alt=self.FLIGHT_ALT_M,
                    speed=guidance.fly_to_speed_mps,
                    loiter_radius=guidance.fly_to_loiter_radius_m,
                ))
            if guidance.aiming_enabled:
                commands = [command for command in commands if command.verb
                            != "component.gimbal_tracking.set_orientation"]
                commands.append(point_gimbal(
                    guidance.gimbal_pan_cmd_deg,
                    guidance.gimbal_tilt_cmd_deg,
                ))
        else:
            self._simple_control.reset()
            if (role == coordinator.MASTER
                    and phase in (coordinator.HOLD, coordinator.ACTIVE)):
                aim_position = coordinator.follow_position
                if self._track.state == self._track.COASTING:
                    # 缺测期间沿既有速度外推，避免云台停在最后一次观测位置。
                    aim_position = self._track.predict_position(now)
                aim = self._simple_control.master_aim(
                    self_position=(obs.self.lat, obs.self.lon),
                    self_alt_m=obs.self.alt,
                    self_heading_deg=obs.self.heading_deg,
                    target_position=aim_position,
                )
                if aim is not None:
                    commands = [command for command in commands if command.verb
                                != "component.gimbal_tracking.set_orientation"]
                    commands.append(point_gimbal(aim.pan_deg, aim.tilt_deg))
                    control_evidence.update({
                        "guidance_enabled": True,
                        "aiming_enabled": True,
                        "aim_target_lat": aim.target_position[0],
                        "aim_target_lon": aim.target_position[1],
                        "target_distance_m": aim.ground_distance_m,
                        "gimbal_pan_cmd_deg": aim.pan_deg,
                        "gimbal_tilt_cmd_deg": aim.tilt_deg,
                    })
        fixed = []
        for command in commands:
            if command.verb == "set_destination":
                params = dict(command.params)
                params["altitude"] = self.FLIGHT_ALT_M
                command = Command(command.verb, params)
            elif command.verb == "set_fov":
                command = Command(command.verb, {"angle": self.SEARCH_FOV_DEG})
            fixed.append(command)
        coordination = coordinator.event_summary
        self._runtime_evidence = {
            **coordination,
            **control_evidence,
            "agent_time_s": now,
            "track_state": self._track.state,
            "track_last_seen_age_s": (
                None if self._track.last_seen <= -1e8
                else max(0.0, now - self._track.last_seen)
            ),
            "master_lost_timeout_s": self.MASTER_ACTIVE_LOST_AFTER_S,
            "track_predict_position": self._track.predict_position(now),
            "confirmation": {
                "count": self._target_gate.count,
                "required": self._target_gate.required_frames,
                "ready": self._target_gate.ready,
            },
            "perception": {
                "frame_id": None if frame is None else frame.frame_id,
                "is_new": bool(is_new),
                "fresh": bool(fresh),
                "is_target": bool(frame is not None and frame.is_target),
                "only_decoys": bool(frame is not None and frame.only_decoys),
            },
            "decoy_only_count": (
                self._decoy_only_count if finished_decoy_only_count is None
                else finished_decoy_only_count
            ),
            "decoy_only_required": self.DECOY_ONLY_END_FRAMES,
        }
        return fixed
