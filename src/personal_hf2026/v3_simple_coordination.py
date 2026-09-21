# 修改时间：2026-09-21（静止完成去重）。
# 修改目的：避免已完成的同一停止目标被不同会话反复重捕获并重复增加协同完成数。
# 修改内容：master_static 在已有完成位置空间门内时广播取消并回到 SEARCH，不记录新的完成会话。
# 修改时间：2026-09-21（静止速度实测校准）。
# 修改目的：适配停止车辆接触点仍有投影摆动、二米每秒连续门限在销毁后始终无法触发的问题。
# 修改内容：依据销毁前零低速连续段和销毁后八帧低速段，把鲁棒速度门限校准为四米每秒。
# 修改时间：2026-09-21（MASTER 等待超时退出）。
# 修改目的：避免 HOLD 中已经跟丢超过五秒的过期会话继续等待从机并延后退出。
# 修改内容：MASTER 在 HOLD 与 ACTIVE 的轨迹 LOST 或 epoch 断裂时统一取消会话并回到 SEARCH。
# 修改时间：2026-09-21（静止速度门限收紧）。
# 修改目的：阻止移动目标在中心投影短暂降到约四米每秒时被误判为明显静止。
# 修改内容：静止专用接触点的鲁棒速度门限收紧为二米每秒，窗口和七帧确认保持不变。
# 修改时间：2026-09-21（静止会话接续修复）。
# 修改目的：让视觉轨迹编号变化时按 H=0 连续性接续，并阻止 HOLD 帧提前贡献完成计数。
# 修改内容：以会话作为稳定身份传入视觉编号，只有 ACTIVE 新帧累计七帧静止确认。
# 修改时间：2026-09-21（MASTER 丢失退出）。
# 修改目的：让 V3 主机超过 ACTIVE 跟踪容忍期后直接结束本次协同跟踪。
# 修改内容：主机滑行时广播预测位置，轨迹 LOST 或 epoch 断裂时广播取消并直接返回 SEARCH。
# 修改时间：2026-09-21（V3 鲁棒静止判定）。
# 修改目的：让 V3 在本地轨迹丢失后仍能用新视觉帧的 H=0 原始位置判断目标静止。
# 修改内容：接入 V3 专用鲁棒停止判定、严格帧去重和可审计的完成边沿证据。
# 修改时间：2026-09-21。
# 修改目的：为 V3 提供不依赖双机轨迹匹配和协同计时的专用简化协调流程。
# 修改内容：实现最近空闲从机选择、零高程目标持续广播、无本机识别接受、结束计数同步和兼容摘要接口。
"""V3 专用简化协调器；保留 V1 搜索外壳，不复用 V1 的轨迹配对计时。"""

from dataclasses import dataclass

from competition.baselines.coop_distributed import _haversine_m

from .coordination import CoopCoordinator, CoordinationUpdate
from .tracking import LocalTrackManager, StoppedTargetDetector, TrackMatchResult


@dataclass(frozen=True)
class _SimpleOffer:
    sender_uid: str
    start_tick: int
    track_epoch: int
    position: tuple
    follower_uid: str = None

    @property
    def priority(self):
        return self.start_tick, self.sender_uid

    @property
    def session(self):
        return self.sender_uid, self.start_tick, self.track_epoch


@dataclass
class _SimplePeerStatus:
    phase: str
    position: tuple
    completed_count: int
    received_at: float


class V3SimpleCoordinator(CoopCoordinator):
    """仅交换本机状态和 MASTER 的零高程目标估计。

    ``step`` 与 ``CoopCoordinator`` 保持兼容，便于 PersonalV1 搜索外壳直接调用。
    FOLLOWER 的本地轨迹、主锁定和重捕获状态不会参与会话存活判定。
    """

    MAX_PAYLOAD_BYTES = 50
    POSITION_SCALE = 100_000.0
    MASTER_DISTANCE_M = 220.0
    TARGET_DISTANCE_M = 200.0
    STATIONARY_WINDOW_S = 2.5
    STATIONARY_MIN_SPAN_S = 2.0
    STATIONARY_MIN_POINTS = 5
    STATIONARY_SPEED_MPS = 4.0
    STATIONARY_MOVING_SPEED_MPS = 8.0
    STATIONARY_CONFIRM_FRAMES = 7

    FINISH_MASTER_STATIC = "master_static"
    FINISH_MASTER_DECOY_ONLY = "master_decoy_only"
    _FINISH_CODE = {
        FINISH_MASTER_STATIC: "S",
        FINISH_MASTER_DECOY_ONLY: "D",
    }
    _CODE_FINISH = {code: reason for reason, code in _FINISH_CODE.items()}

    def __init__(self, my_uid, member_uids, duration_s=22.0,
                 reacquire_timeout_s=5.0, evidence_wait_timeout_s=2.0,
                 *, proposal_gate=None, master_prediction=None):
        super().__init__(
            my_uid, member_uids, duration_s,
            reacquire_timeout_s, evidence_wait_timeout_s,
        )
        # V1 继续使用基类默认判定；只有 V3 简化协调器消费原始视觉 H=0 点。
        self.stopped_target = StoppedTargetDetector(
            required_points=self.STATIONARY_CONFIRM_FRAMES,
            moving_speed_mps=self.STATIONARY_MOVING_SPEED_MPS,
            robust_window_s=self.STATIONARY_WINDOW_S,
            robust_min_span_s=self.STATIONARY_MIN_SPAN_S,
            robust_min_points=self.STATIONARY_MIN_POINTS,
            stationary_speed_mps=self.STATIONARY_SPEED_MPS,
        )
        self._proposal_gate = proposal_gate or (lambda: True)
        self._master_prediction = master_prediction or (lambda now: None)
        self.master_position = None
        self.partner_position = None
        self.target_distance_m = None
        self.partner_distance_m = None
        self.rendezvous_ready = False
        self.stage_reason = "search"
        self.finish_reason = None
        self.last_completed_position = None
        self.revision = 0
        self.proposal_count = 0
        self.follower_accept_count = 0
        self.proposal_revision = None
        self.follower_accept_revision = None
        self.last_transition = {
            "revision": 0,
            "event": "reset",
            "reason": "search",
            "session": None,
            "partner_uid": None,
            "at_s": None,
        }
        self._pending_finish = None
        self._completion_records = {}
        self._announced_completed_count = 0
        self._last_accept_send = -1e9
        self._last_grant_send = -1e9
        self._pending_stationary_observation = None
        self._last_stationary_evidence = dict(self.stopped_target.evidence)
        self.last_match = TrackMatchResult(False, "v3_simple_no_track_matching")

    @property
    def completed_count(self):
        return max(
            len(self.completed_sessions),
            getattr(self, "_announced_completed_count", 0),
        )

    @property
    def event_summary(self):
        """供运行记录器按 ``revision`` 差分的稳定结构化快照。"""
        return {
            "revision": self.revision,
            "event": self.last_transition["event"],
            "phase": self.phase,
            "role": self.role,
            "session": self.current_session,
            "partner_uid": self.partner_uid,
            "proposal_count": self.proposal_count,
            "proposal_revision": self.proposal_revision,
            "follower_accept_count": self.follower_accept_count,
            "follower_accept_revision": self.follower_accept_revision,
            "finish_reason": self.finish_reason,
            "completed_count": self.completed_count,
            "completed_sessions": tuple(sorted(self.completed_sessions)),
            "stage_reason": self.stage_reason,
            "master_position": self.master_position,
            "follow_position": self.follow_position,
            "partner_distance_m": self.partner_distance_m,
            "target_distance_m": self.target_distance_m,
            "rendezvous_ready": self.rendezvous_ready,
            "uses_track_matching": False,
            "uses_coop_timer": False,
            "stationary": dict(self._last_stationary_evidence),
            "last_transition": dict(self.last_transition),
        }

    def _record_transition(self, event, now, reason, *, session=None,
                           partner_uid=None):
        self.revision += 1
        self.last_transition = {
            "revision": self.revision,
            "event": event,
            "reason": reason,
            "session": self.current_session if session is None else session,
            "partner_uid": (self.partner_uid if partner_uid is None
                            else partner_uid),
            "at_s": now,
        }

    @staticmethod
    def _base36(number):
        number = int(number)
        sign = "-" if number < 0 else ""
        number = abs(number)
        digits = "0123456789abcdefghijklmnopqrstuvwxyz"
        if number == 0:
            return "0"
        encoded = ""
        while number:
            number, remainder = divmod(number, 36)
            encoded = digits[remainder] + encoded
        return sign + encoded

    @staticmethod
    def _from_base36(value):
        return int(value, 36)

    @classmethod
    def _encode_position(cls, position):
        return (
            cls._base36(round(float(position[0]) * cls.POSITION_SCALE)),
            cls._base36(round(float(position[1]) * cls.POSITION_SCALE)),
        )

    @classmethod
    def _decode_position(cls, lat_code, lon_code):
        return (
            cls._from_base36(lat_code) / cls.POSITION_SCALE,
            cls._from_base36(lon_code) / cls.POSITION_SCALE,
        )

    @classmethod
    def _bounded(cls, payload):
        if len(payload.encode("utf-8")) > cls.MAX_PAYLOAD_BYTES:
            raise ValueError("V3 简化协同广播超过 50 字节")
        return payload

    def _heartbeat_payload(self, own_position):
        lat, lon = self._encode_position(own_position)
        return self._bounded(
            f"H,{self._PHASE_CODE[self.phase]},{lat},{lon},"
            f"{self._base36(self.completed_count)}"
        )

    def _offer_payload(self):
        if self.follow_position is None or self.selected_follower_uid is None:
            return None
        lat, lon = self._encode_position(self.follow_position)
        return self._bounded(
            f"O,{self._uid_code(self.master_uid)},"
            f"{self._base36(self.session_start_tick)},"
            f"{self._base36(self.master_track_epoch)},{lat},{lon},"
            f"{self._uid_code(self.selected_follower_uid)}"
        )

    def _target_payload(self, own_position):
        if self.follow_position is None or self.partner_uid is None:
            return None
        target_lat, target_lon = self._encode_position(self.follow_position)
        master_lat, master_lon = self._encode_position(own_position)
        return self._bounded(
            f"T,{self._uid_code(self.master_uid)},"
            f"{self._base36(self.session_start_tick)},"
            f"{self._base36(self.master_track_epoch)},"
            f"{target_lat},{target_lon},{master_lat},{master_lon},"
            f"{self._uid_code(self.partner_uid)}"
        )

    def _session_payload(self, kind):
        payload = (
            f"{kind},{self._uid_code(self.master_uid)},"
            f"{self._base36(self.session_start_tick)},"
            f"{self._base36(self.master_track_epoch)}"
        )
        if kind == "G":
            payload += f",{self._uid_code(self.partner_uid)}"
        return self._bounded(payload)

    def _completion_payload(self, session):
        reason, position = self._completion_records[session]
        lat, lon = self._encode_position(position)
        return self._bounded(
            f"K,{self._uid_code(session[0])},{self._base36(session[1])},"
            f"{self._base36(session[2])},{self._FINISH_CODE[reason]},"
            f"{lat},{lon},{self._base36(self.completed_count)}"
        )

    def _parse(self, message):
        fields = str(message.payload).split(",")
        try:
            kind = fields[0]
            if kind == "H" and len(fields) == 5:
                return kind, (
                    self._CODE_PHASE[fields[1]],
                    self._decode_position(fields[2], fields[3]),
                    self._from_base36(fields[4]),
                )
            if kind in ("O", "T") and len(fields) in (7, 9):
                session = (
                    self._code_uid(fields[1]),
                    self._from_base36(fields[2]),
                    self._from_base36(fields[3]),
                )
                target = self._decode_position(fields[4], fields[5])
                if kind == "O":
                    follower_uid = self._code_uid(fields[6])
                    return kind, _SimpleOffer(
                        session[0], session[1], session[2], target, follower_uid)
                master_position = self._decode_position(fields[6], fields[7])
                follower_uid = self._code_uid(fields[8])
                return kind, (session, target, master_position, follower_uid)
            if kind in ("A", "C") and len(fields) == 4:
                return kind, (
                    self._code_uid(fields[1]),
                    self._from_base36(fields[2]),
                    self._from_base36(fields[3]),
                )
            if kind == "G" and len(fields) == 5:
                session = (
                    self._code_uid(fields[1]),
                    self._from_base36(fields[2]),
                    self._from_base36(fields[3]),
                )
                return kind, (session, self._code_uid(fields[4]))
            if kind == "K" and len(fields) == 8:
                session = (
                    self._code_uid(fields[1]),
                    self._from_base36(fields[2]),
                    self._from_base36(fields[3]),
                )
                return kind, (
                    session,
                    self._CODE_FINISH[fields[4]],
                    self._decode_position(fields[5], fields[6]),
                    self._from_base36(fields[7]),
                )
        except (IndexError, KeyError, TypeError, ValueError):
            return None
        return None

    def signal_end(self, reason, position=None):
        """由真实感知显式通知 MASTER 连续只见诱饵；静止结束会在内部判定。"""
        if reason not in self._FINISH_CODE:
            raise ValueError(f"不支持的协同结束原因: {reason}")
        if self.role != self.MASTER or self.phase != self.ACTIVE:
            return False
        self._pending_finish = (reason, position)
        return True

    request_end = signal_end

    def submit_stationary_observation(self, *, frame_key=None, frame_id=None,
                                      source_sim_time=None, track_id=None,
                                      position_h0=None):
        """保存本控制 tick 的 V3 新帧；``step`` 只会消费一次。"""
        if frame_key is None:
            self._pending_stationary_observation = None
            return
        self._pending_stationary_observation = {
            "frame_key": frame_key,
            "frame_id": frame_id,
            "source_sim_time": source_sim_time,
            "track_id": track_id,
            "position_h0": position_h0,
        }

    def _record_completion(self, session, reason, position, announced_count=None):
        if session[0] is None or position is None:
            return False
        added = session not in self.completed_sessions
        self.completed_sessions.add(session)
        self._completion_records.setdefault(session, (reason, position))
        self._kill_events[session] = position
        if all(_haversine_m(*position, *old) >= self.COMPLETED_GATE_M
               for old in self.completed_positions):
            self.completed_positions.append(position)
        if announced_count is not None:
            self._announced_completed_count = max(
                self._announced_completed_count, int(announced_count))
        self._announced_completed_count = max(
            self._announced_completed_count, len(self.completed_sessions))
        self.finish_reason = reason
        self.last_completed_position = position
        return added

    def _return_to_search(self, now, event, reason, *, count_reset=False):
        session = self.current_session
        partner_uid = self.partner_uid
        super()._to_search(count_reset)
        self.master_position = None
        self.partner_position = None
        self.partner_distance_m = None
        self.target_distance_m = None
        self.rendezvous_ready = False
        self.stage_reason = "search"
        self._pending_finish = None
        self._pending_stationary_observation = None
        self._last_stationary_evidence = dict(self.stopped_target.evidence)
        self._record_transition(
            event, now, reason, session=session, partner_uid=partner_uid)

    def _complete_current(self, now, reason, position, payloads):
        session = self.current_session
        partner_uid = self.partner_uid
        stationary_evidence = dict(self.stopped_target.evidence)
        position = position or self.follow_position
        if session is None or position is None:
            return False
        duplicate_static = (
            reason == self.FINISH_MASTER_STATIC
            and any(_haversine_m(*position, *old) < self.COMPLETED_GATE_M
                    for old in self.completed_positions)
        )
        if duplicate_static:
            payloads.append(self._session_payload("C"))
            self.cancelled_sessions.add(session)
            self._return_to_search(
                now, "session_cancelled", "duplicate_completed_position")
            self._last_stationary_evidence = stationary_evidence
            return True
        self._record_completion(session, reason, position)
        payloads.append(self._completion_payload(session))
        super()._to_search(False)
        self.master_position = None
        self.partner_position = None
        self.partner_distance_m = None
        self.target_distance_m = None
        self.rendezvous_ready = False
        self.stage_reason = f"completed_{reason}"
        self._pending_finish = None
        self._pending_stationary_observation = None
        # ``_to_search`` 会重置判定器，完成边沿必须继续保留触发帧快照。
        self._last_stationary_evidence = stationary_evidence
        self._record_transition(
            "coordination_finished", now, reason,
            session=session, partner_uid=partner_uid,
        )
        return True

    def _start_master(self, now, local, own_position):
        super()._reset_timer(False)
        self._last_stationary_evidence = dict(self.stopped_target.evidence)
        self.phase = self.HOLD
        self.role = self.MASTER
        self.session_start_tick = self._tick(now)
        self.master_uid = self.my_uid
        self.master_track_epoch = local.epoch
        self.partner_uid = None
        self.selected_follower_uid = None
        self.selection_started_at = None
        self.session_started_at = now
        self.follow_position = local.position
        self.master_position = own_position
        self.proposal = _SimpleOffer(
            self.my_uid, self.session_start_tick,
            self.master_track_epoch, local.position,
        )
        self.proposal_created_at = now
        self.master_sessions_started += 1
        self.proposal_count += 1
        self.stage_reason = "waiting_follower"
        self._record_transition("proposal_started", now, "five_target_frames")
        self.proposal_revision = self.revision

    def _accept_offer(self, offer, now):
        super()._reset_timer(False)
        self._last_stationary_evidence = dict(self.stopped_target.evidence)
        self.phase = self.INIT
        self.role = self.FOLLOWER
        self.proposal = None
        self.proposal_created_at = None
        self.session_start_tick = offer.start_tick
        self.master_uid = offer.sender_uid
        self.master_track_epoch = offer.track_epoch
        self.partner_uid = offer.sender_uid
        self.selected_follower_uid = self.my_uid
        self.follow_position = offer.position
        status = self.peer_status.get(offer.sender_uid)
        self.master_position = None if status is None else status.position
        self.session_started_at = now
        self.last_master_message = now
        self.follower_sessions_started += 1
        self.follower_accept_count += 1
        self.stage_reason = "follower_accept_sent"
        self._record_transition("follower_accepted", now, "selected_by_nearest_master")
        self.follower_accept_revision = self.revision

    def _select_follower(self, now, own_position, *, exclude_uid=None):
        candidates = []
        for uid in self.peer_uids:
            if uid == exclude_uid:
                continue
            status = self.peer_status.get(uid)
            if (status is None or now - status.received_at > self.STATUS_TIMEOUT_S
                    or status.phase != self.SEARCH):
                continue
            candidates.append((_haversine_m(*own_position, *status.position), uid))
        return min(candidates, default=(None, None))[1]

    def _refresh_distances(self, own_position):
        self.target_distance_m = (
            None if self.follow_position is None
            else _haversine_m(*own_position, *self.follow_position)
        )
        if self.role == self.MASTER:
            self.master_position = own_position
            status = self.peer_status.get(self.partner_uid)
            self.partner_position = None if status is None else status.position
        elif self.role == self.FOLLOWER:
            self.partner_position = self.master_position
        else:
            self.partner_position = None
        self.partner_distance_m = (
            None if self.partner_position is None
            else _haversine_m(*own_position, *self.partner_position)
        )
        self.rendezvous_ready = bool(
            self.role == self.FOLLOWER
            and self.phase in (self.INIT, self.ACTIVE)
            and self.partner_distance_m is not None
            and self.target_distance_m is not None
            and self.partner_distance_m <= self.MASTER_DISTANCE_M
            and self.target_distance_m <= self.TARGET_DISTANCE_M
        )
        if self.phase == self.SEARCH:
            self.stage_reason = "search"
        elif self.role == self.MASTER and self.phase == self.HOLD:
            self.stage_reason = (
                "invite_sent" if self.selected_follower_uid is not None
                else "waiting_follower")
        elif self.role == self.MASTER and self.phase == self.ACTIVE:
            self.stage_reason = "master_tracking"
        elif self.role == self.FOLLOWER:
            if self.master_position is None or self.follow_position is None:
                self.stage_reason = "follower_wait_geometry"
            elif self.partner_distance_m > self.MASTER_DISTANCE_M:
                self.stage_reason = "follower_approach_master"
            elif self.target_distance_m > self.TARGET_DISTANCE_M:
                self.stage_reason = "follower_approach_target"
            else:
                self.stage_reason = "follower_aim_target"

    def should_suppress(self, position, now):
        """零高程估计误差较大，不再用旧空间门限压制搜索候选。"""
        del position, now
        return False

    def _append_completion_gossip(self, now, payloads):
        if (not self.completed_sessions
                or now - self.last_kill_send < self.CHECK_PERIOD_S):
            return
        sessions = sorted(self.completed_sessions, key=lambda item: (item[1], item[0]))
        available = [session for session in sessions
                     if session in self._completion_records]
        if not available:
            return
        session = available[self._kill_index % len(available)]
        self._kill_index += 1
        payloads.append(self._completion_payload(session))
        self.last_kill_send = now

    def step(self, now, local, inbox, can_propose=True, own_position=None,
             target_count=3, local_lock_valid=True, local_lock_expired=False):
        """推进简化会话；FOLLOWER 的 ``local`` 状态不会影响接受或存活。"""
        del local_lock_expired
        own_position = own_position or (0.0, 0.0)
        self._target_count = max(1, int(target_count or 3))
        payloads = []
        reset_local = False
        invitations = []

        master_tracking_position = local.position
        master_coasting = (
            self.role == self.MASTER
            and self.phase == self.ACTIVE
            and local.state == LocalTrackManager.COASTING
        )
        if master_coasting:
            master_tracking_position = (
                self._master_prediction(now) or master_tracking_position)
        if (self.role == self.MASTER
                and self.phase in (self.HOLD, self.ACTIVE)
                and (local_lock_valid or master_coasting)
                and local.state in (LocalTrackManager.CONFIRMED,
                                    LocalTrackManager.COASTING)
                and master_tracking_position is not None):
            self.follow_position = master_tracking_position
            if self.proposal is not None:
                self.proposal = _SimpleOffer(
                    self.my_uid, self.session_start_tick,
                    self.master_track_epoch, master_tracking_position,
                    self.selected_follower_uid,
                )

        for message in inbox:
            sender_uid = str(message.sender_uid)
            if sender_uid not in self.peer_uids:
                continue
            parsed = self._parse(message)
            if parsed is None:
                continue
            kind, data = parsed
            if kind == "H":
                phase, position, completed_count = data
                self.peer_status[sender_uid] = _SimplePeerStatus(
                    phase, position, completed_count, now)
                self._announced_completed_count = max(
                    self._announced_completed_count, completed_count)
                if self.role == self.FOLLOWER and sender_uid == self.master_uid:
                    self.master_position = position
                    self.last_master_message = now
            elif kind == "O":
                offer = data
                if (offer.sender_uid != sender_uid
                        or offer.session in self.cancelled_sessions
                        or offer.session in self.completed_sessions):
                    continue
                latest_tick = self.latest_offer_ticks.get(sender_uid, -1)
                if offer.start_tick < latest_tick:
                    continue
                self.latest_offer_ticks[sender_uid] = offer.start_tick
                self.offers[sender_uid] = (offer, now)
                if (self.role == self.FOLLOWER
                        and self._session_matches(offer.session)):
                    if offer.follower_uid == self.my_uid:
                        self.follow_position = offer.position
                        self.last_master_message = now
                    else:
                        self._return_to_search(
                            now, "session_released", "different_follower")
                        reset_local = True
                elif offer.follower_uid == self.my_uid:
                    invitations.append(offer)
            elif kind == "A":
                if (self.role == self.MASTER and self.phase == self.HOLD
                        and self._session_matches(data)
                        and sender_uid == self.selected_follower_uid):
                    self.phase = self.ACTIVE
                    self.partner_uid = sender_uid
                    self.follower_accept_count += 1
                    self.stage_reason = "master_tracking"
                    self._record_transition(
                        "follower_accept_received", now, "selected_follower_ack")
                    self.follower_accept_revision = self.revision
                    payloads.append(self._session_payload("G"))
                    self._last_grant_send = now
            elif kind == "G":
                session, follower_uid = data
                if (self.role == self.FOLLOWER and self._session_matches(session)
                        and sender_uid == self.master_uid
                        and follower_uid == self.my_uid):
                    if self.phase != self.ACTIVE:
                        self.phase = self.ACTIVE
                        self._record_transition(
                            "follower_activated", now, "master_grant")
                    self.last_master_message = now
            elif kind == "T":
                session, target, master_position, follower_uid = data
                if (session in self.cancelled_sessions
                        or session in self.completed_sessions
                        or sender_uid != session[0]):
                    continue
                if (self.role == self.FOLLOWER and self._session_matches(session)
                        and sender_uid == self.master_uid):
                    if follower_uid != self.my_uid:
                        self.cancelled_sessions.add(session)
                        self._return_to_search(
                            now, "session_released", "different_follower")
                        reset_local = True
                    else:
                        self.follow_position = target
                        self.master_position = master_position
                        self.last_master_message = now
                        if self.phase != self.ACTIVE:
                            self.phase = self.ACTIVE
                            self._record_transition(
                                "follower_activated", now, "master_target_stream")
            elif kind == "C":
                self.cancelled_sessions.add(data)
                if self._session_matches(data):
                    self._return_to_search(now, "session_cancelled", "master_cancel")
                    reset_local = True
            elif kind == "K":
                session, reason, position, completed_count = data
                added = self._record_completion(
                    session, reason, position, completed_count)
                if self._session_matches(session):
                    self._return_to_search(
                        now, "coordination_finished_remote", reason,
                        count_reset=False)
                    reset_local = True
                elif added:
                    self._record_transition(
                        "completion_gossip_received", now, reason,
                        session=session, partner_uid=sender_uid)

        if invitations and self.phase in (self.SEARCH, self.HOLD):
            offer = min(invitations, key=lambda item: item.priority)
            own_wins = (
                self.phase == self.HOLD and self.proposal is not None
                and self.proposal.priority <= offer.priority
            )
            if not own_wins:
                if self.phase == self.HOLD and self.current_session is not None:
                    self.cancelled_sessions.add(self.current_session)
                    payloads.append(self._session_payload("C"))
                self._accept_offer(offer, now)
                payloads.append(self._session_payload("A"))
                self._last_accept_send = now
                reset_local = True

        if (self.phase == self.SEARCH and can_propose
                and bool(self._proposal_gate())
                and local.state == LocalTrackManager.CONFIRMED
                and local.position is not None):
            self._start_master(now, local, own_position)

        if self.role == self.MASTER and self.phase == self.HOLD:
            selected_status = self.peer_status.get(self.selected_follower_uid)
            selection_expired = (
                self.selected_follower_uid is not None
                and self.selection_started_at is not None
                and now - self.selection_started_at >= self.SELECTION_TIMEOUT_S
            )
            selected_stale = (
                self.selected_follower_uid is not None
                and (selected_status is None
                     or now - selected_status.received_at > self.STATUS_TIMEOUT_S)
            )
            if (self.selected_follower_uid is None
                    or selection_expired or selected_stale):
                previous_uid = self.selected_follower_uid
                follower_uid = self._select_follower(
                    now, own_position,
                    exclude_uid=previous_uid if selection_expired else None,
                )
                if follower_uid is None and selection_expired:
                    follower_uid = self._select_follower(now, own_position)
                self.selected_follower_uid = follower_uid
                self.partner_uid = follower_uid
                self.selection_started_at = now if follower_uid is not None else None

        if self.role == self.FOLLOWER and self.phase in (self.INIT, self.ACTIVE):
            if now - self.last_master_message > self.FOLLOWER_TIMEOUT_S:
                session = self.current_session
                payloads.append(self._session_payload("C"))
                self.cancelled_sessions.add(session)
                self._return_to_search(
                    now, "session_timeout", "master_stream_timeout")
                reset_local = True

        if (self.role == self.MASTER and self.phase in (self.HOLD, self.ACTIVE)
                and (local.state == LocalTrackManager.LOST
                     or local.epoch != self.master_track_epoch)):
            session = self.current_session
            if session is not None:
                payloads.append(self._session_payload("C"))
                self.cancelled_sessions.add(session)
            self._return_to_search(
                now, "session_cancelled", "master_track_timeout")
            reset_local = True

        stationary_observation = self._pending_stationary_observation
        self._pending_stationary_observation = None
        stop_ready = False
        if stationary_observation is not None:
            stationary_valid = (
                self.role == self.MASTER
                and self.phase in (self.HOLD, self.ACTIVE)
                and self.current_session is not None
            )
            stop_ready = self.stopped_target.update_observation(
                self.current_session,
                stationary_observation["position_h0"],
                frame_key=stationary_observation["frame_key"],
                frame_id=stationary_observation["frame_id"],
                source_sim_time=stationary_observation["source_sim_time"],
                valid=stationary_valid,
                track_id=stationary_observation["track_id"],
                confirm=self.phase == self.ACTIVE,
            )
            if stationary_valid:
                self._last_stationary_evidence = dict(self.stopped_target.evidence)
        if self.role == self.MASTER and self.phase == self.ACTIVE:
            if self._pending_finish is not None:
                reason, position = self._pending_finish
                if self._complete_current(now, reason, position, payloads):
                    reset_local = True
            elif stop_ready:
                if self._complete_current(
                        now, self.FINISH_MASTER_STATIC,
                        self._last_stationary_evidence.get("position_h0"),
                        payloads):
                    reset_local = True

        self._refresh_distances(own_position)

        if now - self.last_heartbeat_send >= self.SEND_PERIOD_S:
            payloads.append(self._heartbeat_payload(own_position))
            self.last_heartbeat_send = now

        if now - self.last_operational_send >= self.SEND_PERIOD_S:
            operational = None
            if self.role == self.MASTER and self.phase == self.HOLD:
                operational = self._offer_payload()
            elif self.role == self.MASTER and self.phase == self.ACTIVE:
                operational = self._target_payload(own_position)
            elif self.role == self.FOLLOWER and self.phase == self.INIT:
                operational = self._session_payload("A")
            if operational is not None:
                payloads.append(operational)
                self.last_operational_send = now

        self._append_completion_gossip(now, payloads)
        return CoordinationUpdate(tuple(dict.fromkeys(payloads)), reset_local, False)


__all__ = ["V3SimpleCoordinator"]
