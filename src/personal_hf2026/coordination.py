# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13 14:43
# 修改目的：避免重复收件箱内尚未指定伙伴的旧邀请释放已选中的从机。
# 修改内容：邀请释放仅由更新轮次触发，同轮伙伴身份由主机正式广播确认。
# 修改时间：2026-09-13 14:28
# 修改目的：防止邀请改选遗留第二架从机，并让丢失开始广播的伙伴恢复阶段同步。
# 修改内容：改选使用新会话并释放旧邀请，轨迹广播持续携带正式伙伴和协同阶段。
# 修改时间：2026-09-13
# 修改目的：补救官方已消灭并冻结目标而本地协同时间尚未达到阈值的漏计数。
# 修改内容：主机在正常协同下识别连续三次静止观测，并复用原有完成广播与去重流程。
# 修改时间：2026-09-13
# 修改目的：避免临时缺少轨迹证据直接拆散协同并在恢复时补算空档。
# 修改内容：证据不足时暂停并继续检查，超时仅清累计，身份变化仍立即取消。
# 修改时间：2026-09-13
# 修改目的：主锁定失配时继续提供参考轨迹，同时保持协同计时无效。
# 修改内容：分离消息坐标与锁定标志，提供参考轨迹视图，并禁止重置帧发送旧坐标。
# 修改时间：2026-09-13
# 修改目的：避免重捕获请求生效的同一帧继续上报旧轨迹有效。
# 修改内容：本地轨迹将被重置时，将该帧 T 消息的有效标志置为零。
# 修改时间：2026-09-13
# 修改目的：为从机初始捕获提供当前会话主机的参考轨迹。
# 修改内容：从机将已有 T 消息写入轨迹缓冲，并校验发送方的轨迹 epoch。
# 修改时间：2026-09-13
# 修改目的：避免重复重捕获消息反复清空从机轨迹。
# 修改内容：按发送者、会话和接收时间去重 R，并在重置帧跳过旧轨迹绑定。
# 修改时间：2026-09-12
# 修改目的：防止协同阶段单帧主锁定波动立即拆散双机并误计波动时间。
# 修改内容：主从均容忍零点五秒锁定缺失，期间广播无效轨迹并暂停主机协同计时。
"""三机分布式选主、协同跟踪、重复目标避让和任务进度同步。"""

from collections import deque
from dataclasses import dataclass
import math

from competition.baselines.coop_distributed import _haversine_m

from .tracking import (
    LocalTrackManager, TrackMatchResult, TrackPoint, TrackSnapshot,
    compare_track_windows, estimate_track_velocity,
    MatchConfig, StoppedTargetDetector,
)


@dataclass(frozen=True)
class CoordinationUpdate:
    payloads: tuple = ()
    reset_local_track: bool = False
    begin_evade: bool = False


@dataclass(frozen=True)
class _Offer:
    sender_uid: str
    start_tick: int
    track_epoch: int
    seen: float
    position: tuple
    follower_uid: str = None

    @property
    def priority(self):
        return self.start_tick, self.sender_uid

    @property
    def session(self):
        return self.sender_uid, self.start_tick, self.track_epoch


@dataclass
class _PeerStatus:
    phase: str
    position: tuple
    received_at: float


class _PeerTrackBuffer:
    def __init__(self):
        self.epoch = 0
        self.state = LocalTrackManager.LOST
        self.points = deque()
        self.last_seen = -1e9
        self.last_received = -1e9
        self.invalid_since = None

    def clear(self):
        self.epoch = 0
        self.state = LocalTrackManager.LOST
        self.points.clear()
        self.last_seen = -1e9
        self.last_received = -1e9
        self.invalid_since = None

    def ingest(self, epoch, seen, valid, position, now):
        if epoch != self.epoch:
            self.epoch = epoch
            self.points.clear()
        self.last_received = now
        if not valid or position is None:
            if self.invalid_since is None:
                self.invalid_since = now
        else:
            self.invalid_since = None
        if position is None:
            self.state = LocalTrackManager.LOST
            return
        self.state = LocalTrackManager.CONFIRMED
        self.last_seen = seen
        if not self.points or seen > self.points[-1].t:
            self.points.append(TrackPoint(seen, *position))
        while self.points and now - self.points[0].t > 4.0:
            self.points.popleft()

    def snapshot(self, now, *, require_lock=True):
        points = tuple(p for p in self.points if now - p.t <= 3.0)
        # 默认视图用于协同计时；初始关联可单独读取未获主锁定的新鲜轨迹。
        state = (LocalTrackManager.LOST
                 if require_lock and self.invalid_since is not None else self.state)
        return TrackSnapshot(self.epoch, state, points, self.last_seen)


def _offset_m(origin, point):
    lat0, lon0 = origin
    lat, lon = point
    north = (lat - lat0) * 111320.0
    east = (lon - lon0) * 111320.0 * math.cos(math.radians((lat + lat0) * 0.5))
    return east, north


def _move_m(origin, east, north):
    lat, lon = origin
    new_lat = lat + north / 111320.0
    scale = 111320.0 * math.cos(math.radians(lat))
    return new_lat, lon + east / max(scale, 1.0)


def horizontal_perpendicular_destination(first, second, own, distance_m=100.0):
    """从当前位置沿两架协同机连线的外侧水平法向移动。"""
    ax, ay = _offset_m(own, first)
    bx, by = _offset_m(own, second)
    vx, vy = bx - ax, by - ay
    length = math.hypot(vx, vy)
    if length < 1.0:
        # 两机几乎重合时连线方向不稳定，使用固定水平方向退避。
        return _move_m(own, distance_m, 0.0)
    nx, ny = -vy / length, vx / length
    mid_x, mid_y = (ax + bx) * 0.5, (ay + by) * 0.5
    if (-mid_x) * nx + (-mid_y) * ny < 0.0:
        nx, ny = -nx, -ny
    return _move_m(own, nx * distance_m, ny * distance_m)


class CoopCoordinator:
    """每架无人机各持有一个实例，仅通过广播形成同一份任务状态。"""

    SEARCH = "SEARCH"
    HOLD = "HOLD_TARGET"
    INIT = "COOP_INIT"
    ACTIVE = "COOP_ACTIVE"
    EVADE = "DUPLICATE_EVADE"
    DONE = "MISSION_DONE"

    # 保留旧名称，避免实验摘要脚本失效。
    PROPOSING = HOLD

    NONE = "NONE"
    MASTER = "MASTER"
    FOLLOWER = "FOLLOWER"

    SEND_PERIOD_S = 0.5
    CHECK_PERIOD_S = 1.0
    STATUS_TIMEOUT_S = 1.5
    # 最远集结可能跨越数公里；轨迹异常仍立即取消，只有正常赶路允许更长时间。
    SESSION_TIMEOUT_S = 120.0
    ELECTION_DELAY_S = 0.6
    SELECTION_TIMEOUT_S = 5.0
    FOLLOWER_TIMEOUT_S = 5.0
    EVADE_TIMEOUT_S = 6.0
    EVADE_ARRIVAL_M = 15.0
    DUPLICATE_GATE_M = 25.0
    COMPLETED_GATE_M = 30.0

    _PHASE_CODE = {
        SEARCH: "S", HOLD: "H", INIT: "I", ACTIVE: "A", EVADE: "E", DONE: "D",
    }
    _CODE_PHASE = {code: phase for phase, code in _PHASE_CODE.items()}

    def __init__(self, my_uid, member_uids, duration_s=22.0,
                 reacquire_timeout_s=5.0, evidence_wait_timeout_s=2.0):
        self.my_uid = str(my_uid)
        self.member_uids = tuple(str(uid) for uid in member_uids)
        self.peer_uids = tuple(uid for uid in self.member_uids if uid != self.my_uid)
        self.duration_s = duration_s
        self.reacquire_timeout_s = reacquire_timeout_s
        self.evidence_wait_timeout_s = evidence_wait_timeout_s
        self.evidence_wait_since = None
        self.stopped_target = StoppedTargetDetector()
        self.phase = self.SEARCH
        self.role = self.NONE
        self.proposal = None
        self.proposal_created_at = None
        self.selected_follower_uid = None
        self.selection_started_at = None
        self.session_start_tick = None
        self.master_uid = None
        self.master_track_epoch = None
        self.partner_uid = None
        self.follow_position = None
        self.peer_track = _PeerTrackBuffer()
        self.bound_follower_epoch = None
        self.bound_local_epoch = None
        self.coop_seconds = 0.0
        self.peak_seconds = 0.0
        self.resets = 0
        self.last_compare = -1e9
        self.last_success = None
        self.timer_pause_started = None
        self.timer_paused_duration = 0.0
        self.last_evidence = None
        self.last_operational_send = -1e9
        self.last_operational_valid = None
        self.last_heartbeat_send = -1e9
        self.last_kill_send = -1e9
        self.last_master_message = -1e9
        self.master_track_received = False
        self.session_started_at = None
        self.done_at = None
        self.last_match = TrackMatchResult(False, "not_started")
        self.peer_status = {}
        self.offers = {}
        self.latest_offer_ticks = {}
        self.cancelled_sessions = set()
        self._handled_reacquire_messages = set()
        self.completed_sessions = set()
        self.completed_positions = []
        self.master_sessions_started = 0
        self.follower_sessions_started = 0
        self.evade_count = 0
        self._kill_events = {}
        self._kill_index = 0
        self.observed_session = None
        self.observed_members = set()
        self.observed_master_track = _PeerTrackBuffer()
        self.observed_last = -1e9
        self.duplicate_session = None
        self.evade_destination = None
        self.evade_started_at = None
        self._target_count = 3

    @property
    def finished(self):
        return self.phase == self.DONE

    @property
    def completed_count(self):
        return len(self.completed_sessions)

    @property
    def current_session(self):
        if self.master_uid is None:
            return None
        return self.master_uid, self.session_start_tick, self.master_track_epoch

    @staticmethod
    def _tick(t):
        return int(round(t * 10.0))

    def _uid_code(self, uid):
        try:
            return str(self.member_uids.index(str(uid)) + 1)
        except ValueError:
            return "0"

    def _code_uid(self, code):
        index = int(code) - 1
        return self.member_uids[index] if 0 <= index < len(self.member_uids) else None

    def _session_matches(self, session):
        return self.current_session == session

    def _reset_timer(self, count_reset=True):
        self.stopped_target.reset()
        if count_reset and self.coop_seconds > 0.0:
            self.resets += 1
        self.coop_seconds = 0.0
        self.last_success = None
        self.bound_follower_epoch = None
        self.timer_pause_started = None
        self.timer_paused_duration = 0.0
        self.evidence_wait_since = None

    def _set_timer_paused(self, now, paused):
        """记录暂停区间，使恢复后的累计时间排除锁定或证据缺失时长。"""
        if paused and self.timer_pause_started is None:
            self.timer_pause_started = now
        elif not paused and self.timer_pause_started is not None:
            self.timer_paused_duration += max(0.0, now - self.timer_pause_started)
            self.timer_pause_started = None

    def _expire_evidence_wait(self, now):
        """证据空档达到上限只清计时，不解除身份绑定或清空轨迹。"""
        if (self.evidence_wait_since is not None
                and now - self.evidence_wait_since >= self.evidence_wait_timeout_s
                and self.coop_seconds > 0.0):
            self.resets += 1
            self.coop_seconds = 0.0

    def _wait_for_evidence(self, now):
        if self.evidence_wait_since is None:
            # 无法确认上次成功检查之后的空档，从该时刻保守计算等待期限。
            self.evidence_wait_since = self.last_success if self.last_success is not None else now
        self._set_timer_paused(now, True)
        self._expire_evidence_wait(now)

    def _clear_session(self):
        self.role = self.NONE
        self.session_start_tick = None
        self.master_uid = None
        self.master_track_epoch = None
        self.partner_uid = None
        self.selected_follower_uid = None
        self.selection_started_at = None
        self.follow_position = None
        self.peer_track.clear()
        self.bound_local_epoch = None
        self.last_compare = -1e9
        self.last_master_message = -1e9
        self.master_track_received = False
        self.session_started_at = None
        self.last_operational_valid = None

    def _to_search(self, count_reset=True):
        self._reset_timer(count_reset)
        self.phase = self.SEARCH
        self.proposal = None
        self.proposal_created_at = None
        self._clear_session()

    def _finish_mission(self, now):
        self._reset_timer(False)
        self.phase = self.DONE
        self.role = self.NONE
        self.proposal = None
        self.proposal_created_at = None
        self.done_at = now

    def _heartbeat_payload(self, own_position):
        code = self._PHASE_CODE[self.phase]
        return f"H,{code},{own_position[0]:.5f},{own_position[1]:.5f}"

    def _offer_payload(self, local, follower_uid):
        if local.position is None:
            return None
        follower_code = self._uid_code(follower_uid) if follower_uid else "0"
        return (f"O,{self.proposal.start_tick},{self.proposal.track_epoch},"
                f"{self._tick(local.last_seen)},{local.position[0]:.5f},"
                f"{local.position[1]:.5f},{follower_code}")

    def _session_payload(self, kind):
        payload = (f"{kind},{self._uid_code(self.master_uid)},"
                   f"{self.session_start_tick},{self.master_track_epoch}")
        return f"{payload},{self._uid_code(self.partner_uid)}" if kind == "G" else payload

    @staticmethod
    def _track_valid(local, now, lock_valid=True):
        return (lock_valid
                and local.state in (LocalTrackManager.CONFIRMED, LocalTrackManager.COASTING)
                and local.position is not None and now - local.last_seen <= 0.8)

    def _track_payload(self, local, now, lock_valid=True, *, track_available=True):
        reference_valid = track_available and self._track_valid(local, now)
        valid = reference_valid and lock_valid
        prefix = (f"T,{self._uid_code(self.master_uid)},{self.session_start_tick},"
                  f"{self.master_track_epoch},{local.epoch},"
                  f"{self._tick(local.last_seen)},{int(valid)}")
        if self.role == self.MASTER:
            # 无坐标时保留空字段，正式伙伴和阶段也必须持续发送，不能依赖单次 G。
            position = (f"{local.position[0]:.5f},{local.position[1]:.5f}"
                        if reference_valid else ",")
            return (f"{prefix},{position},{self._uid_code(self.partner_uid)},"
                    f"{self._PHASE_CODE[self.phase]}")
        return (f"{prefix},{local.position[0]:.5f},{local.position[1]:.5f}"
                if reference_valid else prefix)

    def _kill_payload(self, session):
        position = self._kill_events.get(session)
        master_uid, start_tick, epoch = session
        prefix = f"K,{self._uid_code(master_uid)},{start_tick},{epoch}"
        if position is None:
            return prefix
        return f"{prefix},{position[0]:.5f},{position[1]:.5f}"

    def _parse(self, message):
        fields = message.payload.split(",")
        try:
            kind = fields[0]
            if kind == "H" and len(fields) == 4:
                return kind, (self._CODE_PHASE[fields[1]],
                              (float(fields[2]), float(fields[3])))
            if kind == "O" and len(fields) == 7:
                follower_uid = self._code_uid(fields[6]) if fields[6] != "0" else None
                return kind, _Offer(
                    message.sender_uid, int(fields[1]), int(fields[2]),
                    int(fields[3]) / 10.0, (float(fields[4]), float(fields[5])),
                    follower_uid,
                )
            if kind == "G" and len(fields) == 5:
                return kind, ((self._code_uid(fields[1]), int(fields[2]), int(fields[3])),
                              self._code_uid(fields[4]))
            if kind in ("A", "C", "R") and len(fields) == 4:
                return kind, (self._code_uid(fields[1]), int(fields[2]), int(fields[3]))
            if kind == "K" and len(fields) in (4, 6):
                position = ((float(fields[4]), float(fields[5])) if len(fields) == 6 else None)
                return kind, ((self._code_uid(fields[1]), int(fields[2]), int(fields[3])),
                              position)
            if kind == "T" and len(fields) in (7, 9, 11):
                valid = bool(int(fields[6]))
                position = ((float(fields[7]), float(fields[8]))
                            if len(fields) >= 9 and fields[7] and fields[8] else None)
                partner = self._code_uid(fields[9]) if len(fields) == 11 else None
                phase = self._CODE_PHASE[fields[10]] if len(fields) == 11 else None
                return kind, ((self._code_uid(fields[1]), int(fields[2]), int(fields[3])),
                              int(fields[4]), int(fields[5]) / 10.0, valid, position, partner, phase)
        except (IndexError, KeyError, TypeError, ValueError):
            return None
        return None

    def _observe_active_track(self, sender_uid, session, epoch, seen, valid, position, now):
        master_uid = session[0]
        if sender_uid == master_uid:
            if self.observed_session != session:
                self.observed_master_track.clear()
                self.observed_members.clear()
            self.observed_session = session
            self.observed_last = now
            self.observed_members.add(sender_uid)
            self.observed_master_track.ingest(epoch, seen, valid, position, now)
        elif self.observed_session == session:
            self.observed_last = now
            self.observed_members.add(sender_uid)

    def _clear_observed(self, session=None):
        if session is not None and self.observed_session != session:
            return
        self.observed_session = None
        self.observed_members.clear()
        self.observed_master_track.clear()
        self.observed_last = -1e9
        if self.duplicate_session == session:
            self.duplicate_session = None

    def _record_kill(self, session, position):
        if session[0] is None:
            return False
        added = session not in self.completed_sessions
        self.completed_sessions.add(session)
        if position is not None:
            self._kill_events[session] = position
            if all(_haversine_m(*position, *old) >= self.COMPLETED_GATE_M
                   for old in self.completed_positions):
                self.completed_positions.append(position)
        else:
            self._kill_events.setdefault(session, None)
        return added

    def _complete_target(self, local, now, payloads):
        """时间达标和目标停止共用完成出口，避免重复计数和不同的释放行为。"""
        session = self.current_session
        self._record_kill(session, local.position)
        payloads.append(self._kill_payload(session))
        self._to_search(False)
        self._clear_observed(session)
        if self.completed_count >= self._target_count:
            self._finish_mission(now)

    def _discard_offer(self, session):
        """只清理对应会话，保留同一主机后来发起的新召集。"""
        offer = self.offers.get(session[0])
        if offer is not None and offer[0].session == session:
            del self.offers[session[0]]

    def _accept_offer(self, offer, now):
        self._reset_timer()
        self.phase = self.INIT
        self.role = self.FOLLOWER
        self.proposal = None
        self.proposal_created_at = None
        self.session_start_tick = offer.start_tick
        self.master_uid = offer.sender_uid
        self.master_track_epoch = offer.track_epoch
        self.partner_uid = offer.sender_uid
        self.follow_position = offer.position
        self.peer_track.clear()
        self.bound_local_epoch = None
        self.last_master_message = now
        self.master_track_received = False
        self.session_started_at = now
        self.follower_sessions_started += 1

    def _become_master(self, follower_uid, now):
        offer = self.proposal
        self._reset_timer()
        self.phase = self.INIT
        self.role = self.MASTER
        self.session_start_tick = offer.start_tick
        self.master_uid = self.my_uid
        self.master_track_epoch = offer.track_epoch
        self.partner_uid = follower_uid
        self.selected_follower_uid = follower_uid
        self.selection_started_at = None
        self.proposal = None
        self.proposal_created_at = None
        self.peer_track.clear()
        self.last_compare = now
        self.session_started_at = now
        self.master_sessions_started += 1

    def _fresh_offers(self, now):
        self.offers = {
            uid: item for uid, item in self.offers.items()
            if now - item[1] <= self.STATUS_TIMEOUT_S
        }
        offers = [item[0] for item in self.offers.values()]
        if self.proposal is not None:
            offers.append(self.proposal)
        return offers

    def _select_follower(self, now, own_position):
        candidates = []
        for uid in self.peer_uids:
            status = self.peer_status.get(uid)
            if status is None or now - status.received_at > self.STATUS_TIMEOUT_S:
                continue
            if status.phase == self.SEARCH:
                candidates.append((_haversine_m(*own_position, *status.position), uid))
            elif status.phase == self.HOLD:
                peer_offer = self.offers.get(uid)
                if peer_offer is not None and self.proposal.priority < peer_offer[0].priority:
                    candidates.append((_haversine_m(*own_position, *status.position), uid))
        return min(candidates, default=(None, None))[1]

    def _plan_evade(self, now, own_position):
        if len(self.observed_members) < 2:
            return None
        positions = []
        for uid in sorted(self.observed_members):
            status = self.peer_status.get(uid)
            if status is None or now - status.received_at > self.STATUS_TIMEOUT_S:
                return None
            positions.append(status.position)
        return horizontal_perpendicular_destination(
            positions[0], positions[1], own_position, distance_m=100.0)

    def _predict_observed_position(self, now):
        snapshot = self.observed_master_track.snapshot(now)
        if snapshot.position is None or now - snapshot.last_seen > self.STATUS_TIMEOUT_S:
            return None
        velocity = estimate_track_velocity(snapshot.points)
        dt = max(0.0, min(now - snapshot.last_seen, self.STATUS_TIMEOUT_S))
        return _move_m(snapshot.position, velocity[0] * dt, velocity[1] * dt)

    def should_suppress(self, position, now):
        """搜索和避让时忽略活动协同目标及已经完成的目标。"""
        if any(_haversine_m(*position, *old) < self.COMPLETED_GATE_M
               for old in self.completed_positions):
            return True
        predicted = self._predict_observed_position(now)
        return (predicted is not None
                and _haversine_m(*position, *predicted) < self.DUPLICATE_GATE_M)

    def step(self, now, local, inbox, can_propose=True, own_position=None,
             target_count=3, local_lock_valid=True, local_lock_expired=False):
        """处理一帧三机通信和状态，返回广播及本地轨迹控制动作。"""
        own_position = own_position or (0.0, 0.0)
        self._target_count = max(1, int(target_count or 3))
        payloads = []
        reset_local = False
        begin_evade = False
        joined_as_follower = False
        accepts = []

        for message in inbox:
            if message.sender_uid not in self.peer_uids:
                continue
            parsed = self._parse(message)
            if parsed is None:
                continue
            kind, data = parsed
            if kind == "H":
                phase, position = data
                self.peer_status[message.sender_uid] = _PeerStatus(phase, position, now)
            elif kind == "O":
                # 收件箱可能重复包含旧消息；结束过的会话不能再次召集。
                if data.session in self.cancelled_sessions or data.session in self.completed_sessions:
                    continue
                if data.start_tick < self.latest_offer_ticks.get(message.sender_uid, -1):
                    continue
                self.latest_offer_ticks[message.sender_uid] = data.start_tick
                if (self.role == self.FOLLOWER and message.sender_uid == self.master_uid
                        and data.start_tick > self.session_start_tick):
                    self.cancelled_sessions.add(self.current_session)
                    self._discard_offer(self.current_session)
                    self._to_search()
                    reset_local = True
                self.offers[message.sender_uid] = (data, now)
                if (self.role == self.FOLLOWER and self._session_matches(data.session)
                        and data.follower_uid == self.my_uid):
                    self.follow_position = data.position
                    self.last_master_message = now
                    payloads.append(self._session_payload("A"))
            elif kind == "A":
                accepts.append((message.sender_uid, data))
            elif kind == "G":
                session, partner = data
                if (self.role == self.FOLLOWER and self._session_matches(session)
                        and message.sender_uid == self.master_uid):
                    if partner == self.my_uid:
                        self.phase = self.ACTIVE
                        self.last_master_message = now
                    else:
                        self.cancelled_sessions.add(session)
                        self._discard_offer(session)
                        self._to_search()
                        reset_local = True
            elif kind == "C":
                self.cancelled_sessions.add(data)
                self._discard_offer(data)
                was_evading_duplicate = (
                    self.phase == self.EVADE and self.duplicate_session == data)
                if self._session_matches(data):
                    self._to_search()
                    reset_local = True
                self._clear_observed(data)
                if was_evading_duplicate:
                    self.phase = self.SEARCH
                    self.evade_destination = None
                    self.evade_started_at = None
            elif (kind == "R" and self.role == self.FOLLOWER and self._session_matches(data)
                  and message.sender_uid == self.master_uid):
                # 同一接收记录只执行一次；后续同内容的新请求仍可触发重捕获。
                receipt = (message.sender_uid, data, message.recv_time)
                if receipt in self._handled_reacquire_messages:
                    continue
                self._handled_reacquire_messages.add(receipt)
                self.bound_local_epoch = None
                reset_local = True
            elif kind == "K":
                session, position = data
                self._discard_offer(session)
                self._record_kill(session, position)
                if self._session_matches(session):
                    self._to_search(False)
                    reset_local = True
                self._clear_observed(session)
            elif kind == "T":
                session, sender_epoch, seen, valid, position, partner, phase = data
                if session in self.cancelled_sessions or session in self.completed_sessions:
                    continue
                if session[1] < self.latest_offer_ticks.get(session[0], -1):
                    continue
                if message.sender_uid == session[0] and partner is not None:
                    self.latest_offer_ticks[session[0]] = session[1]
                    # 即使 C 和改选 O 丢失，新会话或排他的正式伙伴也能释放旧从机。
                    if (self.role == self.FOLLOWER and message.sender_uid == self.master_uid
                            and (session[1] > self.session_start_tick
                                 or (self._session_matches(session) and partner != self.my_uid))):
                        self.cancelled_sessions.add(self.current_session)
                        self._discard_offer(self.current_session)
                        self._to_search()
                        reset_local = True
                self._observe_active_track(
                    message.sender_uid, session, sender_epoch, seen, valid, position, now)
                if not self._session_matches(session):
                    continue
                if self.role == self.MASTER and message.sender_uid == self.partner_uid:
                    self.peer_track.ingest(sender_epoch, seen, valid, position, now)
                elif self.role == self.FOLLOWER and message.sender_uid == self.master_uid:
                    if sender_epoch != self.master_track_epoch:
                        continue
                    self.last_master_message = now
                    self.master_track_received = True
                    if partner == self.my_uid and phase == self.ACTIVE:
                        self.phase = self.ACTIVE
                    self.peer_track.ingest(sender_epoch, seen, valid, position, now)
                    if position is not None:
                        self.follow_position = position

        if self.completed_count >= self._target_count and not self.finished:
            self._finish_mission(now)

        if self.finished:
            if now - self.last_heartbeat_send >= self.SEND_PERIOD_S:
                payloads.append(self._heartbeat_payload(own_position))
                self.last_heartbeat_send = now
            self._append_kill_gossip(now, payloads)
            return CoordinationUpdate(tuple(dict.fromkeys(payloads)), reset_local, begin_evade)

        if self.observed_session is not None and now - self.observed_last > self.STATUS_TIMEOUT_S:
            stale = self.observed_session
            self._clear_observed(stale)

        if self.phase == self.EVADE:
            arrived = (self.evade_destination is not None
                       and _haversine_m(*own_position, *self.evade_destination)
                       <= self.EVADE_ARRIVAL_M)
            if arrived or now - self.evade_started_at >= self.EVADE_TIMEOUT_S:
                self.phase = self.SEARCH
                self.evade_destination = None
                self.evade_started_at = None
                self.duplicate_session = None

        if (self.phase == self.SEARCH and can_propose
                and local.state == LocalTrackManager.CONFIRMED):
            tick = self._tick(now)
            self.proposal = _Offer(
                self.my_uid, tick, local.epoch, local.last_seen, local.position)
            self.proposal_created_at = now
            self.phase = self.HOLD

        # 当应答与超时落在同一帧，先确认仍属于当前邀请的应答，避免无谓改选。
        if self.phase == self.HOLD and self.proposal is not None:
            for sender_uid, session in accepts:
                if (session == self.proposal.session
                        and sender_uid == self.selected_follower_uid
                        and local.state != LocalTrackManager.LOST
                        and local.epoch == self.proposal.track_epoch):
                    self._become_master(sender_uid, now)
                    break

        fresh_offers = self._fresh_offers(now)
        observed_active = self.observed_session is not None

        if self.phase == self.HOLD:
            if local.state == LocalTrackManager.LOST or local.epoch != self.proposal.track_epoch:
                self._to_search()
                reset_local = True
            elif observed_active:
                duplicate = compare_track_windows(
                    local, self.observed_master_track.snapshot(now), now)
                if duplicate.passed:
                    destination = self._plan_evade(now, own_position)
                    if destination is not None:
                        self.last_match = duplicate
                        self.duplicate_session = self.observed_session
                        self.evade_destination = destination
                        self.evade_started_at = now
                        self.evade_count += 1
                        self.phase = self.EVADE
                        self.proposal = None
                        self.proposal_created_at = None
                        reset_local = True
                        begin_evade = True
            else:
                winner = min(fresh_offers, key=lambda offer: offer.priority, default=None)
                if winner is not None and winner.sender_uid != self.my_uid:
                    if (winner.follower_uid == self.my_uid
                            and winner.priority < self.proposal.priority):
                        self._accept_offer(winner, now)
                        joined_as_follower = True
                        payloads.append(self._session_payload("A"))
                        reset_local = True
                elif (winner is not None and winner.sender_uid == self.my_uid
                      and now - self.proposal_created_at >= self.ELECTION_DELAY_S):
                    if (self.selected_follower_uid is not None
                            and now - self.selection_started_at >= self.SELECTION_TIMEOUT_S):
                        old = self.proposal
                        self.cancelled_sessions.add(old.session)
                        payloads.append(f"C,{self._uid_code(self.my_uid)},{old.start_tick},{old.track_epoch}")
                        # 会话编号兼作邀请轮次；旧 A、G、T、C 均不能作用到下一轮。
                        self.proposal = _Offer(
                            self.my_uid, old.start_tick + 1,
                            local.epoch, local.last_seen, local.position)
                        self.proposal_created_at = now
                        self.selected_follower_uid = None
                        self.selection_started_at = None
                    if self.selected_follower_uid is None:
                        self.selected_follower_uid = self._select_follower(now, own_position)
                        if self.selected_follower_uid is not None:
                            self.selection_started_at = now

        elif self.phase == self.SEARCH and not observed_active:
            winner = min(fresh_offers, key=lambda offer: offer.priority, default=None)
            if winner is not None and winner.follower_uid == self.my_uid:
                self._accept_offer(winner, now)
                joined_as_follower = True
                payloads.append(self._session_payload("A"))
                reset_local = True

        if (self.role == self.FOLLOWER and self.phase in (self.INIT, self.ACTIVE)
                and not joined_as_follower and not reset_local):
            alive = local.state in (LocalTrackManager.CONFIRMED, LocalTrackManager.COASTING)
            if self.bound_local_epoch is None and alive:
                self.bound_local_epoch = local.epoch
            elif self.bound_local_epoch is not None and (
                    local.state == LocalTrackManager.LOST
                    or local.epoch != self.bound_local_epoch):
                payloads.append(self._session_payload("C"))
                session = self.current_session
                self._to_search()
                self._clear_observed(session)
                reset_local = True
            elif self.phase == self.ACTIVE and local_lock_expired:
                payloads.append(self._session_payload("C"))
                session = self.current_session
                self._to_search()
                self._clear_observed(session)
                reset_local = True
            elif (self.master_track_received
                  and now - self.last_master_message > self.STATUS_TIMEOUT_S):
                self._to_search()
                reset_local = True
            elif (not self.master_track_received
                  and now - self.last_master_message > self.FOLLOWER_TIMEOUT_S):
                self._to_search()
                reset_local = True

        # 停止判定独立于每秒一次的轨迹配对检查；缺少配对点不等于丢失主锁定。
        peer = self.peer_track.snapshot(now)
        stop_ready = self.stopped_target.update(
            now, (self.current_session, local.epoch, peer.epoch), local,
            valid=(self.role == self.MASTER and self.phase == self.ACTIVE
                   and not reset_local and local_lock_valid and not local_lock_expired
                   and local.epoch == self.master_track_epoch
                   and peer.epoch == self.bound_follower_epoch
                   and peer.state == LocalTrackManager.CONFIRMED
                   and 0.0 <= now - peer.last_seen <= MatchConfig().fresh_s))

        if self.role == self.MASTER and self.phase in (self.INIT, self.ACTIVE):
            lock_wait = (self.phase == self.ACTIVE
                         and (not local_lock_valid or self.peer_track.invalid_since is not None))
            peer_epoch_changed = (self.phase == self.ACTIVE
                                  and self.peer_track.epoch != self.bound_follower_epoch)
            if peer_epoch_changed:
                self.last_match = TrackMatchResult(False, "peer_epoch_changed")
            peer_lock_expired = (
                self.phase == self.ACTIVE
                and self.peer_track.invalid_since is not None
                and now - self.peer_track.invalid_since >= self.reacquire_timeout_s)
            if (local.state == LocalTrackManager.LOST
                    or local.epoch != self.master_track_epoch
                    or peer_epoch_changed
                    or local_lock_expired
                    or peer_lock_expired):
                payloads.append(self._session_payload("C"))
                session = self.current_session
                self._to_search()
                self._clear_observed(session)
                reset_local = True
            elif stop_ready:
                self._complete_target(local, now, payloads)
                reset_local = True
            elif now - self.session_started_at > self.SESSION_TIMEOUT_S and self.phase == self.INIT:
                payloads.append(self._session_payload("C"))
                session = self.current_session
                self._to_search()
                self._clear_observed(session)
                reset_local = True
            else:
                self._set_timer_paused(now, lock_wait or self.evidence_wait_since is not None)
                self._expire_evidence_wait(now)

            if (self.role == self.MASTER
                    and self.phase in (self.INIT, self.ACTIVE)
                    and lock_wait):
                reason = ("local_lock_reacquiring" if not local_lock_valid
                          else "peer_lock_reacquiring")
                self.last_match = TrackMatchResult(False, reason)
            elif (self.role == self.MASTER
                  and self.phase in (self.INIT, self.ACTIVE)
                  and now - self.last_compare >= self.CHECK_PERIOD_S):
                self.last_compare = now
                peer = self.peer_track.snapshot(now)
                self.last_match = (compare_track_windows(local, peer, now)
                                   if local_lock_valid else TrackMatchResult(
                                       False, "local_lock_reacquiring"))
                if self.last_match.passed:
                    self.last_evidence = now
                    if self.evidence_wait_since is not None:
                        # 首次恢复只重新建立计时起点，不追补最后一次成功之后的空档。
                        self._set_timer_paused(now, False)
                        self.evidence_wait_since = None
                        self.last_success = now
                        self.timer_paused_duration = 0.0
                    if self.phase == self.INIT:
                        self.phase = self.ACTIVE
                        self.bound_follower_epoch = peer.epoch
                        self.last_success = now
                        payloads.append(self._session_payload("G"))
                    else:
                        elapsed = max(
                            0.0, now - self.last_success - self.timer_paused_duration)
                        self.coop_seconds += elapsed
                        self.timer_paused_duration = 0.0
                        self.last_success = now
                        self.peak_seconds = max(self.peak_seconds, self.coop_seconds)
                        if self.coop_seconds >= self.duration_s:
                            self._complete_target(local, now, payloads)
                            reset_local = True
                elif self.phase == self.ACTIVE and self.last_match.insufficient_evidence:
                    self._wait_for_evidence(now)
                elif self.phase == self.ACTIVE:
                    payloads.append(self._session_payload("C"))
                    session = self.current_session
                    self._to_search()
                    self._clear_observed(session)
                    reset_local = True
                elif self.last_match.reason in ("latest_position", "position",
                                                 "velocity", "heading"):
                    payloads.append(self._session_payload("R"))
                    self.peer_track.clear()

        if now - self.last_heartbeat_send >= self.SEND_PERIOD_S:
            payloads.append(self._heartbeat_payload(own_position))
            self.last_heartbeat_send = now

        if now - self.last_operational_send >= self.SEND_PERIOD_S:
            operational = None
            if self.phase == self.HOLD and self.proposal is not None:
                operational = self._offer_payload(local, self.selected_follower_uid)
            elif self.role in (self.MASTER, self.FOLLOWER) and self.phase in (self.INIT, self.ACTIVE):
                operational = self._track_payload(
                    local, now, local_lock_valid, track_available=not reset_local)
            if operational is not None:
                payloads.append(operational)
                self.last_operational_send = now
                self.last_operational_valid = self._track_valid(
                    local, now, local_lock_valid and not reset_local)

        if self.role in (self.MASTER, self.FOLLOWER) and self.phase in (self.INIT, self.ACTIVE):
            current_valid = self._track_valid(local, now, local_lock_valid and not reset_local)
            if current_valid != self.last_operational_valid:
                payloads.append(self._track_payload(
                    local, now, local_lock_valid, track_available=not reset_local))
                self.last_operational_send = now
                self.last_operational_valid = current_valid

        self._append_kill_gossip(now, payloads)
        return CoordinationUpdate(tuple(dict.fromkeys(payloads)), reset_local, begin_evade)

    def _append_kill_gossip(self, now, payloads):
        if not self.completed_sessions or now - self.last_kill_send < 1.0:
            return
        sessions = sorted(self.completed_sessions, key=lambda item: (item[1], item[0]))
        session = sessions[self._kill_index % len(sessions)]
        self._kill_index += 1
        payloads.append(self._kill_payload(session))
        self.last_kill_send = now
