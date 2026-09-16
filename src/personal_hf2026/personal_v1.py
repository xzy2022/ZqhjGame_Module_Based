# 修改时间：2026-09-16。
# 修改目的：让实时双机定位只在稳定协同阶段使用离线评测对应的三十度视场角。
# 修改内容：搜索和协同初始化保持四十八度，进入 COOP_ACTIVE 后切换三十度，退出后恢复四十八度。
# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13（采集身份接口）
# 修改目的：允许采集策略限制轨迹资格，同时保留全部车辆的原生锁定竞争。
# 修改内容：新增默认兼容的候选资格接口，并为云台单独保留全体车辆坐标。
# 修改时间：2026-09-13 14:20
# 修改目的：保留已确认且仍有效的搜索目标，避免等待原生主锁定两秒就丢弃。
# 修改内容：确认轨迹按本地丢失门限释放，主锁定稳定仍是发起协同的必要条件。
# 修改时间：2026-09-13
# 修改目的：为正式双机协同目标补充裁判侧目指位置上报。
# 修改内容：主锁定匹配且轨迹本帧更新时按一赫兹调用 report_target，不上报搜索候选。
# 修改时间：2026-09-13
# 修改目的：保持搜索到协同交接期间的云台连续性，避免阶段切换造成目标出视野。
# 修改内容：主从共用限速交接控制器并记录状态，不改飞行和协同计时规则。
# 修改时间：2026-09-13
# 修改目的：用长轴高速普查和实际视野空白补扫提高全域搜索效率。
# 修改内容：接入普查管理器、实际姿态覆盖记账和搜索速度调节，保留候选确认。
# 修改时间：2026-09-13
# 修改目的：在条带搜索时横向巡视并避免侧方候选在确认时立即离开视野。
# 修改内容：接入独立搜索云台控制及日志，保持原航线、速度和协同控制。
# 修改时间：2026-09-13
# 修改目的：让搜索航线按实际到点推进并在协同结束后恢复未完成航段。
# 修改内容：接入条带管理器，记录实际网格访问与首次运动候选确认时间。
# 修改时间：2026-09-13
# 修改目的：让实验日志区分证据等待与主锁定暂停。
# 修改内容：输出证据等待起点及其独立超时参数。
# 修改时间：2026-09-13
# 修改目的：避免主机主锁定失配使从机初始关联失去参考轨迹。
# 修改内容：从机初始关联读取独立于计时锁定标志的主机参考轨迹视图。
# 修改时间：2026-09-13
# 修改目的：避免从机仅凭主机附近的一个坐标抓住错误目标。
# 修改内容：在协同发起阶段收集候选轨迹，唯一匹配主机轨迹后才正式绑定。
# 修改时间：2026-09-13
# 修改目的：让协同计时和飞行纠偏使用排他性的主检测归属结果。
# 修改内容：传递本帧实际接受的轨迹点，仅把明确归属其他候选的检测用于新竞争方向。
# 修改时间：2026-09-13
# 修改目的：让主从协同飞行按接近距离和转向需求调节速度。
# 修改内容：接入速度调节函数并记录指令速度，保持偏移公式及方向计算不变。
# 修改时间：2026-09-13
# 修改目的：为主从协同飞行接入跟随目标的相对位置控制。
# 修改内容：增加竞争偏移调整速率和上限参数，保留远航点作为期望位置方向的执行适配。
# 修改时间：2026-09-12
# 修改目的：避免已有航点导航覆盖协同阶段的直接航向指令。
# 修改内容：将计算航向转换为持续前推的远导航点，半径设零并显式选择转向方向。
# 修改时间：2026-09-12
# 修改目的：验证固定向下宽视野与主从独立竞争方向导航的组合。
# 修改内容：相机固定四十八度向下，协同阶段使用加权改善方向和直接航向指令替代盘旋。
# 修改时间：2026-09-12
# 修改目的：修正协同盘旋分支的状态常量引用以恢复正常决策。
# 修改内容：使用协调器已有的 INIT 和 ACTIVE 常量判断协同阶段。
# 修改时间：2026-09-12
# 修改目的：验证缩小协同阶段盘旋半径能否减少原生主锁定丢失。
# 修改内容：增加十五米名义盘旋半径参数，仅用于协同召唤和协同进行阶段的主从飞行控制。
# 修改时间：2026-09-12
# 修改目的：让主从无人机在协同跟踪时共同容忍短暂的引擎主锁定空值。
# 修改内容：接入零点五秒协同恢复参数，波动期间保留轨迹和云台控制并通知协调器暂停计时。
# 修改时间：2026-09-12
# 修改目的：让新搜索目标只通过云台控制器的一次更新进入十五度锁定视野。
# 修改内容：删除搜索目标接受后的重复绑定调用，由更新方法统一完成新轨迹绑定。
# 修改时间：2026-09-12
# 修改目的：让搜索、云台锁定与协同计时分别使用多目标候选、期望轨迹和引擎主锁定。
# 修改内容：接入多目标搜索中心和动态 FOV 控制，并用单数主检测约束协同有效轨迹。
"""个人基线 v1：三机搜索、动态配对并依次协同跟踪三个目标。"""

import math

from competition.baselines.coop_distributed import (
    CoopDistributedAgent, _bearing_deg, _clamp_to_safebox, _haversine_m, _BBOX,
)
from competition.sdk.core.commands import (
    broadcast, fly_to, point_gimbal, report_target, set_gimbal_fov,
)

from .coordination import CoopCoordinator
from .gimbal_lock import GimbalLockConfig, GimbalLockController
from .search_filter import MultiCandidateSearch
from .survey_search import SurveySearchRoute
from .search_gimbal import SearchGimbalController, GimbalHandoffController
from .tracking import (CandidateTrackSet, LocalTrackManager, TargetAssociationResult,
                       associate_target_track, _move)
from .competition_flight import CompetitionDirectionController, adjust_flight_speed


class PersonalV1Agent(CoopDistributedAgent):
    A = "20001"
    B = "20002"
    C = "20003"
    COOP_DURATION_S = 22.0
    COOP_REACQUIRE_TIMEOUT_S = 5.0
    SEARCH_FOV_DEG = 48.0
    COOP_ACTIVE_FOV_DEG = 30.0
    TARGET_REPORT_PERIOD_S = 1.0
    COMPETITION_UPDATE_PERIOD_S = 0.5
    COMPETITION_OFFSET_STEP_MPS = 5.0
    COMPETITION_MAX_OFFSET_M = 50.0
    COOP_GUIDANCE_LOOKAHEAD_M = 1000.0

    def reset(self):
        super().reset()
        self._track = LocalTrackManager()
        self._follower_candidates = CandidateTrackSet()
        self._follower_session = None
        self._follower_target_bound = False
        self._follower_association = TargetAssociationResult("IDLE")
        self._search_filter = MultiCandidateSearch()
        self._search_gimbal = SearchGimbalController()
        self._gimbal_handoff = GimbalHandoffController()
        self._search_route = SurveySearchRoute(
            _BBOX, (self.A, self.B, self.C).index(self.my_uid), 3)
        self._search_flight_speed = None
        self._first_moving_candidate_confirmed_s = None
        self._gimbal_lock = GimbalLockController(GimbalLockConfig(
            min_fov_deg=self.SEARCH_FOV_DEG, max_fov_deg=self.SEARCH_FOV_DEG,
            preferred_fov_deg=self.SEARCH_FOV_DEG, allow_multiple_targets=True,
            coop_reacquire_timeout_s=self.COOP_REACQUIRE_TIMEOUT_S))
        self._competition_flight = CompetitionDirectionController(
            self.COMPETITION_UPDATE_PERIOD_S, self.COMPETITION_OFFSET_STEP_MPS,
            self.COMPETITION_MAX_OFFSET_M)
        self._coop_flight_speed = None
        self._coordinator = CoopCoordinator(
            self.my_uid, (self.A, self.B, self.C), self.COOP_DURATION_S,
            self.COOP_REACQUIRE_TIMEOUT_S)
        self._state = self._coordinator.phase
        self._home = None
        self._candidate = None
        self._peer = None
        self._peer_received = -1e9
        self._last_seen = -1e9
        self._last_seen_position = None
        self._coop_seconds = 0.0
        self._gap = 0.0
        self._coop_lock_valid = True
        self._coop_lock_expired = False
        self._coop_lock_missing_s = 0.0
        self.done_at = None

    @property
    def finished(self):
        return self._coordinator.finished

    @property
    def completion_summary(self):
        match = self._coordinator.last_match
        return {
            "state": self._state,
            "role": self._coordinator.role,
            "track_state": self._track.state,
            "track_epoch": self._track.epoch,
            "peer_track_epoch": self._coordinator.peer_track.epoch,
            "estimated_coop_seconds": self._coop_seconds,
            "coop_peak_seconds": self._coordinator.peak_seconds,
            "coop_resets": self._coordinator.resets,
            "last_coop_evidence_s": self._coordinator.last_evidence,
            "coop_duration_s": self.COOP_DURATION_S,
            "coop_reacquire_timeout_s": self.COOP_REACQUIRE_TIMEOUT_S,
            "coop_evidence_wait_since_s": self._coordinator.evidence_wait_since,
            "coop_evidence_wait_timeout_s": self._coordinator.evidence_wait_timeout_s,
            "coop_timer_paused": self._coordinator.timer_pause_started is not None,
            "coop_paused_duration_s": self._coordinator.timer_paused_duration,
            "coop_lock_valid": self._coop_lock_valid,
            "coop_lock_expired": self._coop_lock_expired,
            "coop_lock_missing_s": self._coop_lock_missing_s,
            "estimated_destroyed_count": self._coordinator.completed_count,
            "completed_sessions": sorted(self._coordinator.completed_sessions),
            "master_sessions_started": self._coordinator.master_sessions_started,
            "follower_sessions_started": self._coordinator.follower_sessions_started,
            "duplicate_evade_count": self._coordinator.evade_count,
            "partner_uid": self._coordinator.partner_uid,
            "evade_destination": self._coordinator.evade_destination,
            "done_at_s": self.done_at,
            "last_match_reason": match.reason,
            "last_match_position_m": (match.position_median_m
                                      if math.isfinite(match.position_median_m) else None),
            "search_filter_status": self._search_filter.status,
            "search_filter_last_status": self._search_filter.last_result.status,
            "search_filter_last_speed_mps": self._search_filter.last_result.speed_mps,
            "search_filter_last_reason": self._search_filter.last_result.reason,
            "search_track_count": self._search_filter.track_count,
            "search_view_center": self._search_filter.view_center_position,
            "search_route": self._search_route.summary,
            "search_flight_speed_mps": self._search_flight_speed,
            "search_gimbal": self._search_gimbal.summary,
            "gimbal_handoff": self._gimbal_handoff.summary,
            "first_moving_candidate_confirmed_s": self._first_moving_candidate_confirmed_s,
            "gimbal_lock_state": self._gimbal_lock.state,
            "gimbal_lock_primary_matches": self._gimbal_lock.primary_matches,
            "primary_association": self._gimbal_lock.primary_association.status,
            "primary_target_distance_m": self._gimbal_lock.primary_association.target_distance_m,
            "primary_other_distance_m": self._gimbal_lock.primary_association.other_distance_m,
            "follower_association": self._follower_association.status,
            "follower_candidate_id": self._follower_association.track_id,
            "follower_target_bound": self._follower_target_bound,
            "gimbal_lock_visible_count": self._gimbal_lock.visible_count,
            "gimbal_lock_fov_deg": self._gimbal_lock.fov_deg,
            "gimbal_lock_desired_visible": self._gimbal_lock.desired_visible,
            "competition_flight": self._competition_flight.summary,
            "coop_flight_speed_mps": self._coop_flight_speed,
            # 保留旧摘要字段，便于已有结果脚本继续读取。
            "verify_confirmed_count": self._search_filter.accepted_count,
            "verify_static_count": self._search_filter.rejected_count,
            "verify_timeout_count": self._search_filter.timeout_count,
            "verify_sample_span_s": 0.0,
        }

    def _hold(self):
        return [fly_to(*self._home, speed=22.0), point_gimbal(0, -90),
                set_gimbal_fov(self.SEARCH_FOV_DEG)]

    def _tracking_gimbal(self, self_lat, self_lon, self_heading,
                         tgt_lat, tgt_lon, *, self_alt):
        # 光轴固定竖直向下，不使用目标高程估计。
        return 0.0, -90.0

    def _update_compatibility_fields(self, now):
        self._state = self._coordinator.phase
        self._coop_seconds = self._coordinator.coop_seconds
        self.done_at = self._coordinator.done_at
        self._last_seen = self._track.last_seen
        self._last_seen_position = self._track.position
        peer = self._coordinator.peer_track.snapshot(now)
        self._peer_received = self._coordinator.peer_track.last_received
        self._peer = ((peer.last_seen,
                       peer.state in (LocalTrackManager.CONFIRMED, LocalTrackManager.COASTING),
                       peer.position) if peer.position is not None else None)

    def _eligible_positions(self, detections, positions):
        """默认沿用全部检测；专用采集子类可替换目标类别接纳策略。"""
        return positions

    def _competition_positions(self, eligible_positions, all_positions):
        """旧入口沿用原有过滤结果；身份采集子类保留全部竞争车辆。"""
        return eligible_positions

    def decide(self, obs, dt):
        score = getattr(getattr(obs, "briefing", None), "score_view", None)
        now = max(self._t, score.sim_time) if score is not None else self._t + max(0.0, dt)
        self._t = now
        self._tick += 1
        if self._home is None:
            self._home = (obs.self.lat, obs.self.lon)
        self._search_route.observe_search_position(
            (obs.self.lat, obs.self.lon) if self._coordinator.phase == CoopCoordinator.SEARCH else None)
        self._search_flight_speed = None
        self._search_route.coverage.new_cells = []
        if self._coordinator.phase == CoopCoordinator.SEARCH:
            self._search_route.coverage.observe(
                now, (obs.self.lat, obs.self.lon), obs.self.heading_deg,
                obs.self.gimbal_pan, obs.self.gimbal_tilt, obs.self.gimbal_fov_deg)

        # FOLLOWER 只在 MASTER 最新位置附近建立新轨迹，避免追上无关车辆。
        if self._coordinator.role == CoopCoordinator.FOLLOWER:
            if self._follower_session != self._coordinator.current_session:
                self._follower_session = self._coordinator.current_session
                self._follower_candidates.reset()
                self._follower_target_bound = False
                self._follower_association = TargetAssociationResult("NO_REFERENCE")
            self._track.set_acquisition_seed(self._coordinator.follow_position)
        else:
            self._track.clear_acquisition_seed()
            self._follower_session = None
            self._follower_candidates.reset()
            self._follower_target_bound = False
            self._follower_association = TargetAssociationResult("IDLE")

        detections = (obs.self.detections if obs.self.detections
                      else ((obs.self.detection,) if obs.self.detection.detected else ()))
        positions = tuple((detection.target_lat, detection.target_lon)
                          for detection in detections
                          if detection.detected
                          and detection.target_lat is not None
                          and detection.target_lon is not None)
        all_positions = positions
        positions = self._eligible_positions(detections, positions)
        if self._coordinator.phase in (CoopCoordinator.SEARCH, CoopCoordinator.EVADE):
            positions = tuple(position for position in positions
                              if not self._coordinator.should_suppress(position, now))
        all_positions = self._competition_positions(positions, all_positions)

        can_propose = False
        effective_local = None
        self._coop_lock_valid = True
        self._coop_lock_expired = False
        self._coop_lock_missing_s = 0.0
        if self._coordinator.phase == CoopCoordinator.SEARCH:
            if self._gimbal_lock.active and self._track.state != LocalTrackManager.LOST:
                position = self._track.select_position(now, positions)
                local = self._track.update(now, position)
                desired = self._track.predict_position(now)
                lock = self._gimbal_lock.update(
                    now, dt, local.epoch, desired, obs.self.detection,
                    all_positions, obs.self.gimbal_fov_deg,
                    target_observation=local.position if local.last_seen == now else None)
                can_propose = (
                    lock.state == GimbalLockController.LOCKED
                    and local.state == LocalTrackManager.CONFIRMED)
                # 已确认运动目标继续接近；短暂缺测由轨迹自身的丢失门限约束。
                retained = (local.state in (LocalTrackManager.CONFIRMED, LocalTrackManager.COASTING)
                            and now - local.last_seen <= self._track.config.lost_after_s)
                if (lock.expired and not retained) or local.state == LocalTrackManager.LOST:
                    self._track.reset_for_acquisition()
                    self._gimbal_lock.reset()
                    self._search_filter.reset_current()
                    local = self._track.snapshot(now)
            else:
                accepted = self._search_filter.update(now, positions)
                if accepted is not None and self._first_moving_candidate_confirmed_s is None:
                    self._first_moving_candidate_confirmed_s = now
                if accepted is not None:
                    self._track.adopt(accepted)
                    local = self._track.snapshot(now)
                    self._gimbal_lock.update(
                        now, dt, local.epoch, local.position,
                        obs.self.detection, all_positions, obs.self.gimbal_fov_deg,
                        target_observation=local.position if local.last_seen == now else None)
                else:
                    if self._track.state != LocalTrackManager.LOST:
                        self._track.reset_for_acquisition()
                    local = self._track.snapshot(now)
                    self._candidate = self._search_filter.focus_position
            effective_local = local
        else:
            self._search_filter.reset_current()
            if (self._coordinator.role == CoopCoordinator.FOLLOWER
                    and not self._follower_target_bound):
                candidates = self._follower_candidates.update(now, positions)
                reference = self._coordinator.peer_track.snapshot(now, require_lock=False)
                self._follower_association = associate_target_track(reference, candidates, now)
                if self._follower_association.status == "MATCH":
                    self._track.adopt(self._follower_association.track)
                    self._follower_target_bound = True
                    self._follower_candidates.reset()
                local = self._track.snapshot(now)
            else:
                position = self._track.select_position(now, positions)
                local = self._track.update(now, position)
            desired = self._track.predict_position(now)
            if desired is not None:
                lock = self._gimbal_lock.update(
                    now, dt, local.epoch, desired, obs.self.detection,
                    all_positions, obs.self.gimbal_fov_deg,
                    target_observation=local.position if local.last_seen == now else None)
                # 复数检测继续维护轨迹和云台；单数主锁定只控制协同有效性与暂停。
                self._coop_lock_valid = lock.primary_matches
                self._coop_lock_expired = lock.coop_expired
                self._coop_lock_missing_s = lock.missing_s
                effective_local = local
            else:
                self._gimbal_lock.reset()
                self._coop_lock_valid = False
                effective_local = local

        if effective_local is None:
            effective_local = local
        target_count = getattr(getattr(obs, "briefing", None), "target_count", None) or 3
        update = self._coordinator.step(
            now,
            effective_local,
            obs.comm_inbox,
            can_propose,
            own_position=(obs.self.lat, obs.self.lon),
            target_count=target_count,
            local_lock_valid=self._coop_lock_valid,
            local_lock_expired=self._coop_lock_expired,
        )

        if update.reset_local_track:
            seed = (self._coordinator.follow_position
                    if self._coordinator.role == CoopCoordinator.FOLLOWER else None)
            self._track.reset_for_acquisition(seed)
            self._gimbal_lock.reset()
            self._follower_candidates.reset()
            self._follower_target_bound = False
            self._follower_association = TargetAssociationResult("NO_REFERENCE")
            local = self._track.snapshot(now)

        self._update_compatibility_fields(now)
        if self._coordinator.phase != CoopCoordinator.SEARCH:
            self._search_route.pause()
            self._search_gimbal.suspend()
        commands = [broadcast(payload) for payload in update.payloads]
        if self._coordinator.phase not in (CoopCoordinator.INIT, CoopCoordinator.ACTIVE):
            self._gimbal_handoff.reset()
            self._competition_flight.reset()
            self._coop_flight_speed = None

        if self.finished:
            return commands + self._hold()

        if self._coordinator.phase == CoopCoordinator.EVADE:
            destination = self._coordinator.evade_destination
            if destination is None:
                destination = (obs.self.lat, obs.self.lon)
            lat, lon = _clamp_to_safebox(*destination)
            pan, tilt = self._tracking_gimbal(
                obs.self.lat, obs.self.lon, obs.self.heading_deg, lat, lon,
                self_alt=obs.self.alt)
            return commands + [
                fly_to(lat, lon, speed=22.0),
                point_gimbal(pan, tilt),
                set_gimbal_fov(self.SEARCH_FOV_DEG),
            ]

        local_position = self._track.position
        if self._coordinator.role == CoopCoordinator.FOLLOWER:
            self._candidate = local_position or self._coordinator.follow_position
        elif (self._coordinator.phase == CoopCoordinator.SEARCH
              and local_position is None):
            self._candidate = self._search_filter.focus_position
        else:
            self._candidate = local_position

        if (self._coordinator.phase == CoopCoordinator.ACTIVE
                and self._coordinator.role == CoopCoordinator.MASTER
                and self._coop_lock_valid
                and local.position is not None
                and local.last_seen == now
                and now - self._last_report_t >= self.TARGET_REPORT_PERIOD_S):
            self._last_report_t = now
            commands.append(report_target(*local.position))

        if self._coordinator.phase in (CoopCoordinator.INIT, CoopCoordinator.ACTIVE):
            active_fov = (self.COOP_ACTIVE_FOV_DEG
                          if self._coordinator.phase == CoopCoordinator.ACTIVE
                          else self.SEARCH_FOV_DEG)
            # 歧义和空检测不能指定新竞争者，交由飞行控制器短时沿用方向历史。
            primary_position = self._gimbal_lock.primary_association.competitor_position
            heading = self._competition_flight.heading(
                now, (self._coordinator.current_session, self._track.epoch),
                (obs.self.lat, obs.self.lon), obs.self.heading_deg, self._candidate,
                self._gimbal_lock.desired_visible, self._gimbal_lock.primary_matches,
                primary_position)
            # 持续朝目标加偏移的期望位置纠偏，用远航点执行该方向，避免自动盘旋。
            angle = math.radians(heading)
            waypoint = _move((obs.self.lat, obs.self.lon),
                             self.COOP_GUIDANCE_LOOKAHEAD_M * math.sin(angle),
                             self.COOP_GUIDANCE_LOOKAHEAD_M * math.cos(angle))
            turn = (heading - obs.self.heading_deg + 180.0) % 360.0 - 180.0
            self._coop_flight_speed = adjust_flight_speed(
                self._competition_flight.goal_distance_m or 0.0, turn)
            pan, tilt = self._gimbal_handoff.update(
                (self._coordinator.current_session, self._track.epoch), dt,
                (obs.self.lat, obs.self.lon), self._track.predict_position(now),
                obs.self.heading_deg, obs.self.gimbal_pan, obs.self.gimbal_tilt,
                self._gimbal_lock.desired_visible and self._track.last_seen == now)
            return commands + [
                fly_to(*waypoint, speed=self._coop_flight_speed, loiter_radius=0.0,
                       turn_direction="right" if turn >= 0.0 else "left"),
                point_gimbal(pan, tilt), set_gimbal_fov(active_fov),
            ]

        if self._candidate is not None:
            self._search_route.pause()
            lat, lon = self._search_route.clamp_position(self._candidate)
            pan, tilt = self._search_gimbal.observe_candidate(
                (obs.self.lat, obs.self.lon), self._candidate,
                obs.self.heading_deg, obs.self.gimbal_tilt)
            fov = (self._gimbal_lock.fov_deg if self._gimbal_lock.active
                   else self.SEARCH_FOV_DEG)
            commands.extend([
                fly_to(lat, lon, speed=22.0, loiter_radius=100.0),
                point_gimbal(pan, tilt),
                set_gimbal_fov(fov),
            ])
        else:
            lat, lon = self._search_route.target((obs.self.lat, obs.self.lon), obs.self.heading_deg)
            route_heading = _bearing_deg(obs.self.lat, obs.self.lon, lat, lon)
            pan, tilt = self._search_gimbal.scan(
                now, route_heading, obs.self.heading_deg,
                obs.self.gimbal_pan, obs.self.gimbal_tilt)
            turn = (route_heading
                    - obs.self.heading_deg + 180.0) % 360.0 - 180.0
            self._search_flight_speed = adjust_flight_speed(
                _haversine_m(obs.self.lat, obs.self.lon, lat, lon), turn,
                slow_distance_m=80.0, fast_distance_m=250.0)
            commands.extend([
                fly_to(lat, lon, speed=self._search_flight_speed, loiter_radius=0.0,
                       turn_direction="right" if turn >= 0.0 else "left"),
                point_gimbal(pan, tilt),
                set_gimbal_fov(self.SEARCH_FOV_DEG),
            ])
        return commands
