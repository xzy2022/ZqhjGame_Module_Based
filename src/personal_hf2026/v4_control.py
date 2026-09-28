# 修改时间：2026-09-28。
# 修改目的：限制三机同时发起的正式协同任务数量。
# 修改内容：按业务阶段、运行时中间分区和 UID 仲裁 MASTER，补足 CALLING 静止退出与冲突日志。
# 修改时间：2026-09-28。
# 修改目的：固定第三机搜索分区并保护从机入场与 READY 的双机距离。
# 修改内容：移除接管速度，入场复用心跳避碰，READY 增加当前与预测距离门槛。
# 修改时间：2026-09-28。
# 修改目的：让双机入场前后基于新鲜心跳提前分离并持续靠近粗目标。
# 修改内容：接入距离预测、邀请保护、对称避让、远半径回归和每拍日志。
# 修改时间：2026-09-28。
# 修改目的：让双机半径就位即可启动正式协同相位同步。
# 修改内容：移除 READY 相位门槛，记录首次就位几何及双方前视角实际配置。
# 修改时间：2026-09-28。
# 修改目的：让第三机固定35米每秒跟随双机，并让正式轨道协同等速同步相位。
# 修改内容：接入接管速度、主机相位模式、心跳带宽及 TARGET 模式传递和导引日志。
# 修改时间：2026-09-27。
# 修改目的：分开从机槽位捕获与入圆后的双机相位同步。
# 修改内容：捕获阶段使用限幅对置航点，跟踪阶段从既有位置消息选取双方速度并记录对应导引。
# 修改时间：2026-09-27。
# 修改目的：让双机协同飞行由实际位置和实际相位闭环引导。
# 修改内容：统一主从轨道导航、按编队几何判断 READY，并记录每拍导航几何。
# 修改时间：2026-09-26。
# 修改目的：将 V4 搜索与协同接入三机分区 Z 字规划器。
# 修改内容：传递心跳位置与协作角色，处理等待同伴和规划事件，并按固定分区筛选协作无人机。
# 修改时间：2026-09-24。
# 修改目的：确认对象期间继续沿搜索航线飞行，并限制确认阶段时长。
# 修改内容：VERIFY 复用 SEARCH 航点与飞行速度，满 2 秒未进入下一阶段即返回 SEARCH。
# 修改时间：2026-09-24。
# 修改目的：将搜索阶段的三帧候选与正式实体分开计数。
# 修改内容：候选框连续匹配三帧后才创建 Entity，之前不增加实体编号或实体观测帧数。
# 修改时间：2026-09-24。
# 修改目的：搜索时连续三帧确认同一真目标后才进入实体锁定阶段。
# 修改内容：SEARCH 保持航线飞行，候选匹配满三帧才切换 VERIFY，识别中断则重置计数。
# 修改时间：2026-09-24。
# 修改目的：让从机先按主机实时方位飞往目标对侧的外圈入口。
# 修改内容：从机入口半径设为 180 米并以 35 m/s 奔袭，协同旋转半径仍为 130 米。
# 修改时间：2026-09-24。
# 修改目的：让双机从对置槽位进入半径 130 米的同步绕目标轨道。
# 修改内容：主机按当前位置初始化相位，从机直飞对侧入口，并将协同飞行速度设为 35 m/s。
# 修改时间：2026-09-24。
# 修改目的：让协同跟踪阶段的主机向裁判上报实体粗坐标以参与定位评分。
# 修改内容：主机在 COOP_TRACK 中使用已有的 rough.position 每秒发送一次 report_target。
# 修改时间：2026-09-24。
# 修改目的：让从机远程奔袭使用比赛允许的最高飞行速度。
# 修改内容：将 FOLLOWER_APPROACH 的飞行速度设为 40 m/s，协同跟踪仍使用 22 m/s。
# 修改时间：2026-09-24。
# 修改目的：让云台冷却间隔基于实际视觉帧的仿真时间。
# 修改内容：传入快照的 source_sim_time，并仅在产生新纠偏时发送云台命令。
# 修改时间：2026-09-24。
# 修改目的：减少主从已建立会话后的重复邀请广播。
# 修改内容：主机收到 ACCEPT 后停止周期性发送 INVITE。
# 修改时间：2026-09-24。
# 修改目的：防止通信事件名称覆盖紧凑日志的记录类型字段。
# 修改内容：将具体通信类型存到 message_kind 并保留 kind=event。
# 修改时间：2026-09-24。
# 修改目的：保留现有搜索航线按本机实测视野更新覆盖网格的行为。
# 修改内容：搜索时记录实际位置并按半秒采样更新覆盖区域。
# 修改时间：2026-09-24。
# 修改目的：用五个业务状态独立编排 Agent4 的实体、动静和双机控制。
# 修改内容：把新帧消费、消息驱动转移及飞行云台命令集中在单一控制器。
"""Agent4 五状态业务控制。"""
from __future__ import annotations

import math

from competition.baselines.coop_distributed import _BBOX
from competition.sdk.core.commands import fly_to, point_gimbal, report_target, set_gimbal_fov

from .search_gimbal import SearchGimbalController
from .v4_coordination import V4Coordinator
from .v4_entity import EntityManager
from .v4_flight import (CoordinatedSweepRoute, follower_capture_guidance,
                        follower_orbit_guidance, pair_phase_geometry,
                        pair_safety_guidance, pair_safety_mode, phase_sync_mode,
                        radius_rejoin_guidance,
                        COOP_TARGET_SOFT_MAX_RADIUS_M, COOP_TARGET_HARD_MAX_RADIUS_M,
                        formation_ready, ground_distance_m, master_orbit_guidance,
                        solve_ground_aim, visual_waypoint)
from .v4_gimbal import VisualGimbal
from .v4_motion import SingleEntityMotion
from .v4_position import RoughPosition
from .v3_simple_control import bearing_deg


STATES = ("SEARCH", "VERIFY", "CALLING", "FOLLOWER_APPROACH", "COOP_TRACK")
MEMBERS = ("20001", "20002", "20003")
COOP_ORBIT_RADIUS_M = 130.0
# 正常轨道名义半径固定为130米，紧急避让允许暂时偏离。
COOP_SYNC_SPEED_MPS = 25.0
COOP_LOOKAHEAD_S = 1.5
COOP_ORBIT_DIRECTION = 1
COOP_CAPTURE_SPEED_MPS = 35.0
COOP_CAPTURE_MAX_PHASE_STEP_DEG = 30.0
COOP_READY_RADIUS_TOL_M = 40.0
COOP_READY_PAIR_DISTANCE_M = 230.0
COOP_READY_PREDICTED_DISTANCE_M = 220.0
PAIR_INVITE_MIN_DISTANCE_M = 250.0
SEARCH_CONFIRM_FRAMES = 3
VERIFY_TIMEOUT_S = 2.0
MASTER_STATES = ("CALLING", "COOP_TRACK_M")


def master_priority(uid, state, home_sector):
    """阶段先于分区；分区未知时双方均只按 UID 决胜。"""
    return (1 if state == "COOP_TRACK_M" else 0,
            1 if home_sector == 1 else 0, -int(uid))


def master_verdict(own_uid, own_state, own_sector,
                   peer_uid, peer_state, peer_sector):
    if own_state != peer_state:
        reason = ("peer_already_coop_track" if peer_state == "COOP_TRACK_M"
                  else "own_already_coop_track")
    elif own_sector is None or peer_sector is None:
        reason = "uid_fallback_before_sector_init"
        own_sector = peer_sector = None
    elif (own_sector == 1) != (peer_sector == 1):
        reason = "middle_priority"
    else:
        reason = "edge_uid_tiebreak" if own_sector != 1 else "middle_uid_tiebreak"
    own_rank = master_priority(own_uid, own_state, own_sector)
    peer_rank = master_priority(peer_uid, peer_state, peer_sector)
    return (str(own_uid) if own_rank > peer_rank else str(peer_uid)), reason


class V4Control:
    def __init__(self, uid):
        self.uid = str(uid)
        self.state = "SEARCH"
        self.entity = EntityManager(uid)
        self.motion = SingleEntityMotion()
        self.gimbal = VisualGimbal()
        self.rough = RoughPosition()
        self.coord = V4Coordinator(uid)
        self.route = CoordinatedSweepRoute(_BBOX, self.uid, MEMBERS)
        self.search_gimbal = SearchGimbalController()
        self.last_visual_box = None
        self.last_visual_size = None
        self.last_visual_pose = None
        self.last_frame_id = None
        self.events = []
        self.completed_sessions = 0
        self._last_coverage_s = -1e9
        self._last_report_s = -1e9
        self._search_candidate_box = None
        self._search_candidate_frames = 0
        self._verify_started_s = None
        self.last_own_position = None
        self.coop_guidance = None
        self._ready_logged_session = None
        self._pair_distance_sample = None
        self._pair_distance_rate_mps = None
        self._safety_mode = "NORMAL"
        self._pair_safety_session = None
        self._master_conflicts_seen = set()
        self._last_heartbeat_state = None

    def _event(self, name, now, **details):
        self.events.append({"event": name, "time": float(now), "uid": self.uid,
                            "state": self.state, **details})

    def pop_events(self):
        events, self.events = self.events, []
        return events

    def _state(self, state, now, reason):
        if self.state != state:
            old = self.state
            self.state = state
            self._verify_started_s = None
            if old == "SEARCH":
                self._search_candidate_box = None
                self._search_candidate_frames = 0
            self._event("state_changed", now, previous=old, reason=reason)

    def _return_search(self, now, reason, terminal=None):
        if terminal:
            self.coord.queue_message(terminal, now)
        self.coord.clear_session()
        self.entity.clear()
        self.motion.reset()
        self.rough.reset()
        self.gimbal.reset()
        self.last_visual_box = None
        self.coop_guidance = None
        self._ready_logged_session = None
        self._pair_distance_sample = None
        self._pair_distance_rate_mps = None
        self._safety_mode = "NORMAL"
        self._pair_safety_session = None
        self._search_candidate_box = None
        self._search_candidate_frames = 0
        if self.state != "VERIFY":
            self.route.pause()
        self._state("SEARCH", now, reason)

    def _fresh_masters(self, now):
        return [(uid, state, now - seen) for uid, (_, seen, state) in self.coord.peers.items()
                if uid != self.uid and state in MASTER_STATES
                and 0.0 <= now - seen <= 5.0]

    def _master_details(self, own_state, peer_uid, peer_state, peer_age_s, winner_uid, reason):
        sectors = self.route.sector_by_uid if self.route.initialized else {}
        return {"own_uid": self.uid, "own_state": own_state,
                "own_sector": sectors.get(self.uid), "peer_uid": peer_uid,
                "peer_state": peer_state, "peer_sector": sectors.get(peer_uid),
                "winner_uid": winner_uid, "reason": reason,
                "peer_master_age_s": peer_age_s}

    def _master_conflicts(self, now, own_state):
        sectors = self.route.sector_by_uid if self.route.initialized else {}
        for peer_uid, peer_state, age in self._fresh_masters(now):
            winner, reason = master_verdict(
                self.uid, own_state, sectors.get(self.uid),
                peer_uid, peer_state, sectors.get(peer_uid))
            yield peer_uid, peer_state, age, winner, reason

    def _arbitrate_master(self, now):
        own_state = self._heartbeat_state()
        if own_state not in MASTER_STATES:
            self._master_conflicts_seen.clear()
            return
        conflicts = list(self._master_conflicts(now, own_state))
        current = {(uid, state) for uid, state, _, _, _ in conflicts}
        self._master_conflicts_seen.intersection_update(current)
        for peer_uid, peer_state, age, winner, reason in conflicts:
            key = (peer_uid, peer_state)
            details = self._master_details(own_state, peer_uid, peer_state,
                                           age, winner, reason)
            if key not in self._master_conflicts_seen:
                self._event("master_conflict_detected", now, **details)
                self._event("master_conflict_won" if winner == self.uid
                            else "master_conflict_yielded", now, **details)
                self._master_conflicts_seen.add(key)
            if winner != self.uid:
                self._return_search(now, "yield_to_active_master", "CANCEL")
                self._master_conflicts_seen.clear()
                return

    def _update_search_candidate(self, objects, image_size):
        real = [item for item in objects if item.class_name == "real_vehicle"]
        if not real:
            self._search_candidate_box = None
            self._search_candidate_frames = 0
            return None
        center = lambda box: ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
        last_box = self._search_candidate_box
        if last_box is None:
            width, height = image_size
            candidate = min(real, key=lambda item: math.dist(
                center(item.bbox_xyxy), (width * 0.5, height * 0.5)))
            self._search_candidate_frames = 1
        else:
            last_center = center(last_box)
            candidate = min(real, key=lambda item: math.dist(
                center(item.bbox_xyxy), last_center))
            diagonal = math.hypot(last_box[2] - last_box[0], last_box[3] - last_box[1])
            if math.dist(center(candidate.bbox_xyxy), last_center) <= max(220.0, 1.5 * diagonal):
                self._search_candidate_frames += 1
            else:
                width, height = image_size
                candidate = min(real, key=lambda item: math.dist(
                    center(item.bbox_xyxy), (width * 0.5, height * 0.5)))
                self._search_candidate_frames = 1
        self._search_candidate_box = candidate.bbox_xyxy
        return candidate if self._search_candidate_frames >= SEARCH_CONFIRM_FRAMES else None

    def consume_visual(self, snapshot):
        """外层保证同一 frame_id 只调用一次。"""
        now = float(snapshot.source_sim_time)
        self.last_frame_id = snapshot.frame_id
        self._event("visual_frame", now, frame_id=snapshot.frame_id,
                    raw_count=len(snapshot.raw_yolo_objects),
                    effective_count=len(snapshot.effective_yolo_objects))
        if snapshot.error:
            self._event("perception_error", now, error=snapshot.error)
            if self.state == "SEARCH":
                self._search_candidate_box = None
                self._search_candidate_frames = 0
            return
        if self.state == "FOLLOWER_APPROACH" or (
                self.state == "COOP_TRACK" and self.coord.master_uid != self.uid):
            return
        if self.state == "SEARCH":
            candidate = self._update_search_candidate(snapshot.effective_yolo_objects,
                                                      snapshot.image_size)
            if candidate is None:
                return
            entity, event = self.entity.update((candidate,), snapshot.image_size, now)
            self.motion.reset()
            self.rough.reset()
            self._state("VERIFY", now, "real_vehicle_three_frames")
        else:
            entity, event = self.entity.update(snapshot.effective_yolo_objects,
                                               snapshot.image_size, now)
        if event:
            self._event(event, now, frame_id=snapshot.frame_id,
                        entity_id=entity.entity_id, entity_visible=entity.visible,
                        entity_bbox=entity.bbox_xyxy, entity_missing_s=entity.missing_s,
                        entity_observed_frames=entity.observed_frames)
        if event == "entity_lost":
            self._return_search(now, "entity_lost",
                                "CANCEL" if self.state in ("CALLING", "COOP_TRACK") else None)
            return
        if entity is None or not entity.visible:
            return
        self.last_visual_box = entity.bbox_xyxy
        self.last_visual_size = snapshot.image_size
        self.last_visual_pose = dict(snapshot.source_pose)
        self.gimbal.update(entity.bbox_xyxy, snapshot.image_size, snapshot.source_pose, now)
        other = [item.bbox_xyxy for item in snapshot.effective_yolo_objects
                 if item.bbox_xyxy != entity.bbox_xyxy]
        previous_motion = self.motion.decision
        motion = self.motion.update(snapshot.image_bgr, entity.bbox_xyxy, other, now)
        if self.motion.decision != previous_motion:
            self._event("motion_changed", now, entity_id=entity.entity_id,
                        motion_decision=self.motion.decision, motion_evidence=motion)
        if self.state == "VERIFY":
            if self.motion.decision == "STATIC":
                self._return_search(now, "motion_static")
                return
            if entity.observed_frames >= 3 and self.motion.decision == "MOVING":
                blocked = next((conflict for conflict in
                                self._master_conflicts(now, "CALLING")
                                if conflict[3] != self.uid), None)
                if blocked is not None:
                    peer_uid, peer_state, age, winner, reason = blocked
                    details = self._master_details("CALLING", peer_uid, peer_state,
                                                   age, winner, reason)
                    self._event("master_candidate_blocked", now, **details,
                                blocking_master_uid=peer_uid,
                                blocking_master_state=peer_state,
                                blocking_sector=details["peer_sector"])
                    self._return_search(now, "foreign_master_active")
                    return
                self._state("CALLING", now, "motion_moving")
                self.coord.set_master(entity.entity_id)
        if self.state in ("CALLING", "COOP_TRACK"):
            self.rough.update(entity.bbox_xyxy, snapshot.image_size, snapshot.source_pose)
            if self.motion.decision == "STATIC":
                if self.state == "COOP_TRACK":
                    self.completed_sessions += 1
                    self._return_search(now, "motion_static_completed", "DONE")
                else:
                    self._return_search(now, "motion_static_while_calling", "CANCEL")

    def _own_position(self, obs):
        return float(obs.self.lat), float(obs.self.lon)

    def _heartbeat_state(self):
        if self.state == "COOP_TRACK":
            return "COOP_TRACK_M" if self.coord.master_uid == self.uid else "COOP_TRACK_F"
        return self.state

    def _safety_guidance(self, target, own, pose, now, destination, speed, guidance,
                         *, allow_radius_rejoin=True):
        """只使用五秒内同伴心跳，不以 TARGET 携带的旧机位触发避让。"""
        peer_uid = (self.coord.partner_uid if self.coord.master_uid == self.uid
                    else self.coord.master_uid)
        session_key = (self.coord.session, peer_uid)
        if session_key != self._pair_safety_session:
            self._pair_distance_sample = None
            self._pair_distance_rate_mps = None
            self._safety_mode = "NORMAL"
            self._pair_safety_session = session_key
        peer = self.coord.peers.get(peer_uid)
        peer_age = now - peer[1] if peer is not None else None
        radius = ground_distance_m(target, own)
        guidance.update({"peer_age_s": peer_age, "safety_peer_stale": True,
                         "safety_mode": "NORMAL", "pair_distance_rate_mps": None,
                         "pair_distance_m": None,
                         "predicted_pair_distance_m": None,
                         "rough_radius_m": radius,
                         "radius_error_m": COOP_ORBIT_RADIUS_M - radius,
                         "orbit_tangent": None, "away_from_peer": None,
                         "radial_weight": max(-1.0, min(1.0,
                             (COOP_ORBIT_RADIUS_M - radius) / 80.0)),
                         "desired_direction": None,
                         "desired_heading_deg": bearing_deg(own, destination),
                         "heading_error_deg": abs((bearing_deg(own, destination)
                             - pose["heading_deg"] + 180.0) % 360.0 - 180.0),
                         "target_soft_max_radius_m": COOP_TARGET_SOFT_MAX_RADIUS_M,
                         "target_hard_max_radius_m": COOP_TARGET_HARD_MAX_RADIUS_M})
        if peer is None or peer_age > 5.0 or peer_age < 0.0:
            self._pair_distance_sample = None
            self._pair_distance_rate_mps = None
            self._safety_mode = "NORMAL"
            return destination, speed, guidance
        peer_position, seen_time, _ = peer
        distance = ground_distance_m(own, peer_position)
        radial_angle = math.radians(bearing_deg(target, own))
        away_angle = math.radians(bearing_deg(peer_position, own))
        guidance.update({
            "orbit_tangent": (COOP_ORBIT_DIRECTION * math.cos(radial_angle),
                              -COOP_ORBIT_DIRECTION * math.sin(radial_angle)),
            "away_from_peer": (math.sin(away_angle), math.cos(away_angle)),
            "desired_direction": (math.sin(math.radians(guidance["desired_heading_deg"])),
                                  math.cos(math.radians(guidance["desired_heading_deg"]))),
        })
        previous = self._pair_distance_sample
        if previous is None or seen_time > previous[0]:
            self._pair_distance_rate_mps = (None if previous is None else
                (distance - previous[1]) / (seen_time - previous[0]))
            self._pair_distance_sample = (seen_time, distance)
        mode, predicted = pair_safety_mode(
            self._safety_mode, distance, self._pair_distance_rate_mps)
        self._safety_mode = mode
        guidance.update({"safety_peer_stale": False, "safety_mode": mode,
                         "pair_distance_m": distance,
                         "pair_distance_rate_mps": self._pair_distance_rate_mps,
                         "predicted_pair_distance_m": predicted})
        if (mode == "NORMAL" and allow_radius_rejoin
                and radius > COOP_TARGET_SOFT_MAX_RADIUS_M):
            destination, rejoin = radius_rejoin_guidance(
                target, own, pose["heading_deg"], radius_m=COOP_ORBIT_RADIUS_M,
                direction=COOP_ORBIT_DIRECTION)
            guidance.update(rejoin)
        elif mode != "NORMAL":
            destination, speed, safety = pair_safety_guidance(
                target, own, peer_position, pose["heading_deg"], mode, distance,
                radius_m=COOP_ORBIT_RADIUS_M, direction=COOP_ORBIT_DIRECTION)
            guidance.update(safety)
        return destination, speed, guidance

    def _record_coop_guidance(self, role, mode, target, own, destination, speed,
                              *, master=None, follower=None, guidance=None,
                              master_speed=None, follower_speed=None):
        # 两种阶段使用同一实际相位定义；缺少新鲜同伴位置时只记录本机轨道。
        geometry = (pair_phase_geometry(target, master, follower)
                    if master is not None and follower is not None else {})
        self.coop_guidance = {
            "role": role, "guidance_mode": mode, "target": target,
            "master_phase_deg": geometry.get("master_phase_deg", bearing_deg(target, own)),
            "follower_phase_deg": geometry.get("follower_phase_deg"),
            "own_phase_deg": bearing_deg(target, own),
            "desired_phase_deg": geometry.get("desired_phase_deg"),
            "phase_error_deg": geometry.get("phase_error_deg"),
            "master_radius_m": geometry.get("master_radius_m",
                                             ground_distance_m(target, own)),
            "follower_radius_m": geometry.get("follower_radius_m"),
            "own_radius_m": ground_distance_m(target, own),
            "pair_distance_m": (guidance.get("pair_distance_m") if guidance and
                                "pair_distance_m" in guidance else geometry.get("pair_distance_m")),
            "safety_mode": guidance.get("safety_mode") if guidance else None,
            "safety_peer_stale": guidance.get("safety_peer_stale") if guidance else None,
            "pair_distance_rate_mps": guidance.get("pair_distance_rate_mps") if guidance else None,
            "predicted_pair_distance_m": (guidance.get("predicted_pair_distance_m")
                                          if guidance else None),
            "peer_age_s": guidance.get("peer_age_s") if guidance else None,
            "rough_radius_m": guidance.get("rough_radius_m") if guidance else None,
            "radius_error_m": guidance.get("radius_error_m") if guidance else None,
            "radius_rejoin": guidance.get("radius_rejoin", False) if guidance else False,
            "orbit_tangent": guidance.get("orbit_tangent") if guidance else None,
            "away_from_peer": guidance.get("away_from_peer") if guidance else None,
            "radial_weight": guidance.get("radial_weight") if guidance else None,
            "desired_direction": guidance.get("desired_direction") if guidance else None,
            "desired_heading_deg": guidance.get("desired_heading_deg") if guidance else None,
            "heading_error_deg": guidance.get("heading_error_deg") if guidance else None,
            "target_soft_max_radius_m": (guidance.get("target_soft_max_radius_m")
                                          if guidance else None),
            "target_hard_max_radius_m": (guidance.get("target_hard_max_radius_m")
                                          if guidance else None),
            "long_slot": guidance.get("long_slot") if guidance else None,
            "capture_phase_step_deg": (guidance.get("capture_phase_step_deg")
                                       if guidance else None),
            "capture_phase_deg": guidance.get("capture_phase_deg") if guidance else None,
            "short_waypoint": destination,
            "commanded_speed_mps": speed,
            "master_command_speed_mps": master_speed,
            "follower_command_speed_mps": follower_speed,
            "own_command_speed_mps": speed,
            "sync_mode": guidance.get("sync_mode") if guidance else self.coord.sync_mode,
            "base_lookahead_deg": guidance.get("base_lookahead_deg") if guidance else None,
            "effective_lookahead_deg": (guidance.get("effective_lookahead_deg")
                                        if guidance else None),
            "master_effective_lookahead_deg": (guidance.get("master_effective_lookahead_deg")
                                               if guidance else None),
            "follower_effective_lookahead_deg": (guidance.get("follower_effective_lookahead_deg")
                                                 if guidance else None),
            "local_phase_error_deg": (guidance.get("local_phase_error_deg")
                                      if guidance else None),
            "lookahead_deg": guidance.get("lookahead_deg") if guidance else None,
        }

    def step(self, obs, now):
        """消息跨 tick 生效；每拍只从本机观测生成本机 commands。"""
        now = float(now)
        own = self._own_position(obs)
        self.last_own_position = own
        self.coop_guidance = None
        pose = {key: float(getattr(obs.self, key)) for key in (
            "lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt",
            "gimbal_fov_deg")}
        self.last_pose = pose
        incoming = self.coord.ingest(obs.comm_inbox, now, self.state)
        for kind in incoming:
            self._event("comm_in", now, message_kind=kind, session=self.coord.session)
            if kind == "INVITE" and self.state == "SEARCH":
                self._state("FOLLOWER_APPROACH", now, "invite_received")
                self.coord.queue_message("ACCEPT", now)
            elif kind == "READY" and self.state == "CALLING":
                target = self.rough.position
                if target is not None:
                    self._state("COOP_TRACK", now, "follower_ready")
                    self.coord.queue_message("START", now)
            elif kind == "START" and self.state == "FOLLOWER_APPROACH":
                self._state("COOP_TRACK", now, "start_received")
            elif kind in ("DONE", "CANCEL") and self.coord.master_uid != self.uid:
                self._return_search(now, kind.lower())
        self._arbitrate_master(now)
        if (self.state in ("FOLLOWER_APPROACH", "COOP_TRACK")
                and self.coord.master_uid != self.uid
                and now - self.coord.last_master_message_s > 5.0):
            self._return_search(now, "master_communication_timeout")
        if self.state == "VERIFY":
            if self._verify_started_s is None:
                self._verify_started_s = now
            elif now - self._verify_started_s >= VERIFY_TIMEOUT_S:
                self._return_search(now, "verify_timeout")
        heartbeat_state = self._heartbeat_state()
        self.coord.queue_message("H", now, position=own, state=heartbeat_state,
                                 period_s=(0.0 if heartbeat_state != self._last_heartbeat_state
                                           else 1.0))
        self._last_heartbeat_state = heartbeat_state
        commands = []
        if self.state in ("SEARCH", "VERIFY"):
            self.route.observe_search_position(own)
            if now - self._last_coverage_s >= 0.5:
                self.route.coverage.observe(
                    now, own, pose["heading_deg"], pose["gimbal_pan"],
                    pose["gimbal_tilt"], pose["gimbal_fov_deg"])
                self._last_coverage_s = now
            target = self.route.target(own, now, self.coord.peers)
            for name, details in self.route.drain_events():
                self._event(name, now, **details)
            heading = pose["heading_deg"]
            if target is not None:
                heading = bearing_deg(own, target)
                delta = abs((heading - pose["heading_deg"] + 180.0) % 360.0 - 180.0)
                speed = 15.0 if delta > 45.0 else 22.0
                commands.append(fly_to(*target, alt=500.0, speed=speed, loiter_radius=0.0))
            if self.state == "SEARCH":
                pan, tilt = self.search_gimbal.scan(
                    now, heading, pose["heading_deg"], pose["gimbal_pan"], pose["gimbal_tilt"])
                commands.append(point_gimbal(pan, tilt))
            elif self.gimbal.command_pending:
                commands.append(point_gimbal(self.gimbal.pan, self.gimbal.tilt))
                self.gimbal.command_pending = False
            commands.append(set_gimbal_fov(48.0))
        else:
            self.route.pause()
            if self.state == "CALLING" or (
                    self.state == "COOP_TRACK" and self.coord.master_uid == self.uid):
                target = self.rough.position
                if target is not None:
                    peer = self.coord.peers.get(self.coord.partner_uid)
                    follower = (peer[0] if peer is not None
                                and now - peer[1] <= 5.0 else None)
                    master_speed = follower_speed = COOP_SYNC_SPEED_MPS
                    self.coord.sync_mode = "N"
                    if self.state == "COOP_TRACK" and follower is not None:
                        geometry = pair_phase_geometry(target, own, follower)
                        self.coord.sync_mode = phase_sync_mode(geometry["phase_error_deg"])
                    destination, speed, guidance = master_orbit_guidance(
                        target, own, radius_m=COOP_ORBIT_RADIUS_M,
                        speed_mps=master_speed,
                        lookahead_s=COOP_LOOKAHEAD_S,
                        direction=COOP_ORBIT_DIRECTION, sync_mode=self.coord.sync_mode)
                    guidance["sync_mode"] = self.coord.sync_mode
                    guidance["master_effective_lookahead_deg"] = guidance["effective_lookahead_deg"]
                    guidance["follower_effective_lookahead_deg"] = guidance["base_lookahead_deg"]
                    mode = "ORBIT_WAIT"
                    if (self.state == "COOP_TRACK" or
                            (self.state == "CALLING" and self.coord.accepted
                             and self.coord.partner_uid is not None)):
                        destination, speed, guidance = self._safety_guidance(
                            target, own, pose, now, destination, speed, guidance)
                        mode = (guidance["safety_mode"] if self.state == "COOP_TRACK" else
                                "PRE_ENTRY_" + guidance["safety_mode"])
                    commands.append(fly_to(*destination, alt=500.0, speed=speed,
                                           loiter_radius=0.0))
                    self._record_coop_guidance(
                        "MASTER", mode,
                        target, own, destination, speed, master=own, follower=follower,
                        guidance=guidance, master_speed=master_speed,
                        follower_speed=follower_speed)
                elif self.last_visual_box is not None and self.last_visual_size is not None:
                    destination = visual_waypoint(self.last_visual_pose, self.last_visual_box,
                                                  self.last_visual_size, origin_position=own)
                    if destination is not None:
                        commands.append(fly_to(*destination, alt=500.0,
                                               speed=COOP_SYNC_SPEED_MPS,
                                               loiter_radius=0.0))
                if self.gimbal.command_pending:
                    commands.append(point_gimbal(self.gimbal.pan, self.gimbal.tilt))
                    self.gimbal.command_pending = False
            else:
                target = self.coord.target
                if target is not None:
                    master = self.coord.master_position
                    if master is not None:
                        if self.state == "FOLLOWER_APPROACH":
                            destination, speed, guidance = follower_capture_guidance(
                                target, master, own, radius_m=COOP_ORBIT_RADIUS_M,
                                speed_mps=COOP_CAPTURE_SPEED_MPS,
                                max_phase_step_deg=COOP_CAPTURE_MAX_PHASE_STEP_DEG)
                            master_speed, follower_speed = None, speed
                            destination, speed, guidance = self._safety_guidance(
                                target, own, pose, now, destination, speed, guidance,
                                allow_radius_rejoin=False)
                            mode = ("CAPTURE_SLOT" if guidance["safety_mode"] == "NORMAL"
                                    else "PRE_ENTRY_" + guidance["safety_mode"])
                        else:
                            destination, speed, guidance = follower_orbit_guidance(
                                target, master, own, radius_m=COOP_ORBIT_RADIUS_M,
                                speed_mps=COOP_SYNC_SPEED_MPS,
                                sync_mode=self.coord.sync_mode,
                                lookahead_s=COOP_LOOKAHEAD_S,
                                direction=COOP_ORBIT_DIRECTION)
                            guidance["master_effective_lookahead_deg"] = guidance["base_lookahead_deg"]
                            guidance["follower_effective_lookahead_deg"] = guidance["effective_lookahead_deg"]
                            master_speed = guidance["master_command_speed_mps"]
                            follower_speed = guidance["follower_command_speed_mps"]
                            destination, speed, guidance = self._safety_guidance(
                                target, own, pose, now, destination, speed, guidance)
                            mode = guidance["safety_mode"]
                        commands.append(fly_to(*destination, alt=500.0, speed=speed,
                                               loiter_radius=0.0))
                        self._record_coop_guidance(
                            "FOLLOWER", mode, target, own, destination, speed,
                            master=master, follower=own, guidance=guidance,
                            master_speed=master_speed, follower_speed=follower_speed)
                        peer = self.coord.peers.get(self.coord.master_uid)
                        fresh_master = (peer[0] if peer is not None
                                        and 0.0 <= now - peer[1] <= 5.0 else None)
                        if (self.state == "FOLLOWER_APPROACH"
                                and fresh_master is not None
                                and formation_ready(
                                    target, fresh_master, own,
                                    radius_m=COOP_ORBIT_RADIUS_M,
                                    radius_tol_m=COOP_READY_RADIUS_TOL_M,
                                    pair_distance_min_m=COOP_READY_PAIR_DISTANCE_M)
                                and guidance["predicted_pair_distance_m"] is not None
                                and guidance["predicted_pair_distance_m"]
                                    >= COOP_READY_PREDICTED_DISTANCE_M):
                            if self._ready_logged_session != self.coord.session:
                                geometry = pair_phase_geometry(target, fresh_master, own)
                                self._event("formation_ready", now, **{
                                    key: geometry[key] for key in (
                                        "master_radius_m", "follower_radius_m",
                                        "pair_distance_m", "phase_error_deg")},
                                            pair_distance_rate_mps=guidance["pair_distance_rate_mps"],
                                            predicted_pair_distance_m=guidance["predicted_pair_distance_m"])
                                self._ready_logged_session = self.coord.session
                            self.coord.queue_message("READY", now, period_s=1.0)
                    aim = solve_ground_aim(own, pose["alt"], pose["heading_deg"], target)
                    commands.append(point_gimbal(aim.pan_deg, aim.tilt_deg))
            commands.append(set_gimbal_fov(48.0))
        if self.state == "CALLING":
            peer = self.coord.peers.get(self.coord.partner_uid)
            if (not self.coord.accepted and self.coord.partner_uid is not None
                    and (peer is None or now - peer[1] > 5.0
                         or peer[2] not in ("SEARCH", "FOLLOWER_APPROACH"))):
                self.coord.partner_uid = None
            if self.coord.partner_uid is None:
                self.coord.partner_uid = self.coord.select_partner(
                    own, now, allowed_uids=self.route.preferred_partner_uids(),
                    min_distance_m=PAIR_INVITE_MIN_DISTANCE_M)
            if (not self.coord.accepted and self.coord.partner_uid is not None
                    and self.rough.position is not None):
                self.coord.queue_message("INVITE", now, target=self.rough.position,
                                         period_s=1.0)
            if self.coord.accepted and self.rough.position is not None:
                self.coord.queue_message("TARGET", now, target=self.rough.position,
                                         position=own, period_s=0.5,
                                         sync_mode=self.coord.sync_mode)
        elif self.state == "COOP_TRACK" and self.coord.master_uid == self.uid:
            target = self.rough.position
            if target is not None:
                self.coord.queue_message("TARGET", now, target=target,
                                         position=own, period_s=1.0,
                                         sync_mode=self.coord.sync_mode)
                if now - self._last_report_s >= 1.0:
                    commands.append(report_target(*target))
                    self._last_report_s = now
            self.coord.queue_message("START", now, period_s=1.0)
        elif self.state == "FOLLOWER_APPROACH":
            self.coord.queue_message("ACCEPT", now, period_s=1.0)
        command, kind = self.coord.emit(now)
        if command is not None:
            commands.append(command)
            self._event("comm_out", now, message_kind=kind, session=self.coord.session)
        return commands
