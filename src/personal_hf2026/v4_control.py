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
from .v4_flight import (SurveySearchRoute, SimpleCoopControl, follower_ready,
                        solve_ground_aim, visual_waypoint)
from .v4_gimbal import VisualGimbal
from .v4_motion import SingleEntityMotion
from .v4_position import RoughPosition
from .v3_simple_control import bearing_deg


STATES = ("SEARCH", "VERIFY", "CALLING", "FOLLOWER_APPROACH", "COOP_TRACK")
MEMBERS = ("20001", "20002", "20003")


class V4Control:
    def __init__(self, uid):
        self.uid = str(uid)
        self.state = "SEARCH"
        self.entity = EntityManager(uid)
        self.motion = SingleEntityMotion()
        self.gimbal = VisualGimbal()
        self.rough = RoughPosition()
        self.coord = V4Coordinator(uid)
        self.route = SurveySearchRoute(_BBOX, MEMBERS.index(self.uid), len(MEMBERS))
        self.search_gimbal = SearchGimbalController()
        self.coop_geometry = SimpleCoopControl()
        self.last_visual_box = None
        self.last_visual_size = None
        self.last_visual_pose = None
        self.last_frame_id = None
        self.events = []
        self.completed_sessions = 0
        self._last_coverage_s = -1e9
        self._last_report_s = -1e9

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
        self.route.pause()
        self._state("SEARCH", now, reason)

    def consume_visual(self, snapshot):
        """外层保证同一 frame_id 只调用一次。"""
        now = float(snapshot.source_sim_time)
        self.last_frame_id = snapshot.frame_id
        self._event("visual_frame", now, frame_id=snapshot.frame_id,
                    raw_count=len(snapshot.raw_yolo_objects),
                    effective_count=len(snapshot.effective_yolo_objects))
        if snapshot.error:
            self._event("perception_error", now, error=snapshot.error)
            return
        if self.state == "FOLLOWER_APPROACH" or (
                self.state == "COOP_TRACK" and self.coord.master_uid != self.uid):
            return
        entity, event = self.entity.update(snapshot.effective_yolo_objects,
                                           snapshot.image_size, now)
        if event:
            self._event(event, now, frame_id=snapshot.frame_id,
                        entity_id=entity.entity_id, entity_visible=entity.visible,
                        entity_bbox=entity.bbox_xyxy, entity_missing_s=entity.missing_s,
                        entity_observed_frames=entity.observed_frames)
        if event == "entity_created":
            self.motion.reset()
            self.rough.reset()
            self._state("VERIFY", now, "real_vehicle_entity_created")
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
                self._state("CALLING", now, "motion_moving")
                self.coord.set_master(entity.entity_id)
        if self.state in ("CALLING", "COOP_TRACK"):
            self.rough.update(entity.bbox_xyxy, snapshot.image_size, snapshot.source_pose)
            if self.state == "COOP_TRACK" and self.motion.decision == "STATIC":
                self.completed_sessions += 1
                self._return_search(now, "motion_static_completed", "DONE")

    def _own_position(self, obs):
        return float(obs.self.lat), float(obs.self.lon)

    def step(self, obs, now):
        """消息跨 tick 生效；每拍只从本机观测生成本机 commands。"""
        now = float(now)
        own = self._own_position(obs)
        pose = {key: float(getattr(obs.self, key)) for key in (
            "lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt",
            "gimbal_fov_deg")}
        incoming = self.coord.ingest(obs.comm_inbox, now, self.state)
        for kind in incoming:
            self._event("comm_in", now, message_kind=kind, session=self.coord.session)
            if kind == "INVITE" and self.state == "SEARCH":
                self._state("FOLLOWER_APPROACH", now, "invite_received")
                self.coord.queue_message("ACCEPT", now)
            elif kind == "READY" and self.state == "CALLING":
                self._state("COOP_TRACK", now, "follower_ready")
                self.coord.queue_message("START", now)
            elif kind == "START" and self.state == "FOLLOWER_APPROACH":
                self._state("COOP_TRACK", now, "start_received")
            elif kind in ("DONE", "CANCEL") and self.coord.master_uid != self.uid:
                self._return_search(now, kind.lower())
        if (self.state in ("FOLLOWER_APPROACH", "COOP_TRACK")
                and self.coord.master_uid != self.uid
                and now - self.coord.last_master_message_s > 5.0):
            self._return_search(now, "master_communication_timeout")
        self.coord.queue_message("H", now, position=own, state=self.state, period_s=1.0)
        commands = []
        if self.state == "SEARCH":
            self.route.observe_search_position(own)
            if now - self._last_coverage_s >= 0.5:
                self.route.coverage.observe(
                    now, own, pose["heading_deg"], pose["gimbal_pan"],
                    pose["gimbal_tilt"], pose["gimbal_fov_deg"])
                self._last_coverage_s = now
            target = self.route.target(own, pose["heading_deg"])
            heading = bearing_deg(own, target)
            delta = abs((heading - pose["heading_deg"] + 180.0) % 360.0 - 180.0)
            speed = 15.0 if delta > 45.0 else 22.0
            pan, tilt = self.search_gimbal.scan(
                now, heading, pose["heading_deg"], pose["gimbal_pan"], pose["gimbal_tilt"])
            commands.extend((fly_to(*target, alt=500.0, speed=speed, loiter_radius=0.0),
                             point_gimbal(pan, tilt), set_gimbal_fov(48.0)))
        else:
            self.route.pause()
            if self.state in ("VERIFY", "CALLING") or (
                    self.state == "COOP_TRACK" and self.coord.master_uid == self.uid):
                if self.last_visual_box is not None and self.last_visual_size is not None:
                    if self.state == "COOP_TRACK" and self.rough.position is not None:
                        orbit = self.coop_geometry.dual_orbit(
                            target_position=self.rough.position, now_s=now, role="MASTER")
                        target = orbit.fly_to_position
                    else:
                        target = visual_waypoint(self.last_visual_pose, self.last_visual_box,
                                                 self.last_visual_size, origin_position=own)
                    commands.append(fly_to(*target, alt=500.0, speed=22.0,
                                           loiter_radius=0.0))
                if self.gimbal.command_pending:
                    commands.append(point_gimbal(self.gimbal.pan, self.gimbal.tilt))
                    self.gimbal.command_pending = False
            else:
                target = self.coord.target
                if target is not None:
                    if self.state == "COOP_TRACK":
                        orbit = self.coop_geometry.dual_orbit(
                            target_position=target, now_s=now, role="FOLLOWER")
                        destination = orbit.fly_to_position
                    else:
                        destination = target
                    speed = 40.0 if self.state == "FOLLOWER_APPROACH" else 22.0
                    commands.append(fly_to(*destination, alt=500.0, speed=speed,
                                           loiter_radius=0.0))
                    aim = solve_ground_aim(own, pose["alt"], pose["heading_deg"], target)
                    commands.append(point_gimbal(aim.pan_deg, aim.tilt_deg))
                    if self.state == "FOLLOWER_APPROACH" and follower_ready(
                            own, self.coord.master_position, target):
                        self.coord.queue_message("READY", now, period_s=1.0)
            commands.append(set_gimbal_fov(48.0))
        if self.state == "CALLING":
            peer = self.coord.peers.get(self.coord.partner_uid)
            if (not self.coord.accepted and self.coord.partner_uid is not None
                    and (peer is None or now - peer[1] > 5.0
                         or peer[2] not in ("SEARCH", "FOLLOWER_APPROACH"))):
                self.coord.partner_uid = None
            if self.coord.partner_uid is None:
                self.coord.partner_uid = self.coord.select_partner(own, now)
            if (not self.coord.accepted and self.coord.partner_uid is not None
                    and self.rough.position is not None):
                self.coord.queue_message("INVITE", now, target=self.rough.position,
                                         period_s=1.0)
            if self.coord.accepted and self.rough.position is not None:
                self.coord.queue_message("TARGET", now, target=self.rough.position,
                                         position=own, period_s=0.5)
        elif self.state == "COOP_TRACK" and self.coord.master_uid == self.uid:
            target = self.rough.position
            if target is not None:
                self.coord.queue_message("TARGET", now, target=target,
                                         position=own, period_s=0.5)
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
