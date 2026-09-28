# 修改时间：2026-09-28。
# 修改目的：验证近距离合法同伴被预约但在达到安全距离前不会收到邀请。
# 修改内容：将旧的近机排除断言改为预约与邀请分离断言。
# 修改时间：2026-09-28。
# 修改目的：验证从机入场避碰和有距离裕度的 READY 条件。
# 修改内容：覆盖正常、分离、紧急、恢复、预测门槛及不同会话的距离历史。
# 修改时间：2026-09-28。
# 修改目的：验证双机入场前后的距离预测、分离方向、滞回及相位诊断隔离。
# 修改内容：替换旧相位断言并覆盖邀请距离、入场保护、安全速度、远半径回归、心跳和既有状态链。
# 修改时间：2026-09-28。
# 修改目的：验证径向 READY 状态链和反向前视角相位候选。
# 修改内容：覆盖就位与释放，并校验 B 方向的施加对象和等速命令。
# 修改时间：2026-09-28。
# 修改目的：验证主机模式广播和双机等速前视角相位同步。
# 修改内容：覆盖 N/F/M 前视角、固定25米每秒、目标载荷及正反相位动态趋势。
# 修改时间：2026-09-27。
# 修改目的：验证从机槽位捕获及双机相位同步在连续运动中收敛。
# 修改内容：增加正反限幅捕获、主机新鲜心跳变速和移动目标点质量测试。
# 修改时间：2026-09-27。
# 修改目的：验证双机协同从实际位置生成轨道航点和对置速度。
# 修改内容：覆盖固定半径、前视角、主从相位、READY 几何和无时钟 START。
"""V4 双机轨道导航的聚焦测试。"""

import math
import unittest
from types import SimpleNamespace

from personal_hf2026.v3_simple_control import bearing_deg, ground_distance_m, offset_position
from personal_hf2026 import v4_control
from personal_hf2026.v4_control import V4Control
from personal_hf2026.v4_coordination import V4Coordinator
from personal_hf2026.v4_flight import (formation_ready, follower_capture_guidance,
                                       follower_orbit_guidance, master_orbit_guidance,
                                       orbit_point, orbit_short_waypoint,
                                       pair_phase_geometry, pair_safety_guidance,
                                       pair_safety_mode, phase_sync_mode,
                                       radius_rejoin_guidance, wrap180)


TARGET = (27.025, 125.020)


def at(phase, radius=130.0):
    return orbit_point(TARGET, phase, radius)


def observation(position):
    own = SimpleNamespace(lat=position[0], lon=position[1], alt=500.0,
                          heading_deg=0.0, gimbal_pan=0.0, gimbal_tilt=-45.0,
                          gimbal_fov_deg=48.0)
    return SimpleNamespace(self=own, comm_inbox=[])


def move_toward(position, waypoint, distance_m):
    distance = ground_distance_m(position, waypoint)
    heading = math.radians(bearing_deg(position, waypoint))
    step = min(distance_m, distance)
    return offset_position(position, step * math.sin(heading), step * math.cos(heading))


class OrbitGuidanceTest(unittest.TestCase):
    def test_orbit_point_radius_is_fixed_130(self):
        for phase in (0.0, 40.0, 180.0, 359.0):
            self.assertAlmostEqual(ground_distance_m(TARGET, at(phase)), 130.0,
                                   delta=0.2)

    def test_short_waypoint_uses_actual_phase(self):
        own = at(40.0)
        waypoint, theta, base, lookahead = orbit_short_waypoint(TARGET, own, 25.0)
        self.assertEqual(base, lookahead)
        self.assertAlmostEqual(wrap180(theta - 40.0), 0.0, delta=0.2)
        self.assertAlmostEqual(wrap180(bearing_deg(TARGET, waypoint)
                                       - theta - lookahead), 0.0, delta=0.2)
        self.assertAlmostEqual(ground_distance_m(TARGET, waypoint), 130.0,
                               delta=0.2)

    def test_lookahead_matches_speed(self):
        angles = []
        for speed in (15.0, 25.0, 35.0):
            _, _, angle, effective = orbit_short_waypoint(TARGET, at(40.0), speed)
            self.assertAlmostEqual(angle, math.degrees(speed / 130.0 * 1.5))
            self.assertEqual(angle, effective)
            angles.append(angle)
        self.assertEqual(angles, sorted(angles))

    def test_master_uses_actual_phase_and_base_speed(self):
        destination, speed, debug = master_orbit_guidance(TARGET, at(40.0))
        self.assertEqual(speed, 25.0)
        self.assertAlmostEqual(wrap180(bearing_deg(TARGET, destination)
                                       - 40.0 - math.degrees(25.0 / 130.0 * 1.5)),
                               0.0, delta=0.2)
        self.assertAlmostEqual(debug["radius_m_actual"], 130.0, delta=0.2)

    def test_capture_uses_opposite_slot_in_both_directions(self):
        for own_phase, expected_phase, expected_step in ((60.0, 90.0, 30.0),
                                                           (260.0, 230.0, -30.0)):
            destination, speed, debug = follower_capture_guidance(
                TARGET, at(0.0), at(own_phase, 650.0))
            self.assertEqual(speed, 35.0)
            self.assertAlmostEqual(debug["capture_phase_step_deg"], expected_step,
                                   delta=0.2)
            self.assertAlmostEqual(wrap180(bearing_deg(TARGET, destination)
                                           - expected_phase), 0.0, delta=0.2)
            self.assertAlmostEqual(ground_distance_m(TARGET, destination), 130.0,
                                   delta=0.2)
            self.assertGreater(ground_distance_m(destination, debug["long_slot"]), 1.0)

    def test_capture_waypoint_radius_for_different_starting_radii(self):
        for radius in (100.0, 300.0, 800.0):
            destination, _, _ = follower_capture_guidance(
                TARGET, at(0.0), at(60.0, radius))
            self.assertAlmostEqual(ground_distance_m(TARGET, destination), 130.0,
                                   delta=0.2)

    def test_phase_sync_modes_and_equal_speeds(self):
        for error, mode in ((60, "F"), (-60, "M"), (20, "N"), (-20, "N")):
            self.assertEqual(phase_sync_mode(error), mode)
            _, master_speed, master = master_orbit_guidance(
                TARGET, at(0.0), sync_mode=mode)
            _, follower_speed, follower = follower_orbit_guidance(
                TARGET, at(0.0), at(180.0), sync_mode=mode)
            self.assertEqual((master_speed, follower_speed), (25.0, 25.0))
            m_angle = master["effective_lookahead_deg"]
            f_angle = follower["effective_lookahead_deg"]
            self.assertEqual(m_angle, f_angle)
            self.assertEqual(master["base_lookahead_deg"], m_angle)
            self.assertEqual(follower["base_lookahead_deg"], f_angle)

    def test_follower_behind_does_not_change_lookahead(self):
        _, speed, debug = follower_orbit_guidance(
            TARGET, at(40.0), at(180.0), sync_mode="F")
        self.assertGreater(debug["phase_error_deg"], 20.0)
        self.assertEqual(speed, 25.0)
        self.assertEqual(debug["effective_lookahead_deg"], debug["base_lookahead_deg"])

    def test_follower_ahead_does_not_change_lookahead(self):
        _, speed, debug = follower_orbit_guidance(
            TARGET, at(40.0), at(260.0), sync_mode="M")
        self.assertLess(debug["phase_error_deg"], -20.0)
        self.assertEqual(speed, 25.0)
        self.assertEqual(debug["effective_lookahead_deg"], debug["base_lookahead_deg"])

    def test_follower_near_opposite_uses_base_speed(self):
        for phase in (205.0, 220.0, 235.0):
            _, speed, debug = follower_orbit_guidance(TARGET, at(40.0), at(phase))
            self.assertLessEqual(abs(debug["phase_error_deg"]), 20.0)
            self.assertEqual(speed, 25.0)

    def test_follower_sync_waypoint_uses_own_phase(self):
        destination, _, debug = follower_orbit_guidance(TARGET, at(40.0), at(180.0))
        self.assertEqual(destination, debug["short_waypoint"])
        self.assertAlmostEqual(ground_distance_m(TARGET, destination), 130.0,
                               delta=0.2)
        self.assertEqual(debug["master_command_speed_mps"], 25.0)

    def test_continuous_capture_and_sync_moving_target(self):
        target = TARGET
        master = orbit_point(target, 0.0)
        follower = orbit_point(target, 60.0, 650.0)
        ready_at = None
        for tick in range(900):
            target = offset_position(target, 1.0, 0.0)
            m_dest, m_speed, _ = master_orbit_guidance(target, master)
            f_dest, f_speed, _ = follower_capture_guidance(target, master, follower)
            master = move_toward(master, m_dest, m_speed * 0.1)
            follower = move_toward(follower, f_dest, f_speed * 0.1)
            if formation_ready(target, master, follower):
                ready_at = tick * 0.1
                break
        self.assertIsNotNone(ready_at)
        geometry = pair_phase_geometry(target, master, follower)
        self.assertLess(abs(geometry["follower_radius_m"] - 130.0), 40.0)
        self.assertLess(abs(geometry["master_radius_m"] - 130.0), 40.0)

        # 入圆后允许相位误差存在，正式控制只依据实际机间距。
        self.assertIn(pair_safety_mode("NORMAL", geometry["pair_distance_m"], 0.0)[0],
                      ("NORMAL", "SEPARATE", "EMERGENCY"))

    def test_ready_requires_radial_and_pair_distance(self):
        master = at(40.0)
        follower = at(220.0)
        self.assertTrue(formation_ready(TARGET, master, follower))
        self.assertFalse(formation_ready(TARGET, master, at(100.0)))
        self.assertFalse(formation_ready(TARGET, at(0.0), at(108.0)))
        self.assertFalse(formation_ready(TARGET, master, at(220.0, 180.0)))
        self.assertTrue(formation_ready(TARGET, at(0.0, 170.0), at(130.0, 170.0)))
        self.assertFalse(formation_ready(TARGET, at(40.0, 180.0), follower))

        for master_position, own_position, expected in (
                (master, follower, True),
                (master, at(100.0), False),
                (at(40.0, 180.0), follower, False),
                (master, at(220.0, 180.0), False)):
            control = V4Control("20002")
            control.state = "FOLLOWER_APPROACH"
            control.coord.session = "session"
            control.coord.master_uid = "20001"
            control.coord.target = TARGET
            control.coord.master_position = master_position
            control.coord.last_master_message_s = 10.0
            control.coord.peers["20001"] = (master_position, 10.0, "CALLING")
            control.step(observation(own_position), 10.0)
            self.assertEqual("READY" in control.coord.last_queued, expected)

    def test_large_phase_ready_starts_coop_and_report(self):
        master = V4Control("20001")
        follower = V4Control("20002")
        master.state = "CALLING"
        follower.state = "FOLLOWER_APPROACH"
        for control in (master, follower):
            control.coord.session = "session"
            control.coord.master_uid = "20001"
        master.coord.partner_uid = "20002"
        master.coord.accepted = True
        master.rough.points.append(TARGET)
        follower.coord.target = TARGET
        follower.coord.master_position = at(0.0, 170.0)
        follower.coord.last_master_message_s = 10.0
        follower.coord.peers["20001"] = (at(0.0, 170.0), 10.0, "CALLING")

        follower_obs = observation(at(130.0, 170.0))
        follower_commands = follower.step(follower_obs, 10.0)
        ready_event = next(event for event in follower.pop_events()
                           if event["event"] == "formation_ready")
        self.assertAlmostEqual(ready_event["phase_error_deg"], 50.0, delta=0.2)
        for key in ("master_radius_m", "follower_radius_m", "pair_distance_m"):
            self.assertIsNotNone(ready_event[key])
        self.assertEqual(follower.state, "FOLLOWER_APPROACH")
        ready = next(command for command in follower_commands
                     if command.verb == "comm.broadcast")
        self.assertEqual(ready.params["payload"], "V4|READY|session")

        master_obs = observation(at(0.0, 170.0))
        master_obs.comm_inbox = [SimpleNamespace(
            sender_uid="20002", payload="V4|READY|session", recv_time=10.1)]
        master_commands = master.step(master_obs, 10.1)
        self.assertEqual(master.state, "COOP_TRACK")
        self.assertTrue(any(command.verb == "agent.report" for command in master_commands))
        self.assertTrue(any(command.verb == "comm.broadcast" and
                            command.params["payload"] == "V4|START|session"
                            for command in master_commands))

        follower_obs.comm_inbox = [SimpleNamespace(
            sender_uid="20001", payload="V4|START|session", recv_time=10.2)]
        follower.step(follower_obs, 10.2)
        self.assertEqual(follower.state, "COOP_TRACK")

    def test_coop_track_static_completion_returns_to_search(self):
        control = V4Control("20001")
        control.state = "COOP_TRACK"
        control.coord.session = "session"
        control.coord.master_uid = "20001"
        control.motion.decision = "MOVING"
        entity = SimpleNamespace(entity_id="uav_20001_entity_1",
                                 bbox_xyxy=(100, 100, 130, 130), visible=True,
                                 missing_s=0.0, observed_frames=5)
        control.entity.update = lambda objects, size, now: (entity, None)
        control.gimbal.update = lambda *args: None
        control.rough.update = lambda *args: None

        def mark_static(*args):
            control.motion.decision = "STATIC"
            return {}

        control.motion.update = mark_static
        snapshot = SimpleNamespace(
            source_sim_time=20.0, frame_id="static-frame", error=None,
            effective_yolo_objects=[SimpleNamespace(bbox_xyxy=entity.bbox_xyxy)],
            raw_yolo_objects=[], image_size=(640, 480), source_pose={}, image_bgr=None)
        control.consume_visual(snapshot)
        self.assertEqual(control.completed_sessions, 1)
        self.assertEqual(control.state, "SEARCH")
        self.assertIn("DONE", [kind for kind, _ in control.coord.queue])
        self.assertTrue(any(event["event"] == "state_changed" and
                            event["reason"] == "motion_static_completed"
                            for event in control.pop_events()))

    def test_start_has_no_phase_clock(self):
        master = V4Coordinator("20001")
        master.session = "session"
        master.queue_message("START", 10.0)
        self.assertEqual(master.queue[-1][1], "V4|START|session")
        follower = V4Coordinator("20002")
        follower.session = "session"
        follower.master_uid = "20001"
        inbox = [SimpleNamespace(sender_uid="20001", payload=master.queue[-1][1],
                                 recv_time=10.1)]
        self.assertEqual(follower.ingest(inbox, 10.1, "FOLLOWER_APPROACH"), ["START"])
        self.assertTrue(follower.started)

    def test_no_time_based_orbit_state(self):
        control = V4Control("20001")
        for name in ("_orbit_start_s", "_orbit_phase_deg", "_entry_phase_deg"):
            self.assertFalse(hasattr(control, name))
        coordinator = V4Coordinator("20001")
        for name in ("orbit_start_s", "orbit_phase_deg"):
            self.assertFalse(hasattr(coordinator, name))
        self.assertFalse(hasattr(v4_control, "COOP_ORBIT_PERIOD_S"))
        self.assertFalse(hasattr(v4_control, "FOLLOWER_ENTRY_RADIUS_M"))

    def test_coop_states_send_only_short_waypoints(self):
        for state in ("CALLING", "COOP_TRACK"):
            control = V4Control("20001")
            control.state = state
            control.coord.session = "session"
            control.coord.master_uid = "20001"
            control.rough.points.append(TARGET)
            commands = control.step(observation(at(40.0)), 10.0)
            flight = next(command for command in commands
                          if command.verb == "set_destination")
            short = control.coop_guidance["short_waypoint"]
            self.assertAlmostEqual(flight.params["latitude"], short[0])
            self.assertAlmostEqual(flight.params["longitude"], short[1])
            self.assertEqual(flight.params["speed"], 25.0)

        for state in ("FOLLOWER_APPROACH", "COOP_TRACK"):
            control = V4Control("20002")
            control.state = state
            control.coord.session = "session"
            control.coord.master_uid = "20001"
            control.coord.target = TARGET
            control.coord.master_position = at(40.0)
            control.coord.last_master_message_s = 10.0
            commands = control.step(observation(at(180.0)), 10.0)
            flight = next(command for command in commands
                          if command.verb == "set_destination")
            short = control.coop_guidance["short_waypoint"]
            long_slot = control.coop_guidance["long_slot"]
            self.assertAlmostEqual(flight.params["latitude"], short[0])
            self.assertAlmostEqual(flight.params["longitude"], short[1])
            if state == "FOLLOWER_APPROACH":
                self.assertEqual(control.coop_guidance["guidance_mode"], "CAPTURE_SLOT")
                self.assertAlmostEqual(wrap180(bearing_deg(TARGET, short) - 210.0),
                                       0.0, delta=0.2)
                self.assertGreater(ground_distance_m(short, long_slot), 1.0)
                self.assertEqual(flight.params["speed"], 35.0)
            else:
                self.assertEqual(control.coop_guidance["guidance_mode"], "NORMAL")
                self.assertEqual(flight.params["speed"], 25.0)

    def test_master_sync_uses_fresh_follower_heartbeat(self):
        control = V4Control("20001")
        control.state = "COOP_TRACK"
        control.coord.session = "session"
        control.coord.master_uid = "20001"
        control.coord.partner_uid = "20002"
        control.rough.points.append(TARGET)
        control.coord.peers["20002"] = (at(100.0), 10.0, "COOP_TRACK_F")
        commands = control.step(observation(at(0.0)), 10.0)
        flight = next(item for item in commands if item.verb == "set_destination")
        self.assertLessEqual(flight.params["speed"], 35.0)
        self.assertEqual(control.coop_guidance["follower_command_speed_mps"], 25.0)
        self.assertEqual(control.coop_guidance["sync_mode"], "F")
        self.assertEqual(control.coop_guidance["master_effective_lookahead_deg"],
                         control.coop_guidance["follower_effective_lookahead_deg"])
        self.assertEqual(control.coop_guidance["guidance_mode"], "EMERGENCY")
        control.coord.peers["20002"] = (at(100.0), 4.0, "COOP_TRACK_F")
        commands = control.step(observation(at(0.0)), 10.1)
        flight = next(item for item in commands if item.verb == "set_destination")
        self.assertEqual(flight.params["speed"], 25.0)
        self.assertEqual(control.coop_guidance["sync_mode"], "N")
        self.assertTrue(control.coop_guidance["safety_peer_stale"])

    def test_target_carries_master_mode_within_budget(self):
        master = V4Coordinator("20001")
        master.session = "session"
        master.queue_message("TARGET", 10.0, target=TARGET, position=at(0.0),
                             sync_mode="F")
        payload = master.queue[-1][1]
        self.assertTrue(payload.endswith("|F"))
        self.assertLessEqual(len(payload.encode("utf-8")), 50)
        follower = V4Coordinator("20002")
        follower.session = "session"
        follower.master_uid = "20001"
        inbox = [SimpleNamespace(sender_uid="20001", payload=payload, recv_time=10.1)]
        self.assertEqual(follower.ingest(inbox, 10.1, "COOP_TRACK"), ["TARGET"])
        self.assertEqual(follower.sync_mode, "F")
        control = V4Control("20002")
        control.state = "COOP_TRACK"
        control.coord.session = "session"
        control.coord.master_uid = "20001"
        control.coord.sync_mode = follower.sync_mode
        control.coord.target = TARGET
        control.coord.master_position = at(0.0)
        control.coord.last_master_message_s = 10.1
        commands = control.step(observation(at(240.0)), 10.1)
        flight = next(item for item in commands if item.verb == "set_destination")
        self.assertEqual(flight.params["speed"], 25.0)
        self.assertEqual(control.coop_guidance["sync_mode"], "F")
        self.assertLess(control.coop_guidance["local_phase_error_deg"], -20.0)
        self.assertEqual(control.coop_guidance["follower_effective_lookahead_deg"],
                         control.coop_guidance["master_effective_lookahead_deg"])

    def test_coop_track_keeps_master_heartbeats_available(self):
        control = V4Control("20001")
        control.state = "COOP_TRACK"
        control.coord.session = "session"
        control.coord.master_uid = "20001"
        control.coord.partner_uid = "20002"
        control.rough.points.append(TARGET)
        for tick in range(200):
            now = tick * 0.1
            control.coord.peers["20002"] = (at(180.0), now, "COOP_TRACK_F")
            control.step(observation(at(0.0)), now)
        sent = [event["message_kind"] for event in control.pop_events()
                if event["event"] == "comm_out"]
        self.assertGreaterEqual(sent.count("H"), 18)
        self.assertGreaterEqual(sent.count("TARGET"), 18)

    def test_distance_prediction_modes_and_hysteresis(self):
        self.assertEqual(pair_safety_mode("NORMAL", 260.0, 0.0), ("NORMAL", 260.0))
        self.assertEqual(pair_safety_mode("NORMAL", 250.0, -20.0),
                         ("SEPARATE", 210.0))
        self.assertEqual(pair_safety_mode("NORMAL", 210.0, None)[0], "EMERGENCY")
        self.assertEqual(pair_safety_mode("SEPARATE", 245.0, 5.0)[0], "SEPARATE")
        self.assertEqual(pair_safety_mode("SEPARATE", 250.0, 0.0)[0], "NORMAL")
        self.assertEqual(pair_safety_mode("EMERGENCY", 230.0, 2.0)[0], "SEPARATE")

    def test_safety_direction_speed_and_radius(self):
        own = at(0.0)
        peer = orbit_point(TARGET, 180.0, 80.0)
        _, separate_speed, separate = pair_safety_guidance(
            TARGET, own, peer, 90.0, "SEPARATE", 230.0)
        _, emergency_speed, emergency = pair_safety_guidance(
            TARGET, own, peer, 90.0, "EMERGENCY", 205.0)
        for guidance in (separate, emergency):
            self.assertGreater(sum(a * b for a, b in zip(
                guidance["desired_direction"], guidance["away_from_peer"])), 0.0)
        self.assertLessEqual(separate_speed, 35.0)
        self.assertLessEqual(emergency_speed, 35.0)
        _, turning_speed, _ = pair_safety_guidance(
            TARGET, own, peer, (separate["desired_heading_deg"] + 90.0) % 360.0,
            "SEPARATE", 230.0)
        self.assertLessEqual(turning_speed, 18.0)
        self.assertGreater(emergency["desired_direction"][1],
                           separate["desired_direction"][1])
        heading = emergency["desired_heading_deg"]
        _, fast, _ = pair_safety_guidance(TARGET, own, peer, heading,
                                          "EMERGENCY", 205.0)
        self.assertEqual(fast, 40.0)
        far = at(0.0, 270.0)
        _, _, return_guidance = pair_safety_guidance(
            TARGET, far, peer, 0.0, "SEPARATE", 230.0)
        outward = sum(a * b for a, b in zip(
            return_guidance["desired_direction"], (0.0, 1.0)))
        self.assertLess(outward, 0.25)

    def test_heartbeat_distance_rate_only_updates_on_new_seen_time(self):
        control = V4Control("20001")
        control.state = "COOP_TRACK"
        control.coord.session = "session"
        control.coord.master_uid = "20001"
        control.coord.partner_uid = "20002"
        control.rough.points.append(TARGET)
        control.coord.peers["20002"] = (at(180.0), 10.0, "COOP_TRACK_F")
        control.step(observation(at(0.0)), 10.0)
        self.assertIsNone(control.coop_guidance["pair_distance_rate_mps"])
        self.assertEqual(control.coop_guidance["safety_mode"], "NORMAL")
        self.assertEqual(control.coop_guidance["commanded_speed_mps"], 25.0)
        control.step(observation(at(5.0)), 10.5)
        self.assertIsNone(control.coop_guidance["pair_distance_rate_mps"])
        control.coord.peers["20002"] = (at(170.0), 11.0, "COOP_TRACK_F")
        control.step(observation(at(5.0)), 11.0)
        distance = ground_distance_m(at(5.0), at(170.0))
        self.assertAlmostEqual(control.coop_guidance["pair_distance_rate_mps"],
                               distance - 260.0, delta=0.5)
        self.assertAlmostEqual(control.coop_guidance["predicted_pair_distance_m"],
                               distance + 2.0 * control.coop_guidance["pair_distance_rate_mps"])

    def test_both_roles_separate_from_fresh_heartbeats(self):
        master, follower = V4Control("20001"), V4Control("20002")
        for control in (master, follower):
            control.state = "COOP_TRACK"
            control.coord.session = "session"
            control.coord.master_uid = "20001"
        master.coord.partner_uid = "20002"
        master.rough.points.append(TARGET)
        follower.coord.target = TARGET
        follower.coord.master_position = at(0.0)
        follower.coord.last_master_message_s = 10.0
        master.coord.peers["20002"] = (at(90.0), 10.0, "COOP_TRACK_F")
        follower.coord.peers["20001"] = (at(0.0), 10.0, "COOP_TRACK_M")
        for control, position in ((master, at(0.0)), (follower, at(90.0))):
            control.step(observation(position), 10.0)
            guidance = control.coop_guidance
            self.assertEqual(guidance["safety_mode"], "EMERGENCY")
            self.assertFalse(guidance["safety_peer_stale"])
            self.assertGreater(sum(a * b for a, b in zip(
                guidance["desired_direction"], guidance["away_from_peer"])), 0.0)

    def test_calling_master_separates_before_radial_ready(self):
        master = V4Control("20001")
        master.state = "CALLING"
        master.coord.session = "session"
        master.coord.master_uid = "20001"
        master.coord.partner_uid = "20002"
        master.coord.accepted = True
        master.rough.points.append(TARGET)
        master.coord.peers["20002"] = (at(20.0), 10.0, "FOLLOWER_APPROACH")
        commands = master.step(observation(at(0.0)), 10.0)
        flight = next(item for item in commands if item.verb == "set_destination")
        self.assertEqual(master.coop_guidance["guidance_mode"], "PRE_ENTRY_EMERGENCY")
        self.assertLessEqual(flight.params["speed"], 35.0)
        self.assertGreater(sum(a * b for a, b in zip(
            master.coop_guidance["desired_direction"],
            master.coop_guidance["away_from_peer"])), 0.0)

    def test_follower_approach_safety_override_and_recovery(self):
        follower = V4Control("20002")
        follower.state = "FOLLOWER_APPROACH"
        follower.coord.session = "session"
        follower.coord.master_uid = "20001"
        follower.coord.target = TARGET
        follower.coord.master_position = at(180.0, 500.0)
        follower.coord.last_master_message_s = 10.0
        own = at(0.0)
        for now, peer_radius, expected in (
                (10.0, 170.0, "CAPTURE_SLOT"),
                (20.0, 100.0, "PRE_ENTRY_SEPARATE"),
                (21.0, 80.0, "PRE_ENTRY_EMERGENCY"),
                (22.0, 130.0, "CAPTURE_SLOT")):
            follower.coord.last_master_message_s = now
            follower.coord.peers["20001"] = (at(180.0, peer_radius), now, "CALLING")
            commands = follower.step(observation(own), now)
            guidance = follower.coop_guidance
            self.assertEqual(guidance["guidance_mode"], expected)
            self.assertFalse(guidance["safety_peer_stale"])
            self.assertIsNotNone(guidance["pair_distance_m"])
            self.assertIsNotNone(guidance["predicted_pair_distance_m"])
            self.assertIsNotNone(guidance["rough_radius_m"])
            self.assertIsNotNone(guidance["desired_heading_deg"])
            self.assertIsNotNone(guidance["heading_error_deg"])
            flight = next(item for item in commands if item.verb == "set_destination")
            self.assertEqual(flight.params["speed"], guidance["commanded_speed_mps"])
            if expected == "CAPTURE_SLOT":
                self.assertEqual(flight.params["speed"], 35.0)
            else:
                self.assertGreater(sum(a * b for a, b in zip(
                    guidance["desired_direction"], guidance["away_from_peer"])), 0.0)

    def test_ready_prediction_and_session_reset(self):
        follower = V4Control("20002")
        follower.state = "FOLLOWER_APPROACH"
        follower.coord.session = "first"
        follower.coord.master_uid = "20001"
        follower.coord.target = TARGET
        follower.coord.master_position = at(180.0)
        follower.coord.last_master_message_s = 10.0
        follower.coord.peers["20001"] = (at(180.0, 127.5), 9.0, "CALLING")
        follower.step(observation(at(0.0)), 9.0)
        follower.coord.peers["20001"] = (at(180.0, 110.0), 10.0, "CALLING")
        follower.step(observation(at(0.0)), 10.0)
        self.assertGreaterEqual(follower.coop_guidance["pair_distance_m"], 230.0)
        self.assertLess(follower.coop_guidance["predicted_pair_distance_m"], 220.0)
        self.assertEqual(follower.coord.last_queued.get("READY"), 9.0)
        follower.coord.session = "second"
        follower.coord.peers["20001"] = (at(180.0, 130.0), 11.0, "CALLING")
        follower.step(observation(at(0.0)), 11.0)
        self.assertIsNone(follower.coop_guidance["pair_distance_rate_mps"])
        self.assertIn("READY", follower.coord.last_queued)

    def test_follower_safety_ignores_old_target_position_without_heartbeat(self):
        follower = V4Control("20002")
        follower.state = "FOLLOWER_APPROACH"
        follower.coord.session = "session"
        follower.coord.master_uid = "20001"
        follower.coord.target = TARGET
        follower.coord.master_position = at(180.0, 80.0)
        follower.coord.last_master_message_s = 10.0
        follower.step(observation(at(0.0)), 10.0)
        self.assertEqual(follower.coop_guidance["guidance_mode"], "CAPTURE_SLOT")
        self.assertTrue(follower.coop_guidance["safety_peer_stale"])
        self.assertNotIn("READY", follower.coord.last_queued)
        follower.coord.peers["20001"] = (at(180.0, 80.0), 10.1, "CALLING")
        follower.step(observation(at(0.0)), 10.1)
        self.assertEqual(follower.coop_guidance["guidance_mode"],
                         "PRE_ENTRY_EMERGENCY")

    def test_invite_waits_for_near_search_partner(self):
        coordinator = V4Coordinator("20001")
        coordinator.peers["20002"] = (at(90.0), 10.0, "SEARCH")
        coordinator.peers["20003"] = (at(180.0), 10.0, "SEARCH")
        self.assertEqual(coordinator.select_partner(at(0.0), 10.0), "20002")
        master = V4Control("20001")
        master.state = "CALLING"
        master.coord.session = "session"
        master.coord.master_uid = "20001"
        master.rough.points.append(TARGET)
        master.coord.peers["20002"] = (at(90.0), 10.0, "SEARCH")
        master.step(observation(at(0.0)), 10.0)
        self.assertEqual(master.coord.partner_uid, "20002")
        self.assertEqual(master.partner_reservation["reserved_partner_waiting_reason"],
                         "TOO_CLOSE")
        self.assertNotIn("INVITE", master.coord.last_queued)

    def test_normal_far_radius_rejoins_with_tangent_and_inward_direction(self):
        for radius in (210.0, 280.0):
            own = at(0.0, radius)
            waypoint, debug = radius_rejoin_guidance(TARGET, own, 0.0)
            self.assertTrue(debug["radius_rejoin"])
            self.assertLess(debug["desired_direction"][1], 0.0)
            self.assertGreater(debug["desired_direction"][0], 0.0)
            self.assertAlmostEqual(ground_distance_m(own, waypoint), 120.0, delta=0.5)
        master = V4Control("20001")
        master.state = "COOP_TRACK"
        master.coord.session = "session"
        master.coord.master_uid = "20001"
        master.coord.partner_uid = "20002"
        master.rough.points.append(TARGET)
        master.coord.peers["20002"] = (at(180.0, 280.0), 10.0, "COOP_TRACK_F")
        commands = master.step(observation(at(0.0, 280.0)), 10.0)
        flight = next(item for item in commands if item.verb == "set_destination")
        self.assertEqual(flight.params["speed"], 25.0)
        self.assertEqual(master.coop_guidance["safety_mode"], "NORMAL")
        self.assertTrue(master.coop_guidance["radius_rejoin"])


if __name__ == "__main__":
    unittest.main()
