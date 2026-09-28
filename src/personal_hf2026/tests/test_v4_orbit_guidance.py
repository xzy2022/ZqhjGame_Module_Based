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
                                       pair_phase_geometry, phase_sync_mode, wrap180)


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
            self.assertEqual((m_angle > f_angle) - (m_angle < f_angle),
                             {"F": 1, "M": -1, "N": 0}[mode])

    def test_follower_behind_uses_smaller_lookahead(self):
        _, speed, debug = follower_orbit_guidance(
            TARGET, at(40.0), at(180.0), sync_mode="F")
        self.assertGreater(debug["phase_error_deg"], 20.0)
        self.assertEqual(speed, 25.0)
        self.assertLess(debug["effective_lookahead_deg"], debug["base_lookahead_deg"])

    def test_follower_ahead_uses_larger_lookahead(self):
        _, speed, debug = follower_orbit_guidance(
            TARGET, at(40.0), at(260.0), sync_mode="M")
        self.assertLess(debug["phase_error_deg"], -20.0)
        self.assertEqual(speed, 25.0)
        self.assertGreater(debug["effective_lookahead_deg"], debug["base_lookahead_deg"])

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
        self.assertLess(abs(geometry["phase_error_deg"]), 30.0)

        # 移动目标正反各60度，检查误差的整体趋势。
        for signed_error in (60.0, -60.0):
            moving_target = TARGET
            moving_master = at(0.0)
            moving_follower = orbit_point(
                moving_target, 180.0 - signed_error)
            initial_error = abs(pair_phase_geometry(
                moving_target, moving_master, moving_follower)["phase_error_deg"])
            errors = []
            for _ in range(200):
                moving_target = offset_position(moving_target, 1.0, 0.0)
                phase_error = pair_phase_geometry(
                    moving_target, moving_master, moving_follower)["phase_error_deg"]
                mode = phase_sync_mode(phase_error)
                m_dest, m_speed, _ = master_orbit_guidance(
                    moving_target, moving_master, sync_mode=mode)
                f_dest, f_speed, _ = follower_orbit_guidance(
                    moving_target, moving_master, moving_follower, sync_mode=mode)
                self.assertEqual((m_speed, f_speed), (25.0, 25.0))
                moving_master = move_toward(moving_master, m_dest, 2.5)
                moving_follower = move_toward(moving_follower, f_dest, 2.5)
                errors.append(abs(pair_phase_geometry(
                    moving_target, moving_master, moving_follower)["phase_error_deg"]))
            self.assertLess(sum(errors[-100:]) / 100.0, initial_error)

    def test_ready_requires_radial_and_phase_geometry(self):
        master = at(40.0)
        follower = at(220.0)
        self.assertTrue(formation_ready(TARGET, master, follower))
        self.assertFalse(formation_ready(TARGET, at(40.0, 180.0), follower))
        self.assertFalse(formation_ready(TARGET, master, at(220.0, 180.0)))
        self.assertFalse(formation_ready(TARGET, master, at(170.0)))

        for master_position, own_position, expected in (
                (master, follower, True),
                (at(40.0, 180.0), follower, False),
                (master, at(220.0, 180.0), False),
                (master, at(170.0), False)):
            control = V4Control("20002")
            control.state = "FOLLOWER_APPROACH"
            control.coord.session = "session"
            control.coord.master_uid = "20001"
            control.coord.target = TARGET
            control.coord.master_position = master_position
            control.coord.last_master_message_s = 10.0
            control.step(observation(own_position), 10.0)
            self.assertEqual("READY" in control.coord.last_queued, expected)

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
                self.assertEqual(control.coop_guidance["guidance_mode"], "ORBIT_SYNC")
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
        self.assertEqual(flight.params["speed"], 25.0)
        self.assertEqual(control.coop_guidance["follower_command_speed_mps"], 25.0)
        self.assertEqual(control.coop_guidance["sync_mode"], "F")
        self.assertGreater(control.coop_guidance["master_effective_lookahead_deg"],
                           control.coop_guidance["follower_effective_lookahead_deg"])
        self.assertEqual(control.coop_guidance["guidance_mode"], "ORBIT_SYNC")
        control.coord.peers["20002"] = (at(100.0), 4.0, "COOP_TRACK_F")
        commands = control.step(observation(at(0.0)), 10.1)
        flight = next(item for item in commands if item.verb == "set_destination")
        self.assertEqual(flight.params["speed"], 25.0)
        self.assertEqual(control.coop_guidance["sync_mode"], "N")

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
        self.assertLess(control.coop_guidance["follower_effective_lookahead_deg"],
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


if __name__ == "__main__":
    unittest.main()
