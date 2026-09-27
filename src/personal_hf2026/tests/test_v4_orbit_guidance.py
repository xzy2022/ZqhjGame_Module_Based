# 修改时间：2026-09-27。
# 修改目的：验证双机协同从实际位置生成轨道航点和对置速度。
# 修改内容：覆盖固定半径、前视角、主从相位、READY 几何和无时钟 START。
"""V4 双机轨道导航的聚焦测试。"""

import math
import unittest
from types import SimpleNamespace

from personal_hf2026.v3_simple_control import bearing_deg, ground_distance_m
from personal_hf2026 import v4_control
from personal_hf2026.v4_control import V4Control
from personal_hf2026.v4_coordination import V4Coordinator
from personal_hf2026.v4_flight import (formation_ready, follower_orbit_guidance,
                                       master_orbit_guidance, orbit_point,
                                       orbit_short_waypoint, wrap180)


TARGET = (27.025, 125.020)


def at(phase, radius=130.0):
    return orbit_point(TARGET, phase, radius)


def observation(position):
    own = SimpleNamespace(lat=position[0], lon=position[1], alt=500.0,
                          heading_deg=0.0, gimbal_pan=0.0, gimbal_tilt=-45.0,
                          gimbal_fov_deg=48.0)
    return SimpleNamespace(self=own, comm_inbox=[])


class OrbitGuidanceTest(unittest.TestCase):
    def test_orbit_point_radius_is_fixed_130(self):
        for phase in (0.0, 40.0, 180.0, 359.0):
            self.assertAlmostEqual(ground_distance_m(TARGET, at(phase)), 130.0,
                                   delta=0.2)

    def test_short_waypoint_uses_actual_phase(self):
        own = at(40.0)
        waypoint, theta, lookahead = orbit_short_waypoint(TARGET, own, 25.0)
        self.assertAlmostEqual(wrap180(theta - 40.0), 0.0, delta=0.2)
        self.assertAlmostEqual(wrap180(bearing_deg(TARGET, waypoint)
                                       - theta - lookahead), 0.0, delta=0.2)
        self.assertAlmostEqual(ground_distance_m(TARGET, waypoint), 130.0,
                               delta=0.2)

    def test_lookahead_matches_speed(self):
        angles = []
        for speed in (15.0, 25.0, 35.0):
            _, _, angle = orbit_short_waypoint(TARGET, at(40.0), speed)
            self.assertAlmostEqual(angle, math.degrees(speed / 130.0 * 1.5))
            angles.append(angle)
        self.assertEqual(angles, sorted(angles))

    def test_master_uses_actual_phase_and_base_speed(self):
        destination, speed, debug = master_orbit_guidance(TARGET, at(40.0))
        self.assertEqual(speed, 25.0)
        self.assertAlmostEqual(wrap180(bearing_deg(TARGET, destination)
                                       - 40.0 - math.degrees(25.0 / 130.0 * 1.5)),
                               0.0, delta=0.2)
        self.assertAlmostEqual(debug["radius_m_actual"], 130.0, delta=0.2)

    def test_follower_long_slot_is_opposite_actual_master(self):
        _, _, debug = follower_orbit_guidance(TARGET, at(40.0), at(190.0))
        self.assertAlmostEqual(wrap180(debug["desired_follower_phase_deg"] - 220.0),
                               0.0, delta=0.2)
        self.assertAlmostEqual(wrap180(bearing_deg(TARGET, debug["long_slot"])
                                       - 220.0), 0.0, delta=0.2)

    def test_follower_behind_accelerates(self):
        _, speed, debug = follower_orbit_guidance(TARGET, at(40.0), at(180.0))
        self.assertGreater(debug["phase_error_deg"], 20.0)
        self.assertEqual(speed, 35.0)

    def test_follower_ahead_slows(self):
        _, speed, debug = follower_orbit_guidance(TARGET, at(40.0), at(260.0))
        self.assertLess(debug["phase_error_deg"], -20.0)
        self.assertEqual(speed, 15.0)

    def test_follower_near_opposite_uses_base_speed(self):
        for phase in (205.0, 220.0, 235.0):
            _, speed, debug = follower_orbit_guidance(TARGET, at(40.0), at(phase))
            self.assertLessEqual(abs(debug["phase_error_deg"]), 20.0)
            self.assertEqual(speed, 25.0)

    def test_follower_short_waypoint_does_not_equal_long_slot(self):
        destination, _, debug = follower_orbit_guidance(TARGET, at(40.0), at(180.0))
        self.assertGreater(ground_distance_m(destination, debug["long_slot"]), 1.0)
        self.assertEqual(destination, debug["short_waypoint"])
        self.assertAlmostEqual(ground_distance_m(TARGET, destination), 130.0,
                               delta=0.2)

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
            self.assertGreater(ground_distance_m(short, long_slot), 1.0)
            self.assertEqual(flight.params["speed"], 35.0)


if __name__ == "__main__":
    unittest.main()
