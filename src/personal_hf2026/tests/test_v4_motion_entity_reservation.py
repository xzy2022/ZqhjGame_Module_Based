# 修改时间：2026-09-28。
# 修改目的：验证本轮三处简单修改对应的业务边界。
# 修改内容：覆盖四次静止确认、实体中心门和最近合法协作机的预约等待。
"""V4 动静、实体匹配与协作机预约的聚焦测试。"""

import unittest
from types import SimpleNamespace

from personal_hf2026.v3_simple_control import offset_position
from personal_hf2026.v4_control import V4Control
from personal_hf2026.v4_coordination import V4Coordinator
from personal_hf2026.v4_entity import EntityManager
from personal_hf2026.v4_motion import SingleEntityMotion


OWN = (27.025, 125.020)
SECTORS = {"20001": 0, "20002": 2, "20003": 1}


def vehicle(center_x, size=20.0):
    half = size / 2.0
    return SimpleNamespace(class_name="real_vehicle",
                           bbox_xyxy=(center_x-half, 90.0, center_x+half, 90.0+size))


def observation(position=OWN, inbox=()):
    own = SimpleNamespace(lat=position[0], lon=position[1], alt=500.0,
                          heading_deg=0.0, gimbal_pan=0.0, gimbal_tilt=-45.0,
                          gimbal_fov_deg=48.0)
    return SimpleNamespace(self=own, comm_inbox=list(inbox))


def heartbeat(uid, position, state, now):
    coordinator = V4Coordinator(uid)
    coordinator.queue_message("H", now, position=position, state=state)
    return SimpleNamespace(sender_uid=uid, payload=coordinator.queue[-1][1],
                           recv_time=now)


def master():
    control = V4Control("20003")
    control.route.initialized = True
    control.route.sector_by_uid = SECTORS
    control.route.home_sector = 1
    control.state = "CALLING"
    control.coord.set_master("uav_20003_entity_1")
    control.rough.points.append(OWN)
    return control


class MotionConfirmationTest(unittest.TestCase):
    def test_moving_needs_four_consecutive_static_judgments(self):
        motion = SingleEntityMotion()
        self.assertEqual(motion.MOVING_TO_STATIC_CONFIRMATIONS, 4)
        self.assertEqual([motion._confirm_raw("MOVING") for _ in range(2)],
                         ["UNKNOWN", "MOVING"])
        for streak in range(1, 4):
            self.assertEqual(motion._confirm_raw("STATIC"), "UNKNOWN")
            self.assertEqual(motion.decision, "MOVING")
            self.assertEqual(motion._static_raw_streak, streak)
        self.assertEqual(motion._confirm_raw("STATIC"), "STATIC")
        self.assertEqual(motion.decision, "STATIC")

    def test_unknown_and_moving_break_static_streak(self):
        for interruption in ("UNKNOWN", "MOVING"):
            with self.subTest(interruption=interruption):
                motion = SingleEntityMotion()
                motion._confirm_raw("MOVING")
                motion._confirm_raw("MOVING")
                motion._confirm_raw("STATIC")
                motion._confirm_raw("STATIC")
                motion._confirm_raw(interruption)
                self.assertEqual(motion._static_raw_streak, 0)
                for _ in range(3):
                    motion._confirm_raw("STATIC")
                self.assertEqual(motion.decision, "MOVING")
                motion.reset()
                self.assertEqual(motion._static_raw_streak, 0)

    def test_initial_static_still_needs_only_two_judgments(self):
        motion = SingleEntityMotion()
        self.assertEqual(motion._confirm_raw("STATIC"), "UNKNOWN")
        self.assertEqual(motion._confirm_raw("STATIC"), "STATIC")


class EntityMatchTest(unittest.TestCase):
    def test_small_box_matches_170_but_not_190_pixels(self):
        for displacement, expected in ((170.0, "entity_matched"),
                                       (190.0, "entity_missing")):
            with self.subTest(displacement=displacement):
                manager = EntityManager("20001")
                manager.update((vehicle(100.0),), (640, 480), 0.0)
                entity, event = manager.update((vehicle(100.0+displacement),),
                                               (640, 480), 0.1)
                self.assertEqual(event, expected)
                self.assertEqual(entity.visible, displacement == 170.0)

    def test_large_box_keeps_diagonal_gate(self):
        manager = EntityManager("20001")
        manager.update((vehicle(100.0, 160.0),), (640, 480), 0.0)
        _, event = manager.update((vehicle(350.0, 160.0),), (640, 480), 0.1)
        self.assertEqual(event, "entity_matched")


class PartnerReservationTest(unittest.TestCase):
    def test_busy_nearest_waits_then_invites_after_release(self):
        control = master()
        near = offset_position(OWN, 300.0, 0.0)
        far = offset_position(OWN, 1000.0, 0.0)
        control.coord.peers["20001"] = (near, 10.0, "CALLING")
        control.coord.peers["20002"] = (far, 10.0, "SEARCH")
        control.step(observation(), 10.0)
        self.assertEqual(control.coord.partner_uid, "20001")
        self.assertEqual(control.partner_reservation["reserved_partner_waiting_reason"],
                         "BUSY")
        self.assertNotIn("INVITE", control.coord.last_queued)
        self.assertIn("master_conflict_won",
                      [event["event"] for event in control.pop_events()])
        edge = V4Control("20001")
        edge.route.initialized = True
        edge.route.sector_by_uid = SECTORS
        edge.route.home_sector = 0
        edge.route.target = lambda *args: None
        edge.route.coverage.observe = lambda *args: None
        edge.state = "CALLING"
        edge.coord.set_master("uav_20001_entity_1")
        edge.step(observation(near, [heartbeat("20003", OWN, "CALLING", 10.1)]), 10.1)
        self.assertEqual(edge.state, "SEARCH")
        self.assertIsNone(edge.coord.partner_uid)
        control.step(observation(inbox=[heartbeat("20001", near, "SEARCH", 10.5)]),
                     10.5)
        self.assertEqual(control.coord.partner_uid, "20001")
        self.assertEqual(control.coord.last_queued.get("INVITE"), 10.5)
        self.assertTrue(any(event["event"] == "partner_invited"
                            for event in control.pop_events()))

    def test_too_close_stays_reserved_until_safe(self):
        control = master()
        control.coord.peers["20001"] = (offset_position(OWN, 230.0, 0.0),
                                        10.0, "SEARCH")
        control.coord.peers["20002"] = (offset_position(OWN, 1000.0, 0.0),
                                        10.0, "SEARCH")
        control.step(observation(), 10.0)
        self.assertEqual(control.coord.partner_uid, "20001")
        self.assertEqual(control.partner_reservation["reserved_partner_waiting_reason"],
                         "TOO_CLOSE")
        self.assertNotIn("INVITE", control.coord.last_queued)
        control.coord.peers["20001"] = (offset_position(OWN, 270.0, 0.0),
                                        11.0, "SEARCH")
        control.step(observation(), 11.0)
        self.assertEqual(control.coord.partner_uid, "20001")
        self.assertEqual(control.coord.last_queued.get("INVITE"), 11.0)

    def test_stale_nearest_reselects_fresh_allowed_partner(self):
        control = master()
        control.coord.peers["20001"] = (offset_position(OWN, 300.0, 0.0),
                                        10.0, "VERIFY")
        control.coord.peers["20002"] = (offset_position(OWN, 1000.0, 0.0),
                                        10.0, "SEARCH")
        control.step(observation(), 10.0)
        control.pop_events()
        control.coord.peers["20002"] = (offset_position(OWN, 1000.0, 0.0),
                                        15.1, "SEARCH")
        control.step(observation(), 15.1)
        self.assertEqual(control.coord.partner_uid, "20002")
        events = control.pop_events()
        self.assertTrue(any(event["event"] == "partner_reselected"
                            and event["reason"] == "stale" for event in events))
        control._return_search(16.0, "test_end")
        self.assertIsNone(control.coord.partner_uid)


if __name__ == "__main__":
    unittest.main()
