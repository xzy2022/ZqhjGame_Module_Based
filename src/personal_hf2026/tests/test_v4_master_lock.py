# 修改时间：2026-09-28。
# 修改目的：验证三机仅保留一个有效 MASTER 的关键状态转移。
# 修改内容：覆盖运行时分区优先级、候选预检查、双 CALLING 仲裁和静止取消。
"""V4 全局 MASTER 资源锁的聚焦测试。"""

import unittest
from types import SimpleNamespace

from personal_hf2026.v4_control import V4Control, master_verdict
from personal_hf2026.v4_coordination import V4Coordinator


POSITION = (27.025, 125.020)


def control(uid, sectors):
    item = V4Control(uid)
    item.route.initialized = True
    item.route.sector_by_uid = sectors
    item.route.home_sector = sectors[uid]
    item.route.target = lambda *args: None
    item.route.coverage.observe = lambda *args: None
    return item


def observation(inbox=()):
    own = SimpleNamespace(lat=POSITION[0], lon=POSITION[1], alt=500.0,
                          heading_deg=0.0, gimbal_pan=0.0, gimbal_tilt=-45.0,
                          gimbal_fov_deg=48.0)
    return SimpleNamespace(self=own, comm_inbox=list(inbox))


def heartbeat(uid, state, now):
    coordinator = V4Coordinator(uid)
    coordinator.queue_message("H", now, position=POSITION, state=state)
    return SimpleNamespace(sender_uid=uid, payload=coordinator.queue[-1][1],
                           recv_time=now)


def visual_motion(item, decision, now=10.0):
    item.state = "VERIFY" if item.state == "SEARCH" else item.state
    entity = SimpleNamespace(entity_id=f"uav_{item.uid}_entity_1",
                             bbox_xyxy=(100, 100, 130, 130), visible=True,
                             missing_s=0.0, observed_frames=5)
    item.entity.update = lambda *args: (entity, None)
    item.gimbal.update = lambda *args: None
    item.rough.update = lambda *args: None

    def update_motion(*args):
        item.motion.decision = decision
        return {}

    item.motion.update = update_motion
    snapshot = SimpleNamespace(source_sim_time=now, frame_id=f"frame-{now}",
                               error=None, effective_yolo_objects=[entity],
                               raw_yolo_objects=[], image_size=(640, 480),
                               source_pose={}, image_bgr=None)
    item.consume_visual(snapshot)


class MasterLockTest(unittest.TestCase):
    def test_priority_is_stage_then_runtime_sector_then_uid(self):
        cases = [
            ("20001", "CALLING", 0, "20003", "CALLING", 1, "20003", "middle_priority"),
            ("20002", "CALLING", 1, "20003", "CALLING", 2, "20002", "middle_priority"),
            ("20001", "CALLING", 0, "20003", "CALLING", 2, "20001", "edge_uid_tiebreak"),
            ("20001", "COOP_TRACK_M", 0, "20003", "CALLING", 1,
             "20001", "own_already_coop_track"),
            ("20003", "CALLING", None, "20001", "CALLING", 0,
             "20001", "uid_fallback_before_sector_init"),
        ]
        for a, a_state, a_sector, b, b_state, b_sector, winner, reason in cases:
            with self.subTest(a=a, b=b, reason=reason):
                self.assertEqual(master_verdict(a, a_state, a_sector,
                                                b, b_state, b_sector), (winner, reason))
                reverse_reason = ("peer_already_coop_track" if reason ==
                                  "own_already_coop_track" else reason)
                self.assertEqual(master_verdict(b, b_state, b_sector,
                                                a, a_state, a_sector),
                                 (winner, reverse_reason))

    def test_verify_blocks_existing_master_but_middle_can_challenge_edge_calling(self):
        sectors = {"20001": 0, "20002": 2, "20003": 1}
        for peer_state in ("CALLING", "COOP_TRACK_M"):
            edge = control("20001", sectors)
            edge.coord.peers["20003"] = (POSITION, 9.5, peer_state)
            visual_motion(edge, "MOVING")
            self.assertEqual(edge.state, "SEARCH")
            self.assertIsNone(edge.coord.session)
            blocked = [event for event in edge.pop_events()
                       if event["event"] == "master_candidate_blocked"]
            self.assertEqual(len(blocked), 1)
            self.assertEqual(blocked[0]["peer_master_age_s"], 0.5)

        middle = control("20003", sectors)
        middle.coord.peers["20001"] = (POSITION, 9.5, "CALLING")
        visual_motion(middle, "MOVING")
        self.assertEqual(middle.state, "CALLING")
        self.assertEqual(middle.coord.master_uid, middle.uid)

        middle_again = control("20003", sectors)
        middle_again.coord.peers["20001"] = (POSITION, 9.5, "COOP_TRACK_M")
        visual_motion(middle_again, "MOVING")
        self.assertEqual(middle_again.state, "SEARCH")

        stale = control("20001", sectors)
        stale.coord.peers["20003"] = (POSITION, 4.0, "COOP_TRACK_M")
        visual_motion(stale, "MOVING")
        self.assertEqual(stale.state, "CALLING")

    def test_simultaneous_calling_resolves_on_fresh_heartbeats(self):
        sectors = {"20001": 0, "20002": 2, "20003": 1}
        edge, middle = control("20001", sectors), control("20003", sectors)
        for item in (edge, middle):
            item.state = "CALLING"
            item.coord.set_master(f"uav_{item.uid}_entity_1")
        edge_commands = edge.step(
            observation([heartbeat("20003", "CALLING", 10.0)]), 10.0)
        middle.step(observation([heartbeat("20001", "CALLING", 10.0)]), 10.0)
        self.assertEqual((edge.state, middle.state), ("SEARCH", "CALLING"))
        self.assertIsNone(edge.coord.session)
        self.assertTrue(any(command.verb == "comm.broadcast" and
                            "|CANCEL|" in command.params["payload"]
                            for command in edge_commands))
        self.assertEqual(len([event for event in edge.pop_events()
                              if event["event"] == "master_conflict_yielded"]), 1)
        self.assertEqual(len([event for event in middle.pop_events()
                              if event["event"] == "master_conflict_won"]), 1)

    def test_active_coop_beats_new_calling_and_edge_uid_breaks_tie(self):
        sectors = {"20001": 0, "20002": 2, "20003": 1}
        middle = control("20003", sectors)
        middle.state = "CALLING"
        middle.coord.set_master("uav_20003_entity_1")
        middle.step(observation([heartbeat("20001", "COOP_TRACK_M", 10.0)]), 10.0)
        self.assertEqual(middle.state, "SEARCH")

        edge = control("20002", sectors)
        edge.state = "CALLING"
        edge.coord.set_master("uav_20002_entity_1")
        edge.step(observation([heartbeat("20001", "CALLING", 10.0)]), 10.0)
        self.assertEqual(edge.state, "SEARCH")

    def test_calling_static_cancels_without_completion(self):
        item = control("20001", {"20001": 0, "20002": 1, "20003": 2})
        item.state = "CALLING"
        item.coord.set_master("uav_20001_entity_1")
        visual_motion(item, "STATIC")
        self.assertEqual(item.state, "SEARCH")
        self.assertEqual(item.completed_sessions, 0)
        self.assertIn("CANCEL", [kind for kind, _ in item.coord.queue])
        self.assertNotIn("DONE", [kind for kind, _ in item.coord.queue])

    def test_state_change_queues_heartbeat_before_regular_one_second_period(self):
        item = control("20001", {"20001": 0, "20002": 1, "20003": 2})
        item.step(observation(), 10.0)
        item.state = "CALLING"
        item.coord.set_master("uav_20001_entity_1")
        item.step(observation(), 10.1)
        self.assertTrue(any(kind == "H" and payload.endswith("|CALLING")
                            for kind, payload in item.coord.queue))


if __name__ == "__main__":
    unittest.main()
