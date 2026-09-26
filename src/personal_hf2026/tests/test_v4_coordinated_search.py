# 修改时间：2026-09-26。
# 修改目的：验证三机航线的分区、搜索翻转和中区接管主流程。
# 修改内容：新增局部坐标、对齐、双机中点倒退与搜索恢复的聚焦测试。
"""三机协同 Z 字搜索的主流程测试。"""

import unittest

from personal_hf2026.coordinated_search import CoordinatedSweepRoute


MEMBERS = ("20003", "20001", "20002")
NS_LENGTH = 5000.0
EW_LENGTH = 3000.0
BOUNDS = ((0.0, 0.0), (NS_LENGTH / 111320.0, EW_LENGTH / 111320.0))


def point(u, v):
    return v / 111320.0, u / 111320.0


def peers(positions, states=None, now=10.0):
    states = states or {}
    return {uid: (position, now, states.get(uid, "SEARCH"))
            for uid, position in positions.items()}


class CoordinatedSweepRouteTest(unittest.TestCase):
    def setUp(self):
        self.positions = {
            "20001": point(500.0, 1000.0),
            "20002": point(1400.0, 1100.0),
            "20003": point(2500.0, 1200.0),
        }

    def route(self, uid="20003"):
        return CoordinatedSweepRoute(BOUNDS, uid, MEMBERS)

    def start_sweep(self, uid="20003"):
        route = self.route(uid)
        peer_positions = {key: value for key, value in self.positions.items() if key != uid}
        self.assertIsNotNone(route.target(self.positions[uid], 10.0, peers(peer_positions)))
        aligned = {key: point(route._uv(value)[0], 1100.0)
                   for key, value in self.positions.items()}
        own = aligned.pop(uid)
        route.target(own, 10.1, peers(aligned, now=10.1))
        self.assertEqual(route.mode, "SWEEP")
        return route, own, aligned

    def test_short_axis_and_dynamic_sectors(self):
        route = self.route()
        self.assertEqual(route.short_axis, "EW")
        self.assertIsNone(route.target(self.positions["20003"], 10.0, {}))
        self.assertEqual(route.mode, "WAIT_PEERS")
        route.target(self.positions["20003"], 10.0,
                     peers({key: value for key, value in self.positions.items()
                            if key != "20003"}))
        self.assertEqual(route.sector_by_uid,
                         {"20001": 0, "20002": 1, "20003": 2})
        self.assertAlmostEqual(route.align_v, 1100.0)
        self.assertEqual(route.cross_direction, -1)
        self.assertAlmostEqual(route.sector_bounds[0][0], 0.0)
        self.assertAlmostEqual(route.sector_bounds[0][1], route.sector_bounds[1][0])
        self.assertAlmostEqual(route.sector_bounds[1][1], route.sector_bounds[2][0])
        self.assertAlmostEqual(route.sector_bounds[2][1], route.short_length_m)
        self.assertEqual(route.preferred_partner_uids(), ("20002",))

        rotated = CoordinatedSweepRoute(
            ((0.0, 0.0), (1000.0 / 111320.0, 5000.0 / 111320.0)),
            "20003", MEMBERS)
        self.assertEqual(rotated.short_axis, "NS")
        sample = (0.002, 0.03)
        recovered = rotated._position(*rotated._uv(sample))
        self.assertAlmostEqual(recovered[0], sample[0])
        self.assertAlmostEqual(recovered[1], sample[1])

    def test_align_waits_for_all_and_keeps_frontier(self):
        route = self.route()
        others = {key: value for key, value in self.positions.items() if key != "20003"}
        target = route.target(self.positions["20003"], 10.0, peers(others))
        self.assertAlmostEqual(route._uv(target)[0], 2500.0, places=3)
        self.assertAlmostEqual(route._uv(target)[1], 1100.0)
        own = point(2500.0, 1100.0)
        target = route.target(own, 10.1, peers(others, now=10.1))
        self.assertEqual(route.mode, "ALIGN")
        self.assertAlmostEqual(route._uv(target)[1], 1100.0)
        self.assertAlmostEqual(route.frontier_v, 1100.0)

        # 尚未收到三机位置时即使暂停，首次初始化也必须完成共同对齐。
        early = self.route()
        early.pause()
        early.target(self.positions["20003"], 10.0, peers(others))
        self.assertEqual(early.mode, "ALIGN")

    def test_sweep_flip_and_long_axis_reflection(self):
        route, own, others = self.start_sweep()
        target = route.target(own, 10.2, peers(others, now=10.2))
        self.assertAlmostEqual(route._uv(target)[0], 2000.0, places=3)
        route.target(target, 10.3, peers(others, now=10.3))
        self.assertEqual(route.cross_direction, 1)
        self.assertAlmostEqual(route.frontier_v, 1800.0)
        self.assertEqual(route.turn_count, 1)
        route.frontier_v = route.long_length_m - 100.0
        route._advance_frontier()
        self.assertAlmostEqual(route.frontier_v, route.long_length_m)
        self.assertEqual(route.advance_direction, -1)
        route._advance_frontier()
        self.assertAlmostEqual(route.frontier_v, route.long_length_m - 700.0)

    def test_edge_pair_takeover_midpoint_and_backward_motion(self):
        route, own, _ = self.start_sweep()
        pair_positions = {"20001": point(500.0, 1000.0),
                          "20002": point(1500.0, 1200.0)}
        states = {"20001": "COOP_TRACK_M", "20002": "COOP_TRACK_F"}
        route.target(own, 11.0, peers(pair_positions, states, 11.0))
        self.assertEqual(route.mode, "TAKEOVER")
        self.assertEqual(route.trace_state["active_sector"], [1, 2])
        self.assertAlmostEqual(route.frontier_v, 1100.0)
        self.assertAlmostEqual(route._uv(route.pair_midpoint)[1], 1100.0)
        pair_positions["20001"] = point(500.0, 800.0)
        pair_positions["20002"] = point(1500.0, 1000.0)
        route.target(own, 11.1, peers(pair_positions, states, 11.1))
        self.assertAlmostEqual(route.frontier_v, 900.0)

        target = route.last_target
        direction = route.cross_direction
        before = route.frontier_v
        route.target(target, 11.2, peers(pair_positions, states, 11.2))
        self.assertAlmostEqual(route.frontier_v, before)
        self.assertEqual(route.cross_direction, -direction)

    def test_opposite_edge_takeover_and_middle_master_no_takeover(self):
        west, own_west, _ = self.start_sweep("20001")
        east_pair = {"20003": point(2500.0, 1000.0),
                     "20002": point(1500.0, 1200.0)}
        west.target(own_west, 11.0, peers(
            east_pair, {"20003": "CALLING", "20002": "FOLLOWER_APPROACH"}, 11.0))
        self.assertEqual(west.mode, "TAKEOVER")
        self.assertEqual(west.trace_state["active_sector"], [0, 1])

        east, own_east, _ = self.start_sweep()
        middle_pair = {"20001": point(500.0, 1000.0),
                       "20002": point(1500.0, 1200.0)}
        east.target(own_east, 11.0, peers(
            middle_pair, {"20002": "COOP_TRACK_M", "20001": "COOP_TRACK_F"}, 11.0))
        self.assertEqual(east.mode, "SWEEP")
        self.assertEqual(east.trace_state["active_sector"], [2, 2])

    def test_takeover_end_and_pause_resume_from_current_position(self):
        route, own, _ = self.start_sweep()
        pair_positions = {"20001": point(500.0, 1000.0),
                          "20002": point(1500.0, 1200.0)}
        route.target(own, 11.0, peers(
            pair_positions, {"20001": "COOP_TRACK_M", "20002": "COOP_TRACK_F"}, 11.0))
        current = point(2400.0, 1700.0)
        route.target(current, 11.1, peers(pair_positions, now=11.1))
        self.assertEqual(route.mode, "SWEEP")
        self.assertEqual(route.trace_state["active_sector"], [2, 2])
        self.assertAlmostEqual(route.frontier_v, 1700.0)
        route.pause()
        route.pause()
        resumed = point(2350.0, 2300.0)
        route.target(resumed, 12.0, peers(
            pair_positions, {"20001": "COOP_TRACK_M", "20002": "COOP_TRACK_F"}, 12.0))
        self.assertAlmostEqual(route.frontier_v, 2300.0)
        self.assertEqual(route.mode, "SWEEP")

    def test_partner_preferences_and_events(self):
        middle, _, _ = self.start_sweep("20002")
        self.assertEqual(set(middle.preferred_partner_uids()), {"20001", "20003"})
        east, _, _ = self.start_sweep()
        self.assertEqual(east.preferred_partner_uids(), ("20002",))
        events = east.drain_events()
        self.assertEqual([name for name, _ in events],
                         ["search_plan_initialized", "search_mode_changed", "search_mode_changed"])
        self.assertEqual(east.drain_events(), [])
        self.assertIn("covered_fraction", east.summary["coverage"])


if __name__ == "__main__":
    unittest.main()
