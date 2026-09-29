# 修改时间：2026-09-29。
# 修改目的：验证空闲第三机只在原分区同步双机长轴位置和速度。
# 修改内容：覆盖速度变化、重复心跳、边界翻转、固定分区与协同退出后的恢复。
# 修改时间：2026-09-28。
# 修改目的：验证双机协同时第三机始终保持原分区和普通搜索速度。
# 修改内容：删除接管测试，覆盖三种空闲机组合、车道推进与搜索恢复。
# 修改时间：2026-09-28。
# 修改目的：验证第三机按双机心跳速度和位置误差分配35米每秒总速度。
# 修改内容：覆盖平均速度、对齐、落后与超前、两种分区策略和变化心跳。
# 修改时间：2026-09-26。
# 修改目的：核对正式通信的协作人选和五十字节载荷限制。
# 修改内容：增加按允许 UID 筛选最近协作机及各心跳角色长度检查。
# 修改时间：2026-09-26。
# 修改目的：验证三机航线的分区、搜索翻转和中区接管主流程。
# 修改内容：新增局部坐标、对齐、双机中点倒退与搜索恢复的聚焦测试。
"""三机协同 Z 字搜索的主流程测试。"""

import math
import unittest
from types import SimpleNamespace

from personal_hf2026.coordinated_search import CoordinatedSweepRoute
from personal_hf2026.v4_control import V4Control
from personal_hf2026.v4_coordination import V4Coordinator


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

    def test_idle_uav_follows_pair_without_sector_expansion_or_lane_advance(self):
        for idle_uid in ("20001", "20002", "20003"):
            route, _, _ = self.start_sweep(idle_uid)
            pair_uids = [uid for uid in ("20001", "20002", "20003")
                         if uid != idle_uid]
            for master_uid, follower_uid in (pair_uids, pair_uids[::-1]):
                state = {master_uid: "CALLING", follower_uid: "FOLLOWER_APPROACH"}
                pair_positions = {
                    master_uid: point(route._uv(self.positions[master_uid])[0], 1200.0),
                    follower_uid: point(route._uv(self.positions[follower_uid])[0], 1200.0),
                }
                own = point(route._uv(self.positions[idle_uid])[0], 1200.0)
                route.target(own, 11.0, peers(pair_positions, state, 11.0))
                pair_positions[master_uid] = point(route._uv(pair_positions[master_uid])[0], 1206.0)
                pair_positions[follower_uid] = point(route._uv(pair_positions[follower_uid])[0], 1210.0)
                pair_mid_v = 1208.0
                route.target(point(route._uv(own)[0], pair_mid_v), 12.0,
                             peers(pair_positions, state, 12.0))
                trace = route.trace_state
                self.assertEqual(route.mode, "PAIR_PROGRESS_SYNC")
                self.assertEqual(trace["active_sector"], [route.home_sector] * 2)
                self.assertAlmostEqual(trace["pair_mid_v_m"], pair_mid_v)
                self.assertAlmostEqual(trace["pair_long_speed_mps"], 8.0)
                self.assertAlmostEqual(trace["command_long_speed_mps"], 8.0)
                self.assertAlmostEqual(trace["search_total_speed_mps"], 22.0)
                end_u = route._active_bounds()[1 if route.cross_direction > 0 else 0]
                old_turns = route.turn_count
                route.target(point(end_u, pair_mid_v), 12.1,
                             peers(pair_positions, state, 12.0))
                self.assertEqual(route.turn_count, old_turns + 1)
                self.assertAlmostEqual(route.frontier_v, pair_mid_v)
                self.assertEqual(route.trace_state["active_sector"], [route.home_sector] * 2)

    def test_pair_progress_speed_error_and_heartbeat_timing(self):
        route, _, _ = self.start_sweep()
        state = {"20001": "COOP_TRACK_M", "20002": "COOP_TRACK_F"}
        pair_positions = {"20001": point(500.0, 1200.0),
                          "20002": point(1400.0, 1200.0)}
        route.target(point(2500.0, 1200.0), 11.0,
                     peers(pair_positions, state, 11.0))
        pair_positions = {"20001": point(500.0, 1206.0),
                          "20002": point(1400.0, 1210.0)}
        heartbeat = peers(pair_positions, state, 12.0)
        route.target(point(2500.0, 1148.0), 12.0, heartbeat)
        trace = route.trace_state
        self.assertAlmostEqual(trace["long_error_m"], 60.0)
        self.assertAlmostEqual(trace["align_long_speed_mps"], 10.0)
        self.assertAlmostEqual(trace["command_long_speed_mps"], 18.0)
        self.assertAlmostEqual(trace["command_short_speed_mps"] ** 2 +
                               trace["command_long_speed_mps"] ** 2, 22.0 ** 2)
        route.target(point(2500.0, 1268.0), 12.1, heartbeat)
        self.assertAlmostEqual(route.trace_state["align_long_speed_mps"], -10.0)
        self.assertAlmostEqual(route.trace_state["command_long_speed_mps"], -2.0)
        repeated = peers({"20001": point(500.0, 1400.0),
                          "20002": point(1400.0, 1400.0)}, state, 12.0)
        route.target(point(2500.0, 1400.0), 12.2, repeated)
        self.assertAlmostEqual(route.trace_state["pair_long_speed_mps"], 8.0)
        self.assertAlmostEqual(route._peer_long_samples["20001"][1], 1206.0)

    def test_pair_speed_changes_and_restore_from_current_position(self):
        route, _, _ = self.start_sweep()
        state = {"20001": "COOP_TRACK_M", "20002": "COOP_TRACK_F"}
        u = {"20001": 500.0, "20002": 1400.0}
        v = 1200.0
        for now, delta in ((11.0, 0.0), (12.0, 4.0),
                           (13.0, 8.0), (14.0, -3.0)):
            v += delta
            positions = {uid: point(u[uid], v) for uid in u}
            route.target(point(2500.0, v), now, peers(positions, state, now))
            if now > 11.0:
                self.assertAlmostEqual(route.trace_state["pair_long_speed_mps"], delta)
                self.assertAlmostEqual(route.trace_state["command_long_speed_mps"], delta)
        route.target(point(2500.0, 2300.0), 15.0,
                     peers(positions, now=15.0))
        self.assertEqual(route.mode, "SWEEP")
        self.assertAlmostEqual(route.frontier_v, 2300.0)
        self.assertFalse(route.trace_state["pair_progress_sync"])
        route.target(route.last_target, 15.1, peers(positions, now=15.1))
        self.assertAlmostEqual(route.frontier_v, 3000.0)
    def test_search_speed_and_resume_from_current_position(self):
        route, own, _ = self.start_sweep()
        control = V4Control("20003")
        control.route = route
        control.coord.peers = peers(
            {"20001": point(500.0, 1100.0), "20002": point(1500.0, 1100.0)},
            {"20001": "COOP_TRACK_M", "20002": "COOP_TRACK_F"}, 11.0)
        self_obs = SimpleNamespace(
            lat=own[0], lon=own[1], alt=500.0, heading_deg=180.0,
            gimbal_pan=0.0, gimbal_tilt=-45.0, gimbal_fov_deg=48.0)
        commands = control.step(SimpleNamespace(self=self_obs, comm_inbox=[]), 11.0)
        flight = next(item for item in commands if item.verb == "set_destination")
        self.assertIn(flight.params["speed"], (15.0, 22.0))
        trace = route.trace_state
        self.assertTrue(trace["pair_progress_sync"])
        self.assertAlmostEqual(math.hypot(trace["command_long_speed_mps"],
                                      trace["command_short_speed_mps"]),
                               flight.params["speed"])
        route.pause()
        resumed = point(2350.0, 2300.0)
        route.target(resumed, 12.0, control.coord.peers)
        self.assertAlmostEqual(route.frontier_v, 2300.0)
        self.assertEqual(route.trace_state["active_sector"], [2, 2])

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

    def test_partner_filter_and_heartbeat_budget(self):
        coordinator = V4Coordinator("20001")
        own = point(500.0, 1100.0)
        coordinator.peers = {
            "20002": (point(600.0, 1100.0), 10.0, "SEARCH"),
            "20003": (point(1500.0, 1100.0), 10.0, "SEARCH"),
        }
        self.assertEqual(coordinator.select_partner(own, 10.0, {"20003"}), "20003")
        self.assertIsNone(coordinator.select_partner(own, 10.0, {"20004"}))
        self.assertEqual(coordinator.select_partner(own, 10.0), "20002")
        for state in ("SEARCH", "VERIFY", "CALLING", "FOLLOWER_APPROACH",
                      "COOP_TRACK_M", "COOP_TRACK_F"):
            heartbeat = V4Coordinator("20001")
            heartbeat.queue_message("H", 10.0, position=(27.025, 125.020), state=state)
            self.assertLessEqual(len(heartbeat.queue[0][1].encode("utf-8")), 50)


if __name__ == "__main__":
    unittest.main()
