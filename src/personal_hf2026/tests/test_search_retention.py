# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13 14:47
# 修改目的：让搜索阶段不上报的断言检查 SDK 实际使用的指令名。
# 修改内容：将断言中的指令名改为 agent.report。
# 修改时间：2026-09-13 14:19
# 修改目的：验证已确认目标等待主锁定时保留轨迹，真实丢失仍及时释放。
# 修改内容：通过完整决策入口检查超时、短暂缺测、重新锁定和未确认候选。
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS

from personal_hf2026.personal_v1 import PersonalV1Agent
from personal_hf2026.tracking import TrackPoint, TrackSnapshot, _move


def observation(now, position=None, primary=None):
    def detection(point):
        return NS(detected=point is not None, target_lat=point[0] if point else None,
                  target_lon=point[1] if point else None)
    return NS(self=NS(lat=27.0, lon=125.0, alt=500., heading_deg=0.,
                      gimbal_pan=0., gimbal_tilt=-60., gimbal_fov_deg=48.,
                      detection=detection(primary),
                      detections=(detection(position),) if position else ()),
              comm_inbox=(), briefing=NS(score_view=NS(sim_time=now), target_count=3))


def seeded_agent(state="CONFIRMED"):
    agent = PersonalV1Agent("20001")
    agent.configure({})
    agent.reset()
    points = tuple(TrackPoint(i / 10, *_move((27., 125.), i, 0)) for i in range(21))
    agent._track.adopt(TrackSnapshot(1, state, points, 2.))
    agent._gimbal_lock.bind(agent._track.epoch, agent._track.position, 48.)
    agent._gimbal_lock.missing_since = 0.
    return agent


class SearchRetentionTests(unittest.TestCase):
    def test_visible_confirmed_survives_primary_timeout_without_cooperation(self):
        agent = seeded_agent()
        epoch = agent._track.epoch
        for i in range(21, 81):
            cmds = agent.decide(observation(i / 10, _move((27., 125.), i, 0)), .1)
            self.assertEqual(agent._track.epoch, epoch)
            self.assertEqual(agent._track.state, "CONFIRMED")
            self.assertEqual(agent._state, "SEARCH")
            self.assertEqual(agent._coop_seconds, 0)
            self.assertFalse(any(c.verb == "agent.report" for c in cmds))
        self.assertGreater(agent._t - agent._gimbal_lock.missing_since, 2.)

    def test_gap_is_bounded_by_track_loss(self):
        agent = seeded_agent()
        agent.decide(observation(2.1), .1)
        self.assertEqual(agent._track.state, "COASTING")
        agent.decide(observation(2.8), .7)
        self.assertEqual(agent._track.state, "LOST")
        self.assertFalse(agent._gimbal_lock.active)

    def test_primary_recovery_allows_proposal_after_stability(self):
        agent = seeded_agent()
        for i in range(21, 29):
            point = _move((27., 125.), i, 0)
            agent.decide(observation(i / 10, point, point), .1)
        self.assertEqual(agent._state, "HOLD_TARGET")
        self.assertEqual(agent._coop_seconds, 0)

    def test_unconfirmed_timeout_still_releases(self):
        agent = seeded_agent("TENTATIVE")
        agent._track.config = replace(agent._track.config, confirm_span_s=10.)
        agent.decide(observation(2.1, _move((27., 125.), 21, 0)), .1)
        self.assertEqual(agent._track.state, "LOST")


if __name__ == "__main__":
    unittest.main()
