# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：验证无高度关联、照片时间对齐和旁路控制不变。
# 修改内容：增加独立几何样例、歧义拒绝和原版 V1 轨迹回放对照。
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from personal_hf2026.visual_geometry import PixelBox, aligned_sample, bind_boxes
from personal_hf2026.personal_v1 import PersonalV1Agent
from personal_hf2026.personal_v2 import PersonalV2Agent, VisualShadow
from collections import Counter
from competition.sdk.core.observation import Detection, SelfView, Observation, MissionBriefing, ScoreView


OWN = dict(lat=27.0, lon=125.0, alt=500, heading_deg=0, gimbal_pan=0,
           gimbal_tilt=-90, gimbal_fov_deg=48)


class GeometryTests(unittest.TestCase):
    def test_north_without_height_and_class_independence(self):
        sample = {"own": OWN, "tracks": {"n": [27.001, 125], "e": [27, 125.001]}}
        for label in ("true_vehicle", "decoy_vehicle"):
            box = PixelBox(503, 170, 520, 190, 0.9, 1024, 768, label)
            result = bind_boxes([box], sample)[0]
            self.assertEqual((result["status"], result["track_id"]), ("bound", "n"))

    def test_collinear_and_duplicate_boxes_rejected(self):
        box = PixelBox(503, 170, 520, 190, 0.9, 1024, 768)
        sample = {"own": OWN, "tracks": {"a": [27.001, 125], "b": [27.002, 125]}}
        self.assertEqual(bind_boxes([box], sample)[0]["status"], "ambiguous")
        sample["tracks"].pop("b")
        self.assertTrue(all(x["status"] == "multiple_boxes" for x in bind_boxes([box, box], sample)))

    def test_nadir_and_turn_rejected(self):
        sample = {"own": OWN, "tracks": {"a": [27.001, 125]}}
        box = PixelBox(508, 380, 516, 388, 0.9, 1024, 768)
        self.assertEqual(bind_boxes([box], sample)[0]["status"], "near_nadir_or_upward")
        sample["angular_rate_dps"] = 25
        self.assertEqual(bind_boxes([box], sample)[0]["status"], "fast_pose")

    def test_source_time_interpolation_and_wrap(self):
        a = {"t": 1, "own": dict(OWN, heading_deg=359), "tracks": {"1": [27, 125]}}
        b = {"t": 1.2, "own": dict(OWN, heading_deg=1), "tracks": {"1": [27.002, 125], "2": [27, 125]}}
        result, reason = aligned_sample([a, b], 1.1)
        self.assertEqual(reason, "interpolated")
        self.assertAlmostEqual(result["own"]["heading_deg"] % 360, 0)
        self.assertAlmostEqual(result["tracks"]["1"][0], 27.001)
        self.assertNotIn("2", result["tracks"])
        self.assertIsNone(aligned_sample([a, b], 0.9)[0])
        self.assertIsNone(aligned_sample([a, dict(b, t=2)], 1.5)[0])


class ControlTests(unittest.TestCase):
    def test_missing_initial_clock_is_waiting_not_failure(self):
        shadow = VisualShadow.__new__(VisualShadow)
        shadow.stats = Counter()
        shadow.observe(SimpleNamespace(briefing=SimpleNamespace(score_view=None)), None, None)
        self.assertEqual(shadow.stats["waiting_for_clock_ticks"], 1)

    def test_shadow_does_not_change_actions(self):
        first, second = PersonalV1Agent(my_uid="20001"), PersonalV2Agent(my_uid="20001")
        first.configure({})
        second.configure({})
        first.reset()
        second.reset()
        calls = []
        second.shadow = SimpleNamespace(observe=lambda obs, packet, agent: calls.append(obs))
        for tick in range(100):
            t = tick / 10
            d = Detection(True, 1, 27.001 + tick * .00001, 125, target_type="ground_vehicle")
            own = SelfView("20001", 27, 125, 500, 0, 22, 0, -90, 48, d, photo=b"shadow", detections=(d,))
            brief = MissionBriefing("20001", 3, target_count=3, score_view=ScoreView(0, (), False, 0, 3, t))
            obs = Observation(own, (), brief)
            self.assertEqual([asdict(c) for c in first.decide(obs, .1)],
                             [asdict(c) for c in second.decide(obs, .1)])
        self.assertEqual(len(calls), 100)


if __name__ == "__main__":
    unittest.main()
