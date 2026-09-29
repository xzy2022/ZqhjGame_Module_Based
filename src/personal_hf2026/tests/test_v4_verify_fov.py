# 修改时间：2026-09-29。
# 修改目的：验证 VERIFY 的安全变焦、邻近诱饵拒绝及 SEARCH 实测宽视野恢复。
# 修改内容：覆盖配置、内部阶段连续帧、正式状态转移、搜索飞行和原有类别判据。
"""V4 VERIFY 变焦与分类证据的聚焦测试。"""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from collections import Counter
from threading import Condition

import cv2
import numpy as np

from personal_hf2026.v4_control import V4Control
from personal_hf2026.v3_perception import V3PerceptionWorker
from personal_hf2026.v4_perception import (V4PerceptionWorker, VisionDiagnosticV4,
                                           submit_observation)


SIZE = (640, 480)
EDGE = (570.0, 210.0, 610.0, 250.0)
CENTER = (430.0, 210.0, 470.0, 250.0)
FAR = (10.0, 210.0, 50.0, 250.0)


def obj(name="real_vehicle", box=CENTER):
    return SimpleNamespace(class_name=name, bbox_xyxy=box)


def frame(index, objects=(), fov=48.0, now=None):
    return SimpleNamespace(source_sim_time=index * 0.1 if now is None else now,
                           frame_id=f"f{index}", error=None,
                           raw_yolo_objects=tuple(objects),
                           effective_yolo_objects=tuple(objects),
                           image_size=SIZE,
                           source_pose={"gimbal_fov_deg": fov,
                                        "gimbal_pan": 0.0, "gimbal_tilt": -45.0},
                           image_bgr=None)


def observation(fov):
    own = SimpleNamespace(lat=27.025, lon=125.020, alt=500.0,
                          heading_deg=0.0, gimbal_pan=0.0,
                          gimbal_tilt=-45.0, gimbal_fov_deg=fov)
    return SimpleNamespace(self=own, comm_inbox=[])


def requested_fov(commands):
    return next(command.params["angle"] for command in commands
                if command.verb == "set_fov")


def control():
    item = V4Control("20001")
    item.route.target = lambda *args: (27.025, 125.021)
    item.motion.update = lambda *args: {}
    return item


def start_verify(item, box=EDGE):
    for index in range(3):
        item.consume_visual(frame(index, (obj(box=box),)))
    assert item.state == "VERIFY"
    return 3


def reach_confirm(item, index=3):
    for _ in range(2):
        item.consume_visual(frame(index, (obj(),)))
        index += 1
    assert item._verify_phase == "ZOOMING_30"
    for fov in (40.0, 34.0, 31.0, 31.0):
        item.consume_visual(frame(index, (obj(),), fov=fov))
        index += 1
    assert item._verify_phase == "CONFIRM_30"
    return index


class VerifyFovTest(unittest.TestCase):
    def test_v4_submits_real_fov30_frame_while_v3_stays_fov48_only(self):
        worker = V4PerceptionWorker.__new__(V4PerceptionWorker)
        worker._condition = Condition()
        worker._pending = {}
        worker._latest = {}
        worker._last_signature = {}
        worker._uid_order = []
        worker._submission_sequence = 0
        worker._stopping = False
        worker._stats = Counter()
        own = SimpleNamespace(uid="20001", photo=b"synthetic-photo", lat=27.0,
                              lon=125.0, alt=500.0, heading_deg=0.0,
                              gimbal_pan=0.0, gimbal_tilt=-45.0,
                              gimbal_fov_deg=30.0)
        obs = SimpleNamespace(self=own,
                              briefing=SimpleNamespace(score_view=SimpleNamespace(sim_time=1.0)))
        submit_observation(worker, obs)
        self.assertEqual(worker._pending["20001"].fov_deg, 30.0)
        self.assertEqual(worker._pending["20001"].own_pose["gimbal_fov_deg"], 30.0)
        self.assertEqual(worker._stats["rejected_non_fov48"], 0)
        self.assertFalse(V3PerceptionWorker._accept_fov(worker, 30.0))
        self.assertTrue(V3PerceptionWorker._accept_fov(worker, 48.0))

    def test_confidence_and_original_unknown_threshold(self):
        path = (Path(__file__).resolve().parents[3] /
                "configs/detectors/vehicle_prop_v2/vehicle_realtime.json")
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["detector"]["confidence"], 0.05)
        self.assertEqual(config["detector"]["iou"], 0.5)
        self.assertEqual(config["unknown_threshold"], 0.6)

    def test_edge_does_not_zoom_and_two_safe_frames_do(self):
        item = control()
        index = start_verify(item)
        self.assertEqual(item._verify_phase, "CENTER_48")
        self.assertEqual(requested_fov(item.step(observation(48.0), 0.2)), 48.0)
        item.consume_visual(frame(index, (obj(box=EDGE),)))
        self.assertEqual(item._verify_zoom_safe_frames, 0)
        self.assertEqual(requested_fov(item.step(observation(48.0), 0.3)), 48.0)
        item.consume_visual(frame(index + 1, (obj(),)))
        self.assertEqual(item._verify_phase, "CENTER_48")
        item.consume_visual(frame(index + 2, (obj(),)))
        self.assertEqual(item._verify_phase, "ZOOMING_30")
        self.assertEqual(requested_fov(item.step(observation(48.0), 0.5)), 30.0)

    def test_actual_fov_needs_two_valid_frames_and_new_confirm30_frames(self):
        item = control()
        index = start_verify(item)
        for _ in range(2):
            item.consume_visual(frame(index, (obj(),)))
            index += 1
        self.assertEqual(item._verify_phase, "ZOOMING_30")
        for fov in (40.0, 34.0, 31.0):
            item.consume_visual(frame(index, (obj(),), fov=fov))
            self.assertEqual(item._verify_phase, "ZOOMING_30")
            index += 1
        item.consume_visual(frame(index, (obj(),), fov=31.0))
        self.assertEqual(item._verify_phase, "CONFIRM_30")
        self.assertEqual(item._verify_confirm30_frames, 0)
        item.motion.decision = "MOVING"
        for count in (1, 2):
            index += 1
            item.consume_visual(frame(index, (obj(),), fov=30.0))
            self.assertEqual(item._verify_confirm30_frames, count)
            self.assertEqual(item.state, "VERIFY")
        index += 1
        item.consume_visual(frame(index, (obj(),), fov=30.0))
        self.assertEqual(item.state, "CALLING")

    def test_missing_frame_breaks_zoom_safe_and_fov_ready_streaks(self):
        item = control()
        index = start_verify(item)
        item.consume_visual(frame(index, (obj(),)))
        self.assertEqual(item._verify_zoom_safe_frames, 1)
        item.consume_visual(frame(index + 1, ()))
        self.assertEqual(item._verify_zoom_safe_frames, 0)
        for offset in (2, 3):
            item.consume_visual(frame(index + offset, (obj(),)))
        self.assertEqual(item._verify_phase, "ZOOMING_30")
        item.consume_visual(frame(index + 4, (obj(),), fov=31.0))
        self.assertEqual(item._verify_fov30_ready_frames, 1)
        item.consume_visual(frame(index + 5, ()))
        self.assertEqual(item._verify_fov30_ready_frames, 0)
        item.consume_visual(frame(index + 6, (obj(),), fov=31.0))
        self.assertEqual(item._verify_phase, "ZOOMING_30")

    def test_near_model_prop_rejected_immediately_to_search(self):
        item = control()
        index = reach_confirm(item, start_verify(item))
        item.consume_visual(frame(index, (obj("model_prop"),), fov=30.0))
        self.assertEqual(item.state, "VERIFY")
        item.consume_visual(frame(index + 1, (obj("model_prop"),), fov=30.0))
        self.assertEqual(item.state, "SEARCH")
        self.assertTrue(item._search_wait_wide_fov)
        self.assertEqual([event["reason"] for event in item.pop_events()
                          if event["event"] == "state_changed"][-1], "verify_model_prop")

    def test_distant_model_prop_and_uncertain_do_not_reject(self):
        item = control()
        index = reach_confirm(item, start_verify(item))
        item.consume_visual(frame(index, (obj(), obj("model_prop", FAR)), fov=30.0))
        self.assertEqual(item._verify_nearest_object_class, "real_vehicle")
        item.consume_visual(frame(index + 1, (obj("uncertain"),), fov=30.0))
        item.consume_visual(frame(index + 2, (obj("uncertain"),), fov=30.0))
        self.assertEqual(item.state, "VERIFY")
        self.assertEqual(item._verify_model_prop_frames, 0)

    def test_search_wide_recovery_keeps_flying_and_blocks_scan_and_candidate(self):
        item = control()
        reach_confirm(item, start_verify(item))
        item._return_search(1.0, "verify_timeout")
        scans = []
        item.search_gimbal.scan = lambda *args: (scans.append(args) or (0.0, -45.0))
        for index, fov in enumerate((35.0, 43.0, 47.0, 47.0), 20):
            commands = item.step(observation(fov), index * 0.1)
            self.assertTrue(any(command.verb == "set_destination" for command in commands))
            self.assertEqual(requested_fov(commands), 48.0)
            if index < 23:
                self.assertTrue(item._search_wait_wide_fov)
                self.assertEqual(len(scans), 0)
                item.consume_visual(frame(index, (obj(),), fov=fov))
                self.assertEqual(item._search_candidate_frames, 0)
        self.assertFalse(item._search_wait_wide_fov)
        self.assertEqual(len(scans), 1)
        item.consume_visual(frame(24, (obj(),), fov=47.0))
        self.assertEqual(item._search_candidate_frames, 1)

    def test_category_rule_still_uses_max_probability_and_real_tie(self):
        worker = V4PerceptionWorker.__new__(V4PerceptionWorker)
        worker.diagnostic = VisionDiagnosticV4("000")
        worker._unknown_threshold = 0.6
        ok, encoded = cv2.imencode(".jpg", np.zeros((20, 20, 3), dtype=np.uint8))
        self.assertTrue(ok)
        rows = [(1, 1, 5, 5, 0.9, 0.7, 0.7),
                (6, 1, 10, 5, 0.9, 0.1, 0.8),
                (11, 1, 15, 5, 0.9, 0.59, 0.41)]
        detector = SimpleNamespace(predict=lambda image: rows)
        job = SimpleNamespace(photo=encoded.tobytes(), frame_id="classification",
                              source_sim_time=0.0, source_time_basis="test",
                              observed_sim_time=0.0, own_pose={}, diagnostic_metadata=None,
                              uid="20001")
        snapshot = worker._infer(detector, job)
        self.assertEqual([item.class_name for item in snapshot.effective_yolo_objects],
                         ["real_vehicle", "model_prop", "uncertain"])


if __name__ == "__main__":
    unittest.main()
