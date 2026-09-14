# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13（采集后核验）
# 修改目的：确认负帧龄得到显式标记且不被改写为零。
# 修改内容：补充源时间领先观测快照的离线索引断言。
# 修改时间：2026-09-13
# 修改目的：验证旧帧保留、唯一帧去重和照片源时钟关联的采集约定。
# 修改内容：增加临时目录中的采集、角度环绕及默认关闭测试，不启动仿真或分类。
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from personal_hf2026.capture_dataset import DatasetRecorder, build_index, rows


class CaptureDatasetTests(unittest.TestCase):
    def test_default_cache_does_not_create_dataset(self):
        from personal_hf2026.visual_shadow_study import TimedPhotoCache
        with TemporaryDirectory() as folder, patch.object(TimedPhotoCache, "_read", return_value=None):
            cache = TimedPhotoCache(["u"], folder, "127.0.0.1", 6379)
            self.assertIsNone(cache.dataset)
            self.assertFalse((Path(folder) / "dataset").exists())
            cache.audit.close()
            cache.redis.close()

    def test_unique_frames_and_unannotated_projection(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            image = BytesIO()
            Image.new("RGB", (32, 24)).save(image, format="JPEG")
            raw = dict(frame_no=1, source_sim_time=100, received_unix_s=10000,
                       image=image.getvalue(), audit_boxes=[dict(target_id="a", **{"class": "DecoyVehicle"}, bbox=[1, 2, 5, 6])])
            recorder = DatasetRecorder(output)
            recorder.record("u", raw)
            recorder.record("u", raw)
            recorder.record("u", dict(raw, frame_no=2))
            recorder.close()
            saved = list(rows(output / "dataset/frames.jsonl"))
            self.assertEqual(len(saved), 2)
            self.assertEqual((saved[0]["width"], saved[0]["height"]), (32, 24))
            self.assertEqual(saved[0]["image_sha256"], saved[1]["image_sha256"])
            self.assertIsNone(saved[0]["ue_projected_objects"][0]["visibility"])
            self.assertEqual((output / saved[0]["image_path"]).read_bytes(), raw["image"])

    def test_world_time_alignment_and_stale_frame_retained(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "dataset").mkdir()
            def write(name, data):
                (output / name).write_text("".join(json.dumps(r) + "\n" for r in data), encoding="utf-8")
            pose = dict(lat=27, lon=125, alt=500, heading_deg=359, gimbal_pan=0,
                        gimbal_tilt=-55, gimbal_fov_deg=48, detections=[], detection={})
            observations = [dict(uid="u", sim_time=t, t=t-99.9, observed_unix_s=10000,
                self=dict(pose, heading_deg=heading), state="SEARCH", candidate=None,
                local_track={}, search_filter={}) for t, heading in [(100, 359), (100.2, 1)]]
            write("observations.jsonl", observations)
            write("judge.jsonl", [dict(sim_time=t, t=t-100, targets={}, world_targets={},
                                       world_decoys={}) for t in [100, 100.2]])
            frame = dict(uid="u", frame_no=1, source_sim_time=100.1, width=32, height=24,
                         ue_projected_objects=[], visibility=None)
            write("dataset/frames.jsonl", [frame, dict(frame, frame_no=2, source_sim_time=101)])
            write("dataset/deliveries.jsonl", [dict(uid="u", frame_no=1, source_sim_time=100.1, frame_age_s=12),
                dict(uid="u", frame_no=2, source_sim_time=101, frame_age_s=-.2)])
            result = build_index(output)
            saved = list(rows(output / "dataset/samples.jsonl"))
            self.assertAlmostEqual(saved[0]["source_t"], .1)
            self.assertEqual(saved[0]["alignment"], "interpolated")
            self.assertAlmostEqual((saved[0]["source_pose"]["heading_deg"] + 180) % 360 - 180, 0)
            self.assertEqual(saved[0]["first_agent_delivery"]["frame_age_s"], 12)
            self.assertIsNone(saved[1]["source_pose"])
            self.assertIsNone(saved[1]["reference"])
            self.assertEqual(result["counts"]["frames"], 2)
            self.assertEqual(result["counts"]["stale_at_first_delivery"], 1)
            self.assertEqual(result["counts"]["source_ahead_at_first_delivery"], 1)
            self.assertEqual(saved[1]["first_agent_delivery"]["time_status"], "source_ahead_of_observation")
            self.assertEqual(saved[1]["first_agent_delivery"]["frame_age_s"], -.2)


if __name__ == "__main__":
    unittest.main()
