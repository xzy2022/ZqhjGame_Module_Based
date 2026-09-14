# 修改时间：2026-09-13。
# 修改目的：防止连续三次检测计数和失败清零的需求再次偏离。
# 修改内容：用纯关联合成轨迹验证第三次才输出、失败后重新累计与静态背景拒绝。
"""不读取图片、标签或 test 数据的关联计数回归测试。"""
import unittest
import numpy as np
from .motion import build_detector


class ConsecutiveDetectionTest(unittest.TestCase):
    def test_third_candidate_and_failure_reset(self):
        detector = build_detector({})
        matrix = np.array([[1., 0., 4.], [0., 1., 2.], [0., 0., 1.]])

        def step(index):
            candidate = {'box': [100+12*index, 100+2*index, 118+12*index, 116+2*index], 'score': 1.}
            detector.tracks, stable = detector._associate([candidate], matrix, .3)
            return len(stable), detector.tracks[0]['steps']

        self.assertEqual([step(i) for i in range(3)], [(0, 1), (0, 2), (1, 3)])
        failure = detector._skip('homography_failed', {})
        self.assertFalse(failure['processed'])
        self.assertEqual(failure['candidates'], [])
        self.assertEqual(detector.tracks, [])
        self.assertEqual([step(i) for i in range(3, 6)], [(0, 1), (0, 2), (1, 3)])
        detector.reset()
        self.assertEqual(detector.tracks, [])

    def test_empty_frame_breaks_streak_and_background_is_not_motion(self):
        detector = build_detector({})
        matrix = np.eye(3)
        candidate = {'box': [100, 100, 118, 116], 'score': 1.}
        for _ in range(4):
            detector.tracks, stable = detector._associate([candidate], matrix, .3)
            self.assertEqual(stable, [])
            self.assertEqual(detector.tracks[0]['steps'], 1)
        detector.tracks, stable = detector._associate([], matrix, .3)
        self.assertEqual(detector.tracks, [])
        self.assertEqual(stable, [])


if __name__ == '__main__':
    unittest.main()
