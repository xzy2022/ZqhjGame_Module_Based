# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13 14:24
# 修改目的：检查连续急转弯观测可被接受，同时保留跳点和长间隔方向约束。
# 修改内容：覆盖短距离反向运动、超速、预测偏差及缺测后的反向检测。
import unittest

from personal_hf2026.tracking import LocalTrackManager, _move


def moving_track(speed=10.):
    track = LocalTrackManager()
    for i in range(21):
        track.update(i / 10, _move((27., 125.), speed * i / 10, 0))
    return track


class TurnAssociationTests(unittest.TestCase):
    def test_small_continuous_turn_remains_confirmed(self):
        track = moving_track()
        for i in range(1, 21):
            point = _move((27., 125.), 20-.65*i, .65*i)
            chosen = track.select_position(2+i/10, (point, _move(point, 100, 0)))
            state = track.update(2+i/10, chosen)
            self.assertEqual(state.state, 'CONFIRMED')
            self.assertEqual(state.last_seen, 2+i/10)

    def test_long_gap_does_not_bypass_heading(self):
        track = moving_track()
        self.assertFalse(track._compatible(2.5, _move(track.position, -1, 0)))

    def test_large_reverse_displacement_is_rejected(self):
        track = moving_track()
        self.assertFalse(track._compatible(2.25, _move(track.position, -2.5, 0)))

    def test_small_but_impossibly_fast_point_is_rejected(self):
        track = moving_track()
        self.assertFalse(track._compatible(2.01, _move(track.position, -1, 0)))

    def test_small_displacement_with_large_prediction_error_is_rejected(self):
        track = moving_track(20.)
        self.assertFalse(track._compatible(2.25, _move(track.position, -.5, 0)))

    def test_velocity_jump_still_rejected(self):
        track = moving_track(20.)
        self.assertFalse(track._compatible(2.05, _move(track.position, -.5, 0)))


if __name__ == '__main__':
    unittest.main()
