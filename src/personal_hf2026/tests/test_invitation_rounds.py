# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13 14:43
# 修改目的：覆盖实际收件箱重复返回选人前邀请的情况。
# 修改内容：确认同轮旧的未定伙伴消息不会把已应答从机释放。
# 修改时间：2026-09-13 14:32
# 修改目的：复现超时改选的一主两从竞态，并检查丢包和乱序后的伙伴排他性。
# 修改内容：覆盖邀请轮次、延迟应答、旧消息、正式伙伴确认、阶段重传及报文长度。
import unittest
from types import SimpleNamespace as NS

from personal_hf2026.coordination import CoopCoordinator, _Offer, _PeerStatus
from personal_hf2026.tracking import TrackPoint, TrackSnapshot, _move

UIDS = ('20001', '20002', '20003')


def local(now, epoch=3):
    return TrackSnapshot(epoch, 'CONFIRMED', tuple(
        TrackPoint(now-i/10, *_move((27., 125.), (now-i/10)*10, 0))
        for i in reversed(range(31))), now)


def message(uid, payload, now):
    return NS(sender_uid=uid, payload=payload, recv_time=now)


def setup_race(cls=CoopCoordinator):
    b = cls('20002', UIDS)
    b.phase = b.HOLD
    b.proposal = _Offer('20002', 4310, 3, 437., local(437.).position)
    b.proposal_created_at = 431.
    b.selected_follower_uid = '20003'
    b.selection_started_at = 432.133
    c = cls('20003', UIDS)
    offer = b._offer_payload(local(437.), '20003')
    c.step(437.083, local(437.083), [message('20002', offer, 437.083)], False)
    b.peer_status = {'20001': _PeerStatus(b.SEARCH, (27., 125.), 437.18),
                     '20003': _PeerStatus(b.INIT, (27., 125.), 437.18)}
    b.step(437.183, local(437.183), (), False)
    return b, c, offer


class InvitationRoundTests(unittest.TestCase):
    def test_repeated_unselected_offer_does_not_release_chosen_follower(self):
        b, c, chosen = setup_race()
        unselected = chosen.rsplit(',', 1)[0] + ',0'
        for i in range(5):
            now = 437.3+i/10
            c.step(now, local(now), [message('20002', unselected, 436.),
                                   message('20002', chosen, 437.)], False)
            self.assertEqual(c.role, c.FOLLOWER)

    def test_reselection_changes_session_and_releases_old_follower(self):
        b, c, old_offer = setup_race()
        old_session = c.current_session
        self.assertNotEqual(b.proposal.session, old_session)
        self.assertIn(old_session, b.cancelled_sessions)
        new_offer = b._offer_payload(local(437.8), '20001')
        result = c.step(437.8, local(437.8), [message('20002', new_offer, 437.8)], False)
        self.assertEqual(c.role, c.NONE)
        self.assertTrue(result.reset_local_track)
        c.step(438., local(438.), [message('20002', old_offer, 438.)], False)
        self.assertEqual(c.role, c.NONE)

    def test_old_accept_cannot_bind_new_invitation(self):
        b, c, _ = setup_race()
        # 即使迟到应答来自下一轮同一个人，也必须检查会话编号。
        b.selected_follower_uid = '20003'
        b.step(437.3, local(437.3), [message('20003', c._session_payload('A'), 437.3)], False)
        self.assertEqual(b.phase, b.HOLD)

    def test_response_at_timeout_wins_before_reselection(self):
        b, c, _ = setup_race()
        b.proposal = _Offer('20002', 4310, 3, 437., local(437.).position)
        b.selected_follower_uid = '20003'
        b.selection_started_at = 432.133
        b.step(437.2, local(437.2), [message('20003', c._session_payload('A'), 437.2)], False)
        self.assertEqual(b.partner_uid, '20003')
        self.assertEqual(b.role, b.MASTER)

    def test_master_track_releases_old_follower_when_cancel_and_offer_lost(self):
        b, c, _ = setup_race()
        b._become_master('20001', 438.)
        payload = b._track_payload(local(438.), 438.)
        result = c.step(438.1, local(438.1), [message('20002', payload, 438.1)], False)
        self.assertEqual(c.role, c.NONE)
        self.assertTrue(result.reset_local_track)

    def test_phase_recovers_without_g_and_ignores_stale_cancel(self):
        b, _, _ = setup_race()
        a = CoopCoordinator('20001', UIDS)
        offer = b._offer_payload(local(437.8), '20001')
        a.step(437.8, local(437.8), [message('20002', offer, 437.8)], False)
        b.step(438., local(438.), [message('20001', a._session_payload('A'), 438.)], False)
        b.phase = b.ACTIVE
        payload = b._track_payload(local(438.1), 438.1)
        a.step(438.1, local(438.1), [message('20002', payload, 438.1)], False)
        self.assertEqual(a.phase, a.ACTIVE)
        a.step(438.2, local(438.2), [message('20003', 'C,2,4310,3', 438.2)], False)
        self.assertEqual(a.phase, a.ACTIVE)

    def test_g_requires_master_and_selected_partner(self):
        b, c, old_offer = setup_race()
        session = c.current_session
        payload = f'G,2,{session[1]},{session[2]},3'
        c.step(437.3, local(437.3), [message('20001', payload, 437.3)], False)
        self.assertEqual(c.phase, c.INIT)
        c.step(437.4, local(437.4), [message('20002', payload[:-1]+'1', 437.4)], False)
        self.assertEqual(c.role, c.NONE)
        c.step(437.5, local(437.5), [message('20002', old_offer, 437.5)], False)
        self.assertEqual(c.role, c.NONE)

    def test_invalid_track_still_confirms_partner_within_byte_budget(self):
        b, _, _ = setup_race()
        b._become_master('20001', 438.)
        for available in (True, False):
            payload = b._track_payload(local(438.), 438., False, track_available=available)
            self.assertLessEqual(len(payload.encode('utf-8')), 50)
            kind, data = b._parse(message('20002', payload, 438.))
            self.assertEqual(kind, 'T')
            self.assertFalse(data[3])
            self.assertEqual(data[-2:], ('20001', b.INIT))
            self.assertEqual(data[4] is None, not available)


if __name__ == '__main__':
    unittest.main()
