# 修改时间：2026-09-13。
# 修改目的：验证离线发现率的关键分母与时间边界。
# 修改内容：覆盖一对一匹配、初始化跳帧、超时发现及源时间采样。
import unittest
from .common import evaluate,match,sample_frames


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frames=[{'key':str(i),'time':t,'labels':[{'id':'v','box':[0,0,10,10],'moving':True}]} for i,t in enumerate((0.,.3,.6,.9,1.2))]
        self.manifest={'clips':[{'id':'clip','fov':48,'split':'test','start':0,'frames':self.frames}]}

    def records(self,hit_at):
        return [{'key':f['key'],'elapsed_ms':1,'processed':i>0,
                 'candidates':[{'box':[0,0,10,10],'score':1}] if i==hit_at else []} for i,f in enumerate(self.frames)]

    def test_duplicate_is_false_positive(self):
        c=[{'box':[0,0,10,10],'score':1}]*2
        self.assertEqual(len(match(c,self.frames[0]['labels'])),1)

    def test_maximum_cardinality_reassigns(self):
        # 第一个候选可匹配两辆车，第二个只覆盖第一辆，必须允许重分配。
        c=[{'box':[0,0,20,10],'score':1},{'box':[0,0,10,10],'score':.9}]
        g=[{'box':[0,0,10,10]},{'box':[10,0,20,10]}]
        self.assertEqual(len(match(c,g)),2)

    def test_skipped_first_frame_stays_in_denominator(self):
        s=evaluate(self.manifest,self.records(3))
        self.assertEqual(s['frames'],5); self.assertEqual(s['gt'],5)
        self.assertEqual(s['recall'],.2); self.assertEqual(s['processed_ratio'],.8)
        self.assertEqual(s['discovery_1s'],1)

    def test_discovery_does_not_pause_clock(self):
        rows=self.records(4)
        for row in rows[:4]:row['processed']=False
        s=evaluate(self.manifest,rows)
        self.assertEqual(s['recall'],.2); self.assertEqual(s['discovery_1s'],0)

    def test_registered_only_has_explicit_conditional_denominator(self):
        s=evaluate(self.manifest,self.records(3),processed_only=True)
        self.assertEqual(s['frames'],4); self.assertEqual(s['recall'],.25)

    def test_all_skipped_has_zero_recall_not_empty_test(self):
        rows=self.records(-1)
        for row in rows:row['processed']=False
        s=evaluate(self.manifest,rows)
        self.assertEqual(s['frames'],5); self.assertEqual(s['recall'],0)
        self.assertEqual(s['processed_ratio'],0); self.assertEqual(s['discovery_1s'],0)

    def test_absolute_grid_avoids_drift(self):
        c={'start':0,'frames':[{'time':t} for t in (.02,.12,.22,.32,.42,.52,.62)]}
        self.assertEqual([f['time'] for f in sample_frames(c,.3)],[.02,.32,.62])


if __name__=='__main__':unittest.main()
