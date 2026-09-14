# 修改时间：2026-09-13。
# 修改目的：避免融合视频标题遮挡边缘车辆并支持只重建媒体。
# 修改内容：复用图外标题面板并将视频导出独立为函数。
# 修改时间：2026-09-13。
# 修改目的：对已经观察到的互补候选只做一次无模型合并验证。
# 修改内容：沿用静态 NMS 阈值去重固定输出并生成融合指标与视频。
import argparse
from pathlib import Path
import cv2
from .common import read_json,write_json,read_records,save_records,evaluate,iou,sample_frames,load_rgb,overlay
from .compare import panel


def merge_candidates(static,motion,threshold=.65):
    # 两路分数不可直接比较，固定先保留静态候选，再追加未重叠的动态候选。
    kept=[]
    for name,items in [('static',static),('motion',motion)]:
        for c in sorted(items,key=lambda c:-c['score']):
            if all(iou(c['box'],other['box'])<=threshold for other in kept):
                kept.append(dict(c,origin=name))
    return kept

def make_media(manifest,rows,out):
    videos=out/'videos';videos.mkdir(exist_ok=True);index={r['key']:r for r in rows}
    for clip in manifest['clips']:
        if clip['split']!='test':continue
        w=cv2.VideoWriter(str(videos/(clip['id']+'.mp4')),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),10/3,(1024,816))
        if not w.isOpened():raise RuntimeError('H264 writer unavailable')
        for f in sample_frames(clip,.3):w.write(panel(load_rgb(f),f,index[f['key']],'static + motion / fixed NMS'))
        w.release()


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--compare');a=p.parse_args()
    batch=Path(a.batch).resolve();root=Path(a.compare).resolve() if a.compare else batch/'compare'
    out=root/'fusion';out.mkdir(parents=True,exist_ok=True)
    config={'nms_iou':.65,'source_priority':['static','motion'],'model':None,
            'choice':'reuse static NMS threshold; one frozen merge; no threshold sweep'}
    write_json(out/'config.json',config)
    manifest=read_json(batch/'manifest.json');s=read_records(root/'static_records.jsonl');m=read_records(root/'motion_records.jsonl')
    mi={r['key']:r for r in m};assert set(mi)=={r['key'] for r in s}
    from time import perf_counter
    rows=[]
    for sr in s:
        mr=mi[sr['key']];start=perf_counter();candidates=merge_candidates(sr['candidates'],mr['candidates'],config['nms_iou'])
        merge_ms=(perf_counter()-start)*1000
        rows.append(dict(key=sr['key'],clip=sr['clip'],uid=sr['uid'],time=sr['time'],source_time=sr['source_time'],
                         candidates=candidates,processed=True,elapsed_ms=sr['elapsed_ms']+mr['elapsed_ms']+merge_ms,merge_ms=merge_ms))
    save_records(out/'records.jsonl',rows)
    metrics={str(fov):evaluate(manifest,rows,'test',fov) for fov in (48,30,None)}
    write_json(out/'summary.json',metrics)
    make_media(manifest,rows,out)
    lines=['# 一次固定候选合并','',
           '静态优先，随后加入动态稳定候选；沿用静态 NMS IoU=0.65 去重。不训练模型，不扫描阈值。',
           '耗时为两路统一回放耗时之和加实测合并开销，属于顺序执行估计；未运行新的检测器。','',
           '|FOV|全部召回|移动召回|移动1秒发现|误报/帧|顺序耗时毫秒|','|---|---:|---:|---:|---:|---:|']
    for fov in (48,30):
        s=metrics[str(fov)]
        lines.append(f"|{fov}°|{s['recall']:.2%}|{s['moving_recall']:.2%}|{s['moving_found_1s']}/{s['moving_events']}|{s['fp_per_frame']:.2f}|{s['mean_ms']:.1f}|")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (root/'REPORT.md').open('a',encoding='utf-8') as f:f.write('\n## 一次固定候选合并\n\n'+'\n'.join(lines[2:])+'\n')
    print({f:{k:v for k,v in s.items() if k!='event_details'} for f,s in metrics.items()})


if __name__=='__main__':main()
