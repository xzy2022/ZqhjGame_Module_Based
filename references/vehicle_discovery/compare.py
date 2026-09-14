# 修改时间：2026-09-13。
# 修改目的：避免视频标题遮挡原图边缘的小车辆。
# 修改内容：把标题移到图像外侧边栏并同步修正视频尺寸和查看裁剪坐标。
# 修改时间：2026-09-13。
# 修改目的：让总报告同时呈现静态筛选前上限并支持保留原批次重放。
# 修改内容：纳入固定 A0/A1 记录的公共指标并允许指定新的汇总输出目录。
# 修改时间：2026-09-13。
# 修改目的：保留逐片段表现及配准跳过原因以便审计整体指标。
# 修改内容：额外输出逐片段指标、跳过原因计数并检查原始动态候选字段。
# 修改时间：2026-09-13。
# 修改目的：让并排回放视频能使用通用 H.264 解码播放。
# 修改内容：采用本机已验证的 Windows Media Foundation H.264 编码器。
# 修改时间：2026-09-13。
# 修改目的：在汇总分支统一回测已冻结的两条车辆发现路线。
# 修改内容：复用公共评估生成同帧视频、案例、互补性审计和指标报告。
"""只做离线回放；不会连接仿真、读取检测标签或发出控制指令。"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from .common import (read_json,write_json,read_records,save_records,run_detector,evaluate,
                     sample_frames,match,iou,load_rgb,overlay)

def pct(x): return '—' if x is None else f'{100*x:.1f}%'

def brief(s):
    return {k:v for k,v in s.items() if k!='event_details'}

def panel(rgb,frame,record,title):
    """在原图外增加标题条，保留所有原始像素供边缘目标核验。"""
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for g in frame['labels']:
        x1,y1,x2,y2=map(int,g['box']);cv2.rectangle(bgr,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(bgr,('M ' if g['moving'] else 'V ')+g['id'],(x1,max(12,y1-4)),0,.4,(0,255,0),1)
    for c in record.get('candidates',[]):
        x1,y1,x2,y2=map(int,c['box']);cv2.rectangle(bgr,(x1,y1),(x2,y2),(0,140,255),1)
    bgr=cv2.copyMakeBorder(bgr,48,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
    cv2.putText(bgr,f"{title} t={frame['time']:.2f} N={len(record.get('candidates',[]))} ok={record.get('processed',True)}",(8,29),0,.65,(255,255,255),1)
    return bgr

def make_media(manifest,records,out):
    indexes={name:{r['key']:r for r in rows} for name,rows in records.items()}
    media=out/'videos'; media.mkdir(parents=True,exist_ok=True)
    examples=out/'examples'; examples.mkdir(exist_ok=True)
    pools={(name,kind):[] for name in records for kind in ('success','miss','false_positive')}
    for clip in manifest['clips']:
        if clip['split']!='test': continue
        writer=cv2.VideoWriter(str(media/(clip['id']+'.mp4')),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),10/3,(2048,816))
        if not writer.isOpened(): raise RuntimeError('video writer unavailable')
        frames=sample_frames(clip,.3)
        for frame in frames:
            rgb=load_rgb(frame)
            panels=[]
            for name in ('static','motion'):
                row=indexes[name][frame['key']]
                panels.append(panel(rgb,frame,row,name))
                pairs=match(row['candidates'],frame['labels'])
                if pairs:
                    i,j=max(pairs.items(),key=lambda ij:iou(row['candidates'][ij[0]]['box'],frame['labels'][ij[1]]['box']))
                    pools[(name,'success')].append((iou(row['candidates'][i]['box'],frame['labels'][j]['box']),clip,frame,frame['labels'][j]['box']))
                missed=[g for j,g in enumerate(frame['labels']) if j not in pairs.values()]
                if missed:
                    g=max(missed,key=lambda g:(g['moving'],(g['box'][2]-g['box'][0])*(g['box'][3]-g['box'][1])))
                    pools[(name,'miss')].append((int(g['moving'])*10000+(g['box'][2]-g['box'][0])*(g['box'][3]-g['box'][1]),clip,frame,g['box']))
                false=[c for i,c in enumerate(row['candidates']) if i not in pairs]
                if false:
                    c=max(false,key=lambda c:c.get('score',1))
                    pools[(name,'false_positive')].append((len(false),clip,frame,c['box']))
            writer.write(np.hstack(panels))
        writer.release()
    evidence=[]
    for (name,kind),pool in pools.items():
        seen=set()
        for rank,clip,frame,box in sorted(pool,key=lambda x:-x[0]):
            if clip['id'] in seen: continue
            seen.add(clip['id']); rgb=load_rgb(frame)
            panels=[panel(rgb,frame,indexes[n][frame['key']],n) for n in ('static','motion')]
            cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            side=max(80,2*max(box[2]-box[0],box[3]-box[1]))
            x1,y1=max(0,int(cx-side/2)),max(0,int(cy-side/2)); x2,y2=min(1024,int(cx+side/2)),min(768,int(cy+side/2))
            crops=[cv2.resize(p[y1+48:y2+48,x1:x2],(320,320),interpolation=cv2.INTER_NEAREST) for p in panels]
            bottom=np.zeros((352,2048,3),np.uint8)
            bottom[24:344,352:672]=crops[0]; bottom[24:344,1376:1696]=crops[1]
            cv2.putText(bottom,f'{name}: {kind} | green=label orange=candidate | viewing crop only',(12,18),0,.55,(255,255,255),1)
            path=examples/f'{name}_{kind}_{len(seen)}_{clip["id"]}.jpg'
            cv2.imwrite(str(path),np.vstack([np.hstack(panels),bottom]))
            evidence.append({'method':name,'kind':kind,'clip':clip['id'],'time':frame['time'],'key':frame['key'],'path':str(path),'box':box})
            if len(seen)>=3: break
    write_json(out/'examples.json',evidence)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--batch',required=True); parser.add_argument('--output'); args=parser.parse_args()
    batch=Path(args.batch).resolve(); out=Path(args.output).resolve() if args.output else batch/'compare'; out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(batch/'manifest.json')
    from .static import build_detector as static_build
    from .motion import build_detector as motion_build
    cv2.setNumThreads(2)
    try:
        import torch
        torch.set_num_threads(2)
    except ImportError: pass
    configs={name:read_json(batch/name/'config.json') for name in ('static','motion')}
    write_json(out/'frozen_inputs.json',{'manifest_sha256':hashlib.sha256((batch/'manifest.json').read_bytes()).hexdigest(),'configs':configs})
    records={}
    expected={f['key'] for c in manifest['clips'] if c['split']=='test' for f in sample_frames(c,.3)}
    for name,build in [('static',static_build),('motion',motion_build)]:
        print('UNIFIED_REPLAY',name,flush=True)
        rows=run_detector(manifest,lambda:build(configs[name]),.3,'test')
        assert {r['key'] for r in rows}==expected and len(rows)==len(expected)
        assert all(r.get('processed',True) or not r['candidates'] for r in rows)
        if name=='motion': assert all('raw_candidates' in r for r in rows)
        save_records(out/f'{name}_records.jsonl',rows); records[name]=rows
    summaries={}
    for name,rows in records.items():
        summaries[name]={str(fov):evaluate(manifest,rows,'test',fov) for fov in (48,30,None)}
    for variant in ('A0','A1'):
        rows=read_records(batch/'static'/variant/'test_records.jsonl')
        assert {r['key'] for r in rows}==expected and len(rows)==len(expected)
        summaries['static_'+variant]={str(fov):evaluate(manifest,rows,'test',fov) for fov in (48,30,None)}
    summaries['motion_registered_only']={str(fov):evaluate(manifest,records['motion'],'test',fov,True) for fov in (48,30,None)}
    raw=[dict(r,candidates=r.get('raw_candidates',r['candidates'])) for r in records['motion']]
    save_records(out/'motion_raw_records.jsonl',raw)
    summaries['motion_raw']={str(fov):evaluate(manifest,raw,'test',fov) for fov in (48,30,None)}
    per_clip={c['id']:{name:evaluate({'clips':[c]},rows) for name,rows in records.items()}
              for c in manifest['clips'] if c['split']=='test'}
    write_json(out/'per_clip.json',per_clip)
    write_json(out/'registration_reasons.json',dict(Counter(r.get('reason','unspecified') for r in records['motion'])))
    # 互补性只审计固定输出；不在测试标签上选择融合阈值。
    si={r['key']:r for r in records['static']}; mi={r['key']:r for r in records['motion']}
    unique={'static_only':0,'motion_only':0,'both':0,'neither':0}; unique_moving=dict(unique)
    for c in manifest['clips']:
        if c['split']!='test':continue
        for f in sample_frames(c,.3):
            s=set(match(si[f['key']]['candidates'],f['labels']).values()); m=set(match(mi[f['key']]['candidates'],f['labels']).values())
            for j,g in enumerate(f['labels']):
                k='both' if j in s and j in m else 'static_only' if j in s else 'motion_only' if j in m else 'neither'
                unique[k]+=1
                if g['moving']:unique_moving[k]+=1
    write_json(out/'complementarity.json',{'all':unique,'moving':unique_moving})
    write_json(out/'summary.json',summaries)
    make_media(manifest,records,out)
    lines=['# 离线车辆发现统一回测','',
      '两条路线使用固定参数、同一 0.3 秒源时间网格、原始 1024×768 图像。检测器只接收图像和时间。',
      '投影框视为正确 vehicle 标注，空框图视为背景；真假类别完全合并。运动标签来自裁判轨迹约 1 秒实际位移 ≥1 米。','',
      '| 数据 | 路线 | 移动车辆召回 | 全部车辆召回 | 移动车辆1秒发现 | 全部车辆1秒发现 | 误报/帧 | 平均毫秒 | 可处理帧 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for fov in (48,30):
        for name in ('static_A0','static_A1','static','motion_raw','motion','motion_registered_only'):
            s=summaries[name][str(fov)]
            lines.append(f"| {fov}° | {name} | {pct(s['moving_recall'])} ({s['moving_tp']}/{s['moving_gt']}) | {pct(s['recall'])} ({s['tp']}/{s['gt']}) | {pct(s['moving_discovery_1s'])} ({s['moving_found_1s']}/{s['moving_events']}) | {pct(s['discovery_1s'])} ({s['found_1s']}/{s['events']}) | {s['fp_per_frame']:.2f} | {s['mean_ms']:.1f} | {pct(s['processed_ratio'])} |")
    lines+=['','## 口径与边界','',
      '- 一对一最大基数匹配，IoU ≥0.25；误报对全部 vehicle 标注计算，静止车辆命中不算动态误报。',
      '- 单帧召回按车辆框实例计数；首帧初始化和配准失败仍留在动态整体分母中。',
      '- 1 秒发现以原始完整连续帧的可见区间为事件；不可见间隔超过0.4秒开新事件。片段开头已可见的车辆按片段入口计时，因此该列包含片段入口发现；event_details 记录 clip_entry，不能解释为全程首次发现。',
      '- 短于1秒的可见事件也保留，只允许在实际可见期间发现；成功配准子集会排除没有成功配准采样的事件，其条件性结果不能代替整体。',
      '- 平均耗时为本机CPU离线检测时间，含配准与轨迹更新，不含磁盘解码、模型初始化和视频编码；不能当作UE或GPU帧率。',
      '- A0/A1 消融复用分支固定测试记录并以同一评估器复算；static 和 motion 为汇总分支顺序统一回放。raw 与稳定动态结果共用检测耗时，raw 不是另一个单独计时的检测器。',
      '- 数据共两个同seed采集运行，按连续片段隔离并留至少8秒调参/测试间隔；并非跨场景或跨目标ID泛化测试。',
      '- 固定测试集只有少量可见事件，必须同时查看分子分母。未接入协同控制、真假分类或真实变焦。','',
      '## 固定输出互补性','',f'全部车辆框：`{json.dumps(unique,ensure_ascii=False)}`。',f'移动车辆框：`{json.dumps(unique_moving,ensure_ascii=False)}`。','',
      '视频：`videos/` 同片段、同时间，左静态、右动态；绿框=标注，橙框=候选。',
      '案例：`examples/` 与 `examples.json`；裁剪放大仅用于查看，不是实际变焦。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:{f:brief(s) for f,s in v.items()} for k,v in summaries.items()},indent=2),flush=True)

if __name__=='__main__': main()
