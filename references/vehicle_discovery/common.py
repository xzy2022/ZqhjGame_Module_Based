# 修改时间：2026-09-13。
# 修改目的：固定车辆发现的离线数据划分与公平评估。
# 修改内容：实现原图索引、位移标注、采样、候选匹配、发现时延及叠框视频。
"""检测器接口仅接收 RGB 图像与源时间，标签只用于训练和评估。"""
import argparse
import bisect
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

BASELINE = '7d211b83e04668789f4b18215630b4fe19f459a8'
CLIPS = [
    (48, '20003', 24, 'tune', 'vehicles'),
    (48, '20001', 56, 'tune', 'brief_vehicle_view_change'),
    (48, '20001', 112, 'tune', 'background'),
    (48, '20002', 40, 'test', 'vehicles'),
    (48, '20003', 72, 'test', 'vehicles'),
    (48, '20002', 88, 'test', 'view_change'),
    (48, '20002', 184, 'test', 'vehicle_entry'),
    (48, '20001', 128, 'test', 'brief_vehicle'),
    (48, '20001', 160, 'test', 'background'),
    (30, '20003', 16, 'tune', 'vehicle_entry'),
    (30, '20002', 32, 'test', 'vehicles'),
    (30, '20001', 40, 'test', 'background'),
]

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    sources = {48: root/'dataset-200s-seed1-20260913-192854',
               30: root/'dataset-fov30-50s-seed1-20260913-195159'}
    manifest = {'baseline': BASELINE, 'iou': .25, 'moving_rule': 'distance over +/-0.5s >= 1m; no class filtering',
                'split_rule': 'fixed 8s clips; no adjacent tune/test windows within a source run',
                'label_contract': 'all projected boxes=vehicle; empty boxes=background', 'clips': [], 'sources': {}}
    for fov, source in sources.items():
        p = source/'dataset/samples.jsonl'
        rows = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
        trajectory = defaultdict(dict)
        for row in rows:
            ref = row.get('reference') or {}
            t = float(ref.get('sim_time', row['source_sim_time']))
            for obj in ref.get('objects', []):
                trajectory[str(obj['target_id'])][t] = (obj['lat'], obj['lon'])
        tracks = {k: sorted(v.items()) for k,v in trajectory.items()}
        track_times = {k: [r[0] for r in v] for k,v in tracks.items()}
        def displacement(oid, t):
            times = track_times.get(oid, [])
            if not times: return None
            a = max(0, bisect.bisect_left(times, t-.5)-1)
            b = min(len(times)-1, bisect.bisect_left(times, t+.5))
            if times[b]-times[a] < .5: return None
            (lat1,lon1),(lat2,lon2) = tracks[oid][a][1],tracks[oid][b][1]
            return math.hypot((lat2-lat1)*111320, (lon2-lon1)*111320*math.cos(math.radians((lat1+lat2)/2)))
        manifest['sources'][str(fov)] = {'path': str(source), 'index_sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'frames': len(rows)}
        for angle, uid, start, split, tag in CLIPS:
            if angle != fov: continue
            frames = []
            for row in rows:
                if str(row['uid']) != uid or not start <= row['source_t'] < start+8: continue
                labels = []
                for obj in row['ue_projected_objects']:
                    box = list(map(float,obj['ue_projected_bbox'])); oid = str(obj['target_id'])
                    box = [max(0,box[0]),max(0,box[1]),min(row['width'],box[2]),min(row['height'],box[3])]
                    if box[2]<=box[0] or box[3]<=box[1]: continue
                    d = displacement(oid, row['source_sim_time'])
                    labels.append({'id':oid,'box':box,'moving':d is not None and d>=1.,'displacement_m':d})
                pose = row.get('source_pose') or {}
                frames.append({'key':f"{fov}:{uid}:{row['frame_no']}", 'uid':uid,'time':row['source_t'],
                               'source_time':row['source_sim_time'],'image':str(source/row['image_path']),
                               'labels':labels,'pose':pose})
            frames.sort(key=lambda r:r['time'])
            manifest['clips'].append({'id':f'f{fov}_{uid}_{start:03d}', 'fov':fov,'uid':uid,'start':start,'end':start+8,
                                      'split':split,'tag':tag,'frames':frames})
    write_json(output/'manifest.json', manifest)
    summaries = []
    for clip in manifest['clips']:
        frames = clip['frames']
        summaries.append({k:v for k,v in clip.items() if k!='frames'} | {'frames':len(frames),
          'vehicle_frames':sum(bool(f['labels']) for f in frames),'moving_boxes':sum(g['moving'] for f in frames for g in f['labels']),
          'vehicle_boxes':sum(len(f['labels']) for f in frames)})
    write_json(output/'split_summary.json', summaries)
    return manifest

def sample_frames(clip, interval=.3):
    """对固定源时间网格取首个到达帧，避免时间间隔累积漂移。"""
    selected=[]; next_t=clip['start']
    for f in clip['frames']:
        if f['time']+1e-6 >= next_t:
            selected.append(f)
            while next_t<=f['time']+1e-6: next_t+=interval
    return selected

def load_rgb(frame):
    bgr=cv2.imread(frame['image'])
    if bgr is None: raise FileNotFoundError(frame['image'])
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def match(candidates, labels, threshold=.25):
    """以最大基数二分图匹配避免重复候选或贪心顺序改变召回率。"""
    edges = {i:sorted([j for j,g in enumerate(labels) if iou(c['box'],g['box'])>=threshold],
                     key=lambda j:-iou(c['box'],labels[j]['box'])) for i,c in enumerate(candidates)}
    owners={}
    def visit(i,seen):
        for j in edges[i]:
            if j in seen: continue
            seen.add(j)
            if j not in owners or visit(owners[j],seen): owners[j]=i; return True
        return False
    for i in sorted(edges,key=lambda i:-candidates[i].get('score',1)): visit(i,set())
    return {i:j for j,i in owners.items()}

def evaluate(manifest, records, split='test', fov=None, processed_only=False):
    index={r['key']:r for r in records}
    n=tp=gt=mtp=mgt=fp=processed=0; milliseconds=[]; hits={}; included=set()
    clips=[c for c in manifest['clips'] if c['split']==split and (fov is None or c['fov']==fov)]
    for clip in clips:
        for frame in clip['frames']:
            r=index.get(frame['key'])
            if r is None: continue
            ok=r.get('processed',True)
            if processed_only and not ok: continue
            included.add(frame['key']); n+=1; processed+=ok; milliseconds.append(r['elapsed_ms'])
            labels=frame['labels']; pairs=match(r['candidates'],labels)
            tp+=len(pairs); gt+=len(labels); fp+=len(r['candidates'])-len(pairs)
            mgt+=sum(g['moving'] for g in labels); mtp+=sum(labels[j]['moving'] for j in pairs.values())
            hits[frame['key']]={labels[j]['id'] for j in pairs.values()}
    # 发现事件使用完整原始帧的可见区间，跳过帧仍消耗一秒时限。
    events=[]
    for clip in clips:
        episodes=defaultdict(list)
        for f in clip['frames']:
            for g in f['labels']:
                seq=episodes[g['id']]
                if not seq or f['time']-seq[-1][-1][0]['time']>.4: seq.append([])
                seq[-1].append((f,g))
        for oid, seq in episodes.items():
            for episode in seq:
                start=episode[0][0]['time']; stop=min(start+1,episode[-1][0]['time'])
                eligible=[f for f,g in episode if f['key'] in included and f['time']<=start+1+1e-6]
                if processed_only and not eligible: continue
                moving=any(g['moving'] for f,g in episode if f['time']<=start+1)
                found=any(oid in hits.get(f['key'],set()) for f in eligible)
                events.append({'clip':clip['id'],'object':oid,'start':start,'moving':moving,'found_1s':found,
                               'clip_entry':start<=clip['frames'][0]['time']+.1,'visible_duration':episode[-1][0]['time']-start})
    def rate(num,den): return num/den if den else None
    moving_events=[e for e in events if e['moving']]
    return {'frames':n,'gt':gt,'tp':tp,'recall':rate(tp,gt),'moving_gt':mgt,'moving_tp':mtp,'moving_recall':rate(mtp,mgt),
            'false_positives':fp,'fp_per_frame':rate(fp,n),'mean_ms':float(np.mean(milliseconds)) if milliseconds else None,
            'processed_ratio':rate(processed,n),'events':len(events),'found_1s':sum(e['found_1s'] for e in events),
            'discovery_1s':rate(sum(e['found_1s'] for e in events),len(events)),
            'moving_events':len(moving_events),'moving_found_1s':sum(e['found_1s'] for e in moving_events),
            'moving_discovery_1s':rate(sum(e['found_1s'] for e in moving_events),len(moving_events)),
            'event_details':events}

def run_detector(manifest, factory, interval=.3, split=None):
    records=[]
    for clip in manifest['clips']:
        if split and clip['split']!=split: continue
        detector=factory(); detector.reset()
        for frame in sample_frames(clip,interval):
            rgb=load_rgb(frame); start=perf_counter()
            result=detector.detect(rgb,frame['time'])
            elapsed=(perf_counter()-start)*1000
            if isinstance(result,list): result={'candidates':result,'processed':True}
            records.append({'key':frame['key'],'clip':clip['id'],'uid':frame['uid'],'time':frame['time'],
                            'source_time':frame['source_time'],'elapsed_ms':elapsed, **result})
    return records

def save_records(path,records):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')

def read_records(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]

def overlay(rgb,frame,record,title):
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for g in frame['labels']:
        x1,y1,x2,y2=map(int,g['box']); cv2.rectangle(bgr,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(bgr,('M ' if g['moving'] else 'V ')+g['id'],(x1,max(48,y1-4)),0,.4,(0,255,0),1)
    for c in record.get('candidates',[]):
        x1,y1,x2,y2=map(int,c['box']); cv2.rectangle(bgr,(x1,y1),(x2,y2),(0,140,255),1)
    cv2.rectangle(bgr,(0,0),(1024,42),(0,0,0),-1)
    cv2.putText(bgr,f"{title} t={frame['time']:.2f} N={len(record.get('candidates',[]))} ok={record.get('processed',True)}",(8,27),0,.65,(255,255,255),1)
    return bgr

def write_videos(manifest,records,output,title,interval=.3,split='test'):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); index={r['key']:r for r in records}
    for clip in manifest['clips']:
        if clip['split']!=split: continue
        writer=cv2.VideoWriter(str(output/(clip['id']+'.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),1/interval,(1024,768))
        if not writer.isOpened(): raise RuntimeError('video writer unavailable')
        for frame in sample_frames(clip,interval):
            writer.write(overlay(load_rgb(frame),frame,index.get(frame['key'],{}),title))
        writer.release()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); m=prepare(a.data_root,a.output); print(json.dumps(read_json(Path(a.output)/'split_summary.json'),ensure_ascii=False,indent=2))
