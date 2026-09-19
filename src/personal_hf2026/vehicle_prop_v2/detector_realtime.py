# 修改时间：2026-09-19。
# 修改目的：使逐帧记录明确使用的源时间融合上限。
# 修改内容：在正常处理与跳过帧的元数据中均记录 max_source_gap_s。
# 修改时间：2026-09-19。
# 修改目的：让在线旁路和离线回放以显式配置复用 V2 单模型及多机独立时序状态。
# 修改内容：解析仓库内固定权重并支持校验过的外置 engine 与高阈值覆盖。
# 修改时间：2026-09-19。
# 修改目的：将队友 V2 的单模型实时识别实现独立集成到模块仓库。
# 修改内容：固定源码来源、改用包内导入与外置运行目录，并保留原始检测及因果跟踪算法。
"""One shared detector, isolated per-UAV tracking, source-time fusion <= 0.25 s."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from personal_hf2026.paths import PROJECT_ROOT
from .temporal_tracker import TemporalTracker
from .realtime_backend import RealtimeDetector


class FastCameraMotion:
    def __init__(self,width=384):
        self.width=width;self.previous=None;self.previous_shape=None

    def update(self,image):
        h,w=image.shape[:2]
        scale=min(1.,self.width/w)
        current=cv2.cvtColor(cv2.resize(image,(round(w*scale),round(h*scale))),cv2.COLOR_BGR2GRAY)
        previous=self.previous
        previous_shape=self.previous_shape
        self.previous=current;self.previous_shape=(h,w)
        identity=np.eye(2,3,dtype=float)
        info=dict(valid=False,points=0,inliers=0)
        if previous is None or previous.shape!=current.shape or previous_shape!=(h,w):
            return identity,info
        corners=cv2.goodFeaturesToTrack(previous,maxCorners=180,qualityLevel=.015,minDistance=5,blockSize=5)
        if corners is None or len(corners)<12:
            return identity,info
        forward,status,_=cv2.calcOpticalFlowPyrLK(previous,current,corners,None,winSize=(15,15),maxLevel=3)
        if forward is None:
            return identity,info
        reverse,status2,_=cv2.calcOpticalFlowPyrLK(current,previous,forward,None,winSize=(15,15),maxLevel=3)
        if reverse is None:
            return identity,info
        mask=(status[:,0]>0)&(status2[:,0]>0)&(np.linalg.norm(corners[:,0]-reverse[:,0],axis=1)<1.5)
        a,b=corners[:,0][mask],forward[:,0][mask]
        info['points']=len(a)
        if len(a)<12:
            return identity,info
        matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=500,confidence=.99)
        if matrix is None or inliers is None:
            return identity,info
        n=int(inliers.sum());info['inliers']=n
        if n<10 or n/len(a)<.45 or not .85<np.linalg.norm(matrix[0,:2])<1.15:
            return identity,info
        sx,sy=current.shape[1]/w,current.shape[0]/h
        small=np.vstack((matrix,[0,0,1]));S=np.diag([sx,sy,1.])
        info['valid']=True
        return (np.linalg.inv(S)@small@S)[:2],info


@dataclass
class StreamState:
    tracker: TemporalTracker
    camera: FastCameraMotion
    sequence: str|None=None
    previous_t: float|None=None
    image_shape: tuple|None=None


class RealtimePipeline:
    def __init__(self,config=None,device='0',detector=None,*,weights=None,
                 weights_sha256=None,tracker_high=None):
        default_config=PROJECT_ROOT/'configs/detectors/vehicle_prop_v2/vehicle_realtime.json'
        self.config=json.loads(Path(config or default_config).read_text(encoding='utf8')) if not isinstance(config,dict) else dict(config)
        self.config['tracker']=dict(self.config.get('tracker',{}))
        if tracker_high is not None:
            self.config['tracker']['high']=float(tracker_high)
        self.settings=dict(self.config.get('tracker',{}))
        self.settings['max_gap_s']=float(self.config.get('max_source_gap_s',.25))
        if not 0<self.settings['max_gap_s']<=.25:
            raise ValueError('Source-time fusion limit must be positive and <= 0.25 s')
        spec=dict(self.config.get('detector',{}))
        self.weights_path=(Path(weights).resolve() if weights is not None
                           else (PROJECT_ROOT/self.config['weights']).resolve())
        if weights is not None:
            spec['expected_sha256']=weights_sha256
        elif weights_sha256 is not None:
            spec['expected_sha256']=weights_sha256
        self.config['detector']=spec
        self.weights_sha256=spec.get('expected_sha256')
        self.detector=detector or RealtimeDetector(self.weights_path,device=device,**spec)
        self.max_streams=int(self.config.get('max_streams',16))
        self.reset()

    def reset(self,stream_id=None):
        if stream_id is None:
            self.streams={};self.last_metadata={};self.stats=Counter()
        else:
            self.streams.pop(str(stream_id),None)

    def _new_state(self):
        return StreamState(TemporalTracker(**self.settings),FastCameraMotion(self.config.get('motion_width',384)))

    def warmup(self):
        self.detector.warmup()

    def predict(self,image,timestamp=0.,sequence_id='single',stream_id=None):
        """timestamp is image SOURCE simulation time in seconds, never result time.

        Pass stream_id=UAV uid when multiplexing. sequence_id identifies the run
        or camera segment. Call from one worker only; inference is synchronous.
        """
        start=time.perf_counter()
        if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
            raise ValueError('Expected BGR uint8 HWC image')
        timestamp=float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError('Source timestamp must be finite seconds')
        key=str(stream_id) if stream_id is not None else '__default__'
        if key not in self.streams:
            if len(self.streams)>=self.max_streams:
                raise ValueError('Too many streams; reset an ended stream explicitly')
            self.streams[key]=self._new_state()
        state=self.streams[key]
        dt=None if state.previous_t is None else timestamp-state.previous_t
        reason=None
        if state.sequence!=sequence_id:
            reason='sequence_change'
        elif state.image_shape!=image.shape[:2]:
            reason='image_shape_change'
        elif dt is not None and dt<=0:
            # 乱序或重复源帧不增加时序证据。
            self.stats['skipped_nonincreasing_source_time']+=1
            self.last_metadata=dict(stream_id=key,sequence_id=sequence_id,source_timestamp_s=timestamp,
                                    source_dt_s=dt,max_source_gap_s=self.settings['max_gap_s'],
                                    temporal_eligible=False,skipped=True,reset_reason='nonincreasing_source_time')
            return []
        elif dt is not None and dt>self.settings['max_gap_s']+1e-6:
            reason='source_gap'
        eligible=reason is None and dt is not None
        if reason:
            state.camera=FastCameraMotion(self.config.get('motion_width',384))
            # 同一序列在源时间间隔过大时仍保持轨迹编号递增。
            next_id=state.tracker.next_id if state.sequence==sequence_id else 1
            state.tracker.reset();state.tracker.next_id=next_id
            state.tracker.sequence=sequence_id
            self.stats['reset_'+reason]+=1
        detection_start=time.perf_counter()
        a=self.detector.predict(image)
        detection_end=time.perf_counter()
        if self.settings.get('use_motion',True):
            motion,quality=state.camera.update(image)
        else:
            motion,quality=np.eye(2,3),dict(valid=False)
        motion_end=time.perf_counter()
        probs=np.column_stack((a[:,5:7],np.zeros(len(a),dtype=np.float32)))
        result=state.tracker.update(a[:,:5],probs,timestamp,sequence_id,motion,image.shape[:2])
        for d in result:
            d['class_name']=('real_vehicle','model_prop')[d['class_id']] if d['class_confidence']>=self.config.get('unknown_threshold',.6) else 'uncertain'
        state.sequence=sequence_id;state.previous_t=timestamp;state.image_shape=image.shape[:2]
        self.stats['processed']+=1;self.stats['temporal_eligible']+=int(eligible)
        self.last_metadata=dict(stream_id=key,sequence_id=sequence_id,source_timestamp_s=timestamp,
            source_dt_s=dt,max_source_gap_s=self.settings['max_gap_s'],
            temporal_eligible=eligible,skipped=False,reset_reason=reason,
            motion_valid=quality['valid'],fused_detections=sum(d['track_hits']>1 for d in result),
            detector_ms=(detection_end-detection_start)*1000,motion_ms=(motion_end-detection_end)*1000,
            tracker_ms=(time.perf_counter()-motion_end)*1000,inference_wall_ms=(time.perf_counter()-start)*1000)
        return result
