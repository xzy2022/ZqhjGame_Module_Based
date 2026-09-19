# 修改时间：2026-09-19。
# 修改目的：将队友 V2 的单模型实时识别实现独立集成到模块仓库。
# 修改内容：固定源码来源、改用包内导入与外置运行目录，并保留原始检测及因果跟踪算法。
"""Causal two-stage association with image-based camera motion compensation.

Inspired by ByteTrack's high/low detection association and BoT-SORT's camera
motion compensation. This is a compact implementation, not a reproduction.
No reference identities, labels, geographic coordinates, or future frames enter it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def overlaps(a,b):
    a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh.prod(2)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter,1e-9)


class CameraMotion:
    def __init__(self): self.previous=None

    def update(self,image):
        current=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None,fx=.5,fy=.5)
        previous=self.previous;self.previous=current
        identity=np.eye(2,3,dtype=float)
        info=dict(valid=False,points=0,inliers=0)
        if previous is None or previous.shape!=current.shape:return identity,info
        corners=cv2.goodFeaturesToTrack(previous,maxCorners=350,qualityLevel=.015,minDistance=7,blockSize=5)
        if corners is None or len(corners)<15:return identity,info
        forward,status,_=cv2.calcOpticalFlowPyrLK(previous,current,corners,None,winSize=(21,21),maxLevel=3)
        if forward is None:return identity,info
        reverse,status2,_=cv2.calcOpticalFlowPyrLK(current,previous,forward,None,winSize=(21,21),maxLevel=3)
        if reverse is None:return identity,info
        mask=(status[:,0]>0)&(status2[:,0]>0)&(np.linalg.norm(corners[:,0]-reverse[:,0],axis=1)<1.5)
        a=corners[:,0][mask];b=forward[:,0][mask];info['points']=len(a)
        if len(a)<15:return identity,info
        transform,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=1000,confidence=.99)
        if transform is None or inliers is None:return identity,info
        count=int(inliers.sum());info['inliers']=count
        scale=np.linalg.norm(transform[0,:2])
        if count<12 or count/len(a)<.45 or not .85<scale<1.15:return identity,info
        transform[:,2]*=2
        info['valid']=True
        return transform,info


def warp_box(box,matrix):
    x,y,r,b=box
    points=np.array([[x,y,1],[r,y,1],[r,b,1],[x,b,1]],dtype=float)@np.asarray(matrix).T
    return np.r_[points.min(0),points.max(0)]


@dataclass
class Track:
    identity:int
    box:np.ndarray
    score:float
    probabilities:np.ndarray
    hits:int=1
    misses:int=0
    elapsed_missing:float=0.
    velocity:np.ndarray=field(default_factory=lambda:np.zeros(2))
    high_hits:int=1


class TemporalTracker:
    def __init__(self,high=.5,low=.05,background_power=0.,use_motion=True,
                 max_missed=3,max_gap_s=.25,smoothing=.75,recovery=True,distance_gate=1.8,min_high_hits=1,low_distance_gate=None):
        if not 0<=low<=high<=1:raise ValueError('Require 0 <= low <= high <= 1')
        if not np.isfinite(max_gap_s) or max_gap_s<=0:raise ValueError('max_gap_s must be finite and positive')
        self.high=high;self.low=low;self.background_power=background_power;self.use_motion=use_motion
        self.max_missed=max_missed;self.max_gap_s=max_gap_s;self.smoothing=smoothing
        self.recovery=recovery;self.distance_gate=distance_gate
        self.min_high_hits=min_high_hits;self.low_distance_gate=low_distance_gate or distance_gate
        self.reset()

    def reset(self):
        self.tracks=[];self.next_id=1;self.previous_t=None;self.sequence=None

    def update(self,detections,probabilities,timestamp,sequence_id,motion=None,image_shape=(768,1024)):
        if not np.isfinite(timestamp):raise ValueError('timestamp must be finite source seconds')
        if self.sequence!=sequence_id or (self.previous_t is not None and (timestamp<=self.previous_t or timestamp-self.previous_t>self.max_gap_s+1e-6)):
            next_id=self.next_id if self.sequence==sequence_id else 1
            self.reset();self.sequence=sequence_id;self.next_id=next_id
        dt=0. if self.previous_t is None else timestamp-self.previous_t;self.previous_t=timestamp
        d=np.asarray(detections,dtype=float).reshape(-1,5);p=np.asarray(probabilities,dtype=float).reshape(-1,3)
        if len(d)!=len(p) or not np.isfinite(d).all() or not np.isfinite(p).all():raise ValueError('Invalid predictions')
        score=d[:,4]*(np.clip(1-p[:,2],0,1)**self.background_power)
        matrix=np.eye(2,3) if motion is None or not self.use_motion else np.asarray(motion)
        predicted=[];warped=[]
        for tr in self.tracks:
            base=warp_box(tr.box,matrix);warped.append(base)
            predicted.append(base+np.tile(tr.velocity*dt,2))
        assignments=[];free_tracks=set(range(len(self.tracks)))

        def associate(indices,low_stage=False):
            if not free_tracks or not len(indices):return []
            ti=sorted(j for j in free_tracks if not low_stage or self.tracks[j].high_hits>=self.min_high_hits)
            if not ti:return []
            db=d[indices,:4];tb=np.array([predicted[j] for j in ti])
            iou=overlaps(tb,db)
            tc=(tb[:,:2]+tb[:,2:])/2;dc=(db[:,:2]+db[:,2:])/2
            size=np.maximum(8.,np.linalg.norm(tb[:,2:]-tb[:,:2],axis=1))
            distance=np.linalg.norm(tc[:,None]-dc[None,:],axis=2)/size[:,None]
            ratios=np.maximum((tb[:,2:]-tb[:,:2])[:,None,:]/np.maximum(db[None,:,2:]-db[None,:,:2],1),
                              (db[None,:,2:]-db[None,:,:2])/np.maximum((tb[:,2:]-tb[:,:2])[:,None,:],1)).max(2)
            cost=1-iou+.3*distance
            gate=(distance<=(self.low_distance_gate if low_stage else self.distance_gate))&(ratios<3.)
            cost[~gate]=1e6
            rr,cc=linear_sum_assignment(cost);matched=[]
            for a,b in zip(rr,cc):
                if cost[a,b]<1e5:
                    matched.append((ti[a],int(indices[b])));free_tracks.remove(ti[a])
            return matched

        high_indices=np.flatnonzero(score>=self.high)
        assignments+=associate(high_indices)
        if self.recovery:assignments+=associate(np.flatnonzero((score>=self.low)&(score<self.high)),low_stage=True)
        used={j for _,j in assignments};output=[]

        def record(tr,j,recovered):
            q=tr.probabilities[:2]/max(float(tr.probabilities[:2].sum()),1e-9)
            one=p[j,:2]/max(float(p[j,:2].sum()),1e-9)
            return dict(track_id=tr.identity,xyxy=d[j,:4].tolist(),confidence=float(score[j]),
                detector_confidence=float(d[j,4]),class_probabilities=q.tolist(),single_frame_probabilities=one.tolist(),
                background_probability=float(p[j,2]),class_id=int(q.argmax()),class_confidence=float(q.max()),
                track_hits=tr.hits,recovered_low_score=bool(recovered),motion_velocity_px_per_s=tr.velocity.tolist())

        for i,j in assignments:
            tr=self.tracks[i];recovered=score[j]<self.high
            if dt>0:
                residual=((d[j,:2]+d[j,2:4])-(warped[i][:2]+warped[i][2:]))/(2*dt)
                tr.velocity=.5*tr.velocity+.5*residual
            tr.box=d[j,:4].copy();tr.score=float(score[j]);tr.misses=0;tr.elapsed_missing=0;tr.hits+=1
            tr.high_hits+=int(not recovered)
            # 概率平滑只使用当前与过去帧，并随序列重置。
            alpha=self.smoothing
            tr.probabilities=alpha*tr.probabilities+(1-alpha)*p[j]
            output.append(record(tr,j,recovered))
        for i in free_tracks:
            tr=self.tracks[i];tr.box=predicted[i];tr.misses+=1;tr.elapsed_missing+=dt
        self.tracks=[tr for tr in self.tracks if tr.misses<=self.max_missed and tr.elapsed_missing<=self.max_gap_s]
        for j in high_indices:
            if int(j) in used:continue
            tr=Track(self.next_id,d[j,:4].copy(),float(score[j]),p[j].copy());self.next_id+=1
            self.tracks.append(tr);output.append(record(tr,int(j),False))
        # 无当前帧检测支持的预测框仅保留在内部，不作为输出。
        h,w=image_shape
        return [o for o in output if 0<=o['xyxy'][0]<o['xyxy'][2]<=w+1e-4 and 0<=o['xyxy'][1]<o['xyxy'][3]<=h+1e-4]
