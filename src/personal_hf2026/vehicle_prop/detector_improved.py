# 修改时间：2026-09-18
# 修改目的：迁移 FrontierPipeline 使用的视图切片、合并与抑制辅助实现。
# 修改内容：仅调整为包内相对导入，保留队友交付的计算逻辑。
"""SAHI-inspired full-frame + overlapping-tile vehicle detector; pixels only.

Reference: Akyon et al., ICIP 2022, https://arxiv.org/abs/2202.06934.
This is a small project-specific implementation, not the SAHI package.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np

from .common import ROOT, configure_runtime
from .detector import integer_box


def axis_starts(length, size, overlap=.25):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError('size>0 and 0<=overlap<1 required')
    if length <= size:
        return [0]
    step=max(1,round(size*(1-overlap)))
    starts=list(range(0,length-size+1,step))
    if starts[-1]!=length-size:
        starts.append(length-size)
    return starts


def tile_windows(width,height,size=512,overlap=.25):
    return [(x,y,min(width,x+size),min(height,y+size))
            for y in axis_starts(height,size,overlap)
            for x in axis_starts(width,size,overlap)]


def crop_boxes(boxes,window):
    """Clip all intersecting labels; never use labels to select inference windows."""
    x,y,r,b=window
    out=[]
    for a,c,d,e in boxes:
        clipped=[max(a,x)-x,max(c,y)-y,min(d,r)-x,min(e,b)-y]
        if clipped[2]>clipped[0] and clipped[3]>clipped[1]:
            out.append(clipped)
    return out


def suppress(boxes,scores,iou=.5):
    """Pure numpy greedy NMS in original pixel coordinates."""
    boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    scores=np.asarray(scores,dtype=np.float32)
    order=np.argsort(-scores,kind='stable');keep=[]
    area=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
    while len(order):
        i=int(order[0]);keep.append(i);rest=order[1:]
        if not len(rest):break
        wh=np.maximum(0,np.minimum(boxes[i,2:],boxes[rest,2:])-np.maximum(boxes[i,:2],boxes[rest,:2]))
        inter=wh[:,0]*wh[:,1]
        overlap=inter/np.maximum(area[i]+area[rest]-inter,1e-9)
        order=rest[overlap<=iou]
    return keep


def merge_candidates(raw,iou=.5,method='nms'):
    """NMS or WBF-inspired score-weighted coordinates with maximum score.

    Weighted mode uses score squared for coordinates and retains the highest score.
    Overlapping tiles are correlated views, so duplicate votes do not raise confidence.
    """
    a=np.asarray(raw,dtype=np.float32).reshape(-1,5)
    if not len(a):return a
    if method=='nms':return a[suppress(a[:,:4],a[:,4],iou)]
    if method!='weighted':raise ValueError('Unknown merge method')
    remaining=a[np.argsort(-a[:,4],kind='stable')];output=[]
    while len(remaining):
        first=remaining[0]
        wh=np.maximum(0,np.minimum(first[2:4],remaining[:,2:4])-np.maximum(first[:2],remaining[:,:2]))
        inter=wh[:,0]*wh[:,1];areas=np.prod(remaining[:,2:4]-remaining[:,:2],axis=1)
        overlap=inter/np.maximum(areas+areas[0]-inter,1e-9)
        mask=overlap>iou;mask[0]=True;group=remaining[mask]
        box=np.average(group[:,:4],axis=0,weights=np.maximum(group[:,4]**2,1e-12))
        output.append([*box,float(first[4])]);remaining=remaining[~mask]
    return np.asarray(output,dtype=np.float32)


def extract_candidate(image,detection,padding=.5,min_side=32):
    """Undrawn original-pixel crop + geometry for a future true/decoy classifier."""
    h,w=image.shape[:2];x1,y1,x2,y2=detection['xyxy'];cx=(x1+x2)/2;cy=(y1+y2)/2
    cw=max(min_side,(x2-x1)*(1+2*padding));ch=max(min_side,(y2-y1)*(1+2*padding))
    x,y,r,b=integer_box([cx-cw/2,cy-ch/2,cx+cw/2,cy+ch/2],w,h)
    return image[y:b,x:r].copy(),dict(crop_window_xyxy=[x,y,r,b],box_in_crop_xyxy=[x1-x,y1-y,x2-x,y2-y])


def records_from_arrays(boxes,scores,width,height,conf):
    result=[]
    for box,score in zip(boxes,scores):
        if float(score)<conf:continue
        x1,y1,x2,y2=map(float,box)
        box=[max(0.,min(width,x1)),max(0.,min(height,y1)),max(0.,min(width,x2)),max(0.,min(height,y2))]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        result.append(dict(class_id=0,class_name='vehicle',confidence=float(score),xyxy=box,
            xyxy_int=integer_box(box,width,height),center_xy=[(box[0]+box[2])/2,(box[1]+box[3])/2]))
    return result


class ImprovedVehicleDetector:
    def __init__(self,weights=None,device='0',imgsz=768,tile_size=512,overlap=.25,
                 conf=.25,iou=.5,full_frame=True,tile_batch=12,config=None,flip=False,merge='nms'):
        if config is None and weights is None:
            config=ROOT/'configs/vehicle_improved.json'
        if config:
            settings=json.loads(Path(config).read_text(encoding='utf-8'))
            weights=ROOT/settings['weights'];imgsz=settings['imgsz'];tile_size=settings['tile_size']
            overlap=settings.get('overlap',overlap);conf=settings['confidence'];iou=settings['iou']
            full_frame=settings['full_frame'];flip=settings.get('flip',False)
            merge=settings.get('merge','nms')
        configure_runtime()
        from ultralytics import YOLO
        self.weights=Path(weights or ROOT/'weights/vehicle_improved.pt')
        if config and settings.get('weights_sha256'):
            if hashlib.sha256(self.weights.read_bytes()).hexdigest()!=settings['weights_sha256']:
                raise ValueError('The weights do not match the validated configuration checksum')
        self.model=YOLO(str(self.weights))
        if self.model.names!={0:'vehicle'}:raise ValueError('Expected single-class vehicle weights')
        self.device,self.imgsz,self.tile_size,self.overlap=device,imgsz,tile_size,overlap
        self.conf,self.iou,self.full_frame,self.tile_batch,self.flip=conf,iou,full_frame,tile_batch,flip
        self.merge=merge

    def predict_components(self,images,raw_conf=.001):
        """Return separate full/tile candidates for validation-only configuration search."""
        for im in images:
            if not isinstance(im,np.ndarray) or im.ndim!=3 or im.shape[2]!=3 or im.dtype!=np.uint8:
                raise ValueError('Expected uint8 HxWx3 BGR arrays')
        collected=[{'full':[],'tiles':[],'full_original':[],'full_flipped':[]} for _ in images]
        jobs=[]
        for index,im in enumerate(images):
            h,w=im.shape[:2]
            if self.full_frame:
                jobs.append((index,'full',(0,0,w,h),False,im))
                if self.flip:jobs.append((index,'full',(0,0,w,h),True,np.ascontiguousarray(im[:,::-1])))
            if self.tile_size:
                for window in tile_windows(w,h,self.tile_size,self.overlap):
                    x,y,r,b=window;patch=im[y:b,x:r].copy()
                    jobs.append((index,'tiles',window,False,patch))
                    if self.flip:jobs.append((index,'tiles',window,True,np.ascontiguousarray(patch[:,::-1])))
        # Keep square tiles and whole frames separate to avoid unnecessary padding.
        for kind in ('full','tiles'):
            work=[j for j in jobs if j[1]==kind]
            batch=min(self.tile_batch,2) if self.imgsz>=1280 else self.tile_batch
            for start in range(0,len(work),batch):
                group=work[start:start+batch]
                predictions=self.model.predict([j[4] for j in group],imgsz=self.imgsz,device=self.device,
                    conf=raw_conf,iou=.7,max_det=100,verbose=False,rect=True)
                for job,pred in zip(group,predictions):
                    index,_,window,flipped,patch=job;x,y,r,b=window
                    boxes=pred.boxes.xyxy.cpu().numpy();scores=pred.boxes.conf.cpu().numpy()
                    for box,score in zip(boxes,scores):
                        a,c,d,e=map(float,box)
                        if flipped:a,d=patch.shape[1]-d,patch.shape[1]-a
                        # Reject boxes visibly truncated by INTERNAL tile boundaries.
                        # Overlap supplies a second, complete view; true image edges remain valid.
                        if kind=='tiles' and ((x>0 and a<1) or (y>0 and c<1) or
                            (r<images[index].shape[1] and d>patch.shape[1]-1) or
                            (b<images[index].shape[0] and e>patch.shape[0]-1)):
                            continue
                        record=[a+x,c+y,d+x,e+y,float(score)]
                        collected[index][kind].append(record)
                        if kind=='full':collected[index]['full_flipped' if flipped else 'full_original'].append(record)
        return collected

    def predict_batch(self,images):
        components=self.predict_components(images,raw_conf=min(.001,self.conf))
        outputs=[]
        for im,parts in zip(images,components):
            candidates=parts['full']+parts['tiles']
            if candidates:
                a=merge_candidates(candidates,self.iou,self.merge)
                h,w=im.shape[:2];outputs.append(records_from_arrays(a[:,:4],a[:,4],w,h,self.conf))
            else:outputs.append([])
        return outputs

    def predict(self,image):return self.predict_batch([image])[0]
