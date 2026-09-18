# 修改时间：2026-09-18
# 修改目的：迁移 FrontierPipeline 间接依赖的检测框辅助实现。
# 修改内容：改为包内相对导入，其余交付源码保持不变。
"""纯图像推理接口。模型不接收标签、真伪类别、物体 ID 或地理坐标。"""
from __future__ import annotations

import math
import os
from pathlib import Path

from .common import ROOT, configure_runtime


def integer_box(xyxy, width, height):
    """原点为左上角，返回半开整数矩形 [x_min,y_min,x_max,y_max)。"""
    x1,y1,x2,y2=map(float,xyxy)
    return [max(0,min(width,math.floor(x1))),max(0,min(height,math.floor(y1))),
            max(0,min(width,math.ceil(x2))),max(0,min(height,math.ceil(y2)))]


class VehicleDetector:
    def __init__(self,weights=ROOT/"weights"/"vehicle_best.pt",device="0",imgsz=1024,conf=0.25,iou=0.5):
        configure_runtime()
        from ultralytics import YOLO
        weights=Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"缺少已训练权重：{weights}，请先运行 train.py。")
        self.model=YOLO(str(weights))
        names=self.model.names
        if len(names)!=1 or names[0]!="vehicle":
            raise ValueError("请使用本项目训练的单类 vehicle 权重，不能直接把 COCO 权重当成完成训练的模型。")
        self.device,self.imgsz,self.conf,self.iou=device,imgsz,conf,iou

    def predict_batch(self,images):
        """images: OpenCV BGR ndarray 列表。返回每幅图的检测列表，坐标均对应原图。"""
        import numpy as np
        for im in images:
            if not isinstance(im,np.ndarray) or im.ndim!=3 or im.shape[2]!=3 or im.dtype!=np.uint8:
                raise ValueError("输入必须为 H×W×3、uint8 的 BGR 图像")
        results=self.model.predict(images,imgsz=self.imgsz,conf=self.conf,iou=self.iou,
                                   device=self.device,verbose=False,rect=True,max_det=100)
        output=[]
        for result,im in zip(results,images):
            h,w=im.shape[:2]
            detections=[]
            for box,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                xyxy=[max(0.,min(float(w),box[0])),max(0.,min(float(h),box[1])),
                      max(0.,min(float(w),box[2])),max(0.,min(float(h),box[3]))]
                if xyxy[2]<=xyxy[0] or xyxy[3]<=xyxy[1]:
                    continue
                detections.append({"class_id":0,"class_name":"vehicle","confidence":float(score),
                                   "xyxy":xyxy,"xyxy_int":integer_box(xyxy,w,h),
                                   "center_xy":[(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2]})
            output.append(detections)
        return output

    def predict(self,image):
        return self.predict_batch([image])[0]


def read_image(path):
    """支持 Windows 中文路径，不依赖 cv2.imread 的路径编码行为。"""
    import cv2
    import numpy as np
    im=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"无法解码图像：{path}")
    return im


def draw_detections(image,detections):
    import cv2
    canvas=image.copy()
    for d in detections:
        x1,y1,x2,y2=d['xyxy_int']
        cv2.rectangle(canvas,(x1,y1),(max(x1,x2-1),max(y1,y2-1)),(0,230,80),2)
        label=f"vehicle {d['confidence']:.2f} ({x1},{y1},{x2},{y2})"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
        tx=max(0,min(x1,image.shape[1]-tw-4))
        ty=y1-4 if y1>=th+8 else min(image.shape[0]-4,y2+th+6)
        cv2.rectangle(canvas,(tx,ty-th-3),(tx+tw+4,ty+3),(0,50,15),-1)
        cv2.putText(canvas,label,(tx+2,ty),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1,cv2.LINE_AA)
    return canvas
