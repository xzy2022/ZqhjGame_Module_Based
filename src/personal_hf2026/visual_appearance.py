# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：在独立 V2 旁路中复用队友 v31 外观模型。
# 修改内容：保留候选、裁剪和网络实现，仅替换本地数据类导入。
"""复用 v31 本机图像候选与外观分类，保持权重和预处理一致。"""
from io import BytesIO
import math
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from .visual_geometry import PixelBox


def vehicle_proposals(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,31,12)
    mask[gray>110]=0
    masks=[mask,*[(gray<threshold).astype(np.uint8)*255 for threshold in (35,45,55,65,75,85)]]
    contours=[]
    for candidate_mask in masks:
        found,_=cv2.findContours(candidate_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    result=[]
    for c in contours:
        area=cv2.contourArea(c)
        if not 10<=area<=1600:continue
        (_, _),(w,h),_=cv2.minAreaRect(c);short,long=sorted((w,h))
        if short<2.5 or not 8<=long<=90 or not 1.5<=long/short<=4.8:continue
        if area/max(w*h,1)<.53:continue
        hull=cv2.contourArea(cv2.convexHull(c))
        if area/max(hull,1)<.70:continue
        x,y,bw,bh=cv2.boundingRect(c)
        box=(x,y,x+bw,y+bh)
        if any(max(0,min(box[2],b[2])-max(box[0],b[0]))*max(0,min(box[3],b[3])-max(box[1],b[1]))/
               max(1,bw*bh+(b[2]-b[0])*(b[3]-b[1])-max(0,min(box[2],b[2])-max(box[0],b[0]))*max(0,min(box[3],b[3])-max(box[1],b[1])))>.65 for b in result):continue
        result.append(box)
        if len(result)>=512:break
    return result[:512]


def vehicle_patch(rgb,box):
    x1,y1,x2,y2=box;cx,cy=(x1+x2)/2,(y1+y2)/2;side=max(x2-x1,y2-y1)*1.5
    # 保持局部上下文与宽高比例，与来源图像尺寸无关。
    transform=np.array([[side/64,0,cx-side/2],[0,side/64,cy-side/2]],dtype=np.float32)
    patch=cv2.warpAffine(rgb,transform,(64,64),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REFLECT_101)
    return patch


class VehicleAppearance(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,48,3,2,1),nn.ReLU(),nn.Conv2d(48,64,3,2,1),nn.ReLU())
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(1024,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,x):return self.head(self.features(x))


class PatchPhotoDetector:
    def __init__(self,model,confidence=.65):
        self.model=model.eval();self.confidence=confidence
        self.names=('true_vehicle','decoy_vehicle');self.size=64
    def warmup(self):
        with torch.inference_mode():self.model(torch.zeros(1,3,64,64))
    def __call__(self,photo):
        if not isinstance(photo,bytes) or not photo or len(photo)>16_000_000:raise ValueError('bounded own photo required')
        with Image.open(BytesIO(photo)) as im:
            if not 1<im.width<=4096 or not 1<im.height<=4096:raise ValueError('unsupported photo dimensions')
            rgb=np.asarray(im.convert('RGB'))
        boxes=vehicle_proposals(rgb)
        if not boxes:return []
        patches=np.stack([vehicle_patch(rgb,b) for b in boxes]).transpose(0,3,1,2).copy()
        with torch.inference_mode():scores=self.model(torch.from_numpy(patches).float()/255).softmax(1).numpy()
        h,w=rgb.shape[:2];result=[]
        for b,s in zip(boxes,scores):
            label=int(s.argmax())
            if label==2 or s[label]<self.confidence:continue
            margin=float(s[label]-max(s[i] for i in range(3) if i!=label))
            result.append(PixelBox(*b,float(s[label]),w,h,('true_vehicle','decoy_vehicle')[label],margin))
        kept=[]
        for box in sorted(result,key=lambda b:-b.confidence):
            duplicate=False
            for other in kept:
                intersection=max(0,min(box.x2,other.x2)-max(box.x1,other.x1))*max(0,min(box.y2,other.y2)-max(box.y1,other.y1))
                union=(box.x2-box.x1)*(box.y2-box.y1)+(other.x2-other.x1)*(other.y2-other.y1)-intersection
                if intersection/max(1,union)>.35:duplicate=True;break
            if not duplicate:kept.append(box)
            if len(kept)>=32:break
        return kept
