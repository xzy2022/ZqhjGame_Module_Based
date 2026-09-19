# 修改时间：2026-09-19。
# 修改目的：使 V2 运行记录能区分实际后端及固定输入约束。
# 修改内容：暴露运行权重、后端名并核对 engine 的真实两类原始输出形状。
# 修改时间：2026-09-19。
# 修改目的：将队友 V2 的单模型实时识别实现独立集成到模块仓库。
# 修改内容：固定源码来源、改用包内导入与外置运行目录，并保留原始检测及因果跟踪算法。
"""Single-view, batch-one raw-score PyTorch and TensorRT inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ..vehicle_prop.frontier_runtime import activate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def letterbox(image, shape):
    h,w=image.shape[:2];oh,ow=map(int,shape)
    gain=min(oh/h,ow/w)
    nw,nh=round(w*gain),round(h*gain)
    left,top=(ow-nw)//2,(oh-nh)//2
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR) if (nw,nh)!=(w,h) else image
    canvas=cv2.copyMakeBorder(resized,top,oh-nh-top,left,ow-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
    rgb=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1))
    return rgb,(gain,left,top)


def decode(raw, original_shape, geometry, confidence=.05, iou=.5, max_det=100):
    if isinstance(raw,(list,tuple)):
        raw=raw[0]
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 6:
        raise ValueError(f'Expected [1,6,anchors] raw two-class head, got {tuple(raw.shape)}')
    a=raw[0].transpose(0,1).float()
    scores=a[:,4:6].amax(1)
    mask=(scores>=confidence)&torch.isfinite(a).all(1)
    a,scores=a[mask],scores[mask]
    if not len(a):
        return np.empty((0,7),dtype=np.float32)
    boxes=torch.cat((a[:,:2]-a[:,2:4]/2,a[:,:2]+a[:,2:4]/2),dim=1)
    keep=nms(boxes,scores,iou)[:max_det]
    boxes=boxes[keep];q=a[keep,4:6];q=q/q.sum(1,keepdim=True).clamp_min(1e-9)
    gain,left,top=geometry
    boxes[:,[0,2]]=(boxes[:,[0,2]]-left)/gain
    boxes[:,[1,3]]=(boxes[:,[1,3]]-top)/gain
    boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,original_shape[1])
    boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,original_shape[0])
    output=torch.cat((boxes,scores[keep,None],q),dim=1)
    output=output[(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])]
    return output.cpu().numpy()


class TorchBackend:
    def __init__(self,weights,shape=(1152,1536),device='0',half=True):
        activate()
        from ultralytics import YOLO
        self.device=torch.device('cpu' if str(device)=='cpu' else f'cuda:{device}')
        self.shape=tuple(shape)
        self.model=YOLO(str(weights)).model
        if self.model.names != {0:'real_vehicle',1:'model_prop'}:
            raise ValueError('Wrong checkpoint class mapping')
        self.model.end2end=False
        self.model=self.model.to(self.device).float().eval().fuse(verbose=False)
        self.dtype=torch.float16 if half and self.device.type=='cuda' else torch.float32
        self.model.to(dtype=self.dtype)
        self.parameters=sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def __call__(self,x):
        return self.model(x)[0]


class TensorRTBackend:
    def __init__(self,engine_path,device='0'):
        if str(device)=='cpu':
            raise ValueError('TensorRT requires an NVIDIA GPU; select the .pt fallback for CPU')
        import tensorrt as trt
        self.device=torch.device(f'cuda:{device}')
        torch.cuda.set_device(self.device)
        self.metadata=json.loads(Path(engine_path).with_suffix('.engine.json').read_text(encoding='utf8'))
        if self.metadata.get('format')!='raw_two_class_v1' or self.metadata.get('class_names')!=['real_vehicle','model_prop']:
            raise ValueError('Unexpected engine class/score contract')
        if sha256(engine_path)!=self.metadata['engine_sha256']:
            raise ValueError('TensorRT engine checksum mismatch')
        if trt.__version__!=self.metadata['tensorrt_version']:
            raise RuntimeError('Rebuild the engine for this TensorRT version using export_realtime.py')
        if torch.cuda.get_device_capability(self.device)!=tuple(self.metadata['compute_capability']):
            raise RuntimeError('Rebuild the engine for this GPU using export_realtime.py')
        if torch.cuda.get_device_name(self.device)!=self.metadata['gpu']:
            raise RuntimeError('This engine was built for a different GPU model; rebuild it on the target device')
        self.logger=trt.Logger(trt.Logger.ERROR)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError('Engine could not be deserialized; rebuild on the target GPU')
        self.context=self.engine.create_execution_context()
        self.input_name=None;self.outputs={}
        types={trt.float32:torch.float32,trt.float16:torch.float16,trt.int32:torch.int32}
        for i in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(i)
            shape=tuple(self.engine.get_tensor_shape(name))
            if min(shape)<=0:
                raise ValueError('This runtime requires a fixed-shape, batch-one engine')
            dtype=types[self.engine.get_tensor_dtype(name)]
            if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:
                if self.input_name is not None or shape[:2]!=(1,3):
                    raise ValueError('Expected one RGB input with batch one')
                self.input_name=name;self.shape=shape[2:];self.dtype=dtype
            else:
                self.outputs[name]=torch.empty(shape,dtype=dtype,device=self.device)
        if self.input_name is None or len(self.outputs)!=1:
            raise ValueError('Unexpected engine IO signature')
        if [1,3,*self.shape]!=self.metadata['input_shape']:
            raise ValueError('Engine metadata shape does not match actual bindings')
        output_shape=list(next(iter(self.outputs.values())).shape)
        if output_shape[:2]!=[1,6] or output_shape!=self.metadata['output_shape']:
            raise ValueError('Expected raw two-class output matching engine metadata')

    @torch.inference_mode()
    def __call__(self,x):
        if x.device!=self.device or x.dtype!=self.dtype or tuple(x.shape)!=(1,3,*self.shape):
            raise ValueError('TensorRT input shape/device/dtype mismatch')
        self.context.set_tensor_address(self.input_name,x.data_ptr())
        for name,value in self.outputs.items():
            self.context.set_tensor_address(name,value.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError('TensorRT execute_async_v3 failed')
        return next(iter(self.outputs.values()))


class RealtimeDetector:
    def __init__(self,weights,shape=None,device='0',half=True,
                 confidence=.05,iou=.5,expected_sha256=None):
        weights=Path(weights)
        if weights.suffix not in ('.pt','.engine'):
            raise ValueError('Use a trained .pt or a raw-score .engine exported by export_realtime.py')
        if expected_sha256 and sha256(weights)!=expected_sha256:
            raise ValueError('Weights checksum mismatch')
        self.backend=(TensorRTBackend(weights,device) if weights.suffix=='.engine'
                      else TorchBackend(weights,shape or (1152,1536),device,half))
        self.weights=weights.resolve()
        self.backend_name='tensorrt_raw_two_class' if weights.suffix=='.engine' else 'pytorch_raw_two_class'
        if shape is not None and tuple(self.backend.shape)!=tuple(shape):
            raise ValueError('Configured shape does not match the fixed engine input')
        self.confidence,self.iou=confidence,iou

    @torch.inference_mode()
    def predict(self,image):
        rgb,geometry=letterbox(image,self.backend.shape)
        x=torch.from_numpy(rgb).unsqueeze(0).to(self.backend.device,dtype=self.backend.dtype)/255
        return decode(self.backend(x),image.shape[:2],geometry,self.confidence,self.iou)

    def warmup(self,count=5):
        image=np.zeros((768,1024,3),dtype=np.uint8)
        for _ in range(count):
            self.predict(image)
        if self.backend.device.type=='cuda':
            torch.cuda.synchronize(self.backend.device)
