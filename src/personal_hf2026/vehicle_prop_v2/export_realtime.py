# 修改时间：2026-09-19。
# 修改目的：将队友 V2 的单模型实时识别实现独立集成到模块仓库。
# 修改内容：固定源码来源、改用包内导入与外置运行目录，并保留原始检测及因果跟踪算法。
"""Build a target-GPU TensorRT FP16 engine preserving both raw class scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import time
from ..vehicle_prop.frontier_runtime import activate
activate()
import torch
from ultralytics import YOLO
from .realtime_backend import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--height',type=int,default=1152)
    p.add_argument('--width',type=int,default=1536)
    p.add_argument('--workspace-gb',type=float,default=2)
    p.add_argument('--device',type=int,default=0)
    args=p.parse_args()
    import tensorrt as trt
    import onnx
    if args.output.exists():
        raise FileExistsError('Use a new output name; engines are immutable experiment artifacts')
    if args.height%32 or args.width%32:
        raise ValueError('Height and width must be divisible by 32')
    torch.cuda.set_device(args.device);torch.set_num_threads(4)
    device=torch.device(f'cuda:{args.device}')
    source=YOLO(str(args.weights)).model
    if source.names!={0:'real_vehicle',1:'model_prop'}:
        raise ValueError('Expected two-class vehicle checkpoint')
    source.end2end=False
    source=source.to(device).float().eval().fuse(verbose=False)
    parameters=sum(p.numel() for p in source.parameters())
    head=source.model[-1]
    head.export=True;head.format='onnx';head.dynamic=False;head.xyxy=False
    example=torch.zeros((1,3,args.height,args.width),device=device)
    with torch.inference_mode():
        raw=source(example)
        if raw.ndim!=3 or raw.shape[1]!=6:
            raise ValueError('Raw class probabilities were not preserved')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    onnx_path=args.output.with_suffix('.onnx')
    torch.onnx.export(source,example,str(onnx_path),input_names=['images'],output_names=['raw_scores'],
                      opset_version=17,dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(str(onnx_path))
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,int(args.workspace_gb*(1<<30)))
    if not builder.platform_has_fast_fp16:
        raise RuntimeError('Target GPU does not support fast FP16')
    config.set_flag(trt.BuilderFlag.FP16)
    # 小目标的像素坐标解码保持 FP32，避免 FP16 在 x=1024 附近的像素量化误差。
    # 卷积仍允许 FP16，仅约束检测头中的坐标和概率运算。
    protected=[]
    head_prefix=f'/model.{len(source.model)-1}/'
    arithmetic={trt.LayerType.ELEMENTWISE,trt.LayerType.ACTIVATION,trt.LayerType.SOFTMAX}
    for i in range(network.num_layers):
        layer=network.get_layer(i)
        if (head_prefix in layer.name and layer.type in arithmetic
                and '/cv2' not in layer.name and '/cv3' not in layer.name
                and any(layer.get_input(j) is not None and layer.get_input(j).dtype==trt.float32 for j in range(layer.num_inputs))):
            layer.precision=trt.float32
            for j in range(layer.num_outputs):
                if layer.get_output(j).dtype==trt.float32:layer.set_output_type(j,trt.float32)
            protected.append(layer.name)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    start=time.perf_counter()
    serialized=builder.build_serialized_network(network,config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')
    args.output.write_bytes(bytes(serialized))
    meta=dict(format='raw_two_class_v1',precision='FP16 builder with FP32 IO and automatic FP32 fallbacks',
        batch=1,input_shape=[1,3,args.height,args.width],output_shape=list(raw.shape),
        class_names=['real_vehicle','model_prop'],box_format='xywh_input_pixels',
        preprocess='BGR->RGB, letterbox constant114, CHW /255',flip=False,
        weights_sha256=sha256(args.weights),engine_sha256=sha256(args.output),onnx_sha256=sha256(onnx_path),
        parameters=parameters,tensorrt_version=trt.__version__,torch_version=torch.__version__,cuda=torch.version.cuda,
        fp32_decode_layers=protected,
        gpu=torch.cuda.get_device_name(device),compute_capability=list(torch.cuda.get_device_capability(device)),
        platform=platform.platform(),build_seconds=time.perf_counter()-start,
        portability='Rebuild on target GPU and TensorRT version; do not assume cross-device portability.')
    runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(bytes(serialized))
    inspector=engine.create_engine_inspector()
    args.output.with_suffix('.layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON),encoding='utf8')
    args.output.with_suffix('.engine.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
