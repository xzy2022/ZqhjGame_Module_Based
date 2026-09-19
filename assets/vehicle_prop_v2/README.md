# VehicleProp V2 实时识别

本目录固定队友 `yzy/0919/UE_test_used` 的 FOV30+FOV48 混合训练权重与来源哈希；
`Training_Project` 提供完整训练实现。PT 权重 SHA-256 为
`a30c839d72a6d996eac62b0ab15a147ee03d3c28c0a3853bb25b16c04445cfc2`。
源 Python 文件与权重清单见 `source_sha256.json`；运行源码在
`src/personal_hf2026/vehicle_prop_v2`，不依赖队友目录、独立 worker 或其 `.runtime`。

配置 `configs/detectors/vehicle_prop_v2/vehicle_realtime.json` 固定可移植 PT；
`vehicle_realtime_pt.json` 与之等价。在线 `V2` 档位将显式覆盖为本机生成的 TensorRT engine，
engine 和 ONNX 属于外置实验产物，不纳入 Git。队友交付的 RTX 5070 Ti / TensorRT 10.13 engine
不能直接当作本机 engine；运行时核对 TensorRT 版本、GPU 型号、算力与资源哈希。

```python
from personal_hf2026.vehicle_prop import create_detector

detector = create_detector(profile="V2", device="0", weights=engine_path,
                           weights_sha256=engine_sha256)
predictions = detector.predict(image_bgr, timestamp=source_sim_time,
                               sequence_id=run_id, stream_id=uav_uid)
```

只有一个同步模型；三机通过 `stream_id` 保留独立 FastCameraMotion 与 TemporalTracker。
输入固定 `[1,3,1152,1536]`，单视图、raw 两类分数、保留队友 `high=0.4`、`low=0.1`、
拒识阈值 `0.6`。时序融合使用源仿真时间，最大间隔为 `0.25 s`；重复或乱序源帧跳过。
`image_rotation_deg` 默认为 0，可选 90/180/270 仅供诊断，输出框及速度还原到原图。
V1 工厂及其既有旋转行为保留；旧 `v1/v2/v3` 是 `V1-v1/V1-v2/V1-v3` 的别名。

在模块代码可导入的环境中，用目标 GPU 重新导出，输出使用仓库外的新目录：

```powershell
$trtPython = 'D:\venvs\hf26trt\Scripts\python.exe'
& $trtPython -m personal_hf2026.vehicle_prop_v2.export_realtime `
  --weights '.\assets\vehicle_prop_v2\yolo26s_fovmix_blend50.pt' `
  --output '..\output\personal_v2\new-export\vehicle_realtime_fp16.engine'

& $trtPython -m personal_hf2026.vehicle_prop_v2.verify_engine `
  --engine '..\output\personal_v2\new-export\vehicle_realtime_fp16.engine' `
  --manifest 'D:\path\to\UE_test_used\validation\frames.jsonl' `
  --image-root 'D:\path\to\UE_test_used' `
  --output '..\output\personal_v2\new-export\parity.json'
```

导出保留队友方法：卷积允许 FP16，检测头解码的坐标与概率运算约束为 FP32。
核验入口只读取真实验证图片与标签，保存 PT/engine 逐帧结果、资源哈希及每种 FOV 指标；
它不改部署配置，不代表在线检测精度结论。验证指标复用本仓库评估器，与队友原评价器的匹配细节可能不同。

Ultralytics 版本固定为 `8.4.154`。其 AGPL-3.0 许可证已保存于
`licenses/vehicle_prop/ULTRALYTICS_AGPL-3.0.txt`。队友源码与权重未声明额外独立许可证，
继续作为竞赛团队内部材料使用，不据此推断额外授权。
