# VehicleProp FrontierPipeline 迁移说明

本目录保存队友于 2026-09-17 交付的 YOLO26s 双类别固定权重。迁移来源为
`VehicleProp_Best_Windows_x64_20260917/VehicleProp_Best_Windows_x64_20260917`，
运行入口为 `personal_hf2026.vehicle_prop:create_detector`。

## 固定资源

- 权重：`yolo26s_two_class.pt`
- 权重 SHA-256：`de1736560d235151c031063876fdcf6a1b1ec8a2857dcf47cd72177c1d81a1f6`
- 配置：`configs/detectors/vehicle_prop/vehicle_frontier.json`
- Ultralytics：严格固定为 `8.4.154`
- 类别：`0=real_vehicle`，`1=model_prop`

配置除权重相对路径从交付包的 `weights/` 改为本仓库的 `assets/vehicle_prop/`
外，推理、融合、跟踪和拒识参数未改动。模型加载时会再次核对权重哈希。

## API

```python
from personal_hf2026.vehicle_prop import create_detector

detector = create_detector(device="0")
detections = detector.predict(image_bgr, timestamp=0.0, sequence_id="frame")
```

每项检测至少包含 `bbox_xyxy`、`score`、`class_id` 和 `class_name`；同时保留
`track_id`、`class_confidence`、`class_probabilities`、`detector_confidence` 等原始
审计字段。连续序列可以逐帧传入递增的时间戳，也可以调用 `predict_sequence`。

## 来源哈希

以下是迁移前交付包记录并在本机复核的 SHA-256。Python 文件因增加中文修改记录、
改为包内导入及切换到项目虚拟环境而不再逐字节相同；算法主体保持原实现。

| 交付包文件 | 原始 SHA-256 |
|---|---|
| `frontier_runtime.py` | `c2c2049f57f1fe4acbe38776e1bb41fafa746590e4e1b6809df8783bb19e8684` |
| `detector_frontier.py` | `744c5c55f2cd1cafe450db6e95d674cc47018922d8b9a5f744708ff792e9cc2b` |
| `temporal_tracker.py` | `a809aa0565918db2317ba8ba821f7be6286d5b569aac1bdea642f4dbc88ce218` |
| `detector_improved.py` | `9785da7719bed1d46e4c4a3a059cc0b993e9617ecb30d7ccd44f3e2403d7ab5d` |
| `detector.py` | `6e9ecc3a523757685e17a37b9435a10a152a157ca4ab1ef9adcb3307ab4e445c` |
| `common.py` | `ba7470238c913c6b5288bf92ba1ccbc67979a0062a193077634f759fd864d69b` |
| `configs/vehicle_frontier.json` | `1afbc61a2e4e08cfa58e94c1742738480ff0b53a4d89dbd3771174668b4b3329` |
| `weights/yolo26s_two_class.pt` | `de1736560d235151c031063876fdcf6a1b1ec8a2857dcf47cd72177c1d81a1f6` |

## 许可证边界

交付包没有为队友源码和模型权重声明独立许可证，本仓库不据此推断额外授权；它们按
竞赛团队内部开发材料使用。运行依赖 Ultralytics 8.4.154，其许可证为 AGPL-3.0，
完整文本保存在 `licenses/vehicle_prop/ULTRALYTICS_AGPL-3.0.txt`。交付包的第三方
说明原文保存在 `licenses/vehicle_prop/SOURCE_THIRD_PARTY_NOTICES.md`。
