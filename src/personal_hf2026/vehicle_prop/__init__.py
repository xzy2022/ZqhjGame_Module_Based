# 修改时间：2026-09-18。
# 修改目的：让在线旁路以显式档位参数创建 PT 或 TensorRT 检测器。
# 修改内容：透传权重、翻转和 tracker.high 覆盖，并在运行时元数据中记录实际生效值。
# 修改时间：2026-09-18
# 修改目的：为离线评估和后续集成提供稳定的单帧、序列检测入口。
# 修改内容：封装 FrontierPipeline 并统一输出 bbox、分数、类别与审计字段。
"""队友交付的车辆与模型道具检测流水线。"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
from os import PathLike
from pathlib import Path

import numpy as np

from .detector_frontier import FrontierPipeline
from .frontier_runtime import CONFIG_PATH


class VehiclePropDetector:
    """面向评估器的轻量 API；内部算法仍由原 FrontierPipeline 执行。"""

    def __init__(
        self,
        config: str | PathLike[str] | None = None,
        device: str = "0",
        *,
        profile: str | None = None,
        weights: str | PathLike[str] | None = None,
        weights_sha256: str | None = None,
        flip: bool | None = None,
        tracker_high: float | None = None,
    ):
        self.config_path = Path(config or CONFIG_PATH).resolve()
        self.device_requested = str(device)
        self.profile = profile
        self.pipeline = FrontierPipeline(
            config=self.config_path,
            device=device,
            weights=weights,
            weights_sha256=weights_sha256,
            flip=flip,
            tracker_high=tracker_high,
        )

    def reset(self) -> None:
        """清空时序跟踪状态，开始一个独立序列。"""
        self.pipeline.reset()

    def predict(
        self,
        image_bgr: np.ndarray,
        timestamp: float = 0.0,
        sequence_id: str = "single",
    ) -> list[dict]:
        """检测一帧 BGR uint8 图像，并保留原始流水线审计字段。"""
        detections = self.pipeline.predict(
            image_bgr,
            timestamp=float(timestamp),
            sequence_id=str(sequence_id),
        )
        output = []
        for detection in detections:
            record = dict(detection)
            record["bbox_xyxy"] = list(detection["xyxy"])
            record["score"] = float(detection["confidence"])
            output.append(record)
        return output

    def predict_sequence(
        self,
        images_bgr: Iterable[np.ndarray],
        fps: float,
        sequence_id: str = "sequence",
        start_timestamp: float = 0.0,
    ) -> list[list[dict]]:
        """按固定帧率顺序处理图像；返回与输入逐帧对齐的检测列表。"""
        if fps <= 0:
            raise ValueError("fps 必须大于 0")
        self.reset()
        return [
            self.predict(image, start_timestamp + index / fps, sequence_id)
            for index, image in enumerate(images_bgr)
        ]

    def runtime_metadata(self) -> dict:
        """返回评估结果复现所需的固定资源和运行时版本。"""
        import cv2
        import scipy
        import torch
        import ultralytics

        weights_path = self.pipeline.weights_path
        return {
            "detector": "VehicleProp FrontierPipeline",
            "profile": self.profile,
            "device_requested": self.device_requested,
            "config_path": str(self.config_path),
            "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            "weights_path": str(weights_path),
            "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "model_format": (
                "tensorrt_engine" if weights_path.suffix.lower() == ".engine" else "pytorch_pt"
            ),
            "effective_options": {
                "flip_enabled": bool(self.pipeline.detector.flip),
                "tracker_high_confidence_threshold": float(self.pipeline.tracker.high),
                "tracker_low_confidence_threshold": float(self.pipeline.tracker.low),
                "unknown_class_confidence_threshold": float(
                    self.pipeline.config["unknown_threshold"]
                ),
            },
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }


def create_detector(
    config: str | PathLike[str] | None = None,
    device: str = "0",
    *,
    profile: str | None = None,
    weights: str | PathLike[str] | None = None,
    weights_sha256: str | None = None,
    flip: bool | None = None,
    tracker_high: float | None = None,
) -> VehiclePropDetector:
    """创建使用固定配置与权重的检测器。"""
    return VehiclePropDetector(
        config=config,
        device=device,
        profile=profile,
        weights=weights,
        weights_sha256=weights_sha256,
        flip=flip,
        tracker_high=tracker_high,
    )


__all__ = ["FrontierPipeline", "VehiclePropDetector", "create_detector"]
