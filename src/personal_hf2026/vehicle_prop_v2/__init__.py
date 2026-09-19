# 修改时间：2026-09-19。
# 修改目的：提供可直接用于现有旁路与评估器的 V2 两类检测接口。
# 修改内容：保留单模型多路独立跟踪、旋转坐标还原及可复核的真实后端元数据。
"""队友 V2 的同步实时流水线；进程调度由现有旁路负责。"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np

from personal_hf2026.paths import PROJECT_ROOT
from ..vehicle_prop import VehiclePropDetector
from .detector_realtime import RealtimePipeline
from .realtime_backend import sha256


CONFIG_PATH = PROJECT_ROOT / "configs/detectors/vehicle_prop_v2/vehicle_realtime.json"


class RealtimeVehiclePropDetector(VehiclePropDetector):
    """共享一个模型，内部维护每个 UAV 的跟踪器与相机运动状态。"""

    manages_streams = True

    def __init__(
        self,
        config: str | PathLike[str] | None = None,
        device: str = "0",
        *,
        profile: str | None = "V2",
        weights: str | PathLike[str] | None = None,
        weights_sha256: str | None = None,
        flip: bool | None = None,
        tracker_high: float | None = None,
        image_rotation_deg: int = 0,
    ):
        if flip:
            raise ValueError("V2 固定单视图，不支持 flip=True")
        if image_rotation_deg not in (0, 90, 180, 270):
            raise ValueError("image_rotation_deg 必须为 0、90、180 或 270")
        self.config_path = Path(config or CONFIG_PATH).resolve()
        self.device_requested = str(device)
        self.profile = profile
        self.image_rotation_deg = image_rotation_deg
        self.pipeline = RealtimePipeline(
            config=self.config_path,
            device=device,
            weights=weights,
            weights_sha256=weights_sha256,
            tracker_high=tracker_high,
        )

    def reset(self, stream_id: str | None = None) -> None:
        """清空全部流或某一路的时序状态，模型继续共享。"""
        self.pipeline.reset(stream_id=stream_id)

    def warmup(self) -> None:
        """按部署尺寸预热共享模型，且不产生任何跟踪状态。"""
        self.pipeline.warmup()

    def predict(
        self,
        image_bgr: np.ndarray,
        timestamp: float = 0.0,
        sequence_id: str = "single",
        stream_id: str | None = None,
    ) -> list[dict]:
        """同步推理；timestamp 使用源仿真时间，stream_id 使用 UAV 编号。"""
        height, width = image_bgr.shape[:2]
        inference_image = (
            np.ascontiguousarray(np.rot90(image_bgr, self.image_rotation_deg // 90))
            if self.image_rotation_deg else image_bgr
        )
        detections = self.pipeline.predict(
            inference_image,
            timestamp=float(timestamp),
            sequence_id=str(sequence_id),
            stream_id=stream_id,
        )
        output = []
        for detection in detections:
            record = dict(detection)
            record["bbox_xyxy"] = list(detection["xyxy"])
            if self.image_rotation_deg:
                # 跟踪器沿用旋转后的图像坐标，对外框与速度还原到原始图像。
                a, b, c, d = record["bbox_xyxy"]
                if self.image_rotation_deg == 90:
                    box = [width - d, a, width - b, c]
                elif self.image_rotation_deg == 180:
                    box = [width - c, height - d, width - a, height - b]
                else:
                    box = [b, height - c, d, height - a]
                record["bbox_xyxy"] = box
                record["xyxy"] = box
                if "motion_velocity_px_per_s" in record:
                    vx, vy = record["motion_velocity_px_per_s"]
                    record["motion_velocity_px_per_s"] = {
                        90: [-vy, vx], 180: [-vx, -vy], 270: [vy, -vx]
                    }[self.image_rotation_deg]
            record["score"] = float(detection["confidence"])
            output.append(record)
        return output

    def runtime_metadata(self) -> dict:
        """记录加载资源、实际计算后端、固定分辨率与时序规则。"""
        import cv2
        import scipy
        import torch
        import ultralytics

        pipeline = self.pipeline
        backend = pipeline.detector.backend
        weights_path = pipeline.weights_path
        engine_metadata = getattr(backend, "metadata", None)
        result = {
            "detector": "VehicleProp V2 RealtimePipeline",
            "profile": self.profile,
            "manages_streams": True,
            "device_requested": self.device_requested,
            "device_actual": str(backend.device),
            "config_path": str(self.config_path),
            "config_sha256": sha256(self.config_path),
            "weights_path": str(weights_path),
            "weights_sha256": sha256(weights_path),
            "model_format": "tensorrt_engine" if engine_metadata else "pytorch_pt",
            "backend": pipeline.detector.backend_name,
            "backend_implementation": type(backend).__name__,
            "input_shape": [1, 3, *backend.shape],
            "input_dtype": str(backend.dtype),
            "batch_size": 1,
            "model_instances": 1,
            "class_names": ["real_vehicle", "model_prop"],
            "effective_options": {
                "image_rotation_deg": self.image_rotation_deg,
                "flip_enabled": False,
                "inference_shape": list(backend.shape),
                "raw_confidence_threshold": float(pipeline.detector.confidence),
                "tracker_high_confidence_threshold": float(pipeline.settings["high"]),
                "tracker_low_confidence_threshold": float(pipeline.settings["low"]),
                "unknown_class_confidence_threshold": float(pipeline.config["unknown_threshold"]),
                "max_source_gap_s": float(pipeline.settings["max_gap_s"]),
                "motion_width": int(pipeline.config.get("motion_width", 384)),
            },
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
        if engine_metadata:
            import tensorrt

            result["tensorrt"] = tensorrt.__version__
            result["engine_metadata"] = engine_metadata
            result["engine_metadata_sha256"] = sha256(weights_path.with_suffix(".engine.json"))
            result["source_weights_sha256"] = engine_metadata["weights_sha256"]
        return result


def create_detector(*args, **kwargs) -> RealtimeVehiclePropDetector:
    """独立 V2 工厂，供仅设置 module:factory 的离线评估器使用。"""
    return RealtimeVehiclePropDetector(*args, **kwargs)


__all__ = ["RealtimePipeline", "RealtimeVehiclePropDetector", "create_detector"]
