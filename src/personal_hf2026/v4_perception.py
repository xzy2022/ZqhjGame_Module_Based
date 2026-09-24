# 修改时间：2026-09-24。
# 修改目的：让 Agent4 只从本机像素取得单帧 YOLO 结果并隔离开发诊断真值。
# 修改内容：复用最新帧调度和实时模型后端，保存诊断前后对象及合法提交姿态。
"""Agent4 的单帧像素感知；诊断真值仅在 Runner 显式注入时使用。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import math
import time
from typing import Mapping

import cv2
import numpy as np

from .paths import PROJECT_ROOT
from .v3_perception import V3PerceptionWorker


@dataclass(frozen=True)
class V4Object:
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float
    real_probability: float
    decoy_probability: float
    class_name: str


@dataclass(frozen=True)
class V4Snapshot:
    uid: str
    frame_id: str
    source_sim_time: float
    source_time_basis: str
    observed_sim_time: float
    image_size: tuple[int, int]
    source_pose: Mapping[str, float]
    raw_yolo_objects: tuple[V4Object, ...]
    effective_yolo_objects: tuple[V4Object, ...]
    vision_diagnostic: Mapping[str, object]
    inference_wall_ms: float
    completed_perf_counter: float
    image_bgr: np.ndarray | None = None
    motion_error: str | None = None
    error: str | None = None


def _object(record, image_size):
    width, height = image_size
    raw = record.get("bbox_xyxy", record.get("xyxy"))
    if raw is None or len(raw) != 4:
        return None
    box = tuple(float(value) for value in raw)
    if not (all(math.isfinite(value) for value in box)
            and 0 <= box[0] < box[2] <= width
            and 0 <= box[1] < box[3] <= height):
        return None
    probs = record.get("class_probabilities", (0.5, 0.5))
    real, decoy = float(probs[0]), float(probs[1])
    return V4Object(box, float(record.get("detector_confidence", record.get("confidence", 0.0))),
                    real, decoy, str(record.get("class_name", "uncertain")))


class VisionDiagnosticV4:
    """只在 Runner 感知边界接收同帧 UE 投影框。"""

    def __init__(self, mode="000"):
        if len(str(mode)) != 3 or any(bit not in "01" for bit in str(mode)):
            raise ValueError("--vision-diagnostic 必须为 000 到 111")
        self.mode = str(mode)
        self.counts = Counter()

    @property
    def enabled(self):
        return self.mode != "000"

    @property
    def summary(self):
        return {"mode": self.mode, "enabled": self.enabled, "counts": dict(self.counts),
                "truth_boundary": "runner_diagnostic_only"}

    @staticmethod
    def _intersection(left, right):
        return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
            0.0, min(left[3], right[3]) - max(left[1], right[1]))

    def apply(self, frame_id, raw, image_size, metadata):
        if not self.enabled:
            return tuple(raw), False
        self.counts["completed_frames_seen"] += 1
        if not isinstance(metadata, dict) or str(metadata.get("frame_id")) != str(frame_id):
            self.counts["frames_truth_unavailable"] += 1
            return tuple(raw), False
        self.counts["frames_truth_available"] += 1
        width, height = image_size
        truth = []
        for entry in metadata.get("ue_projected_objects", ()):
            name = {"TargetVehicle": "real_vehicle", "DecoyVehicle": "model_prop"}.get(
                str(entry.get("class", ""))) if isinstance(entry, dict) else None
            box = entry.get("bbox") if isinstance(entry, dict) else None
            if name is None or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            box = tuple(float(x) for x in box)
            if (all(math.isfinite(x) for x in box) and 0 <= box[0] < box[2] <= width
                    and 0 <= box[1] < box[3] <= height):
                truth.append((box, name))
        overlap = [[self._intersection(obj.bbox_xyxy, item[0]) for item in truth] for obj in raw]
        effective = []
        for index, obj in enumerate(raw):
            matches = [(area, j) for j, area in enumerate(overlap[index]) if area > 0]
            if self.mode[1] == "1" and not matches:
                self.counts["unmatched_predictions_removed"] += 1
                continue
            if self.mode[0] == "1" and matches:
                j = min(matches, key=lambda pair: (-pair[0], pair[1]))[1]
                name = truth[j][1]
                obj = V4Object(obj.bbox_xyxy, obj.detector_confidence,
                               1.0 if name == "real_vehicle" else 0.0,
                               0.0 if name == "real_vehicle" else 1.0, name)
                self.counts["labels_replaced"] += 1
            effective.append(obj)
        if self.mode[2] == "1":
            for j, (box, name) in enumerate(truth):
                if any(row[j] > 0 for row in overlap):
                    continue
                effective.append(V4Object(box, 1.0, 1.0 if name == "real_vehicle" else 0.0,
                                          0.0 if name == "real_vehicle" else 1.0, name))
                self.counts["unmatched_truth_added"] += 1
        changed = tuple(effective) != tuple(raw)
        self.counts["frames_changed"] += int(changed)
        return tuple(effective), changed


class V4PerceptionWorker(V3PerceptionWorker):
    """仅复用 V3 的共享 latest-only 队列；推理和对象定义均为 Agent4 专用。"""

    def __init__(self, *, device="0", config=None, weights=None, diagnostic=None):
        self.diagnostic = diagnostic or VisionDiagnosticV4()
        super().__init__(detector_kwargs={"device": device, "config": config, "weights": weights})

    def _create_detector(self):
        from .vehicle_prop_v2.realtime_backend import RealtimeDetector
        import torch

        config_path = Path(self._detector_kwargs["config"] or
                           PROJECT_ROOT / "configs/detectors/vehicle_prop_v2/vehicle_realtime.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        spec = dict(config["detector"])
        weights = self._detector_kwargs["weights"]
        if weights is None:
            weights = PROJECT_ROOT / config["weights"]
        else:
            spec["expected_sha256"] = None
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        detector = RealtimeDetector(weights, device=self._detector_kwargs["device"], **spec)
        detector.warmup()
        self._unknown_threshold = float(config.get("unknown_threshold", 0.6))
        return detector

    def _infer(self, detector, job):
        started = time.perf_counter()
        size = (0, 0)
        image = None
        raw = effective = ()
        changed = False
        error = None
        try:
            image = cv2.imdecode(np.frombuffer(job.photo, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法解码相机图片")
            height, width = image.shape[:2]
            size = (int(width), int(height))
            records = detector.predict(image)
            objects = []
            for row in records:
                box = tuple(float(value) for value in row[:4])
                real, decoy = float(row[5]), float(row[6])
                name = ("real_vehicle" if real >= decoy else "model_prop") if max(
                    real, decoy) >= self._unknown_threshold else "uncertain"
                item = _object({"bbox_xyxy": box, "detector_confidence": float(row[4]),
                                "class_probabilities": (real, decoy), "class_name": name}, size)
                if item is not None:
                    objects.append(item)
            raw = tuple(objects)
            effective, changed = self.diagnostic.apply(
                job.frame_id, raw, size, job.diagnostic_metadata)
        except Exception as exc:
            error = repr(exc)
        completed = time.perf_counter()
        return V4Snapshot(job.uid, job.frame_id, job.source_sim_time, job.source_time_basis,
                          job.observed_sim_time, size, dict(job.own_pose or {}), raw, effective,
                          {"mode": self.diagnostic.mode, "changed": changed},
                          (completed - started) * 1000, completed, image, error=error)


def submit_observation(worker, obs, *, diagnostic_metadata=None):
    own = obs.self
    score = obs.briefing.score_view
    now = float(score.sim_time)
    if own.photo:
        pose = {name: float(getattr(own, name)) for name in (
            "lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt",
            "gimbal_fov_deg")}
        worker.submit(str(own.uid), own.photo, observed_sim_time=now, source_sim_time=now,
                      source_time_basis="observation_sim_time_not_verified_exposure",
                      fov_deg=pose["gimbal_fov_deg"], own_pose=pose,
                      diagnostic_metadata=diagnostic_metadata)
    return worker.latest(str(own.uid), now_sim_time=now, max_age_s=1.5)
