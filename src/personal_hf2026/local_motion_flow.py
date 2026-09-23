# 修改时间：2026-09-23。
# 修改目的：缩短局部光流确认运动目标所需的视觉时间。
# 修改内容：将有效转换窗口从八次调整为五次，其余判定阈值保持不变。
# 修改时间：2026-09-23。
# 修改目的：让小目标周围有足够道路角点，并避免单次光流失效抹除已有动静证据。
# 修改内容：统一扩大局部 ROI，在跳过坏帧时保留窗口并补齐窗口内的目标锚点断层。
# 修改时间：2026-09-23。
# 修改目的：让有限角点集中覆盖小目标周围，并防止无效轨迹编号的检测框污染背景光流。
# 修改内容：局部道路 ROI 限定全图单次提点区域，同时在前后帧遮蔽所有合法车辆框。
# 修改时间：2026-09-23。
# 修改目的：只凭本机连续图像和同帧 YOLO 轨迹保守区分地面车辆的动静状态。
# 修改内容：新增局部背景 KLT、前后向校验、仿射 RANSAC 和八次有效转换的双窗口确认。
"""单机局部背景光流动静判定，不读取位姿、地理投影或 Runner 真值。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _box(value: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    raw = _field(value, "bbox_xyxy", _field(value, "xyxy"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        return None
    try:
        box = tuple(float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = box
    if not (all(math.isfinite(v) for v in box) and 0 <= x1 < x2 <= width
            and 0 <= y1 < y2 <= height):
        return None
    return box  # type: ignore[return-value]


def _anchor(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.array(((box[0] + box[2]) * 0.5, box[3]), dtype=np.float64)


@dataclass(frozen=True)
class LocalMotionParameters:
    """第一版阈值；像素与框高度同时约束判定。"""

    max_corners: int = 600
    fb_error_px: float = 1.5
    min_local_points: int = 10
    min_inliers: int = 8
    min_inlier_ratio: float = 0.55
    max_rmse_px: float = 2.0
    window_transitions: int = 5
    max_transition_gap_s: float = 1.5


@dataclass
class _TrackWindow:
    transitions: deque = field(default_factory=lambda: deque(maxlen=8))
    last_raw: str = "UNKNOWN"


class LocalMotionDetector:
    """每架 UAV 持有一个实例；调用方按图像采集顺序提交新帧。"""

    def __init__(self, parameters: LocalMotionParameters | None = None) -> None:
        self.parameters = parameters or LocalMotionParameters()
        self.reset()

    def reset(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._prev_boxes: dict[int, tuple[float, float, float, float]] = {}
        self._prev_all_boxes: tuple[tuple[float, float, float, float], ...] = ()
        self._prev_frame_id: str | None = None
        self._prev_time: float | None = None
        self._tracks: dict[int, _TrackWindow] = {}

    @staticmethod
    def _unknown(reason: str, **evidence: Any) -> dict[str, Any]:
        return {"decision": "UNKNOWN", "raw_decision": "UNKNOWN",
                "confirmed_decision": "UNKNOWN", "reason": reason,
                "local_feature_count": 0, "affine_inlier_count": 0,
                "affine_inlier_ratio": None, "affine_rmse_px": None,
                "window_valid_frames": 0, "static_predicted_pixel": None,
                "actual_bottom_center": None, "endpoint_error_px": None,
                "bbox_height_px": None, "normalized_error": None,
                "T_static": None, "T_moving": None, **evidence}

    @staticmethod
    def _detect_boxes(detections: Sequence[Any], width: int, height: int) -> dict[int, tuple[float, float, float, float]]:
        boxes = {}
        for detection in detections:
            track_id = _field(detection, "track_id")
            box = _box(detection, width, height)
            try:
                key = int(track_id)
            except (TypeError, ValueError):
                continue
            if key < 0 or box is None:
                continue
            boxes[key] = box
        return boxes

    @staticmethod
    def _local_roi(
        box: tuple[float, float, float, float], image_width: int, image_height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx = (x1 + x2) * 0.5
        half_w = max(2.0 * bw, 50.0)
        bottom_extra = max(1.5 * bh, 45.0)
        xa = max(0, int(math.floor(cx - half_w)))
        xb = min(image_width, int(math.ceil(cx + half_w)))
        ya = max(0, int(math.floor(y1 + 0.3 * bh)))
        yb = min(image_height, int(math.ceil(y2 + bottom_extra)))
        return xa, ya, xb, yb

    @staticmethod
    def _background_mask(
        shape: tuple[int, int],
        boxes: Sequence[tuple[float, float, float, float]],
        focus_boxes: Sequence[tuple[float, float, float, float]] = (),
    ) -> np.ndarray:
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8) if focus_boxes else np.full(
            (height, width), 255, dtype=np.uint8)
        # 有候选目标时把角点预算集中到其脚下及两侧。
        for box in focus_boxes:
            xa, ya, xb, yb = LocalMotionDetector._local_roi(box, width, height)
            mask[ya:yb, xa:xb] = 255
        for x1, y1, x2, y2 in boxes:
            dx, dy = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
            xa, ya = max(0, int(math.floor(x1 - dx))), max(0, int(math.floor(y1 - dy)))
            xb, yb = min(width, int(math.ceil(x2 + dx))), min(height, int(math.ceil(y2 + dy)))
            mask[ya:yb, xa:xb] = 0
        return mask

    def _background_flow(
        self,
        gray: np.ndarray,
        all_boxes: tuple[tuple[float, float, float, float], ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._prev_gray is not None
        previous_mask = self._background_mask(
            gray.shape, self._prev_all_boxes, tuple(self._prev_boxes.values()))
        corners = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=self.parameters.max_corners,
            qualityLevel=0.01, minDistance=7, blockSize=7, mask=previous_mask,
        )
        empty = np.empty((0, 2), dtype=np.float32)
        if corners is None:
            return empty, empty
        forward, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, corners, None, winSize=(31, 31), maxLevel=4,
        )
        if forward is None or forward_ok is None:
            return empty, empty
        valid_forward = (forward_ok[:, 0] == 1) & np.isfinite(forward[:, 0]).all(axis=1)
        corners, forward = corners[valid_forward], forward[valid_forward]
        if len(corners) == 0:
            return empty, empty
        backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, forward, None, winSize=(31, 31), maxLevel=4,
        )
        if backward is None or backward_ok is None:
            return empty, empty
        old, new, back = corners[:, 0], forward[:, 0], backward[:, 0]
        good = ((backward_ok[:, 0] == 1)
                & np.isfinite(back).all(axis=1)
                & (np.linalg.norm(old - back, axis=1) < self.parameters.fb_error_px))
        # 当前帧的目标框也要排除，防止背景点落入移动车辆或遮挡处。
        current_mask = self._background_mask(gray.shape, all_boxes)
        x = np.clip(np.rint(new[:, 0]).astype(np.int32), 0, gray.shape[1] - 1)
        y = np.clip(np.rint(new[:, 1]).astype(np.int32), 0, gray.shape[0] - 1)
        good &= current_mask[y, x] != 0
        good &= ((new[:, 0] >= 0) & (new[:, 0] < gray.shape[1])
                 & (new[:, 1] >= 0) & (new[:, 1] < gray.shape[0]))
        return old[good], new[good]

    def _transition(
        self,
        track_id: int,
        old_box: tuple[float, float, float, float],
        new_box: tuple[float, float, float, float],
        old_points: np.ndarray,
        new_points: np.ndarray,
    ) -> dict[str, Any]:
        assert self._prev_gray is not None
        image_height, image_width = self._prev_gray.shape
        xa, ya, xb, yb = self._local_roi(old_box, image_width, image_height)
        local = ((old_points[:, 0] >= xa)
                 & (old_points[:, 0] < xb)
                 & (old_points[:, 1] >= ya)
                 & (old_points[:, 1] < yb))
        p0, p1 = old_points[local], new_points[local]
        count = len(p0)
        if count < self.parameters.min_local_points:
            return self._unknown("insufficient_local_features", local_feature_count=count)
        affine, inlier_mask = cv2.estimateAffine2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0,
            maxIters=1000, confidence=0.99, refineIters=10,
        )
        if affine is None or inlier_mask is None or not np.isfinite(affine).all():
            return self._unknown("affine_failed", local_feature_count=count)
        inliers = inlier_mask[:, 0].astype(bool)
        inlier_count = int(np.sum(inliers))
        ratio = inlier_count / count
        estimate = p0[inliers] @ affine[:, :2].T + affine[:, 2]
        rmse = float(np.sqrt(np.mean(np.sum((estimate - p1[inliers]) ** 2, axis=1)))) if inlier_count else math.inf
        evidence = {"local_feature_count": count, "affine_inlier_count": inlier_count,
                    "affine_inlier_ratio": ratio, "affine_rmse_px": rmse}
        if (inlier_count < self.parameters.min_inliers
                or ratio <= self.parameters.min_inlier_ratio
                or rmse >= self.parameters.max_rmse_px):
            return self._unknown("weak_local_affine", **evidence)

        window = self._tracks.setdefault(track_id, _TrackWindow(
            transitions=deque(maxlen=self.parameters.window_transitions)))
        window.transitions.append((affine.copy(), _anchor(old_box), _anchor(new_box),
                                   float(new_box[3] - new_box[1]), rmse))
        steps = tuple(window.transitions)
        evidence["window_valid_frames"] = len(steps)
        if len(steps) < self.parameters.window_transitions:
            window.last_raw = "UNKNOWN"
            return self._unknown("warmup", **evidence)
        predicted = steps[0][1].copy()
        previous_endpoint = None
        for matrix, old_anchor, new_anchor, _, _ in steps:
            # 坏帧没有仿射证据，用两侧已观测锚点补齐断层，不把该帧运动计入窗口。
            if previous_endpoint is not None:
                predicted += old_anchor - previous_endpoint
            predicted = matrix[:, :2] @ predicted + matrix[:, 2]
            previous_endpoint = new_anchor
        actual = steps[-1][2]
        endpoint_error = float(np.linalg.norm(actual - predicted))
        median_height = float(np.median([step[3] for step in steps]))
        sigma = float(np.median([step[4] for step in steps]))
        threshold_static = max(0.06 * median_height, 3.0, 2.0 * sigma)
        threshold_moving = max(0.18 * median_height, 6.0, 4.0 * sigma)
        raw = ("STATIC" if endpoint_error < threshold_static else
               "MOVING" if endpoint_error > threshold_moving else "UNKNOWN")
        confirmed = raw if raw != "UNKNOWN" and raw == window.last_raw else "UNKNOWN"
        window.last_raw = raw
        evidence.update({
            "static_predicted_pixel": predicted.tolist(),
            "actual_bottom_center": actual.tolist(),
            "endpoint_error_px": endpoint_error,
            "bbox_height_px": median_height,
            "normalized_error": endpoint_error / median_height,
            "T_static": threshold_static,
            "T_moving": threshold_moving,
        })
        return {"decision": confirmed, "raw_decision": raw,
                "confirmed_decision": confirmed,
                "reason": "confirmed" if confirmed != "UNKNOWN" else "awaiting_confirmation",
                **evidence}

    def update(
        self,
        frame_bgr: np.ndarray | None,
        detections: Sequence[Any],
        frame_time_s: float,
        *,
        frame_id: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        """返回按字符串 track_id 索引的证据；仅新且递增的同尺寸帧可累计。"""
        try:
            time_s = float(frame_time_s)
        except (TypeError, ValueError):
            self.reset()
            return {}
        if (frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
                or not math.isfinite(time_s)):
            self.reset()
            return {}
        frame_id = frame_id or str(time_s)
        height, width = frame_bgr.shape[:2]
        boxes = self._detect_boxes(detections, width, height)
        all_boxes = tuple(box for detection in detections
                          if (box := _box(detection, width, height)) is not None)
        if frame_id == self._prev_frame_id:
            return {tid: {"track_id": tid, **self._unknown("duplicate_frame")}
                    for tid in boxes}
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if (self._prev_gray is None or gray.shape != self._prev_gray.shape
                or self._prev_time is None or time_s <= self._prev_time
                or time_s - self._prev_time > self.parameters.max_transition_gap_s):
            self.reset()
            self._prev_gray, self._prev_boxes = gray, boxes
            self._prev_all_boxes = all_boxes
            self._prev_frame_id, self._prev_time = frame_id, time_s
            return {tid: {"track_id": tid, **self._unknown("first_or_discontinuous_frame")}
                    for tid in boxes}
        old_points, new_points = self._background_flow(gray, all_boxes)
        results = {}
        for tid, box in boxes.items():
            old_box = self._prev_boxes.get(tid)
            if old_box is None:
                self._tracks.pop(tid, None)
                results[tid] = {"track_id": tid, **self._unknown("new_track")}
                continue
            result = self._transition(tid, old_box, box, old_points, new_points)
            results[tid] = {"track_id": tid, **result}
        self._tracks = {tid: state for tid, state in self._tracks.items() if tid in boxes}
        self._prev_gray, self._prev_boxes = gray, boxes
        self._prev_all_boxes = all_boxes
        self._prev_frame_id, self._prev_time = frame_id, time_s
        return results


__all__ = ["LocalMotionDetector", "LocalMotionParameters"]
