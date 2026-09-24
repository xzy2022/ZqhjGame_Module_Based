# 修改时间：2026-09-24。
# 修改目的：将局部背景光流判定绑定到本机唯一 Entity 并保留短暂缺帧证据。
# 修改内容：复用 KLT 与仿射判定数学，移除按轨迹编号存储的窗口。
"""单 Entity 的局部背景光流动静判定。"""
from __future__ import annotations

from collections import deque
import math

import cv2
import numpy as np

from .local_motion_flow import LocalMotionDetector, LocalMotionParameters, _anchor


class SingleEntityMotion:
    def __init__(self, parameters=None):
        self.parameters = parameters or LocalMotionParameters()
        self.reset()

    def reset(self):
        self._prev_gray = None
        self._prev_box = None
        self._prev_all_boxes = ()
        self._prev_time = None
        self._transitions = deque(maxlen=self.parameters.window_transitions)
        self._last_raw = "UNKNOWN"
        self.decision = "UNKNOWN"
        self.evidence = {"decision": self.decision, "reason": "reset"}

    def _flow(self, gray, all_boxes):
        previous_mask = LocalMotionDetector._background_mask(
            gray.shape, self._prev_all_boxes, (self._prev_box,))
        corners = cv2.goodFeaturesToTrack(
            self._prev_gray, maxCorners=self.parameters.max_corners,
            qualityLevel=0.01, minDistance=7, blockSize=7, mask=previous_mask)
        empty = np.empty((0, 2), np.float32)
        if corners is None:
            return empty, empty
        forward, ok, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, corners, None, winSize=(31, 31), maxLevel=4)
        if forward is None or ok is None:
            return empty, empty
        valid = (ok[:, 0] == 1) & np.isfinite(forward[:, 0]).all(axis=1)
        corners, forward = corners[valid], forward[valid]
        if not len(corners):
            return empty, empty
        backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, forward, None, winSize=(31, 31), maxLevel=4)
        if backward is None or backward_ok is None:
            return empty, empty
        old, new, back = corners[:, 0], forward[:, 0], backward[:, 0]
        good = ((backward_ok[:, 0] == 1) & np.isfinite(back).all(axis=1)
                & (np.linalg.norm(old - back, axis=1) < self.parameters.fb_error_px))
        mask = LocalMotionDetector._background_mask(gray.shape, all_boxes)
        x = np.clip(np.rint(new[:, 0]).astype(np.int32), 0, gray.shape[1] - 1)
        y = np.clip(np.rint(new[:, 1]).astype(np.int32), 0, gray.shape[0] - 1)
        good &= mask[y, x] != 0
        good &= (new[:, 0] >= 0) & (new[:, 0] < gray.shape[1])
        good &= (new[:, 1] >= 0) & (new[:, 1] < gray.shape[0])
        return old[good], new[good]

    def _transition(self, old_box, box, old_points, new_points):
        h, w = self._prev_gray.shape
        xa, ya, xb, yb = LocalMotionDetector._local_roi(old_box, w, h)
        selected = ((old_points[:, 0] >= xa) & (old_points[:, 0] < xb)
                    & (old_points[:, 1] >= ya) & (old_points[:, 1] < yb))
        p0, p1 = old_points[selected], new_points[selected]
        count = len(p0)
        if count < self.parameters.min_local_points:
            return {"decision": self.decision, "reason": "insufficient_local_features",
                    "local_feature_count": count}
        affine, inlier_mask = cv2.estimateAffine2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0,
            maxIters=1000, confidence=0.99, refineIters=10)
        if affine is None or inlier_mask is None or not np.isfinite(affine).all():
            return {"decision": self.decision, "reason": "affine_failed"}
        inliers = inlier_mask[:, 0].astype(bool)
        n = int(np.sum(inliers))
        ratio = n / count
        estimate = p0[inliers] @ affine[:, :2].T + affine[:, 2]
        rmse = float(np.sqrt(np.mean(np.sum((estimate - p1[inliers]) ** 2, axis=1)))) if n else math.inf
        evidence = {"local_feature_count": count, "affine_inlier_count": n,
                    "affine_inlier_ratio": ratio, "affine_rmse_px": rmse}
        if (n < self.parameters.min_inliers or ratio <= self.parameters.min_inlier_ratio
                or rmse >= self.parameters.max_rmse_px):
            return {"decision": self.decision, "reason": "weak_local_affine", **evidence}
        self._transitions.append((affine.copy(), _anchor(old_box), _anchor(box),
                                  float(box[3] - box[1]), rmse))
        steps = tuple(self._transitions)
        evidence["window_valid_frames"] = len(steps)
        if len(steps) < self.parameters.window_transitions:
            return {"decision": self.decision, "reason": "warmup", **evidence}
        predicted = steps[0][1].copy()
        previous = None
        for matrix, old_anchor, new_anchor, _, _ in steps:
            if previous is not None:
                predicted += old_anchor - previous
            predicted = matrix[:, :2] @ predicted + matrix[:, 2]
            previous = new_anchor
        error = float(np.linalg.norm(steps[-1][2] - predicted))
        height = float(np.median([step[3] for step in steps]))
        sigma = float(np.median([step[4] for step in steps]))
        static_limit = max(0.06 * height, 3.0, 2.0 * sigma)
        moving_limit = max(0.18 * height, 6.0, 4.0 * sigma)
        raw = ("STATIC" if error < static_limit else
               "MOVING" if error > moving_limit else "UNKNOWN")
        confirmed = raw if raw != "UNKNOWN" and raw == self._last_raw else "UNKNOWN"
        self._last_raw = raw
        if confirmed != "UNKNOWN":
            self.decision = confirmed
        evidence.update({"decision": self.decision, "raw_decision": raw,
                         "confirmed_decision": confirmed, "endpoint_error_px": error,
                         "bbox_height_px": height, "T_static": static_limit,
                         "T_moving": moving_limit, "reason": "confirmed" if confirmed != "UNKNOWN"
                         else "awaiting_confirmation"})
        return evidence

    def update(self, image_bgr, box, other_boxes, time_s):
        """仅实体可见时调用；缺帧时调用方不触碰窗口。"""
        if box is None or image_bgr is None:
            return self.evidence
        time_s = float(time_s)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        all_boxes = tuple([box, *other_boxes])
        if (self._prev_gray is None or gray.shape != self._prev_gray.shape
                or self._prev_time is None or time_s <= self._prev_time
                or time_s - self._prev_time > self.parameters.max_transition_gap_s):
            # 时间断层只更新参考帧，不删除已确认的运动窗口。
            self._prev_gray, self._prev_box = gray, box
            self._prev_all_boxes, self._prev_time = all_boxes, time_s
            self.evidence = {"decision": self.decision, "reason": "first_or_discontinuous_frame",
                             "window_valid_frames": len(self._transitions)}
            return self.evidence
        old_points, new_points = self._flow(gray, all_boxes)
        self.evidence = self._transition(self._prev_box, box, old_points, new_points)
        self._prev_gray, self._prev_box = gray, box
        self._prev_all_boxes, self._prev_time = all_boxes, time_s
        return self.evidence
