# 修改时间：2026-09-22。
# 修改目的：让 PersonalV3 能在只使用本机相机像素与完整相机位姿的前提下保守判定动静目标。
# 修改内容：新增横移视差射线拟合器，输出可直接作为协同放行门的证据字典。
"""单机横移视差动静判定器。

本模块的在线输入边界刻意很窄：当前本机 ``PerceptionSnapshot`` 的
``source_pose``、已报告 ``real_vehicle`` 框的底边中心、调用方提供的完整
世界相机位姿，以及该帧的仿真时间。它不读取 SDK 理想检测、目标编号、UE
投影框、队友消息或 Runner 真值。相机完整位姿目前尚未由 V3 感知快照公开，
所以接线层必须显式提供；缺失时本模块只拒绝，不会退回不完整的 heading/pan
近似。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True)
class ParallaxParameters:
    """一次横移判定的保守固定门槛。"""

    min_span_s: float = 1.4
    max_span_s: float = 2.7
    max_detection_gap_s: float = 0.45
    min_rays: int = 5
    min_transverse_baseline_m: float = 30.0
    max_transverse_baseline_m: float = 60.0
    min_ray_angle_deg: float = 1.0
    moving_median_residual_m: float = 2.7
    moving_p95_residual_m: float = 5.5
    moving_median_reprojection_deg: float = 0.32
    moving_p95_reprojection_deg: float = 0.65


@dataclass(frozen=True)
class CameraPose:
    """世界系相机位姿，坐标轴约定为相机右、下、前对应 x、y、z。"""

    center_world_m: Vector3
    camera_to_world: Matrix3


@dataclass(frozen=True)
class _RayObservation:
    time_s: float
    track_id: str
    frame_id: str
    center_world_m: Vector3
    direction_world: Vector3


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _vector(value: Any) -> Vector3 | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    numbers = tuple(_finite_number(item) for item in value)
    return numbers if all(item is not None for item in numbers) else None  # type: ignore[return-value]


def _matrix(value: Any) -> Matrix3 | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    rows = tuple(_vector(row) for row in value)
    return rows if all(row is not None for row in rows) else None  # type: ignore[return-value]


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(left * right for left, right in zip(first, second))


def _sub(first: Vector3, second: Vector3) -> Vector3:
    return tuple(left - right for left, right in zip(first, second))  # type: ignore[return-value]


def _scale(vector: Vector3, factor: float) -> Vector3:
    return tuple(factor * item for item in vector)  # type: ignore[return-value]


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: Vector3) -> Vector3 | None:
    length = _norm(vector)
    if length <= 1e-9:
        return None
    return _scale(vector, 1.0 / length)


def _matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(_dot(row, vector) for row in matrix)  # type: ignore[return-value]


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * percent / 100.0
    low, high = math.floor(location), math.ceil(location)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> Vector3 | None:
    """解射线最小二乘正规方程；病态矩阵直接拒绝。"""
    augmented = [row[:] + [item] for row, item in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-8:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [item / divisor for item in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            multiplier = augmented[row][column]
            augmented[row] = [item - multiplier * base for item, base in zip(augmented[row], augmented[column])]
    return tuple(augmented[row][3] for row in range(3))  # type: ignore[return-value]


def _fit_static_point(window: list[_RayObservation]) -> tuple[Vector3 | None, list[float], list[float], bool]:
    """拟合所有射线共有的静止点，并给出垂距和重投影角残差。"""
    matrix = [[0.0] * 3 for _ in range(3)]
    vector = [0.0] * 3
    for observation in window:
        direction = observation.direction_world
        projection = [
            [(1.0 if row == column else 0.0) - direction[row] * direction[column] for column in range(3)]
            for row in range(3)
        ]
        for row in range(3):
            for column in range(3):
                matrix[row][column] += projection[row][column]
            vector[row] += sum(projection[row][column] * observation.center_world_m[column] for column in range(3))
    point = _solve_3x3(matrix, vector)
    if point is None:
        return None, [], [], False
    distance_residuals, reprojection_deg, all_positive_depth = [], [], True
    for observation in window:
        offset = _sub(point, observation.center_world_m)
        range_m = _norm(offset)
        depth_m = _dot(offset, observation.direction_world)
        all_positive_depth = all_positive_depth and depth_m > 1e-6
        perpendicular = _sub(offset, _scale(observation.direction_world, depth_m))
        residual_m = _norm(perpendicular)
        distance_residuals.append(residual_m)
        reprojection_deg.append(
            math.degrees(math.asin(min(1.0, residual_m / range_m))) if range_m > 1e-6 else 90.0
        )
    return point, distance_residuals, reprojection_deg, all_positive_depth


def _view_geometry(window: list[_RayObservation]) -> tuple[float, float]:
    baseline = _sub(window[-1].center_world_m, window[0].center_world_m)
    mean_direction = _unit(tuple(sum(item.direction_world[index] for item in window) for index in range(3)))
    if mean_direction is None:
        return 0.0, 0.0
    transverse = _sub(baseline, _scale(mean_direction, _dot(baseline, mean_direction)))
    pair_angles = [
        math.degrees(math.acos(max(-1.0, min(1.0, _dot(first.direction_world, second.direction_world)))))
        for index, first in enumerate(window)
        for second in window[index + 1:]
    ]
    return _norm(transverse), max(pair_angles, default=0.0)


class StaticMotionParallaxEstimator:
    """仅在局部连续候选内累积观测的保守横移视差门。"""

    def __init__(self, parameters: ParallaxParameters | None = None) -> None:
        self.parameters = parameters or ParallaxParameters()
        self._observations: list[_RayObservation] = []
        self._track_id: str | None = None

    def reset(self) -> None:
        """离开当前区域或协同取消时清空局部证据，绝不形成持久记忆。"""
        self._observations = []
        self._track_id = None

    def observe(
        self,
        snapshot: Any,
        camera_pose: CameraPose | Mapping[str, Any] | None,
        *,
        frame_time_s: float | None = None,
    ) -> dict[str, Any]:
        """吸收一张本机快照，返回协同门可直接消费的判定证据。"""
        ray, reason = self._ray_from_snapshot(snapshot, camera_pose, frame_time_s)
        if ray is None:
            return self._evidence("rejected", reason)
        if self._track_id != ray.track_id:
            self.reset()
            self._track_id = ray.track_id
        if self._observations:
            previous = self._observations[-1]
            if ray.frame_id == previous.frame_id:
                return self._evidence("collecting", "duplicate_frame")
            gap_s = ray.time_s - previous.time_s
            if gap_s <= 0.0 or gap_s > self.parameters.max_detection_gap_s:
                self.reset()
                self._track_id = ray.track_id
        self._observations.append(ray)
        self._trim_window()
        return self._evaluate()

    def _ray_from_snapshot(
        self,
        snapshot: Any,
        camera_pose: CameraPose | Mapping[str, Any] | None,
        frame_time_s: float | None,
    ) -> tuple[_RayObservation | None, str]:
        detection = _field(snapshot, "detection")
        if detection is None or _field(detection, "class_name") != "real_vehicle":
            return None, "no_reported_target"
        source_pose = _field(snapshot, "source_pose")
        image_size = _field(snapshot, "image_size")
        bbox = _field(detection, "bbox_xyxy")
        track_id, frame_id = _field(detection, "track_id"), _field(snapshot, "frame_id")
        time_s = _finite_number(frame_time_s if frame_time_s is not None else _field(snapshot, "source_sim_time"))
        if not isinstance(source_pose, Mapping) or time_s is None or track_id is None or not isinstance(frame_id, str):
            return None, "missing_snapshot_geometry"
        if not isinstance(image_size, Sequence) or isinstance(image_size, (str, bytes)) or len(image_size) != 2:
            return None, "missing_image_size"
        width, height = _finite_number(image_size[0]), _finite_number(image_size[1])
        if width is None or height is None or width <= 1.0 or height <= 1.0:
            return None, "invalid_image_size"
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            return None, "missing_bbox"
        box = tuple(_finite_number(item) for item in bbox)
        if any(item is None for item in box):
            return None, "invalid_bbox"
        left, top, right, bottom = box  # type: ignore[misc]
        if not (0.0 <= left < right <= width and 0.0 <= top < bottom <= height):
            return None, "invalid_bbox"
        pose = self._camera_pose(camera_pose)
        if pose is None:
            return None, "missing_complete_camera_pose"
        intrinsics = self._intrinsics(source_pose, width, height)
        if intrinsics is None:
            return None, "missing_camera_intrinsics"
        fx, fy, cx, cy = intrinsics
        pixel_x, pixel_y = (left + right) * 0.5, bottom
        direction_camera = _unit(((pixel_x - cx) / fx, (pixel_y - cy) / fy, 1.0))
        direction_world = _unit(_matvec(pose.camera_to_world, direction_camera)) if direction_camera is not None else None
        if direction_world is None:
            return None, "invalid_camera_rotation"
        return _RayObservation(time_s, str(track_id), frame_id, pose.center_world_m, direction_world), "accepted"

    @staticmethod
    def _camera_pose(value: CameraPose | Mapping[str, Any] | None) -> CameraPose | None:
        if isinstance(value, CameraPose):
            center, rotation = value.center_world_m, value.camera_to_world
        elif isinstance(value, Mapping):
            center = _vector(value.get("center_world_m"))
            rotation = _matrix(value.get("camera_to_world"))
        else:
            return None
        if center is None or rotation is None:
            return None
        # 三个轴均须为单位正交轴，拒绝用 heading/pan 拼凑的不完整姿态。
        rows = rotation
        if any(abs(_norm(row) - 1.0) > 1e-3 for row in rows):
            return None
        if any(abs(_dot(rows[left], rows[right])) > 1e-3 for left in range(3) for right in range(left + 1, 3)):
            return None
        return CameraPose(center, rotation)

    @staticmethod
    def _intrinsics(source_pose: Mapping[str, Any], width: float, height: float) -> tuple[float, float, float, float] | None:
        raw = source_pose.get("camera_intrinsics")
        if isinstance(raw, Mapping):
            values = tuple(_finite_number(raw.get(key)) for key in ("fx", "fy", "cx", "cy"))
            if all(value is not None for value in values) and values[0] > 0.0 and values[1] > 0.0:
                return values  # type: ignore[return-value]
        fov_deg = _finite_number(source_pose.get("gimbal_fov_deg"))
        if fov_deg is None or not 1.0 < fov_deg < 179.0:
            return None
        focal = width / (2.0 * math.tan(math.radians(fov_deg * 0.5)))
        return focal, focal, (width - 1.0) * 0.5, (height - 1.0) * 0.5

    def _trim_window(self) -> None:
        while self._observations and self._observations[-1].time_s - self._observations[0].time_s > self.parameters.max_span_s:
            self._observations.pop(0)

    def _evaluate(self) -> dict[str, Any]:
        if len(self._observations) < self.parameters.min_rays:
            return self._evidence("collecting", "insufficient_rays")
        span_s = self._observations[-1].time_s - self._observations[0].time_s
        if span_s < self.parameters.min_span_s:
            return self._evidence("collecting", "insufficient_time_span")
        transverse_m, ray_angle_deg = _view_geometry(self._observations)
        if transverse_m < self.parameters.min_transverse_baseline_m:
            return self._evidence("collecting", "insufficient_transverse_baseline", transverse_m, ray_angle_deg)
        if transverse_m > self.parameters.max_transverse_baseline_m:
            return self._evidence("rejected", "transverse_baseline_exceeded", transverse_m, ray_angle_deg)
        if ray_angle_deg < self.parameters.min_ray_angle_deg:
            return self._evidence("rejected", "insufficient_ray_angle", transverse_m, ray_angle_deg)
        point, residuals, reprojection, positive_depth = _fit_static_point(self._observations)
        if point is None or not positive_depth:
            return self._evidence("rejected", "invalid_static_fit", transverse_m, ray_angle_deg, point=point)
        median_residual, p95_residual = _percentile(residuals, 50.0), _percentile(residuals, 95.0)
        median_reprojection, p95_reprojection = _percentile(reprojection, 50.0), _percentile(reprojection, 95.0)
        moving = bool(
            median_residual is not None and p95_residual is not None
            and median_reprojection is not None and p95_reprojection is not None
            and median_residual >= self.parameters.moving_median_residual_m
            and p95_residual >= self.parameters.moving_p95_residual_m
            and median_reprojection >= self.parameters.moving_median_reprojection_deg
            and p95_reprojection >= self.parameters.moving_p95_reprojection_deg
        )
        return self._evidence(
            "moving" if moving else "rejected",
            "moving_residuals_confirmed" if moving else "static_fit_or_insufficient_motion_evidence",
            transverse_m,
            ray_angle_deg,
            point=point,
            median_residual_m=median_residual,
            p95_residual_m=p95_residual,
            median_reprojection_deg=median_reprojection,
            p95_reprojection_deg=p95_reprojection,
        )

    def _evidence(
        self,
        decision: str,
        reason: str,
        transverse_m: float | None = None,
        ray_angle_deg: float | None = None,
        *,
        point: Vector3 | None = None,
        median_residual_m: float | None = None,
        p95_residual_m: float | None = None,
        median_reprojection_deg: float | None = None,
        p95_reprojection_deg: float | None = None,
    ) -> dict[str, Any]:
        first = self._observations[0] if self._observations else None
        last = self._observations[-1] if self._observations else None
        return {
            "schema_version": 1,
            "decision": decision,
            "allow_cooperation": decision == "moving",
            "reason": reason,
            "track_id": self._track_id,
            "start_time_s": first.time_s if first is not None else None,
            "end_time_s": last.time_s if last is not None else None,
            "span_s": last.time_s - first.time_s if first is not None and last is not None else None,
            "ray_count": len(self._observations),
            "transverse_baseline_m": transverse_m,
            "max_ray_angle_deg": ray_angle_deg,
            "fit_point_world_m": list(point) if point is not None else None,
            "median_ray_residual_m": median_residual_m,
            "p95_ray_residual_m": p95_residual_m,
            "median_reprojection_deg": median_reprojection_deg,
            "p95_reprojection_deg": p95_reprojection_deg,
            "parameters": {
                "span_s": [self.parameters.min_span_s, self.parameters.max_span_s],
                "transverse_baseline_m": [self.parameters.min_transverse_baseline_m, self.parameters.max_transverse_baseline_m],
                "min_ray_angle_deg": self.parameters.min_ray_angle_deg,
                "moving_residual_gate": [
                    self.parameters.moving_median_residual_m,
                    self.parameters.moving_p95_residual_m,
                    self.parameters.moving_median_reprojection_deg,
                    self.parameters.moving_p95_reprojection_deg,
                ],
            },
            "agent_input_boundary": (
                "local snapshot.source_pose, reported target bbox bottom-center, "
                "explicit complete camera pose, and frame time only"
            ),
        }
