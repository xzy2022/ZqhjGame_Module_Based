# 修改时间：2026-09-22。
# 修改目的：将 SDK 若公开的完整相机世界位姿原样绑定到同一像素快照，供横移视差估计使用。
# 修改内容：仅透传 camera_pose 或 center_world_m/camera_to_world，不由 heading 或云台角近似构造姿态。
# 修改时间：2026-09-22。
# 修改目的：使 V3 开发诊断能在 YOLO 输出转成观测前按同帧 Runner 元数据矫正检测列表。
# 修改内容：FrameJob 可携带仅显式注入的诊断元数据和变换器，并在 detector.predict 后、select_observations 前调用。
# 修改时间：2026-09-20（异步姿态绑定与初始化门控）。
# 修改目的：避免用推理返回时姿态解释旧图片，并让模型加载失败可在启动仿真前暴露。
# 修改内容：快照固化提交时相机位姿，并增加可等待 ready 或明确失败的启动接口。
# 修改时间：2026-09-20（离线回放修正）。
# 修改目的：避免尚无主目标时把普通对象误称为主目标的竞争对象。
# 修改内容：closest_others 仅在 detection 已存在时从其余零高程投影对象中选择。
# 修改时间：2026-09-20（sensor 契约适配）。
# 修改目的：确保 V3 无结果时不会静默回退原生理想感知。
# 修改内容：增加只读合法观测的提交适配，并把主目标显式转换为空列表或单个 SDK Detection。
# 修改时间：2026-09-21（静止位置投影修复）。
# 修改目的：避免车辆包围框中心的物体高度在无人机绕飞时制造虚假的 H=0 环形运动。
# 修改内容：额外把包围框底边中心投影到 H=0，作为静止判断专用的地面接触位置。
# 修改时间：2026-09-20。
# 修改目的：为 V3 提供只依赖真实相机像素的模块化在线感知结果。
# 修改内容：接入 YOLO-V2 单工作线程与每机最新帧槽，并输出目标、最近对象和最大竞争对象的可追溯像素及零高程投影信息。
"""V3 真实像素感知层。

正式推理只接收相机图片、本机相机位姿和仿真时间，不接收 UE 投影框、目标
编号、类别真值或原生 ``obs.self.detection``。一个实例应由 runner 创建并共享给
全部 UAV；后台只有一个工作线程和一个 YOLO-V2 模型，每架 UAV 只保留最新待
处理帧。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from threading import Condition, Thread
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from .visual_geometry import pixel_ray


FOV_DEG = 48.0
FOV_TOLERANCE_DEG = 0.2
MAX_PHOTO_BYTES = 16_000_000


@dataclass(frozen=True)
class PixelObservation:
    """同一真实像素帧中的一个 YOLO-V2 对象。"""

    track_id: int
    class_name: str
    detector_confidence: float
    real_probability: float
    decoy_probability: float
    real_score: float
    bbox_xyxy: tuple[float, float, float, float]
    pixel_center: tuple[float, float]
    image_size: tuple[int, int]
    track_hits: int
    motion_velocity_px_per_s: tuple[float, float]
    ground_point_h0: tuple[float, float, float] | None
    ground_contact_h0: tuple[float, float, float] | None
    ground_distance_m: float | None


@dataclass(frozen=True)
class PerceptionSnapshot:
    """一张已完成图片的完整感知结果；frame_id 可用于连续帧去重。"""

    uid: str
    frame_id: str
    source_sim_time: float
    source_time_basis: str
    observed_sim_time: float
    fov_deg: float
    image_size: tuple[int, int]
    source_pose: Mapping[str, float] | None
    detection: PixelObservation | None
    track_predict: PixelObservation | None
    closest_others: PixelObservation | None
    objects: tuple[PixelObservation, ...]
    inference_wall_ms: float
    completed_perf_counter: float
    error: str | None = None


@dataclass(frozen=True)
class _FrameJob:
    uid: str
    frame_id: str
    photo: bytes
    source_sim_time: float
    source_time_basis: str
    observed_sim_time: float
    fov_deg: float
    sequence_id: str
    own_pose: Mapping[str, float] | None
    submission_sequence: int
    diagnostic_metadata: Mapping | None


def _probabilities(record: Mapping) -> tuple[float, float]:
    probabilities = record.get("class_probabilities")
    if isinstance(probabilities, Sequence) and len(probabilities) >= 2:
        real, decoy = float(probabilities[0]), float(probabilities[1])
        total = real + decoy
        if total > 0.0 and math.isfinite(total):
            return max(0.0, real / total), max(0.0, decoy / total)
    class_name = str(record.get("class_name", "uncertain"))
    confidence = float(record.get("class_confidence", 0.0))
    if class_name == "real_vehicle":
        return confidence, max(0.0, 1.0 - confidence)
    if class_name == "model_prop":
        return max(0.0, 1.0 - confidence), confidence
    return 0.5, 0.5


def _ground_projection(
    center: tuple[float, float],
    image_size: tuple[int, int],
    own_pose: Mapping[str, float] | None,
) -> tuple[tuple[float, float, float] | None, float | None]:
    """把 bbox 中心光线投影到高度 0 平面；只使用本机相机位姿。"""
    if own_pose is None:
        return None, None
    required = (
        "lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt",
        "gimbal_fov_deg",
    )
    try:
        own = {key: float(own_pose[key]) for key in required}
    except (KeyError, TypeError, ValueError):
        return None, None
    if not all(math.isfinite(value) for value in own.values()) or own["alt"] <= 0.0:
        return None, None
    width, height = image_size
    ray, _ = pixel_ray(center[0], center[1], width, height, own)
    if ray[2] >= -1e-6:
        return None, None
    scale = own["alt"] / -ray[2]
    east, north = ray[0] * scale, ray[1] * scale
    cos_lat = math.cos(math.radians(own["lat"]))
    if abs(cos_lat) < 1e-6:
        return None, None
    lat = own["lat"] + north / 111_320.0
    lon = own["lon"] + east / (111_320.0 * cos_lat)
    distance = math.sqrt(east * east + north * north + own["alt"] * own["alt"])
    return (lat, lon, 0.0), distance


def select_observations(
    detections: Sequence[Mapping],
    image_size: tuple[int, int],
    own_pose: Mapping[str, float] | None,
) -> tuple[
    PixelObservation | None,
    PixelObservation | None,
    PixelObservation | None,
    tuple[PixelObservation, ...],
]:
    """把 YOLO-V2 输出变成 V3 选择结果，不读取任何裁判或理想检测字段。"""
    width, height = image_size
    objects = []
    for record in detections:
        raw_box = record.get("bbox_xyxy", record.get("xyxy"))
        if not isinstance(raw_box, Sequence) or len(raw_box) != 4:
            continue
        try:
            box = tuple(float(value) for value in raw_box)
        except (TypeError, ValueError):
            continue
        if (not all(math.isfinite(value) for value in box)
                or not 0.0 <= box[0] < box[2] <= width + 1e-4
                or not 0.0 <= box[1] < box[3] <= height + 1e-4):
            continue
        center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        ground_point, distance = _ground_projection(center, image_size, own_pose)
        contact_pixel = (center[0], max(box[1], box[3] - 1.0))
        ground_contact, _ = _ground_projection(
            contact_pixel, image_size, own_pose)
        real_probability, decoy_probability = _probabilities(record)
        detector_confidence = float(record.get("detector_confidence",
                                               record.get("confidence", 0.0)))
        velocity = record.get("motion_velocity_px_per_s", (0.0, 0.0))
        objects.append(PixelObservation(
            track_id=int(record.get("track_id", -1)),
            class_name=str(record.get("class_name", "uncertain")),
            detector_confidence=detector_confidence,
            real_probability=real_probability,
            decoy_probability=decoy_probability,
            real_score=detector_confidence * real_probability,
            bbox_xyxy=box,
            pixel_center=center,
            image_size=image_size,
            track_hits=int(record.get("track_hits", 1)),
            motion_velocity_px_per_s=(float(velocity[0]), float(velocity[1])),
            ground_point_h0=ground_point,
            ground_contact_h0=ground_contact,
            ground_distance_m=distance,
        ))
    ordered = tuple(sorted(objects, key=lambda item: item.real_score, reverse=True))
    real_objects = [item for item in ordered if item.class_name == "real_vehicle"]
    detection = real_objects[0] if real_objects else None
    projected = [item for item in ordered if item.ground_distance_m is not None]
    track_predict = min(projected, key=lambda item: item.ground_distance_m) if projected else None
    competitors = ([item for item in projected if item is not detection]
                   if detection is not None else [])
    closest_others = min(competitors, key=lambda item: item.ground_distance_m) if competitors else None
    return detection, track_predict, closest_others, ordered


def submit_observation(
    worker: "V3PerceptionWorker",
    obs,
    *,
    sequence_id: str = "run",
    diagnostic_metadata: Mapping | None = None,
) -> PerceptionSnapshot | None:
    """供 ``sensor()`` 调用的轻量适配；不会读取原有 detection(s)。"""
    score_view = getattr(getattr(obs, "briefing", None), "score_view", None)
    own = getattr(obs, "self", None)
    if score_view is None or own is None:
        return None
    now = float(score_view.sim_time)
    photo = own.photo
    if photo:
        source_pose = {key: getattr(own, key) for key in (
            "lat", "lon", "alt", "heading_deg", "gimbal_pan",
            "gimbal_tilt", "gimbal_fov_deg",
        )}
        complete_pose = getattr(own, "camera_pose", None)
        if isinstance(complete_pose, Mapping):
            source_pose["camera_pose"] = dict(complete_pose)
        else:
            center = getattr(own, "center_world_m", None)
            rotation = getattr(own, "camera_to_world", None)
            if center is not None and rotation is not None:
                source_pose["camera_pose"] = {
                    "center_world_m": center,
                    "camera_to_world": rotation,
                }
        worker.submit(
            own.uid,
            photo,
            observed_sim_time=now,
            # SDK 未公开相机曝光时刻；不得把此时间伪称为已验证曝光时间。
            source_sim_time=now,
            source_time_basis="observation_sim_time_not_verified_exposure",
            fov_deg=float(own.gimbal_fov_deg),
            sequence_id=sequence_id,
            own_pose=source_pose,
            diagnostic_metadata=diagnostic_metadata,
        )
    return worker.latest(own.uid, now_sim_time=now, max_age_s=1.5)


def snapshot_to_sensor_detections(snapshot: PerceptionSnapshot | None) -> list:
    """把主目标转为 SDK sensor 列表；空结果返回 ``[]``，绝不回退理想感知。"""
    error = (snapshot.get("error") if isinstance(snapshot, Mapping)
             else getattr(snapshot, "error", None))
    item = (snapshot.get("detection") if isinstance(snapshot, Mapping)
            else getattr(snapshot, "detection", None))
    item = item if snapshot is not None and error is None else None
    point = (item.get("ground_point_h0") if isinstance(item, Mapping)
             else getattr(item, "ground_point_h0", None))
    if item is None or point is None:
        return []
    real_score = (item.get("real_score", 0.0) if isinstance(item, Mapping)
                  else getattr(item, "real_score", 0.0))
    from competition.sdk.core.observation import Detection
    return [Detection(
        detected=True,
        confidence=max(0.0, min(1.0, float(real_score))),
        target_lat=point[0],
        target_lon=point[1],
        target_type="ground_vehicle",
    )]


class V3PerceptionWorker:
    """共享单模型的非阻塞最新帧 worker。

    ``submit`` 只复制合法像素和元数据；若该 UAV 的旧帧尚未开始，旧帧会被
    覆盖。``latest`` 只读取已完成快照，不等待 GPU。
    """

    def __init__(
        self,
        *,
        detector_factory: Callable[[], object] | None = None,
        detector_kwargs: Mapping | None = None,
        diagnostic_transform: Callable | None = None,
    ) -> None:
        self._detector_factory = detector_factory
        self._detector_kwargs = dict(detector_kwargs or {})
        self._diagnostic_transform = diagnostic_transform
        self._condition = Condition()
        self._pending: dict[str, _FrameJob] = {}
        self._latest: dict[str, PerceptionSnapshot] = {}
        self._last_signature: dict[str, tuple[str, str]] = {}
        self._uid_order: list[str] = []
        self._cursor = 0
        self._submission_sequence = 0
        self._stopping = False
        self._ready = False
        self._worker_error: str | None = None
        self._stats = Counter()
        self._thread = Thread(target=self._run, name="v3-perception", daemon=True)
        self._thread.start()

    @property
    def ready(self) -> bool:
        with self._condition:
            return self._ready

    @property
    def stats(self) -> dict[str, int | str | bool | None]:
        with self._condition:
            return {**dict(self._stats), "ready": self._ready,
                    "worker_error": self._worker_error,
                    "pending_uavs": len(self._pending)}

    def submit(
        self,
        uid: str,
        photo: bytes,
        *,
        observed_sim_time: float,
        source_sim_time: float | None = None,
        source_time_basis: str | None = None,
        fov_deg: float = FOV_DEG,
        sequence_id: str = "run",
        own_pose: Mapping[str, float] | None = None,
        diagnostic_metadata: Mapping | None = None,
    ) -> str | None:
        """提交真实图片；返回像素哈希 frame_id，不等待推理。"""
        if not isinstance(photo, bytes) or not photo or len(photo) > MAX_PHOTO_BYTES:
            with self._condition:
                self._stats["rejected_photo"] += 1
            return None
        observed_sim_time = float(observed_sim_time)
        source_sim_time = (observed_sim_time if source_sim_time is None
                           else float(source_sim_time))
        if not (math.isfinite(observed_sim_time) and math.isfinite(source_sim_time)):
            with self._condition:
                self._stats["rejected_time"] += 1
            return None
        if abs(float(fov_deg) - FOV_DEG) > FOV_TOLERANCE_DEG:
            with self._condition:
                self._stats["rejected_non_fov48"] += 1
            return None
        uid, sequence_id = str(uid), str(sequence_id)
        frame_id = hashlib.sha256(photo).hexdigest()
        signature = (sequence_id, frame_id)
        with self._condition:
            if self._stopping:
                return None
            if self._last_signature.get(uid) == signature:
                self._stats["duplicate_frames"] += 1
                return frame_id
            self._last_signature[uid] = signature
            self._submission_sequence += 1
            if uid in self._pending:
                self._stats["replaced_frames"] += 1
            if uid not in self._uid_order:
                self._uid_order.append(uid)
            self._pending[uid] = _FrameJob(
                uid=uid,
                frame_id=frame_id,
                photo=photo,
                source_sim_time=source_sim_time,
                source_time_basis=(source_time_basis or
                                   "observation_sim_time_not_verified_exposure"),
                observed_sim_time=observed_sim_time,
                fov_deg=float(fov_deg),
                sequence_id=sequence_id,
                own_pose=dict(own_pose) if own_pose is not None else None,
                submission_sequence=self._submission_sequence,
                diagnostic_metadata=(
                    dict(diagnostic_metadata)
                    if diagnostic_metadata is not None else None
                ),
            )
            self._stats["submitted_frames"] += 1
            self._condition.notify()
        return frame_id

    def latest(
        self,
        uid: str,
        *,
        now_sim_time: float | None = None,
        max_age_s: float | None = None,
    ) -> PerceptionSnapshot | None:
        """返回最近完成结果；可按源仿真时间限制陈旧结果。"""
        with self._condition:
            result = self._latest.get(str(uid))
        if (result is not None and now_sim_time is not None and max_age_s is not None
                and float(now_sim_time) - result.source_sim_time > float(max_age_s)):
            return None
        return result

    def close(self, timeout_s: float = 10.0) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, float(timeout_s)))

    def wait_until_ready(self, timeout_s: float = 120.0) -> None:
        """在启动仿真前等待模型加载/预热，失败或超时立即终止本轮。"""
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        with self._condition:
            while not self._ready and self._worker_error is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise TimeoutError("V3 感知模型初始化超时")
                self._condition.wait(timeout=remaining)
            if self._worker_error is not None:
                raise RuntimeError(f"V3 感知模型初始化失败：{self._worker_error}")

    def _next_job(self) -> _FrameJob | None:
        if not self._pending or not self._uid_order:
            return None
        count = len(self._uid_order)
        for offset in range(count):
            index = (self._cursor + offset) % count
            uid = self._uid_order[index]
            if uid in self._pending:
                self._cursor = (index + 1) % count
                return self._pending.pop(uid)
        return None

    def _create_detector(self):
        if self._detector_factory is not None:
            return self._detector_factory()
        from .vehicle_prop_v2 import create_detector
        return create_detector(**self._detector_kwargs)

    def _run(self) -> None:
        try:
            detector = self._create_detector()
            with self._condition:
                self._ready = True
                self._condition.notify_all()
        except Exception as exc:  # 初始化失败后保持 sensor 非阻塞并明确暴露状态。
            with self._condition:
                self._worker_error = repr(exc)
                self._stats["initialization_failures"] += 1
                self._condition.notify_all()
            return
        while True:
            with self._condition:
                while not self._stopping and not self._pending:
                    self._condition.wait()
                if self._stopping:
                    return
                job = self._next_job()
            if job is None:
                continue
            result = self._infer(detector, job)
            with self._condition:
                self._latest[job.uid] = result
                self._stats["completed_frames"] += 1
                if result.error is not None:
                    self._stats["inference_failures"] += 1

    def _infer(self, detector, job: _FrameJob) -> PerceptionSnapshot:
        started = time.perf_counter()
        image_size = (0, 0)
        try:
            import cv2
            encoded = np.frombuffer(job.photo, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法解码相机图片")
            height, width = image.shape[:2]
            image_size = (int(width), int(height))
            detections = detector.predict(
                image,
                timestamp=job.source_sim_time,
                sequence_id=job.sequence_id,
                stream_id=job.uid,
            )
            if self._diagnostic_transform is not None:
                detections = self._diagnostic_transform(
                    job.uid, job.frame_id, detections, image_size,
                    job.diagnostic_metadata,
                )
            detection, track_predict, closest_others, objects = select_observations(
                detections, image_size, job.own_pose)
            error = None
        except Exception as exc:
            detection = track_predict = closest_others = None
            objects = ()
            error = repr(exc)
        completed = time.perf_counter()
        return PerceptionSnapshot(
            uid=job.uid,
            frame_id=job.frame_id,
            source_sim_time=job.source_sim_time,
            source_time_basis=job.source_time_basis,
            observed_sim_time=job.observed_sim_time,
            fov_deg=job.fov_deg,
            image_size=image_size,
            source_pose=(dict(job.own_pose) if job.own_pose is not None else None),
            detection=detection,
            track_predict=track_predict,
            closest_others=closest_others,
            objects=objects,
            inference_wall_ms=(completed - started) * 1000.0,
            completed_perf_counter=completed,
            error=error,
        )


__all__ = [
    "FOV_DEG",
    "PerceptionSnapshot",
    "PixelObservation",
    "V3PerceptionWorker",
    "select_observations",
    "snapshot_to_sensor_detections",
    "submit_observation",
]
