# 修改时间：2026-09-16。
# 修改目的：为采集帧补充同一 sim:state 时刻的完整机体与云台姿态。
# 修改内容：原子记录位置、三轴姿态和云台 pan/tilt/FOV，并保留旧姿态字段兼容性。
# 修改时间：2026-09-16。
# 修改目的：为高分辨率 GT-crop 采集补充飞机姿态与相机针孔参数元数据。
# 修改内容：提供世界状态姿态提取、逐帧字段组装和单轮相机标定文件写入接口。
"""采集元数据辅助模块。

本模块只记录 runner 已持有的世界状态真值，并按显式假设推导针孔参数；它不把
姿态或相机参数传给参赛 Agent，也不把推导值称为经过标定的完整内参。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading
from typing import Any, Mapping


ATTITUDE_SOURCE = "world_state.entities[uid].raw.platform.attitude"
POSITION_SOURCE = "world_state.entities[uid].raw.platform.position"
GIMBAL_SOURCE = "world_state.entities[uid].raw.gimbal_tracking"


def _entity_raw(entity: Any) -> Mapping[str, Any]:
    """兼容 EntityTruth 和原始实体字典，返回含 platform 的原始结构。"""
    if isinstance(entity, Mapping):
        raw = entity.get("raw", entity)
    else:
        raw = getattr(entity, "raw", {})
    return raw if isinstance(raw, Mapping) else {}


def _number(mapping: Mapping[str, Any], key: str) -> float | None:
    """读取原始数值；缺字段时保留 None，不采用 SDK 的零值回退。"""
    value = mapping.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _status(values: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    return ("recorded_from_engine_state" if all(values[key] is not None for key in keys)
            else "missing_schema_fields")


def aircraft_position(entity: Any) -> dict[str, Any]:
    """直接从 sim:state 的 platform.position 读取经纬高。"""
    raw = _entity_raw(entity)
    platform = raw.get("platform", {})
    platform = platform if isinstance(platform, Mapping) else {}
    position = platform.get("position", {})
    position = position if isinstance(position, Mapping) else {}
    values = {
        "lat": _number(position, "latitude"),
        "lon": _number(position, "longitude"),
        "alt": _number(position, "altitude"),
    }
    values.update(
        horizontal_unit="degree",
        altitude_unit="meter",
        source=POSITION_SOURCE,
        value_status=_status(values, ("lat", "lon", "alt")),
        altitude_reference_status="unverified",
    )
    return values


def aircraft_attitude(entity: Any) -> dict[str, Any]:
    """从官方状态 schema 对应路径提取 roll/pitch/yaw，不用零值掩盖缺失。"""
    raw = _entity_raw(entity)
    platform = raw.get("platform", {})
    platform = platform if isinstance(platform, Mapping) else {}
    attitude = platform.get("attitude", {})
    attitude = attitude if isinstance(attitude, Mapping) else {}
    values = {key: _number(attitude, key) for key in ("roll", "pitch", "yaw")}
    values.update(
        unit="degree",
        source=ATTITUDE_SOURCE,
        value_status=_status(values, ("roll", "pitch", "yaw")),
        coordinate_convention_status="unverified",
    )
    return values


def gimbal_state(entity: Any) -> dict[str, Any]:
    """直接从 sim:state 的 gimbal_tracking 读取云台角和 FOV。"""
    raw = _entity_raw(entity)
    gimbal = raw.get("gimbal_tracking", {})
    gimbal = gimbal if isinstance(gimbal, Mapping) else {}
    fov_key = "fov" if gimbal.get("fov") is not None else "fov_deg"
    values = {
        "pan": _number(gimbal, "pan_angle"),
        "tilt": _number(gimbal, "tilt_angle"),
        "fov": _number(gimbal, fov_key),
    }
    values.update(
        unit="degree",
        source=GIMBAL_SOURCE,
        source_fields={"pan": "pan_angle", "tilt": "tilt_angle", "fov": fov_key},
        value_status=_status(values, ("pan", "tilt", "fov")),
        coordinate_convention_status="unverified",
    )
    return values


def capture_pose(entity: Any) -> dict[str, Any]:
    """生成同一个世界状态 tick 内的机体位置、姿态与云台状态。"""
    position = aircraft_position(entity)
    attitude = aircraft_attitude(entity)
    gimbal = gimbal_state(entity)
    groups = (position, attitude, gimbal)
    return {
        "aircraft_position": position,
        "aircraft_attitude": attitude,
        "gimbal_state": gimbal,
        "source": "sim:state",
        "value_status": ("recorded_from_engine_state" if all(
            group["value_status"] == "recorded_from_engine_state" for group in groups)
            else "missing_schema_fields"),
        "camera_extrinsics_status": "unknown",
    }


def world_state_pose_sample(world_state: Any, uid: str) -> dict[str, Any]:
    """生成完整姿态样本，供照片源时间离线对齐。"""
    entities = getattr(world_state, "entities", {})
    entity = entities.get(uid) if isinstance(entities, Mapping) else None
    if entity is None:
        raise KeyError(f"world_state 中不存在实体 {uid}")
    pose = capture_pose(entity)
    timestamp = getattr(world_state, "timestamp", None)
    return {
        "uid": str(uid),
        "sim_time": float(getattr(world_state, "sim_time")),
        "state_timestamp": float(timestamp) if timestamp is not None else None,
        "capture_pose": pose,
        "aircraft_attitude": pose["aircraft_attitude"],
    }


def world_state_attitude_sample(world_state: Any, uid: str) -> dict[str, Any]:
    """兼容旧调用名；返回值已包含完整 capture_pose。"""
    return world_state_pose_sample(world_state, uid)


def add_frame_attitude(frame_row: Mapping[str, Any], attitude: Mapping[str, Any],
                       alignment: str) -> dict[str, Any]:
    """把已经按照片源时间对齐的姿态加入 samples.jsonl 行。"""
    row = dict(frame_row)
    row["aircraft_attitude"] = dict(attitude)
    row["aircraft_attitude_alignment"] = str(alignment)
    return row


def add_frame_capture_pose(frame_row: Mapping[str, Any], state_row: Mapping[str, Any],
                           alignment: str) -> dict[str, Any]:
    """把同一原始状态样本的完整姿态与兼容字段加入 samples.jsonl。"""
    row = add_frame_attitude(
        frame_row, state_row["aircraft_attitude"], alignment)
    row["capture_pose"] = dict(state_row["capture_pose"])
    row["capture_pose_alignment"] = str(alignment)
    row["capture_pose_state_sim_time"] = float(state_row["sim_time"])
    row["capture_pose_state_timestamp"] = state_row.get("state_timestamp")
    return row


def derive_camera_calibration(width: int, height: int, fov_deg: float,
                              fov_source: str) -> dict[str, Any]:
    """按“FOV 为水平视场角、方形像素、主点居中”假设推导针孔参数。"""
    width = int(width)
    height = int(height)
    fov_deg = float(fov_deg)
    if width <= 0 or height <= 0 or not 0.0 < fov_deg < 180.0:
        raise ValueError("图像宽高必须为正数，FOV 必须位于 0～180 度")

    focal_px = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    vertical_fov_deg = math.degrees(2.0 * math.atan(
        (height / width) * math.tan(math.radians(fov_deg / 2.0))))
    return {
        "schema_version": 1,
        "model": "pinhole",
        "image": {"width_px": width, "height_px": height,
                  "dimensions_source": "decoded_captured_frame"},
        "fov": {
            "value_deg": fov_deg,
            "value_source": str(fov_source),
            "axis_assumption": "horizontal",
            "axis_verification_status": "unverified",
            "derived_vertical_fov_deg": vertical_fov_deg,
        },
        "intrinsics": {
            "fx_px": focal_px,
            "fy_px": focal_px,
            "cx_px": (width - 1.0) / 2.0,
            "cy_px": (height - 1.0) / 2.0,
            "matrix": [
                [focal_px, 0.0, (width - 1.0) / 2.0],
                [0.0, focal_px, (height - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            "status": "derived_under_unverified_assumptions",
        },
        "derivation": {
            "fx": "(width_px / 2) / tan(horizontal_fov_deg / 2)",
            "fy": "fx (square-pixel assumption)",
            "cx": "(width_px - 1) / 2 (pixel-center coordinates)",
            "cy": "(height_px - 1) / 2 (pixel-center coordinates)",
        },
        "assumptions": {
            "square_pixels": True,
            "principal_point_at_image_center": True,
            "pixel_coordinate_origin": "top_left_pixel_center",
        },
        "verification": {
            "complete_intrinsics": False,
            "ue_fov_axis": "unverified_assumed_horizontal",
            "principal_point": "unverified_assumed_image_center",
            "pixel_aspect_ratio": "unverified_assumed_square",
            "lens_distortion": "unknown",
            "camera_extrinsics": "unknown",
            "camera_to_gimbal_transform": "unknown",
            "gimbal_to_aircraft_transform": "unknown",
        },
        "distortion": {"model": None, "coefficients": None, "status": "unknown"},
        "extrinsics": {"rotation": None, "translation": None, "status": "unknown"},
    }


class CameraCalibrationWriter:
    """在首张真实图像到达后写一次 camera_calibration.json。"""

    def __init__(self, run_output: Path | str, fov_deg: float, fov_source: str):
        self.path = Path(run_output) / "camera_calibration.json"
        self.fov_deg = float(fov_deg)
        self.fov_source = str(fov_source)
        self._lock = threading.Lock()
        self._calibration: dict[str, Any] | None = None

    def observe_frame(self, width: int, height: int) -> dict[str, Any]:
        """用解码后的实际宽高建档；后续帧只核对本轮相机参数未改变。"""
        candidate = derive_camera_calibration(
            width, height, self.fov_deg, self.fov_source)
        signature = (candidate["image"]["width_px"], candidate["image"]["height_px"],
                     candidate["fov"]["value_deg"])
        with self._lock:
            if self._calibration is None and self.path.is_file():
                self._calibration = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if self._calibration is not None:
                existing = (self._calibration["image"]["width_px"],
                            self._calibration["image"]["height_px"],
                            self._calibration["fov"]["value_deg"])
                if existing != signature:
                    raise RuntimeError(f"本轮图像尺寸或 FOV 发生变化：{existing} -> {signature}")
                return self._calibration
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8") as stream:
                json.dump(candidate, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            self._calibration = candidate
            return candidate
