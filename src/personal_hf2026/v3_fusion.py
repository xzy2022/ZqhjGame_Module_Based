# 修改时间：2026-09-20（乱序收件保护）。
# 修改目的：避免同拍首次看到多条 F3 消息时旧测量覆盖发送者的新测量。
# 修改内容：每个发送者和会话只保留 source_sim_time 不回退的最新包。
# 修改时间：2026-09-20。
# 修改目的：为 V3 提供只依赖合法本机观测和机间通信的单机投影及双机定位能力。
# 修改内容：实现零高度投影、50 字节逻辑单播协议、收件去重和成功三角化后的上报候选。
"""V3 像素几何与通信融合；不读取 UE 投影框、目标真值或中央跨机状态。"""
from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections import deque
import math
from collections.abc import Mapping
import struct
import zlib


EARTH_RADIUS_M = 6_378_137.0
PROTOCOL_PREFIX = "F3"
MAX_PAYLOAD_BYTES = 50
# 34 字节正文加 2 字节校验，经 URL-safe base64 编码后为 48 字节，加前缀恰为 50 字节。
_PACKET_BODY = struct.Struct("!HIBHiiHHhhHHHHB")
_PACKET = struct.Struct("!HIBHiiHHhhHHHHBH")


def _field(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数")
    return number


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second))


def _add(first, second):
    return tuple(a + b for a, b in zip(first, second))


def _subtract(first, second):
    return tuple(a - b for a, b in zip(first, second))


def _scale(value, factor):
    return tuple(item * factor for item in value)


def _norm(value):
    return math.sqrt(_dot(value, value))


def _normalized(value):
    length = _norm(value)
    if length <= 0.0:
        raise ValueError("零长度射线")
    return _scale(value, 1.0 / length)


class LocalFrame:
    """小范围 WGS84 经纬度与局部 ENU 的近似转换。"""

    def __init__(self, latitude_deg, longitude_deg, altitude_m=0.0):
        self.latitude_deg = _finite(latitude_deg, "latitude_deg")
        self.longitude_deg = _finite(longitude_deg, "longitude_deg")
        self.altitude_m = _finite(altitude_m, "altitude_m")
        self.latitude_scale = math.pi * EARTH_RADIUS_M / 180.0
        self.longitude_scale = self.latitude_scale * math.cos(
            math.radians(self.latitude_deg))
        if abs(self.longitude_scale) <= 1e-9:
            raise ValueError("极区不支持局部经纬度近似")

    def to_enu(self, position):
        return (
            (_finite(_field(position, "lon"), "lon") - self.longitude_deg)
            * self.longitude_scale,
            (_finite(_field(position, "lat"), "lat") - self.latitude_deg)
            * self.latitude_scale,
            _finite(_field(position, "alt"), "alt") - self.altitude_m,
        )

    def to_geodetic(self, value):
        return {
            "lat": self.latitude_deg + value[1] / self.latitude_scale,
            "lon": self.longitude_deg + value[0] / self.longitude_scale,
            "alt": self.altitude_m + value[2],
        }


def _pose(pose, *, fallback_fov=None):
    fov = _field(pose, "gimbal_fov_deg", _field(pose, "fov_deg", fallback_fov))
    return {
        "lat": _finite(_field(pose, "lat"), "lat"),
        "lon": _finite(_field(pose, "lon"), "lon"),
        "alt": _finite(_field(pose, "alt"), "alt"),
        "heading_deg": _finite(_field(pose, "heading_deg"), "heading_deg"),
        "gimbal_pan": _finite(_field(pose, "gimbal_pan"), "gimbal_pan"),
        "gimbal_tilt": _finite(_field(pose, "gimbal_tilt"), "gimbal_tilt"),
        "gimbal_fov_deg": _finite(fov, "gimbal_fov_deg"),
    }


def _image_size(snapshot, observation):
    size = _field(observation, "image_size", _field(snapshot, "image_size"))
    if size is not None:
        width, height = size
    else:
        width = _field(observation, "image_width", _field(snapshot, "image_width"))
        height = _field(observation, "image_height", _field(snapshot, "image_height"))
    width, height = int(width), int(height)
    if not 1 <= width <= 65_535 or not 1 <= height <= 65_535:
        raise ValueError("图像宽高必须位于 1..65535")
    return width, height


def _pixel_center(observation):
    center = _field(observation, "pixel_center")
    if center is not None:
        return _finite(center[0], "pixel_x"), _finite(center[1], "pixel_y")
    bbox = _field(observation, "bbox_xyxy")
    if bbox is None or len(bbox) != 4:
        raise ValueError("像素观测缺少 pixel_center 或 bbox_xyxy")
    x1, y1, x2, y2 = (_finite(value, "bbox") for value in bbox)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def measurement_from_snapshot(
    snapshot,
    pose,
    *,
    observation=None,
    observation_key="detection",
    require_real=True,
):
    """把 ``PerceptionSnapshot`` 或普通字典收敛为几何测量字典。"""
    selected = observation if observation is not None else _field(snapshot, observation_key)
    if selected is None:
        return None
    class_name = str(_field(selected, "class_name", ""))
    if require_real and class_name != "real_vehicle":
        return None
    width, height = _image_size(snapshot, selected)
    pixel_x, pixel_y = _pixel_center(selected)
    if not 0.0 <= pixel_x <= width or not 0.0 <= pixel_y <= height:
        raise ValueError("像素中心位于图像范围外")
    fov = _field(snapshot, "fov_deg", _field(selected, "fov_deg"))
    source_time = _finite(
        _field(snapshot, "source_sim_time", _field(snapshot, "observed_sim_time")),
        "source_sim_time",
    )
    quality = _field(selected, "real_score", _field(selected, "detector_confidence", 0.0))
    quality = max(0.0, min(1.0, _finite(quality, "quality")))
    return {
        "uid": str(_field(snapshot, "uid", "")),
        "frame_id": str(_field(snapshot, "frame_id", "")),
        "source_sim_time": source_time,
        "observed_sim_time": _finite(
            _field(snapshot, "observed_sim_time", source_time), "observed_sim_time"),
        "source_pose": _pose(pose, fallback_fov=fov),
        "pixel_center": (pixel_x, pixel_y),
        "width": width,
        "height": height,
        "quality": quality,
        "track_id": _field(selected, "track_id"),
        "class_name": class_name,
    }


def _pixel_ray(measurement):
    """按 heading+pan 和云台 tilt 构造射线，不访问机身姿态字段。"""
    pose = measurement["source_pose"]
    pixel_x, pixel_y = measurement["pixel_center"]
    width, height = float(measurement["width"]), float(measurement["height"])
    horizontal_fov_deg = _finite(pose["gimbal_fov_deg"], "gimbal_fov_deg")
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("水平视场角必须位于 0..180 度")
    focal_pixels = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    normalized_center = measurement.get("normalized_center")
    if normalized_center is None:
        image_x = (float(pixel_x) - (width - 1.0) / 2.0) / focal_pixels
        image_y = (float(pixel_y) - (height - 1.0) / 2.0) / focal_pixels
    else:
        # 协议只传归一化中心和宽高比，足以恢复针孔射线且节省校验码空间。
        image_x = (float(normalized_center[0]) - 0.5) * width / focal_pixels
        image_y = (float(normalized_center[1]) - 0.5) * height / focal_pixels
    azimuth = math.radians(float(pose["heading_deg"]) + float(pose["gimbal_pan"]))
    tilt = math.radians(float(pose["gimbal_tilt"]))
    forward = (
        math.cos(tilt) * math.sin(azimuth),
        math.cos(tilt) * math.cos(azimuth),
        math.sin(tilt),
    )
    right = (math.cos(azimuth), -math.sin(azimuth), 0.0)
    down = (
        math.sin(tilt) * math.sin(azimuth),
        math.sin(tilt) * math.cos(azimuth),
        -math.cos(tilt),
    )
    return _normalized(_add(_add(forward, _scale(right, image_x)),
                            _scale(down, image_y)))


def project_to_ground(measurement, *, ground_alt_m=0.0):
    """把单机像素射线投影到指定高度平面，供 V1 轨迹/控制逻辑使用。"""
    pose = measurement["source_pose"]
    ray = _pixel_ray(measurement)
    ground_alt_m = _finite(ground_alt_m, "ground_alt_m")
    if abs(ray[2]) <= 1e-9:
        return {"status": "failed", "estimate": None,
                "failure_reason": "ray_parallel_to_ground"}
    distance = (ground_alt_m - float(pose["alt"])) / ray[2]
    if distance <= 0.0:
        return {"status": "failed", "estimate": None,
                "failure_reason": "ground_behind_camera"}
    frame = LocalFrame(pose["lat"], pose["lon"], pose["alt"])
    estimate = frame.to_geodetic(_scale(ray, distance))
    estimate["alt"] = ground_alt_m
    return {"status": "ok", "estimate": estimate, "failure_reason": None,
            "slant_distance_m": distance}


def estimate_pair(first, second, *, minimum_ray_angle_deg=3.0, max_pair_delta_s=0.25):
    """先由两条最近射线估高，再在该高度平面分别投影并平均经纬度。"""
    time_delta = abs(float(first["source_sim_time"]) - float(second["source_sim_time"]))
    if time_delta > max_pair_delta_s + 1e-9:
        return {"status": "failed", "estimate": None,
                "failure_reason": "pair_time_delta_too_large",
                "geometry": {"pair_time_delta_s": time_delta}}
    first_pose, second_pose = first["source_pose"], second["source_pose"]
    frame = LocalFrame(
        (float(first_pose["lat"]) + float(second_pose["lat"])) / 2.0,
        (float(first_pose["lon"]) + float(second_pose["lon"])) / 2.0,
    )
    first_origin, second_origin = frame.to_enu(first_pose), frame.to_enu(second_pose)
    first_ray, second_ray = _pixel_ray(first), _pixel_ray(second)
    offset = _subtract(first_origin, second_origin)
    ray_dot = _dot(first_ray, second_ray)
    denominator = 1.0 - ray_dot * ray_dot
    ray_angle = math.degrees(math.acos(max(-1.0, min(1.0, abs(ray_dot)))))
    geometry = {
        "pair_time_delta_s": time_delta,
        "ray_angle_deg": ray_angle,
        "baseline_m": _norm(offset),
        "baseline_horizontal_m": math.hypot(offset[0], offset[1]),
    }
    if denominator <= 1e-12:
        return {"status": "failed", "estimate": None,
                "failure_reason": "parallel_rays", "geometry": geometry}
    first_distance = (
        ray_dot * _dot(second_ray, offset) - _dot(first_ray, offset)
    ) / denominator
    second_distance = (
        _dot(second_ray, offset) - ray_dot * _dot(first_ray, offset)
    ) / denominator
    if first_distance <= 0.0 or second_distance <= 0.0:
        return {"status": "failed", "estimate": None,
                "failure_reason": "intersection_behind_camera", "geometry": geometry}
    if ray_angle < minimum_ray_angle_deg:
        return {"status": "failed", "estimate": None,
                "failure_reason": "ray_angle_too_small", "geometry": geometry}
    first_closest = _add(first_origin, _scale(first_ray, first_distance))
    second_closest = _add(second_origin, _scale(second_ray, second_distance))
    height = (first_closest[2] + second_closest[2]) / 2.0
    plane_points = []
    for origin, ray in ((first_origin, first_ray), (second_origin, second_ray)):
        if abs(ray[2]) <= 1e-9:
            return {"status": "failed", "estimate": None,
                    "failure_reason": "ray_parallel_to_height_plane", "geometry": geometry}
        distance = (height - origin[2]) / ray[2]
        if distance <= 0.0:
            return {"status": "failed", "estimate": None,
                    "failure_reason": "height_plane_behind_camera", "geometry": geometry}
        plane_points.append(_add(origin, _scale(ray, distance)))
    point = _scale(_add(plane_points[0], plane_points[1]), 0.5)
    point = (point[0], point[1], height)
    ray_separation = _norm(_subtract(first_closest, second_closest))
    plane_disagreement = math.hypot(
        plane_points[0][0] - plane_points[1][0],
        plane_points[0][1] - plane_points[1][1],
    )
    geometry.update({
        "distance_first_m": first_distance,
        "distance_second_m": second_distance,
        "ray_separation_m": ray_separation,
        "height_plane_disagreement_m": plane_disagreement,
    })
    # 该值只是排序启发式，不是概率或校准置信度。
    parallax_quality = min(1.0, ray_angle / 10.0)
    separation_quality = 1.0 / (1.0 + ray_separation / 5.0)
    time_quality = max(0.0, 1.0 - time_delta / max(max_pair_delta_s, 1e-9))
    input_quality = min(float(first.get("quality", 0.0)), float(second.get("quality", 0.0)))
    quality = max(0.0, min(1.0, input_quality * parallax_quality
                           * separation_quality * (0.5 + 0.5 * time_quality)))
    return {"status": "ok", "estimate": frame.to_geodetic(point),
            "failure_reason": None, "geometry": geometry, "quality": quality}


def session_token(session):
    """把协调器会话压缩为 32 位令牌；原始会话仍由本机控制层保留。"""
    if isinstance(session, (tuple, list)):
        value = "\x1f".join(str(item) for item in session)
    else:
        value = str(session)
    return zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF


def _uid_number(uid):
    number = int(str(uid))
    if not 0 <= number <= 65_535:
        raise ValueError("逻辑单播 UID 必须是 0..65535 的数字")
    return number


def _centidegrees(value, *, signed):
    number = round(float(value) * 100.0)
    low, high = ((-32_768, 32_767) if signed else (0, 65_535))
    if not low <= number <= high:
        raise ValueError("角度超出协议量化范围")
    return number


def encode_measurement(destination_uid, session, sequence, measurement):
    """编码固定 50 UTF-8 字节广播载荷，接收机按完整 UID 做逻辑单播过滤。"""
    pose = measurement["source_pose"]
    width, height = int(measurement["width"]), int(measurement["height"])
    pixel_x, pixel_y = measurement["pixel_center"]
    if not 0.0 <= pixel_x <= width or not 0.0 <= pixel_y <= height:
        raise ValueError("像素中心位于图像范围外")
    x_quantized = round(pixel_x / width * 65_535.0)
    y_quantized = round(pixel_y / height * 65_535.0)
    aspect_quantized = round(height / width * 10_000.0)
    values = (
        _uid_number(destination_uid),
        session_token(session),
        int(sequence) & 0xFF,
        round(float(measurement["source_sim_time"]) * 100.0),
        round(float(pose["lat"]) * 1_000_000.0),
        round(float(pose["lon"]) * 1_000_000.0),
        round(float(pose["alt"]) * 10.0),
        _centidegrees(float(pose["heading_deg"]) % 360.0, signed=False),
        _centidegrees(pose["gimbal_pan"], signed=True),
        _centidegrees(pose["gimbal_tilt"], signed=True),
        _centidegrees(pose["gimbal_fov_deg"], signed=False),
        aspect_quantized,
        x_quantized,
        y_quantized,
        round(max(0.0, min(1.0, float(measurement.get("quality", 0.0)))) * 255.0),
    )
    try:
        body = _PACKET_BODY.pack(*values)
        packed = body + struct.pack("!H", zlib.crc32(body) & 0xFFFF)
    except struct.error as exc:
        raise ValueError(f"测量超出通信协议量化范围: {exc}") from exc
    payload = PROTOCOL_PREFIX + urlsafe_b64encode(packed).decode("ascii").rstrip("=")
    if len(payload.encode("utf-8")) != MAX_PAYLOAD_BYTES:
        raise AssertionError("V3 融合协议必须恰为 50 UTF-8 字节")
    return payload


def decode_measurement(payload):
    """解码 V3 载荷；发送者 UID 只信任 SDK ``Message.sender_uid``。"""
    if not isinstance(payload, str) or not payload.startswith(PROTOCOL_PREFIX):
        return None
    if len(payload.encode("utf-8")) != MAX_PAYLOAD_BYTES:
        return None
    try:
        packed = urlsafe_b64decode(payload[len(PROTOCOL_PREFIX):] + "===")
        values = _PACKET.unpack(packed)
    except (ValueError, struct.error):
        return None
    (destination, token, sequence, source_centiseconds, lat, lon, alt,
     heading, pan, tilt, fov, aspect, pixel_x, pixel_y, quality, checksum) = values
    if zlib.crc32(packed[:-2]) & 0xFFFF != checksum:
        return None
    if not (0 < aspect <= 65_535 and 0 < fov < 18_000
            and -90_000_000 <= lat <= 90_000_000
            and -180_000_000 <= lon <= 180_000_000):
        return None
    width, height = 10_000, aspect
    normalized_center = (pixel_x / 65_535.0, pixel_y / 65_535.0)
    return {
        "destination_uid": str(destination),
        "session_token": token,
        "sequence": sequence,
        "source_sim_time": source_centiseconds / 100.0,
        "source_pose": {
            "lat": lat / 1_000_000.0,
            "lon": lon / 1_000_000.0,
            "alt": alt / 10.0,
            "heading_deg": heading / 100.0,
            "gimbal_pan": pan / 100.0,
            "gimbal_tilt": tilt / 100.0,
            "gimbal_fov_deg": fov / 100.0,
        },
        "pixel_center": (normalized_center[0] * width,
                         normalized_center[1] * height),
        "normalized_center": normalized_center,
        "image_aspect_ratio": aspect / 10_000.0,
        "width": width,
        "height": height,
        "quality": quality / 255.0,
    }


class V3Fusion:
    """管理 ACTIVE 期融合槽、广播收件去重和 MASTER 上报候选。"""

    def __init__(
        self,
        self_uid,
        *,
        send_period_s=0.5,
        max_pair_delta_s=0.25,
        minimum_ray_angle_deg=3.0,
        max_seen_messages=512,
    ):
        if send_period_s < 0.25:
            raise ValueError("发送间隔不得小于 0.25 秒")
        self.self_uid = str(self_uid)
        self.send_period_s = float(send_period_s)
        self.max_pair_delta_s = float(max_pair_delta_s)
        self.minimum_ray_angle_deg = float(minimum_ray_angle_deg)
        self.max_seen_messages = int(max_seen_messages)
        self.reset()

    def reset(self):
        self._sequence = 0
        self._last_send_time = -math.inf
        self._seen = set()
        self._seen_order = deque()
        self._latest = {}

    def prepare_broadcast(
        self,
        now,
        destination_uid,
        session,
        measurement,
        *,
        shared_slot_granted=False,
    ):
        """只在控制层授予共享通信槽时生成载荷；替换 ACTIVE heartbeat，保留 operational。"""
        now = _finite(now, "now")
        if not shared_slot_granted or now - self._last_send_time < self.send_period_s - 1e-9:
            return None
        payload = encode_measurement(destination_uid, session, self._sequence, measurement)
        self._sequence = (self._sequence + 1) & 0xFF
        self._last_send_time = now
        return payload

    def ingest(self, comm_inbox, now, *, session=None):
        """过滤自回环、非目标 UID、旧会话和重复 inbox，只保留每个发送者最新测量。"""
        now = _finite(now, "now")
        expected_token = None if session is None else session_token(session)
        accepted = []
        for message in comm_inbox:
            sender_uid = str(_field(message, "sender_uid", ""))
            if sender_uid == self.self_uid:
                continue
            decoded = decode_measurement(_field(message, "payload"))
            if decoded is None or decoded["destination_uid"] != self.self_uid:
                continue
            if expected_token is not None and decoded["session_token"] != expected_token:
                continue
            key = (sender_uid, decoded["session_token"], decoded["sequence"])
            if key in self._seen:
                continue
            self._seen.add(key)
            self._seen_order.append(key)
            while len(self._seen_order) > self.max_seen_messages:
                self._seen.discard(self._seen_order.popleft())
            decoded["sender_uid"] = sender_uid
            decoded["received_sim_time"] = now
            latest_key = (sender_uid, decoded["session_token"])
            previous = self._latest.get(latest_key)
            if (previous is None
                    or decoded["source_sim_time"] >= previous["source_sim_time"]):
                self._latest[latest_key] = decoded
            accepted.append(decoded)
        return accepted

    def report_candidate(self, local_measurement, session, *, role, peer_uid=None):
        """仅 MASTER 在双机估高成功后获得 report_target 候选，绝不回退到零高度结果。"""
        if str(role).upper() != "MASTER":
            return None
        token = session_token(session)
        candidates = [
            value for (sender, candidate_token), value in self._latest.items()
            if candidate_token == token and (peer_uid is None or sender == str(peer_uid))
        ]
        if not candidates:
            return None
        peer = max(candidates, key=lambda value: value["source_sim_time"])
        result = estimate_pair(
            local_measurement,
            peer,
            minimum_ray_angle_deg=self.minimum_ray_angle_deg,
            max_pair_delta_s=self.max_pair_delta_s,
        )
        if result["status"] != "ok" or result["estimate"].get("alt") is None:
            return None
        estimate = result["estimate"]
        return {
            "lat": estimate["lat"],
            "lon": estimate["lon"],
            "alt": estimate["alt"],
            "quality": result["quality"],
            "session": session,
            "peer_uid": peer["sender_uid"],
            "source_sim_time": max(
                float(local_measurement["source_sim_time"]),
                float(peer["source_sim_time"]),
            ),
            "geometry": result["geometry"],
        }


__all__ = [
    "MAX_PAYLOAD_BYTES",
    "PROTOCOL_PREFIX",
    "LocalFrame",
    "V3Fusion",
    "decode_measurement",
    "encode_measurement",
    "estimate_pair",
    "measurement_from_snapshot",
    "project_to_ground",
    "session_token",
]
