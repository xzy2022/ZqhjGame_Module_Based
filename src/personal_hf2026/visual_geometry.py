# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13
# 修改目的：在不读取车辆高度和身份的条件下关联图像与坐标轨迹。
# 修改内容：实现照片时刻插值、无高度方向匹配和一对一歧义拒绝。
"""V2 旁路几何；采用水平 FOV、机头相对 pan 和零相机 roll 假设。"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PixelBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    width: int
    height: int
    category: str = "vehicle_candidate"
    class_margin: float = 0.0

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


def angle_delta(a, b):
    return (a - b + 180) % 360 - 180


def aligned_sample(history, t, max_gap=0.35):
    """只在历史覆盖的时间内插值；候选必须在两端均存在。"""
    for row in history:
        if abs(row["t"] - t) < 1e-5:
            return row, "exact"
    for a, b in zip(history, history[1:]):
        if not a["t"] <= t <= b["t"]:
            continue
        dt = b["t"] - a["t"]
        if dt > max_gap or dt <= 0:
            return None, "pose_gap"
        if abs(a["own"]["gimbal_fov_deg"] - b["own"]["gimbal_fov_deg"]) > 0.2:
            return None, "fov_transition"
        ratio = (t - a["t"]) / dt
        own = {}
        for key, value in a["own"].items():
            delta = b["own"][key] - value
            if key in ("heading_deg", "gimbal_pan"):
                delta = angle_delta(b["own"][key], value)
            own[key] = value + ratio * delta
        common = a["tracks"].keys() & b["tracks"].keys()
        tracks = {key: [a["tracks"][key][j] + ratio *
                       (b["tracks"][key][j] - a["tracks"][key][j]) for j in (0, 1)]
                  for key in common}
        rate = max(abs(angle_delta(b["own"][k], a["own"][k])) / dt
                   for k in ("heading_deg", "gimbal_pan", "gimbal_tilt"))
        return {"t": t, "own": own, "tracks": tracks, "angular_rate_dps": rate}, "interpolated"
    return None, "time_not_bracketed"


def pixel_ray(u, v, width, height, own):
    focal = width / (2 * math.tan(math.radians(own["gimbal_fov_deg"] / 2)))
    x, y = (u - (width - 1) / 2) / focal, (v - (height - 1) / 2) / focal
    yaw = math.radians(own["heading_deg"] + own["gimbal_pan"])
    pitch = math.radians(own["gimbal_tilt"])
    forward = (math.cos(pitch) * math.sin(yaw), math.cos(pitch) * math.cos(yaw), math.sin(pitch))
    right = (math.cos(yaw), -math.sin(yaw), 0)
    down = (math.sin(pitch) * math.sin(yaw), math.sin(pitch) * math.cos(yaw), -math.cos(pitch))
    return tuple(forward[i] + x * right[i] + y * down[i] for i in range(3)), focal


def bind_boxes(boxes, sample):
    """类别不参与几何选择；一框多轨或多框同轨都拒绝。"""
    own, tracks = sample["own"], sample["tracks"]
    results = []
    for index, box in enumerate(boxes):
        ray, focal = pixel_ray(*box.center, box.width, box.height, own)
        horizontal = math.hypot(ray[0], ray[1])
        row = {"box_index": index, "track_id": None, "status": "unmatched", "candidates": []}
        results.append(row)
        if sample.get("angular_rate_dps", 0) > 20:
            row["status"] = "fast_pose"
            continue
        if horizontal * focal < 12 or ray[2] >= 0:
            row["status"] = "near_nadir_or_upward"
            continue
        bearing = math.degrees(math.atan2(ray[0], ray[1]))
        radius = math.hypot(box.x2 - box.x1, box.y2 - box.y1) / 2
        tolerance = min(15.0, 2.0 + math.degrees(math.atan2(radius, horizontal * focal)))
        row.update(bearing_deg=bearing, tolerance_deg=tolerance)
        for track_id, (lat, lon) in tracks.items():
            north = (lat - own["lat"]) * 111320
            east = (lon - own["lon"]) * 111320 * math.cos(math.radians(own["lat"]))
            if math.hypot(north, east) < 15:
                continue
            error = abs(angle_delta(bearing, math.degrees(math.atan2(east, north))))
            if error <= tolerance:
                row["candidates"].append({"track_id": track_id, "error_deg": error})
        if len(row["candidates"]) == 1:
            row["track_id"] = row["candidates"][0]["track_id"]
            row["status"] = "bound"
        elif len(row["candidates"]) > 1:
            row["status"] = "ambiguous"
    counts = {}
    for row in results:
        if row["status"] == "bound":
            counts[row["track_id"]] = counts.get(row["track_id"], 0) + 1
    for row in results:
        if row["status"] == "bound" and counts[row["track_id"]] > 1:
            row.update(status="multiple_boxes", track_id=None)
    return results
