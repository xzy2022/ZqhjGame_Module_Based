# 修改时间：2026-09-27。
# 修改目的：防止提交版感知异常触发官方默认识别器并引入 Runner 真值。
# 修改内容：入口感知捕获异常后明确返回空检测，保持像素输入边界。
# 修改时间：2026-09-27。
# 修改目的：把当前 V4 控制模块整理为赛题二的单文件提交入口。
# 修改内容：按固定顺序合并所需定义，并接入仅使用本机照片的推理适配层。
"""生成 v9aget 单文件提交版；运行时不需要本脚本。"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/personal_hf2026"
DEST = ROOT / "submission/v9aget_赛题二/v9aget.py"
MODULES = (
    "visual_geometry", "search_route", "survey_search", "coordinated_search",
    "search_gimbal", "v3_simple_control", "local_motion_flow", "v4_entity",
    "v4_coordination", "v4_gimbal", "v4_motion", "v4_position",
    "v4_flight", "v4_control",
)

HEADER = '''# 修改时间：2026-09-27。
# 修改目的：防止感知异常触发官方默认识别器并引入 Runner 真值。
# 修改内容：入口感知捕获异常后明确返回空检测，保持像素输入边界。
# 修改时间：2026-09-27。
# 修改目的：基于 V4 生成符合赛题二提交规范的独立单文件 Agent。
# 修改内容：整合控制与推理逻辑，只消费本机照片、姿态、通信和允许的仿真时间。
"""v9aget：赛题二正式提交入口，三架无人机各实例独立控制。"""
from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from threading import Condition, Lock, Thread
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from competition.sdk.core.commands import broadcast, fly_to, point_gimbal, report_target, set_gimbal_fov
from competition.sdk.core.observation import Detection
from competition.sdk.scenarios.coop_decoy import CoopAgent


# 任务区域来自官方赛题二基线中的静态地图边界，不读取场景或航线文件。
_BBOX = ((26.982, 124.980), (27.025, 125.020))


def _haversine_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
'''

PROJECTION = '''
def _ground_projection(center, image_size, own_pose):
    """只按本机姿态把像素射线投到高度零的粗平面。"""
    if own_pose is None:
        return None, None
    keys = ("lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt", "gimbal_fov_deg")
    try:
        own = {key: float(own_pose[key]) for key in keys}
    except (KeyError, TypeError, ValueError):
        return None, None
    if not all(math.isfinite(value) for value in own.values()) or own["alt"] <= 0:
        return None, None
    ray, _ = pixel_ray(center[0], center[1], image_size[0], image_size[1], own)
    if ray[2] >= -1e-6:
        return None, None
    scale = own["alt"] / -ray[2]
    east, north = ray[0] * scale, ray[1] * scale
    cos_lat = math.cos(math.radians(own["lat"]))
    if abs(cos_lat) < 1e-6:
        return None, None
    lat = own["lat"] + north / 111320.0
    lon = own["lon"] + east / (111320.0 * cos_lat)
    distance = math.sqrt(east * east + north * north + own["alt"] * own["alt"])
    return (lat, lon, 0.0), distance
'''

RUNTIME = '''
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
    image_size: tuple[int, int]
    source_pose: Mapping[str, float]
    raw_yolo_objects: tuple[V4Object, ...]
    effective_yolo_objects: tuple[V4Object, ...]
    image_bgr: np.ndarray | None = None
    error: str | None = None


def _letterbox(image, shape):
    height, width = image.shape[:2]
    out_h, out_w = shape
    gain = min(out_h / height, out_w / width)
    new_w, new_h = round(width * gain), round(height * gain)
    left, top = (out_w - new_w) // 2, (out_h - new_h) // 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, top, out_h - new_h - top,
                                left, out_w - new_w - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
    return rgb, (gain, left, top)


def _decode(raw, original_shape, geometry):
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 6:
        raise ValueError("模型输出不是双类别原始预测")
    anchors = raw[0].transpose(0, 1).float()
    scores = anchors[:, 4:6].amax(1)
    keep = (scores >= 0.05) & torch.isfinite(anchors).all(1)
    anchors, scores = anchors[keep], scores[keep]
    if not len(anchors):
        return np.empty((0, 7), dtype=np.float32)
    boxes = torch.cat((anchors[:, :2] - anchors[:, 2:4] / 2,
                       anchors[:, :2] + anchors[:, 2:4] / 2), dim=1)
    selected = nms(boxes, scores, 0.5)[:100]
    boxes = boxes[selected]
    probs = anchors[selected, 4:6]
    probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-9)
    gain, left, top = geometry
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_shape[0])
    output = torch.cat((boxes, scores[selected, None], probs), dim=1)
    output = output[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    return output.cpu().numpy()


class _Detector:
    def __init__(self):
        from ultralytics import YOLO
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(str(Path(__file__).resolve().with_name("yolo.pt"))).model
        if self.model.names != {0: "real_vehicle", 1: "model_prop"}:
            raise ValueError("模型类别与 V4 不一致")
        self.model.end2end = False
        self.model = self.model.to(self.device).float().eval().fuse(verbose=False)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model.to(dtype=self.dtype)

    @torch.inference_mode()
    def predict(self, image):
        rgb, geometry = _letterbox(image, (1152, 1536))
        tensor = torch.from_numpy(rgb).unsqueeze(0).to(self.device, dtype=self.dtype) / 255
        return _decode(self.model(tensor)[0], image.shape[:2], geometry)


class _VisionWorker:
    """三架 Agent 共用模型；每架只保留尚未处理的最新照片。"""
    def __init__(self):
        self.condition = Condition()
        self.pending = {}
        self.completed = {}
        self.last_submitted = {}
        self.failure = None
        Thread(target=self._run, daemon=True, name="v9aget-vision").start()

    def submit(self, key, uid, photo, now, pose):
        if not isinstance(photo, bytes) or not photo:
            return
        frame_id = hashlib.sha256(photo).hexdigest()
        with self.condition:
            if self.last_submitted.get(key) == frame_id:
                return
            self.last_submitted[key] = frame_id
            self.pending[key] = (uid, frame_id, photo, now, pose)
            self.condition.notify()

    def latest(self, key, now):
        with self.condition:
            snapshot = self.completed.get(key)
        if snapshot is None or not 0 <= now - snapshot.source_sim_time <= 1.5:
            return None
        return snapshot

    def _run(self):
        try:
            detector = _Detector()
        except Exception as exc:
            with self.condition:
                self.failure = repr(exc)
            return
        while True:
            with self.condition:
                while not self.pending:
                    self.condition.wait()
                key = next(iter(self.pending))
                uid, frame_id, photo, now, pose = self.pending.pop(key)
            try:
                image = cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("无法解码照片")
                rows = detector.predict(image)
                height, width = image.shape[:2]
                objects = []
                for row in rows:
                    box = tuple(float(value) for value in row[:4])
                    real, decoy = float(row[5]), float(row[6])
                    if not (0 <= box[0] < box[2] <= width and
                            0 <= box[1] < box[3] <= height):
                        continue
                    class_name = ("real_vehicle" if real >= decoy else "model_prop") if max(
                        real, decoy) >= 0.6 else "uncertain"
                    objects.append(V4Object(box, float(row[4]), real, decoy, class_name))
                found = tuple(objects)
                snapshot = V4Snapshot(uid, frame_id, now, (width, height), pose,
                                      found, found, image)
            except Exception as exc:
                snapshot = V4Snapshot(uid, frame_id, now, (0, 0), pose,
                                      (), (), error=repr(exc))
            with self.condition:
                self.completed[key] = snapshot


_worker_lock = Lock()
_worker = None


def _vision_worker():
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _VisionWorker()
        return _worker


class V9aget(CoopAgent):
    """v9aget：V4 三机协同识别正式提交版。"""
    def __init__(self, my_uid):
        super().__init__(my_uid)
        self.reset()

    def reset(self):
        self.control = V4Control(self.my_uid)
        self._vision_key = object()
        self._last_snapshot = None
        self._consumed_frame_id = None
        self._sensor_error = None
        self._t = 0.0

    def sensor(self, obs, dt):
        try:
            return self._sensor_pixels(obs, dt)
        except Exception as exc:
            # 官方 resolver 在异常时会走默认识别器；必须明确返回空检测。
            self._sensor_error = repr(exc)
            return []

    def _sensor_pixels(self, obs, dt):
        score = getattr(getattr(obs, "briefing", None), "score_view", None)
        now = float(score.sim_time) if score is not None else self._t + max(0.0, float(dt))
        photo = obs.self.photo
        if photo:
            worker = _vision_worker()
            pose = {key: float(getattr(obs.self, key)) for key in (
                "lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt",
                "gimbal_fov_deg")}
            worker.submit(self._vision_key, str(self.my_uid), photo, now, pose)
        else:
            worker = _worker
        if worker is None:
            return []
        if worker.failure is not None:
            raise RuntimeError(f"视觉模型初始化失败：{worker.failure}")
        snapshot = worker.latest(self._vision_key, now)
        if snapshot is None:
            return []
        self._last_snapshot = snapshot
        if snapshot.error:
            return []
        real = [item for item in snapshot.effective_yolo_objects
                if item.class_name == "real_vehicle"]
        if not real:
            return []
        width, height = snapshot.image_size
        item = min(real, key=lambda obj: ((obj.bbox_xyxy[0] + obj.bbox_xyxy[2] - width) ** 2
                                         + (obj.bbox_xyxy[1] + obj.bbox_xyxy[3] - height) ** 2))
        box = item.bbox_xyxy
        point, _ = _ground_projection(((box[0] + box[2]) * 0.5,
                                       (box[1] + box[3]) * 0.5),
                                      snapshot.image_size, snapshot.source_pose)
        if point is None:
            return []
        return [Detection(detected=True,
                          confidence=item.detector_confidence * item.real_probability,
                          target_lat=point[0], target_lon=point[1],
                          target_type="ground_vehicle")]

    def decide(self, obs, dt):
        score = getattr(getattr(obs, "briefing", None), "score_view", None)
        now = float(score.sim_time) if score is not None else self._t + max(0.0, float(dt))
        self._t = now
        snapshot = self._last_snapshot
        if snapshot is not None and snapshot.frame_id != self._consumed_frame_id:
            self._consumed_frame_id = snapshot.frame_id
            self.control.consume_visual(snapshot)
        commands = self.control.step(obs, now)
        self.control.pop_events()
        return commands
'''


def source_definitions(name):
    path = SOURCE / f"{name}.py"
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    parts = [f"\n\n# ===== V4 来源：{name}.py =====\n"]
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets):
            continue
        decorators = getattr(node, "decorator_list", ())
        start = min([node.lineno, *(item.lineno for item in decorators)]) - 1
        piece = "".join(lines[start:node.end_lineno])
        # 已合并的几何函数在全局可见，删除原模块函数内的相对导入。
        if name == "v4_coordination":
            piece = piece.replace("        from .v3_simple_control import ground_distance_m\n", "")
        parts.append(piece.rstrip() + "\n\n")
    return "".join(parts)


def main():
    sections = [HEADER]
    for name in MODULES:
        if name == "v4_position":
            sections.append(PROJECTION)
        sections.append(source_definitions(name))
    sections.append(RUNTIME)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(sections).rstrip() + "\n"
    ast.parse(content)
    DEST.write_text(content, encoding="utf-8")
    print(f"已生成 {DEST}，共 {len(content.splitlines())} 行")


if __name__ == "__main__":
    main()
