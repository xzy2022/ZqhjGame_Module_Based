# 修改时间：2026-09-21。
# 修改目的：为 V3 真实像素实验提供可按需启用且不回流给 Agent 的图像、预测和真值审计日志。
# 修改内容：只记录完成 YOLO 推理的原图及快照，并保存裁判侧世界状态供离线作图。
"""V3 离线可视化日志。

所有真值只由 Runner 在裁判侧写入文件，既不放入 Observation，也不传给 Agent。
默认不开启本模块；启用后，各 JSONL 文件以 frame_id 关联。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _safe_json(value: Any):
    """把数据类、路径和非有限数转换为稳定的 JSON 值。"""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    if hasattr(value, "__dict__"):
        return {str(key): _safe_json(item) for key, item in vars(value).items()
                if not str(key).startswith("_")}
    return str(value)


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _pose(self_view: Any) -> dict[str, Any]:
    return {key: _safe_json(getattr(self_view, key, None)) for key in (
        "lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt", "gimbal_fov_deg")}


class V3VisualLog:
    """运行线程的轻量 JSONL 记录器；只在显式参数开启时实例化。"""

    def __init__(self, output: Path | str, *, save_images: bool, detailed_log: bool) -> None:
        self.output = Path(output)
        self.save_images, self.detailed_log = bool(save_images), bool(detailed_log)
        self._seen_frames: set[tuple[str, str]] = set()
        self._seen_predictions: set[tuple[str, str]] = set()
        self._closed = False
        self._counts = {"frame_rows": 0, "images_saved": 0, "prediction_rows": 0,
                        "truth_samples": 0, "frame_errors": 0}
        self._manifest = ((self.output / "image_manifest.jsonl").open(
            "x", encoding="utf-8", buffering=1) if self.save_images else None)
        self._frames = ((self.output / "visual_frames.jsonl").open(
            "x", encoding="utf-8", buffering=1) if self.detailed_log else None)
        self._predictions = ((self.output / "visual_predictions.jsonl").open(
            "x", encoding="utf-8", buffering=1) if self.detailed_log else None)
        self._truth = ((self.output / "visual_truth.jsonl").open(
            "x", encoding="utf-8", buffering=1) if self.detailed_log else None)

    @staticmethod
    def _write(stream, row: Mapping[str, Any]) -> None:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                allow_nan=False) + "\n")

    @staticmethod
    def _suffix(photo: bytes) -> str:
        return ".png" if photo.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg"

    def _record_frame(self, uid: str, photo: bytes, metadata: Mapping[str, Any],
                      observed_t: float | None, own_pose: Mapping[str, Any] | None) -> None:
        """把一张已完成 YOLO 推理的相机帧作为图像/审计记录写出。"""
        uid, frame_id = str(uid), hashlib.sha256(photo).hexdigest()
        frame_no = metadata.get("frame_no")
        unique = str(frame_no) if isinstance(frame_no, int) else frame_id
        if (uid, unique) in self._seen_frames:
            return
        self._seen_frames.add((uid, unique))
        stem = f"frame_{frame_no:07d}_{frame_id[:12]}" if isinstance(frame_no, int) else frame_id
        image_path = None
        try:
            if self.save_images:
                relative = Path("images") / uid / f"{stem}{self._suffix(photo)}"
                destination = self.output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(photo)
                image_path = relative.as_posix()
                self._counts["images_saved"] += 1
            row = {
                "schema_version": 1, "kind": "camera_frame", "uid": uid, "frame_id": frame_id,
                "frame_no": frame_no, "image_path": image_path, "image_bytes": len(photo),
                "observed_sim_time_s": _safe_json(observed_t),
                "source_sim_time": _safe_json(metadata.get("source_sim_time")),
                "source_time_basis": metadata.get("source_time_basis",
                    "observation_sim_time_not_verified_exposure"),
                "own_pose": _safe_json(own_pose),
                "ue_projected_objects": _safe_json(metadata.get("ue_projected_objects", [])),
                "ue_metadata_status": metadata.get("ue_metadata_status", "not_captured"),
            }
            if self._manifest is not None:
                self._write(self._manifest, row)
            if self._frames is not None:
                self._write(self._frames, row)
            self._counts["frame_rows"] += 1
        except OSError:
            self._counts["frame_errors"] += 1
            raise

    def record_processed_frame(self, uid: str, photo: bytes, metadata: Mapping[str, Any],
                               snapshot: Any) -> None:
        """将原始图片和对应的已完成 YOLO 快照原子关联到同一个 frame_id。"""
        self._record_frame(
            uid, photo, dict(metadata), _field(snapshot, "observed_sim_time"),
            _safe_json(_field(snapshot, "source_pose")),
        )
        self.record_prediction(uid, snapshot)

    def record_frame(self, uid: str, obs: Any, metadata: Mapping[str, Any] | None = None) -> None:
        """兼容手动调用；正式 Runner 只通过 record_processed_frame 记录完成帧。"""
        photo = getattr(getattr(obs, "self", None), "photo", None)
        if not isinstance(photo, bytes) or not photo:
            return
        score = getattr(getattr(obs, "briefing", None), "score_view", None)
        self._record_frame(uid, photo, dict(metadata or {}),
                           getattr(score, "sim_time", None), _pose(obs.self))

    def record_prediction(self, uid: str, snapshot: Any) -> None:
        """写出已完成的 YOLO 快照；latest-only 被覆盖帧没有预测行。"""
        if self._predictions is None or snapshot is None:
            return
        frame_id = _field(snapshot, "frame_id")
        if not frame_id or (str(uid), str(frame_id)) in self._seen_predictions:
            return
        self._seen_predictions.add((str(uid), str(frame_id)))
        row = {
            "schema_version": 1, "kind": "yolo_prediction", "uid": str(uid),
            "frame_id": str(frame_id), "source_sim_time": _safe_json(_field(snapshot, "source_sim_time")),
            "observed_sim_time": _safe_json(_field(snapshot, "observed_sim_time")),
            "source_time_basis": _field(snapshot, "source_time_basis"),
            "image_size": _safe_json(_field(snapshot, "image_size")),
            "source_pose": _safe_json(_field(snapshot, "source_pose")),
            "objects": _safe_json(_field(snapshot, "objects", ())),
            "detection": _safe_json(_field(snapshot, "detection")),
            "track_predict": _safe_json(_field(snapshot, "track_predict")),
            "closest_others": _safe_json(_field(snapshot, "closest_others")),
            "inference_wall_ms": _safe_json(_field(snapshot, "inference_wall_ms")),
            "error": _safe_json(_field(snapshot, "error")),
        }
        self._write(self._predictions, row)
        self._counts["prediction_rows"] += 1

    def record_truth(self, sim_time_s: float, world_state: Any) -> None:
        """记录当前控制拍裁判侧车辆真值，供离线按时间关联。"""
        if self._truth is None:
            return
        entities = []
        for category, mapping in (("target", getattr(world_state, "targets", {})),
                                  ("decoy", getattr(world_state, "decoys", {}))):
            for uid, entity in mapping.items():
                entities.append({"uid": str(uid), "category": category,
                                 "lat": _safe_json(getattr(entity, "lat", None)),
                                 "lon": _safe_json(getattr(entity, "lon", None)),
                                 "alt": _safe_json(getattr(entity, "alt", None)),
                                 "status": _safe_json(getattr(entity, "status", None))})
        self._write(self._truth, {"schema_version": 1, "kind": "judge_world_truth",
                                  "sim_time_s": _safe_json(sim_time_s),
                                  "basis": "judge_side_world_state_not_exposed_to_agent",
                                  "entities": entities})
        self._counts["truth_samples"] += 1

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "enabled": True, "save_images": self.save_images,
            "detailed_log": self.detailed_log,
            "image_manifest": "image_manifest.jsonl" if self._manifest else None,
            "frames_path": "visual_frames.jsonl" if self._frames else None,
            "predictions_path": "visual_predictions.jsonl" if self._predictions else None,
            "truth_path": "visual_truth.jsonl" if self._truth else None,
            "truth_boundary": "judge_side_only_not_exposed_to_agent",
            "latest_only_note": "仅已完成的 YOLO 推理帧会有 visual_predictions 行",
            **self._counts,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for stream in (self._manifest, self._frames, self._predictions, self._truth):
            if stream is not None:
                stream.close()
        (self.output / "visual_log_summary.json").write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")


__all__ = ["V3VisualLog"]
