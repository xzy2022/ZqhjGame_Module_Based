# 修改时间：2026-09-16。
# 修改目的：让实时研究评估保留并返回全部满足门控的双机定位结果。
# 修改内容：移除每会话每目标仅一次的锁，批量返回本帧估计并明确记录限额与正式上报边界。
# 修改时间：2026-09-16。
# 修改目的：让一次短 UE 运行可量化帧新鲜度、双机时差和姿态取样错配。
# 修改内容：为候选配对写入受记录数与字节双重限制的紧凑时间审计日志。
# 修改时间：2026-09-16。
# 修改目的：用协同阶段 Redis 真实框实时评估既有双机经纬度与高度估计算法。
# 修改内容：实现严格主从配对、三度夹角门控、紧凑限额 JSONL 与最终误差汇总。
"""实时双机定位评估核心；Redis 读取和 UE 生命周期由 runner 管理。"""
from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import fmean
import threading
import time
from typing import Any, Mapping

from PIL import Image

from .camera_metadata import derive_camera_calibration
from .paired_geolocation_triangulation import (
    CONVENTIONS,
    LocalFrame,
    estimate_pair,
    norm,
    percentile,
)


ACTIVE_PHASE = "COOP_ACTIVE"
MASTER = "MASTER"
FOLLOWER = "FOLLOWER"
FIXED_CONVENTION_NAME = "x+1_y+1_pan+1_yaw+0"
FIXED_CONVENTION = next(
    item for item in CONVENTIONS if item.name == FIXED_CONVENTION_NAME)
POSE_FIELDS = (
    "lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt",
    "gimbal_fov_deg",
)


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {key: getattr(value, key) for key in dir(value) if not key.startswith("_")}


def _session(value: Any) -> tuple[str, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return str(value[0]), int(value[1]), int(value[2])
    except (TypeError, ValueError):
        return None


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("数值不是有限数")
    return number


def _pose(value: Any) -> dict[str, float]:
    raw = _plain_mapping(value)
    return {key: _finite(raw[key]) for key in POSE_FIELDS}


def _attitude(value: Any, pose: Mapping[str, float]) -> dict[str, float | None]:
    raw = _plain_mapping(value)
    # 现有算法只读取 yaw；缺少引擎姿态时保留其既有 heading 约定，不假造 roll/pitch。
    yaw = raw.get("yaw", pose["heading_deg"])
    output = {"yaw": _finite(yaw)}
    for key in ("roll", "pitch"):
        try:
            output[key] = _finite(raw[key]) if raw.get(key) is not None else None
        except (TypeError, ValueError):
            output[key] = None
    return output


def _dimensions(redis_frame: Mapping[str, Any]) -> tuple[int, int]:
    if redis_frame.get("width") is not None and redis_frame.get("height") is not None:
        width, height = int(redis_frame["width"]), int(redis_frame["height"])
    else:
        image = redis_frame.get("image")
        if not isinstance(image, (bytes, bytearray, memoryview)):
            raise ValueError("Redis 帧缺少图像字节或显式宽高")
        with Image.open(BytesIO(bytes(image))) as decoded:
            width, height = decoded.size
    if width <= 0 or height <= 0:
        raise ValueError("图像宽高必须为正数")
    return width, height


def _detections(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        value = json.loads(bytes(value).decode("utf-8") if not isinstance(value, str) else value)
    if not isinstance(value, list):
        raise ValueError("Redis detections 必须为列表")
    output = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        target_id, target_class, bbox = item.get("target_id"), item.get("class"), item.get("bbox")
        if target_id is None or target_class is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            coordinates = [_finite(number) for number in bbox]
        except (TypeError, ValueError):
            continue
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            continue
        output.append({
            "target_id": str(target_id),
            "class": str(target_class),
            "bbox_xyxy": coordinates,
        })
    return output


def _truth(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    raw = _plain_mapping(value)
    try:
        return {key: _finite(raw[key]) for key in ("lat", "lon", "alt")}
    except (KeyError, TypeError, ValueError):
        return None


def _metric(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


class LivePairedGeolocationEvaluator:
    """累积实时帧，写出并返回所有通过三度门控的一对一帧估计。"""

    def __init__(
        self,
        output: Path | str,
        *,
        angle_threshold_deg: float = 3.0,
        max_records: int = 1000,
        max_output_bytes: int = 8 * 1024 * 1024,
        max_pair_delta_s: float = 0.1,
    ):
        self.output = Path(output)
        self.angle_threshold_deg = float(angle_threshold_deg)
        self.max_records = int(max_records)
        self.max_output_bytes = int(max_output_bytes)
        self.max_pair_delta_s = float(max_pair_delta_s)
        if not 0.0 < self.angle_threshold_deg < 90.0:
            raise ValueError("angle_threshold_deg 必须位于 0～90 度")
        if self.max_records <= 0 or self.max_output_bytes <= 0 or self.max_pair_delta_s < 0.0:
            raise ValueError("记录数、字节上限必须为正，配对时间差不能为负")
        self.output.mkdir(parents=True, exist_ok=True)
        self.predictions_path = self.output / "paired_geolocation_predictions.jsonl"
        self.timing_path = self.output / "paired_geolocation_timing.jsonl"
        self.summary_path = self.output / "paired_geolocation_summary.json"
        self._stream = self.predictions_path.open("x", encoding="utf-8", buffering=1)
        self._timing_stream = self.timing_path.open("x", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()
        self._allowed_uids: set[str] | None = None
        self._latest: dict[tuple[str, str], dict[str, Any]] = {}
        self._seen_pair_frames: set[tuple[Any, ...]] = set()
        self._used_frames: set[tuple[str, int, float]] = set()
        self._counts: Counter[str] = Counter()
        self._failure_reasons: Counter[str] = Counter()
        self._written_bytes = 0
        self._timing_written_bytes = 0
        self._errors = {name: [] for name in ("horizontal_m", "vertical_abs_m", "three_d_m")}
        self._geometry = {name: [] for name in ("convergence_angle_deg", "ray_gap_m", "baseline_horizontal_m")}
        self._closed_summary: dict[str, Any] | None = None

    def start(self, uids: list[str] | tuple[str, ...] | set[str]) -> None:
        """可选地限制本轮允许进入核心的无人机。"""
        with self._lock:
            if self._closed_summary is not None:
                raise RuntimeError("评估器已经关闭")
            self._allowed_uids = {str(uid) for uid in uids}

    def observe_frame(
        self,
        uid: str,
        redis_frame: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """接收一张原子读取的 Redis 帧，返回本次形成的全部研究估计记录。"""
        with self._lock:
            evaluator_receive_unix_s = time.time()
            evaluator_receive_monotonic_s = time.monotonic()
            if self._closed_summary is not None:
                raise RuntimeError("评估器已经关闭")
            uid = str(uid)
            self._counts["frames_observed"] += 1
            if self._allowed_uids is not None and uid not in self._allowed_uids:
                self._counts["frames_uid_not_allowed"] += 1
                return []
            try:
                active = bool(context.get("active"))
                phase = str(context.get("phase"))
                role = str(context.get("role"))
                partner_uid = str(context["partner_uid"])
                session_id = _session(context.get("session_id"))
                if not (active and phase == ACTIVE_PHASE and role in (MASTER, FOLLOWER)
                        and session_id is not None and partner_uid != uid):
                    self._counts["frames_outside_strict_cooperation"] += 1
                    return []
                self._counts["frames_strict_cooperation"] += 1
                frame_no = int(redis_frame["frame_no"])
                source_sim_time = _finite(redis_frame["source_sim_time"])
                width, height = _dimensions(redis_frame)
                pose = _pose(context["pose"])
                attitude = _attitude(context.get("aircraft_attitude"), pose)
                detections = _detections(redis_frame.get("detections", []))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                self._counts["frames_invalid"] += 1
                return []
            if not detections:
                self._counts["frames_without_valid_detection"] += 1
                return []

            truths = context.get("truth_by_target_id")
            truths = truths if isinstance(truths, Mapping) else {}
            candidate_base = {
                "uid": uid,
                "frame_no": frame_no,
                "source_sim_time": source_sim_time,
                "width": width,
                "height": height,
                "pose": pose,
                "aircraft_attitude": attitude,
                "active": active,
                "phase": phase,
                "session_id": session_id,
                "role": role,
                "partner_uid": partner_uid,
                "time_alignment": str(context.get(
                    "time_alignment", "nearest_current_state_unverified")),
                "pose_time_delta_s": self._optional_finite(
                    context.get("pose_time_delta_s")),
                "world_truth_time_delta_s": self._optional_finite(
                    context.get("world_truth_time_delta_s")),
                "pose_sample_sim_time": self._optional_finite(
                    context.get("pose_sample_sim_time")),
                "truth_sample_sim_time": self._optional_finite(
                    context.get("truth_sample_sim_time")),
                "context_published_unix_s": self._optional_finite(
                    context.get("context_published_unix_s")),
                "context_published_monotonic_s": self._optional_finite(
                    context.get("context_published_monotonic_s")),
                "redis_read_unix_s": self._optional_finite(
                    redis_frame.get("redis_read_unix_s")),
                "redis_read_monotonic_s": self._optional_finite(
                    redis_frame.get("redis_read_monotonic_s")),
                "evaluator_receive_unix_s": evaluator_receive_unix_s,
                "evaluator_receive_monotonic_s": evaluator_receive_monotonic_s,
            }
            records = []
            for detection in detections:
                candidate = dict(candidate_base, detection=detection,
                                 truth=_truth(truths.get(detection["target_id"])))
                self._latest[(uid, detection["target_id"])] = candidate
                record = self._try_pair(candidate)
                if record is not None:
                    records.append(record)
            return records

    def _try_pair(self, current: Mapping[str, Any]) -> dict | None:
        detection = current["detection"]
        other = self._latest.get((current["partner_uid"], detection["target_id"]))
        if other is None:
            self._counts["detections_waiting_for_partner"] += 1
            return None
        self._counts["candidate_pairs_seen"] += 1
        self._write_timing_record(current, other)
        if not (other["active"] and other["phase"] == ACTIVE_PHASE
                and other["session_id"] == current["session_id"]
                and other["partner_uid"] == current["uid"]
                and {other["role"], current["role"]} == {MASTER, FOLLOWER}):
            self._counts["pairs_rejected_cooperation_mismatch"] += 1
            return None
        if other["detection"]["class"] != detection["class"]:
            self._counts["pairs_rejected_class_mismatch"] += 1
            return None
        delta = abs(float(current["source_sim_time"]) - float(other["source_sim_time"]))
        if delta > self.max_pair_delta_s:
            self._counts["pairs_rejected_time_delta"] += 1
            return None
        frame_tokens = tuple(sorted((
            (str(current["uid"]), int(current["frame_no"]), float(current["source_sim_time"])),
            (str(other["uid"]), int(other["frame_no"]), float(other["source_sim_time"])),
        )))
        pair_key = (*frame_tokens, detection["target_id"])
        if pair_key in self._seen_pair_frames:
            self._counts["pairs_duplicate"] += 1
            return None
        self._seen_pair_frames.add(pair_key)
        if any(token in self._used_frames for token in frame_tokens):
            self._counts["pairs_rejected_frame_already_used"] += 1
            return None
        by_role = {current["role"]: current, other["role"]: other}
        master, follower = by_role[MASTER], by_role[FOLLOWER]
        if (master["width"], master["height"], master["pose"]["gimbal_fov_deg"]) != (
                follower["width"], follower["height"], follower["pose"]["gimbal_fov_deg"]):
            self._counts["pairs_rejected_camera_mismatch"] += 1
            return None
        self._used_frames.update(frame_tokens)
        self._counts["pairs_estimation_attempted"] += 1
        calibration = derive_camera_calibration(
            master["width"], master["height"], master["pose"]["gimbal_fov_deg"],
            "context.pose.gimbal_fov_deg",
        )
        local_frame = LocalFrame(
            master["pose"]["lat"], master["pose"]["lon"], master["pose"]["alt"])
        row = {
            "master": self._algorithm_side(master),
            "follower": self._algorithm_side(follower),
        }
        try:
            estimated = estimate_pair(
                row, calibration, local_frame, FIXED_CONVENTION,
                self.angle_threshold_deg,
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            reason = f"invalid_input:{type(exc).__name__}"
            self._failure_reasons[reason] += 1
            self._counts["pairs_estimation_failed"] += 1
            return None
        if estimated["status"] != "ok":
            reason = str(estimated.get("failure_reason") or "unknown")
            self._failure_reasons[reason] += 1
            self._counts["pairs_estimation_failed"] += 1
            return None
        angle = float(estimated["geometry"]["convergence_angle_deg"])
        if angle <= self.angle_threshold_deg:
            self._failure_reasons["convergence_not_strictly_greater"] += 1
            self._counts["pairs_estimation_failed"] += 1
            return None

        truth = master.get("truth") or follower.get("truth")
        error = self._error(estimated, truth, local_frame) if truth is not None else None
        estimate_index = self._counts["estimates_produced"] + 1
        record = self._record(
            master, follower, detection, estimated, truth, error, delta,
            estimate_index,
        )
        self._counts["estimates_produced"] += 1
        for key in self._geometry:
            self._geometry[key].append(float(estimated["geometry"][key]))
        if error is not None:
            self._counts["estimates_scored"] += 1
            for key in self._errors:
                self._errors[key].append(float(error[key]))
        else:
            self._counts["estimates_without_truth"] += 1
        self._write_record(record)
        return record

    @staticmethod
    def _optional_finite(value: Any) -> float | None:
        try:
            return _finite(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _algorithm_side(candidate: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "bbox_xyxy": candidate["detection"]["bbox_xyxy"],
            "width": candidate["width"],
            "height": candidate["height"],
            "source_pose": candidate["pose"],
            "aircraft_attitude": candidate["aircraft_attitude"],
        }

    @staticmethod
    def _difference(later: Any, earlier: Any) -> float | None:
        if later is None or earlier is None:
            return None
        return float(later) - float(earlier)

    @classmethod
    def _absolute_difference(cls, left: Any, right: Any) -> float | None:
        difference = cls._difference(left, right)
        return abs(difference) if difference is not None else None

    def _write_timing_record(self, current: Mapping[str, Any], other: Mapping[str, Any]) -> None:
        """记录配对发生时已有的时间事实；不把 Redis sim_time 宣称为曝光时刻。"""
        matched_unix_s = time.time()
        matched_monotonic_s = time.monotonic()

        def view(candidate: Mapping[str, Any]) -> dict[str, Any]:
            source = float(candidate["source_sim_time"])
            pose_time = candidate.get("pose_sample_sim_time")
            truth_time = candidate.get("truth_sample_sim_time")
            redis_mono = candidate.get("redis_read_monotonic_s")
            evaluator_mono = candidate.get("evaluator_receive_monotonic_s")
            return {
                "uid": candidate["uid"],
                "role": candidate["role"],
                "frame_no": candidate["frame_no"],
                "source_sim_time": source,
                "pose_sample_sim_time": pose_time,
                "truth_sample_sim_time": truth_time,
                "frame_age_sim_s": self._difference(pose_time, source),
                "pose_time_mismatch_s": self._difference(pose_time, source),
                "truth_time_mismatch_s": self._difference(truth_time, source),
                "redis_read_unix_s": candidate.get("redis_read_unix_s"),
                "evaluator_receive_unix_s": candidate.get("evaluator_receive_unix_s"),
                "context_published_unix_s": candidate.get("context_published_unix_s"),
                "redis_read_to_evaluator_s": self._difference(evaluator_mono, redis_mono),
                "context_to_redis_read_wall_s": self._difference(
                    candidate.get("redis_read_monotonic_s"),
                    candidate.get("context_published_monotonic_s")),
                "redis_read_to_pair_match_s": self._difference(matched_monotonic_s, redis_mono),
                "source_pose": candidate["pose"],
                "aircraft_attitude": candidate["aircraft_attitude"],
            }

        by_role = {current["role"]: current, other["role"]: other}
        if set(by_role) == {MASTER, FOLLOWER}:
            views = {"master": view(by_role[MASTER]), "follower": view(by_role[FOLLOWER])}
        else:
            views = {"current": view(current), "other": view(other)}
        record = {
            "schema_version": 1,
            "kind": "candidate_pair_timing",
            "session_id": list(current["session_id"]),
            "target_id": current["detection"]["target_id"],
            "matched_unix_s": matched_unix_s,
            "source_time_delta_s": abs(
                float(current["source_sim_time"]) - float(other["source_sim_time"])),
            "redis_read_wall_delta_s": self._absolute_difference(
                current.get("redis_read_unix_s"), other.get("redis_read_unix_s")),
            "evaluator_receive_wall_delta_s": self._absolute_difference(
                current.get("evaluator_receive_unix_s"),
                other.get("evaluator_receive_unix_s")),
            "views": views,
            "semantics": {
                "source_sim_time": "redis_renderer_sim_time_not_verified_exposure_time",
                "frame_age_sim_s": "pose_sample_sim_time_minus_source_sim_time",
                "wall_clock": "local_python_observation_only_not_camera_transport_latency",
            },
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        encoded_size = len(line.encode("utf-8"))
        if self._counts["timing_records_written"] >= self.max_records:
            self._counts["timing_records_suppressed_record_limit"] += 1
            return
        if self._timing_written_bytes + encoded_size > self.max_output_bytes:
            self._counts["timing_records_suppressed_byte_limit"] += 1
            return
        self._timing_stream.write(line)
        self._timing_written_bytes += encoded_size
        self._counts["timing_records_written"] += 1

    @staticmethod
    def _error(estimated: Mapping[str, Any], truth: Mapping[str, float],
               local_frame: LocalFrame) -> dict[str, float]:
        truth_enu = local_frame.to_enu(truth["lat"], truth["lon"], truth["alt"])
        delta = tuple(float(estimated["estimate_enu_m"][i]) - truth_enu[i] for i in range(3))
        horizontal = math.hypot(delta[0], delta[1])
        vertical = abs(delta[2])
        return {
            "horizontal_m": horizontal,
            "vertical_abs_m": vertical,
            "three_d_m": norm(delta),
        }

    def _record(self, master: Mapping[str, Any], follower: Mapping[str, Any],
                detection: Mapping[str, Any], estimated: Mapping[str, Any],
                truth: Mapping[str, float] | None, error: Mapping[str, float] | None,
                source_delta_s: float, estimate_index: int) -> dict[str, Any]:
        def view(candidate: Mapping[str, Any]) -> dict[str, Any]:
            return {
                "uid": candidate["uid"],
                "frame_no": candidate["frame_no"],
                "source_sim_time": candidate["source_sim_time"],
                "bbox_xyxy": candidate["detection"]["bbox_xyxy"],
                "source_pose": candidate["pose"],
                "aircraft_yaw_deg": candidate["aircraft_attitude"]["yaw"],
                "time_alignment": candidate["time_alignment"],
                "pose_time_delta_s": candidate["pose_time_delta_s"],
                "world_truth_time_delta_s": candidate["world_truth_time_delta_s"],
            }

        record = {
            "schema_version": 2,
            "estimate_index": estimate_index,
            "session_id": list(master["session_id"]),
            "target_id": detection["target_id"],
            "class": detection["class"],
            "views": {"master": view(master), "follower": view(follower)},
            "source_time_delta_s": source_delta_s,
            "estimate": estimated["estimate"],
            "geometry": {
                key: estimated["geometry"][key]
                for key in ("convergence_angle_deg", "ray_gap_m", "baseline_horizontal_m")
            },
        }
        if truth is not None:
            record["ground_truth"] = truth
            record["error"] = error
        return record

    def _write_record(self, record: Mapping[str, Any]) -> None:
        if self._counts["records_written"] >= self.max_records:
            self._counts["records_suppressed_record_limit"] += 1
            return
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        encoded_size = len(line.encode("utf-8"))
        if self._written_bytes + encoded_size > self.max_output_bytes:
            self._counts["records_suppressed_byte_limit"] += 1
            return
        self._stream.write(line)
        self._written_bytes += encoded_size
        self._counts["records_written"] += 1

    def close(self, status: str = "completed", error: str | None = None) -> dict[str, Any]:
        """幂等关闭；没有合格估计是有效测试结果，不伪装为运行失败。"""
        with self._lock:
            if self._closed_summary is not None:
                return self._closed_summary
            self._stream.close()
            self._timing_stream.close()
            final_status = str(status)
            if final_status == "completed" and self._counts["estimates_produced"] == 0:
                final_status = "completed_with_no_estimate"
            summary = {
                "schema_version": 2,
                "status": final_status,
                "error": str(error) if error is not None else None,
                "algorithm": {
                    "entrypoint": "paired_geolocation_triangulation.estimate_pair",
                    "convention": FIXED_CONVENTION_NAME,
                    "minimum_convergence_deg_exclusive": self.angle_threshold_deg,
                    "ground_truth_used_for_estimation": False,
                },
                "pairing": {
                    "phase": ACTIVE_PHASE,
                    "require_same_session": True,
                    "require_master_follower_reciprocal": True,
                    "identity_source": "redis_frame.detections.target_id",
                    "class_source": "redis_frame.detections.class",
                    "max_source_time_delta_s": self.max_pair_delta_s,
                    "estimate_policy": "all_qualifying_one_to_one_frame_pairs",
                },
                "reporting": {
                    "formal_report_target_emitted_by_evaluator": False,
                    "observe_frame_returns_all_new_estimates": True,
                    "runner_must_count_formal_report_target_commands_separately": True,
                },
                "timing_observability": {
                    "path": self.timing_path.name,
                    "record_scope": "candidate_pairs_before_pair_rejection",
                    "source_sim_time": "redis_renderer_sim_time_not_verified_exposure_time",
                    "frame_age_formula": "pose_sample_sim_time - source_sim_time",
                    "local_receive_clock": "time.time_at_redis_hmget_completion",
                    "matching_clock": "time.monotonic_inside_evaluator",
                    "missing_aircraft_roll_pitch": "null_not_zero_filled",
                    "camera_transport_latency_directly_measurable": False,
                },
                "limits": {
                    "cap_scope": "prediction_and_timing_streams_independently",
                    "max_records": self.max_records,
                    "max_output_bytes": self.max_output_bytes,
                    "written_bytes": self._written_bytes,
                    "estimates_produced_includes_records_suppressed_by_limits": True,
                    "timing_written_bytes": self._timing_written_bytes,
                    "prediction_stream": {
                        "max_records": self.max_records,
                        "max_output_bytes": self.max_output_bytes,
                        "written_bytes": self._written_bytes,
                    },
                    "timing_stream": {
                        "max_records": self.max_records,
                        "max_output_bytes": self.max_output_bytes,
                        "written_bytes": self._timing_written_bytes,
                    },
                },
                "counts": dict(sorted(self._counts.items())),
                "failure_reasons": dict(sorted(self._failure_reasons.items())),
                "error_metrics": {key: _metric(values) for key, values in self._errors.items()},
                "geometry_metrics": {key: _metric(values) for key, values in self._geometry.items()},
                "limitations": [
                    "redis_sim_time_is_not_verified_camera_exposure_time",
                    "camera_intrinsics_and_extrinsics_are_derived_under_unverified_assumptions",
                    "context_pose_is_nearest_current_state_without_exposure_time_interpolation",
                    "ground_truth_is_nearest_current_state_without_exposure_time_interpolation",
                ],
            }
            self.summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            self._closed_summary = summary
            return summary


__all__ = ["LivePairedGeolocationEvaluator"]
