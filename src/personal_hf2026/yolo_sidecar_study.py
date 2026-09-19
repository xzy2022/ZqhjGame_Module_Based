# 修改时间：2026-09-19。
# 修改目的：在保持同一 v3 模型的前提下验证输入方向敏感性的候选缓解方案。
# 修改内容：新增默认关闭的单视图旋转参数并记录实际方向，不增加模型推理次数。
# 修改时间：2026-09-19。
# 修改目的：区分单帧模型偏置和时序融合带来的类别变化。
# 修改内容：旁路结果保留融合前后类别概率及低分恢复标记。
# 修改时间：2026-09-19。
# 修改目的：保留在线错分诊断所需的真实推理图像以支持同帧重放。
# 修改内容：新增可选的已处理帧原始字节保存及图像路径和保存耗时记录。
# 修改时间：2026-09-18。
# 修改目的：让在线 YOLO 旁路可通过 CLI 对比原始 PT、无翻转 PT 与无翻转 TensorRT FP16 三档。
# 修改内容：新增 profile 解析、TensorRT engine 覆盖、高置信阈值相对提高 20% 及逐层资源身份日志。
# 修改时间：2026-09-18（run_official 集成复测）
# 修改目的：让 Windows spawn 在入口以 runpy 的 __main__ 名称执行时仍能导入旁路目标。
# 修改内容：创建子进程时固定引用规范包模块中的 worker 函数，避免 __main__ 无法反序列化。
# 修改时间：2026-09-18
# 修改目的：在保持 PersonalV1 理想控制不变的前提下评估三机共享的 YOLO 视觉旁路。
# 修改内容：新增 Windows spawn 单进程最新帧槽、帧级真值与时延审计以及可汇总的专用 Runner。
"""PersonalV1 控制加单进程 YOLO 旁路的在线审计入口。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import threading
import time

import redis

from competition.sdk.core.perception import DetectionResolver
from competition.sdk.core.runner import ScenarioConfig

from .control_test_runner import MultiTargetIdealDetector
from .dropout_capture import StudyRenderer
from .paths import PROJECT_ROOT, RUNTIME_ROOT, SCENARIO_ROOT
from .personal_v1 import PersonalV1Agent
from .timing_study import StudyRunner


WEATHERS = (
    "Clear_Skies",
    "Partly_Cloudy",
    "Rain",
    "Foggy",
    "Snow_Light",
    "Sand_Dust_Calm",
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/detectors/vehicle_prop/vehicle_frontier.json"
DEFAULT_LAYOUT = PROJECT_ROOT / "configs/scenarios/coop_decoy/static-decoys.json"
DEFAULT_TRT_ENGINE = Path(
    "D:/Workspace/00_MyRepo/red_m_competiton/output/personal_v2/"
    "yolo-offline-accel/trt-fp16-build-20260918-214857-153/"
    "yolo26s_two_class.engine"
)
YOLO_PROFILES = ("v1", "v2", "v3")
V3_TRACKER_HIGH_MULTIPLIER = 1.2
SLOT_CAPACITY_BYTES = 32 * 1024 * 1024
SLOT_HEADER = struct.Struct("<I")


def _json_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _normalise_ground_truth(raw_boxes):
    """保留 Redis 相机帧自带的 UE 投影框，不把它送入 Agent 或模型。"""
    output = []
    for raw in raw_boxes or ():
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            bbox = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in bbox):
            continue
        output.append({
            "target_id": str(raw.get("target_id")),
            "class": str(raw.get("class", "")),
            "bbox": bbox,
        })
    return output


class AuditPhotoCache:
    """轮询三机最新相机帧，并把图像与同一 Redis hash 中的投影框绑定。"""

    def __init__(self, uids, output, host, port):
        self.uids = tuple(uids)
        self.redis = redis.Redis(host=host, port=port, socket_timeout=2)
        self.latest = {}
        self.delivered = {}
        self.stop_event = threading.Event()
        self.error_count = 0
        self.closed = False
        self.baseline = {uid: self._read(uid) for uid in self.uids}
        self.activated = set()
        self.audit = (Path(output) / "camera_frames.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _read(self, uid):
        keys = list(self.redis.scan_iter(match=f"sync_camera:{uid}:frame:*", count=100))
        if not keys:
            return None
        key = max(keys, key=lambda item: int(item.rsplit(b":", 1)[1]))
        image, source, boxes = self.redis.hmget(key, "image", "sim_time", "detections")
        if not image or source is None:
            return None
        return {
            "uid": str(uid),
            "frame_no": int(key.rsplit(b":", 1)[1]),
            "source_sim_time": float(source),
            "image": image,
            "ground_truth": _normalise_ground_truth(
                json.loads(boxes) if boxes else []
            ),
            "received_unix_s": time.time(),
        }

    def start(self):
        self.thread.start()

    def _loop(self):
        last_logged = {}
        while not self.stop_event.is_set():
            for uid in self.uids:
                try:
                    packet = self._read(uid)
                    if packet is None:
                        continue
                    signature = (packet["frame_no"], packet["source_sim_time"])
                    baseline = self.baseline[uid]
                    if uid not in self.activated:
                        if baseline and signature == (
                            baseline["frame_no"], baseline["source_sim_time"]
                        ):
                            continue
                        self.activated.add(uid)
                    self.latest[uid] = packet
                    if signature != last_logged.get(uid):
                        _json_line(self.audit, {
                            key: value for key, value in packet.items() if key != "image"
                        })
                        last_logged[uid] = signature
                except Exception as exc:  # noqa: BLE001
                    self.error_count += 1
                    _json_line(self.audit, {"uid": str(uid), "error": repr(exc)})
            self.stop_event.wait(0.05)

    def get(self, uid):
        packet = self.latest.get(uid)
        self.delivered[uid] = packet
        return packet["image"] if packet else None

    def stop(self):
        if self.closed:
            return
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.error_count += 1
            return
        self.audit.close()
        self.redis.close()
        self.closed = True


def _serialise_predictions(raw_predictions):
    """只跨进程返回评估所需字段，避免把模型内部对象写入日志。"""
    output = []
    for raw in raw_predictions:
        output.append({
            "bbox_xyxy": [float(value) for value in raw["bbox_xyxy"]],
            "class_id": int(raw["class_id"]),
            "class_name": str(raw["class_name"]),
            "score": float(raw["score"]),
            "class_confidence": float(raw.get("class_confidence", 0.0)),
            "detector_confidence": float(raw.get("detector_confidence", raw["score"])),
            "track_id": int(raw.get("track_id", 0)),
            "track_hits": int(raw.get("track_hits", 0)),
            "single_frame_probabilities": [
                float(value) for value in raw.get("single_frame_probabilities", [])
            ],
            "class_probabilities": [
                float(value) for value in raw.get("class_probabilities", [])
            ],
            "recovered_low_score": bool(raw.get("recovered_low_score", False)),
        })
    return output


def _worker_main(
    slots,
    slot_lengths,
    slot_sequences,
    slot_lock,
    slot_ready,
    stop_event,
    result_pipe,
    config,
    device,
    resources,
):
    """spawn 子进程：一次只推理共享槽中尚未开始的最新一帧。"""
    try:
        from .vehicle_prop import create_detector
        from .vehicle_prop.temporal_tracker import CameraMotion, TemporalTracker

        started = time.perf_counter()
        effective = resources["effective_options"]
        detector = create_detector(
            config=config,
            device=device,
            profile=resources["profile"],
            weights=resources["weights_path"],
            weights_sha256=resources["weights_sha256"],
            flip=effective["flip_enabled"],
            tracker_high=effective["tracker_high_confidence_threshold"],
            image_rotation_deg=effective.get("image_rotation_deg", 0),
        )
        stream_states = {}
        frame_output = resources.get("processed_frames_dir")
        if frame_output:
            frame_output = Path(frame_output)
            frame_output.mkdir(parents=True, exist_ok=True)
        result_pipe.send({
            "event": "worker_ready",
            "initialization_ms": (time.perf_counter() - started) * 1000.0,
            "runtime": detector.runtime_metadata(),
            "pid": os.getpid(),
        })
        last_sequences = [0] * len(slots)
        next_slot = 0
        while not stop_event.is_set():
            if not slot_ready.wait(timeout=0.1):
                continue
            with slot_lock:
                pending = {
                    index for index, sequence in enumerate(slot_sequences)
                    if int(sequence.value) != last_sequences[index]
                }
                if not pending:
                    slot_ready.clear()
                    continue
                selected = next(
                    (index for offset in range(len(slots))
                     if (index := (next_slot + offset) % len(slots)) in pending)
                )
                sequence = int(slot_sequences[selected].value)
                payload = bytes(slots[selected][: int(slot_lengths[selected].value)])
                last_sequences[selected] = sequence
                next_slot = (selected + 1) % len(slots)
                if all(
                    int(value.value) == last_sequences[index]
                    for index, value in enumerate(slot_sequences)
                ):
                    slot_ready.clear()
            if stop_event.is_set():
                break
            metadata_length = SLOT_HEADER.unpack_from(payload)[0]
            metadata_start = SLOT_HEADER.size
            metadata_end = metadata_start + metadata_length
            metadata = json.loads(payload[metadata_start:metadata_end].decode("utf-8"))
            image_bytes = payload[metadata_end:]
            result_pipe.send({
                "event": "started",
                "submission_id": sequence,
                "started_perf_counter": time.perf_counter(),
            })

            import cv2
            import numpy as np

            decode_started = time.perf_counter()
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("相机帧无法解码")
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            inference_started = time.perf_counter()
            uid = str(metadata["uid"])
            if uid not in stream_states:
                stream_states[uid] = {
                    "camera": CameraMotion(),
                    "tracker": TemporalTracker(**detector.pipeline.config["tracker"]),
                    "previous_t": None,
                    "sequence": uid,
                }
            state = stream_states[uid]
            source_time = float(metadata["source_sim_time"])
            previous_source_time = state["previous_t"]
            source_gap_s = (
                source_time - previous_source_time
                if previous_source_time is not None else None
            )
            tracker_reset_reason = None
            if source_gap_s is not None:
                if source_gap_s <= 0.0:
                    tracker_reset_reason = "non_increasing_source_time"
                elif source_gap_s > state["tracker"].max_gap_s:
                    tracker_reset_reason = "source_gap_exceeds_tracker_max_gap"
            # 三机共享同一个 YOLO 模型，但各自保留时序跟踪与相机运动状态。
            detector.pipeline.camera = state["camera"]
            detector.pipeline.tracker = state["tracker"]
            detector.pipeline.previous_t = state["previous_t"]
            detector.pipeline.sequence = state["sequence"]
            predictions = detector.predict(
                image,
                timestamp=source_time,
                sequence_id=uid,
            )
            state.update({
                "camera": detector.pipeline.camera,
                "tracker": detector.pipeline.tracker,
                "previous_t": detector.pipeline.previous_t,
                "sequence": detector.pipeline.sequence,
            })
            inference_completed = time.perf_counter()
            image_path = None
            save_started = time.perf_counter()
            if frame_output:
                # 保留模型实际收到的压缩字节，避免再次编码改变离线重放的像素。
                suffix = ".png" if image_bytes.startswith(b"\x89PNG") else ".jpg"
                saved_frame = frame_output / f"{sequence:07d}{suffix}"
                saved_frame.write_bytes(image_bytes)
                image_path = str(saved_frame.resolve())
            result_pipe.send({
                "event": "completed",
                "submission_id": sequence,
                "worker_completed_perf_counter": inference_completed,
                "decode_ms": decode_ms,
                "inference_wall_ms": (inference_completed - inference_started) * 1000.0,
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "image_path": image_path,
                "frame_save_ms": (time.perf_counter() - save_started) * 1000.0,
                "previous_processed_source_sim_time": previous_source_time,
                "processed_source_gap_s": source_gap_s,
                "tracker_reset_reason": tracker_reset_reason,
                "predictions": _serialise_predictions(predictions),
            })
    except Exception as exc:  # noqa: BLE001
        try:
            result_pipe.send({"event": "worker_error", "error": repr(exc)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        result_pipe.close()


def _iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _match_counts(ground_truth, predictions, class_aware, threshold=0.50):
    expected = {"TargetVehicle": "real_vehicle", "DecoyVehicle": "model_prop"}
    candidates = []
    for gt_index, truth in enumerate(ground_truth):
        for pred_index, prediction in enumerate(predictions):
            if class_aware and prediction["class_name"] != expected.get(truth["class"]):
                continue
            overlap = _iou(truth["bbox"], prediction["bbox_xyxy"])
            if overlap >= threshold:
                candidates.append((overlap, gt_index, pred_index))
    matched_gt, matched_predictions = set(), set()
    for _, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in matched_gt or pred_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)
    return {
        "tp": len(matched_gt),
        "fp": len(predictions) - len(matched_predictions),
        "fn": len(ground_truth) - len(matched_gt),
    }


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _metric_summary(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0.0
            else None
        ),
    }


def _percentiles(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


class YoloSidecar:
    """runner 所有的单子进程旁路；每机只保留最新待处理帧并公平轮询。"""

    def __init__(self, output, uids, config, device, resources, log):
        self.output = Path(output)
        self.log = log
        self.resources = dict(resources)
        self.profile = str(resources["profile"])
        self.uids = tuple(str(uid) for uid in uids)
        self.slot_by_uid = {uid: index for index, uid in enumerate(self.uids)}
        self.context = multiprocessing.get_context("spawn")
        self.slots = [
            self.context.RawArray(ctypes.c_ubyte, SLOT_CAPACITY_BYTES)
            for _ in self.uids
        ]
        self.slot_lengths = [self.context.Value("Q", 0) for _ in self.uids]
        self.slot_sequences = [self.context.Value("Q", 0) for _ in self.uids]
        self.submission_sequence = self.context.Value("Q", 0)
        self.slot_lock = self.context.Lock()
        self.slot_ready = self.context.Event()
        self.stop_event = self.context.Event()
        self.result_parent, result_child = self.context.Pipe(duplex=False)
        # run_official 用 runpy 把入口执行为 __main__；spawn 必须拿到可重新导入的规范模块函数。
        worker_target = importlib.import_module(
            "personal_hf2026.yolo_sidecar_study"
        )._worker_main
        self.process = self.context.Process(
            target=worker_target,
            name="hf2026-yolo-sidecar",
            args=(
                self.slots,
                self.slot_lengths,
                self.slot_sequences,
                self.slot_lock,
                self.slot_ready,
                self.stop_event,
                result_child,
                str(Path(config).resolve()),
                str(device),
                self.resources,
            ),
        )
        self.process.start()
        result_child.close()
        self.submission_log = (self.output / "yolo_sidecar_submissions.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self.result_log = (self.output / "yolo_sidecar_results.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self.event_log = (self.output / "yolo_sidecar_events.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self.submissions = {}
        self.status = {}
        self.results = []
        self.worker = {"pid": self.process.pid, "ready": False, "errors": []}
        self.last_observed_sim_time = None
        self.closed = False

    def submit(self, packet, image_received_sim_time, observed_gimbal_fov_deg):
        image = packet["image"]
        submission_id = int(self.submission_sequence.value) + 1
        metadata = {
            "submission_id": submission_id,
            "job_id": f"job-{submission_id:08d}",
            "yolo_profile": self.profile,
            "uid": packet["uid"],
            "frame_no": packet["frame_no"],
            "source_sim_time": packet["source_sim_time"],
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "image_bytes": len(image),
            "image_received_sim_time": float(image_received_sim_time),
            "observed_gimbal_fov_deg": float(observed_gimbal_fov_deg),
            "fov_observation_alignment": "runner_observation_not_verified_exposure_time",
            "submitted_unix_s": time.time(),
            "submitted_perf_counter": time.perf_counter(),
            "ground_truth": packet["ground_truth"],
            "ground_truth_source": "redis_sync_camera_same_frame_ue_projected_boxes",
        }
        metadata_bytes = json.dumps(
            metadata, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        payload = SLOT_HEADER.pack(len(metadata_bytes)) + metadata_bytes + image
        if len(payload) > SLOT_CAPACITY_BYTES:
            self.worker["errors"].append(
                f"frame_too_large:{packet['uid']}:{packet['frame_no']}:{len(payload)}"
            )
            _json_line(self.event_log, {
                "event": "rejected",
                "reason": "frame_too_large",
                "uid": packet["uid"],
                "frame_no": packet["frame_no"],
                "payload_bytes": len(payload),
            })
            return False
        with self.slot_lock:
            # 序号只在成功写入后递增，子进程由序号差判断被覆盖的提交。
            submission_id = int(self.submission_sequence.value) + 1
            metadata["submission_id"] = submission_id
            metadata["job_id"] = f"job-{submission_id:08d}"
            metadata_bytes = json.dumps(
                metadata, ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            payload = SLOT_HEADER.pack(len(metadata_bytes)) + metadata_bytes + image
            slot_index = self.slot_by_uid[packet["uid"]]
            self.slots[slot_index][: len(payload)] = payload
            self.slot_lengths[slot_index].value = len(payload)
            self.slot_sequences[slot_index].value = submission_id
            self.submission_sequence.value = submission_id
            self.slot_ready.set()
        self.submissions[submission_id] = metadata
        self.status[submission_id] = "submitted"
        _json_line(self.submission_log, metadata)
        _json_line(self.event_log, {
            "event": "accepted",
            "submission_id": submission_id,
            "job_id": metadata["job_id"],
            "uid": metadata["uid"],
            "frame_no": metadata["frame_no"],
            "image_received_sim_time": metadata["image_received_sim_time"],
        })
        return True

    def poll(self, observed_sim_time):
        if observed_sim_time is not None:
            self.last_observed_sim_time = float(observed_sim_time)
        while True:
            try:
                available = self.result_parent.poll()
            except (BrokenPipeError, OSError):
                break
            if not available:
                break
            try:
                event = self.result_parent.recv()
            except (BrokenPipeError, EOFError, OSError):
                break
            kind = event["event"]
            if kind == "worker_ready":
                self.worker.update(event)
                self.worker["ready"] = True
                _json_line(self.event_log, event)
                continue
            if kind == "worker_error":
                self.worker["errors"].append(event["error"])
                _json_line(self.event_log, event)
                continue
            submission_id = int(event["submission_id"])
            if kind == "started":
                self.status[submission_id] = "started"
                uid = self.submissions[submission_id]["uid"]
                for older_id in sorted(self.status):
                    if older_id >= submission_id:
                        break
                    if (self.status[older_id] == "submitted"
                            and self.submissions[older_id]["uid"] == uid):
                        self.status[older_id] = "superseded_before_start"
                        _json_line(self.event_log, {
                            "event": "terminal",
                            "status": "superseded_before_start",
                            "submission_id": older_id,
                            "superseded_by": submission_id,
                        })
                _json_line(self.event_log, event)
                continue
            if kind != "completed":
                _json_line(self.event_log, event)
                continue
            metadata = self.submissions[submission_id]
            row = dict(metadata)
            row.update({
                "inference_completed_sim_time": self.last_observed_sim_time,
                "result_observed_sim_time": self.last_observed_sim_time,
                "result_observed_unix_s": time.time(),
                "result_observed_perf_counter": time.perf_counter(),
                "inference_wall_ms": float(event["inference_wall_ms"]),
                "decode_ms": float(event["decode_ms"]),
                "image_width": int(event["image_width"]),
                "image_height": int(event["image_height"]),
                "image_path": event.get("image_path"),
                "frame_save_ms": event.get("frame_save_ms", 0.0),
                "previous_processed_source_sim_time": event[
                    "previous_processed_source_sim_time"
                ],
                "processed_source_gap_s": event["processed_source_gap_s"],
                "tracker_reset_reason": event["tracker_reset_reason"],
                "predictions": event["predictions"],
            })
            if self.last_observed_sim_time is not None:
                row["observation_latency_sim_s"] = (
                    self.last_observed_sim_time - metadata["image_received_sim_time"]
                )
            else:
                row["observation_latency_sim_s"] = None
            self.status[submission_id] = "completed"
            self.results.append(row)
            _json_line(self.result_log, row)
            _json_line(self.event_log, {
                "event": "collected",
                "status": "completed",
                "submission_id": submission_id,
                "job_id": metadata["job_id"],
                "result_observed_sim_time": self.last_observed_sim_time,
            })

    def summary(self):
        counts = Counter(self.status.values())
        by_uid = Counter(row["uid"] for row in self.results)
        reset_reasons = Counter(
            row["tracker_reset_reason"] for row in self.results
            if row["tracker_reset_reason"] is not None
        )
        hashes = [item["image_sha256"] for item in self.submissions.values()]
        completed_hashes = [row["image_sha256"] for row in self.results]
        submitted_hashes_by_uid = {
            uid: [
                item["image_sha256"] for item in self.submissions.values()
                if item["uid"] == uid
            ]
            for uid in self.uids
        }
        agnostic = {"tp": 0, "fp": 0, "fn": 0}
        aware = {"tp": 0, "fp": 0, "fn": 0}
        by_uid_metrics = defaultdict(
            lambda: {
                "frames": 0,
                "class_agnostic": {"tp": 0, "fp": 0, "fn": 0},
                "class_aware": {"tp": 0, "fp": 0, "fn": 0},
            }
        )
        for row in self.results:
            frame_agnostic = _match_counts(
                row["ground_truth"], row["predictions"], class_aware=False
            )
            frame_aware = _match_counts(
                row["ground_truth"], row["predictions"], class_aware=True
            )
            for key in agnostic:
                agnostic[key] += frame_agnostic[key]
                aware[key] += frame_aware[key]
                by_uid_metrics[row["uid"]]["class_agnostic"][key] += frame_agnostic[key]
                by_uid_metrics[row["uid"]]["class_aware"][key] += frame_aware[key]
            by_uid_metrics[row["uid"]]["frames"] += 1
        for value in by_uid_metrics.values():
            value["class_agnostic"] = _metric_summary(value["class_agnostic"])
            value["class_aware"] = _metric_summary(value["class_aware"])
        return {
            "yolo_profile": self.profile,
            "resources": self.resources,
            "process_model": "one_windows_spawn_process_one_latest_pending_frame_per_uid",
            "stream_state_model": (
                "one_shared_yolo_model_with_independent_tracker_camera_state_per_uid"
            ),
            "worker": self.worker,
            "counts": {
                "expected_uids": list(self.uids),
                "submitted": len(self.submissions),
                "completed": len(self.results),
                "by_final_status": dict(sorted(counts.items())),
                "completed_by_uid": dict(sorted(by_uid.items())),
            },
            "image_hashes": {
                "submitted_unique": len(set(hashes)),
                "submitted_duplicates": len(hashes) - len(set(hashes)),
                "completed_unique": len(set(completed_hashes)),
                "completed_duplicates": len(completed_hashes) - len(set(completed_hashes)),
                "deduplication_applied": False,
                "submitted_consecutive_by_uid": {
                    uid: {
                        "frames": len(uid_hashes),
                        "comparisons": max(0, len(uid_hashes) - 1),
                        "same_hash": sum(
                            left == right
                            for left, right in zip(uid_hashes, uid_hashes[1:])
                        ),
                        "same_hash_rate": _ratio(
                            sum(
                                left == right
                                for left, right in zip(uid_hashes, uid_hashes[1:])
                            ),
                            max(0, len(uid_hashes) - 1),
                        ),
                    }
                    for uid, uid_hashes in submitted_hashes_by_uid.items()
                },
            },
            "latency": {
                "inference_wall_ms": _percentiles(
                    row["inference_wall_ms"] for row in self.results
                ),
                "result_observation_sim_s": _percentiles(
                    row["observation_latency_sim_s"]
                    for row in self.results
                    if row["observation_latency_sim_s"] is not None
                ),
                "processed_source_gap_s_by_uid": {
                    uid: _percentiles(
                        row["processed_source_gap_s"] for row in self.results
                        if row["uid"] == uid
                        and row["processed_source_gap_s"] is not None
                    )
                    for uid in self.uids
                },
                "tracker_reset_reasons": dict(sorted(reset_reasons.items())),
                "semantics": (
                    "image_received_sim_time 与 inference_completed_sim_time 均来自 "
                    "obs.briefing.score_view.sim_time；后者表示 runner 首次观测到子进程结果的时刻。"
                ),
                "throughput_caveat": (
                    "单 worker 公平轮询三机最新帧；若同一 UID 的已处理 source gap 超过 "
                    "TemporalTracker.max_gap_s，原算法会按设计重置，因此在线结果不等同离线连续帧。"
                ),
            },
            "metrics_iou_0_50": {
                "class_agnostic": _metric_summary(agnostic),
                "class_aware": _metric_summary(aware),
                "by_uid": dict(sorted(by_uid_metrics.items())),
                "ground_truth_caveat": (
                    "ground_truth 是相机 Redis 帧附带的 UE 投影框，属于审计/oracle 元数据；"
                    "未人工验证遮挡与可见性，不能直接称为官方端到端检测成绩。"
                ),
            },
        }

    def close(self, observed_sim_time=None):
        if self.closed:
            return self.summary()
        self.poll(observed_sim_time)
        self.stop_event.set()
        self.slot_ready.set()
        self.process.join(timeout=30)
        if self.process.is_alive():
            self.worker["errors"].append("worker_join_timeout_terminated")
            self.process.terminate()
            self.process.join(timeout=5)
        self.poll(observed_sim_time)
        for submission_id, status in list(self.status.items()):
            if status in ("submitted", "started"):
                self.status[submission_id] = "unfinished_at_shutdown"
                _json_line(self.event_log, {
                    "event": "terminal",
                    "status": "unfinished_at_shutdown",
                    "submission_id": submission_id,
                })
        self.worker["exitcode"] = self.process.exitcode
        self.result_parent.close()
        self.submission_log.close()
        self.result_log.close()
        self.event_log.close()
        self.closed = True
        value = self.summary()
        (self.output / "yolo_sidecar_summary.json").write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )
        return value


class YoloSidecarRunner(StudyRunner):
    """保留 PersonalV1 理想检测控制，只从 runner 将 RGB 复制到旁路。"""

    def __init__(self, cfg, output, runtime_root, weather, config, device, resources, log):
        super().__init__(cfg, PersonalV1Agent, output, log)
        self.runtime_root = Path(runtime_root).resolve()
        self.weather = weather
        self.config = Path(config).resolve()
        self.device = str(device)
        self.resources = dict(resources)
        self.camera = None
        self.sidecar = None
        self.sidecar_summary = None
        self.delivered_frames = set()

    def prepare_scenario(self):
        super().prepare_scenario()
        self._scenario_cfg.setdefault("weather", {})["type"] = self.weather
        self.cfg.weather = self.weather
        for entity in self._scenario_cfg.get("entities", []):
            if entity.get("type") != "FixedWingUAV":
                continue
            components = entity.setdefault("components", {})
            gimbal = components.setdefault("gimbal_tracking", {})
            gimbal.setdefault("params", {})["fov"] = PersonalV1Agent.SEARCH_FOV_DEG
        self.log(f"[yolo-sidecar] prepared scenario weather = {self.weather}")
        self.log(
            "[yolo-sidecar] prepared UAV initial FOV = "
            f"{PersonalV1Agent.SEARCH_FOV_DEG:.1f} deg"
        )

    def _build_perception(self, uids):
        self.camera = AuditPhotoCache(
            uids, self.output, self.cfg.redis_host, self.cfg.redis_port
        )
        self.renderer = StudyRenderer(
            self.runtime_root,
            self.output,
            self.cfg.redis_host,
            self.cfg.redis_port,
            self.log,
        )
        self.renderer.start(self._scenario_cfg, uids)
        self.camera.start()
        self.sidecar = YoloSidecar(
            self.output, uids, self.config, self.device, self.resources, self.log
        )
        return self.camera, DetectionResolver(default_detector=MultiTargetIdealDetector())

    def make_agent_for(self, entity_type, entity_uid, world_state):
        agent = super().make_agent_for(entity_type, entity_uid, world_state)
        controlled_decide = agent.decide

        def decide(obs, dt):
            score = getattr(getattr(obs, "briefing", None), "score_view", None)
            observed_sim_time = score.sim_time if score is not None else None
            if self.sidecar:
                self.sidecar.poll(observed_sim_time)
            packet = self.camera.delivered.get(entity_uid) if self.camera else None
            if packet is not None and observed_sim_time is not None:
                signature = (
                    entity_uid,
                    packet["frame_no"],
                    packet["source_sim_time"],
                )
                if signature not in self.delivered_frames:
                    self.sidecar.submit(
                        packet, observed_sim_time, obs.self.gimbal_fov_deg
                    )
                    self.delivered_frames.add(signature)
            # 不改 obs.self.detection/detections；PersonalV1 继续消费理想控制输入。
            return controlled_decide(obs, dt)

        agent.decide = decide
        return agent

    def should_finish(self, agents):
        # 更新 V1 完成摘要，但视觉评估必须覆盖用户请求的完整时长。
        super().should_finish(agents)
        return False

    def close(self):
        if self.sidecar:
            self.sidecar_summary = self.sidecar.close(
                self.sidecar.last_observed_sim_time
            )
        if self.renderer:
            self.renderer.close()
        if self.camera:
            self.camera.stop()
        self.trace.close()
        self.judge.close()


def _resource_metadata(config, profile, trt_engine):
    config = Path(config).resolve()
    detector_config = json.loads(config.read_text(encoding="utf-8"))
    configured_weights = (PROJECT_ROOT / detector_config["weights"]).resolve()
    if profile == "v3":
        weights = Path(trt_engine).resolve()
        if weights.suffix.lower() != ".engine":
            raise ValueError("v3 的 --trt-engine 必须指向 .engine 文件")
        flip_enabled = False
        tracker_high_multiplier = V3_TRACKER_HIGH_MULTIPLIER
    else:
        weights = configured_weights
        flip_enabled = profile == "v1"
        tracker_high_multiplier = 1.0
    weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    if profile != "v3" and weights_sha256 != detector_config["sha256"]:
        raise ValueError("基础 PT 权重与配置中的 sha256 不一致")
    base_tracker_high = float(detector_config["tracker"]["high"])
    return {
        "profile": profile,
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "weights_path": str(weights),
        "weights_sha256": weights_sha256,
        "model_format": (
            "tensorrt_engine" if weights.suffix.lower() == ".engine" else "pytorch_pt"
        ),
        "effective_options": {
            "flip_enabled": flip_enabled,
            "tracker_high_confidence_threshold": (
                base_tracker_high * tracker_high_multiplier
            ),
            "tracker_low_confidence_threshold": float(
                detector_config["tracker"]["low"]
            ),
            "unknown_class_confidence_threshold": float(
                detector_config["unknown_threshold"]
            ),
            "tracker_high_multiplier": tracker_high_multiplier,
        },
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--runtime-root", default=str(RUNTIME_ROOT))
    result.add_argument("--layout", default=str(DEFAULT_LAYOUT))
    result.add_argument("--weather", choices=WEATHERS, default="Clear_Skies")
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--duration", type=float, default=40.0)
    result.add_argument("--fov", type=float, default=48.0)
    result.add_argument("--output", required=True)
    result.add_argument("--config", default=str(DEFAULT_CONFIG))
    result.add_argument("--device", default="0")
    result.add_argument("--image-rotation-deg", type=int, choices=(0, 90, 180, 270),
                        default=0, help="模型输入逆时针旋转角；输出框还原到原图，默认 0 保持原 v3")
    result.add_argument(
        "--save-processed-frames", action="store_true",
        help="保存实际推理的原始相机图像供错分诊断；磁盘写入在worker内，单独记录耗时",
    )
    result.add_argument(
        "--yolo-profile",
        choices=YOLO_PROFILES,
        default="v1",
        help="v1=原始 PT+翻转，v2=原始 PT+无翻转，v3=TensorRT FP16+无翻转+tracker.high 相对提高 20%%",
    )
    result.add_argument(
        "--trt-engine",
        default=str(DEFAULT_TRT_ENGINE),
        help="v3 使用的 TensorRT FP16 .engine；其余档位忽略此参数",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 5.0 <= args.fov <= 50.0:
        raise SystemExit("--fov 必须位于比赛允许的 5～50 度范围")
    if args.duration <= 0:
        raise SystemExit("--duration 必须大于 0")
    PersonalV1Agent.SEARCH_FOV_DEG = float(args.fov)
    resources = _resource_metadata(args.config, args.yolo_profile, args.trt_engine)
    resources["effective_options"]["image_rotation_deg"] = args.image_rotation_deg
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    resources["processed_frames_dir"] = (
        str(output / "processed_frames") if args.save_processed_frames else None
    )
    metadata = {
        "schema_version": 1,
        "mode": "personal_v1_control_yolo_sidecar_audit",
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "layout": str(Path(args.layout).resolve()),
        "runtime_root": str(Path(args.runtime_root).resolve()),
        "weather": args.weather,
        "seed": args.seed,
        "duration": args.duration,
        "requested_fov_deg": args.fov,
        "effective_fov_deg": PersonalV1Agent.SEARCH_FOV_DEG,
        "device": args.device,
        "yolo_profile": args.yolo_profile,
        "multiprocessing_start_method": "spawn",
        "slot_capacity_bytes": SLOT_CAPACITY_BYTES,
        "truth_source": "redis_sync_camera_same_frame_ue_projected_boxes",
        "truth_limit": "ue_projected_boxes_visibility_and_exposure_unverified",
        "time_source": "obs.briefing.score_view.sim_time",
        "resources": resources,
        "argv": os.sys.argv,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    os.environ["OPENSIM_SIM_STDERR"] = str(output / "engine.log")
    cfg = ScenarioConfig(
        "coop_decoy",
        args.layout,
        args.duration,
        output_dir=str(output),
        sim_binary=str(Path(args.runtime_root) / "opensim-sim.exe"),
        start_sim_flag=True,
        photo_mode="on",
        seed=args.seed,
    )
    runner = None
    result = None
    with (output / "run.log").open("w", encoding="utf-8", buffering=1) as stream:
        def log(message):
            stream.write(str(message) + "\n")
            print(message, flush=True)

        runner = YoloSidecarRunner(
            cfg,
            output,
            args.runtime_root,
            args.weather,
            args.config,
            args.device,
            resources,
            log,
        )
        try:
            result = runner.run()
        finally:
            runner.close()
            completed_by_uid = (
                runner.sidecar_summary["counts"]["completed_by_uid"]
                if runner.sidecar_summary else {}
            )
            sidecar_ok = bool(
                runner.sidecar_summary
                and runner.sidecar_summary["worker"].get("ready")
                and not runner.sidecar_summary["worker"].get("errors")
                and runner.sidecar_summary["worker"].get("exitcode") == 0
                and all(
                    completed_by_uid.get(uid, 0) > 0
                    for uid in runner.sidecar_summary["counts"]["expected_uids"]
                )
                and runner.camera is not None
                and runner.camera.closed
                and runner.camera.error_count == 0
            )
            summary = {
                "status": (
                    "completed"
                    if result is not None and not result.get("error") and sidecar_ok
                    else "failed"
                ),
                "weather": args.weather,
                "seed": args.seed,
                "duration": args.duration,
                "fov_deg": args.fov,
                "yolo_profile": args.yolo_profile,
                "resources": resources,
                "v1": getattr(runner, "v1_summary", None),
                "n_destroyed": result.get("n_destroyed", 0) if result else None,
                "runner_error": result.get("error") if result else "runner_exception",
                "sidecar_ok": sidecar_ok,
                "camera_error_count": runner.camera.error_count if runner.camera else None,
                "yolo_sidecar": runner.sidecar_summary,
            }
            (output / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2),
                encoding="utf-8",
            )
        if result and result.get("error"):
            raise RuntimeError(result["error"])
        if not sidecar_ok:
            raise RuntimeError("YOLO 旁路未让三架无人机各正常完成至少一帧")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
