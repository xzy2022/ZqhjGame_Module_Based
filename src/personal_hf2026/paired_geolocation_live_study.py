# 修改时间：2026-09-16。
# 修改目的：避免 Windows 低分辨率 monotonic 时钟把 Redis 子毫秒阶段量化为零。
# 修改内容：在保留兼容钟字段的同时记录 QueryPerformanceCounter 高分辨率同进程时钟。
# 修改时间：2026-09-16。
# 修改目的：排除实时图像时间语义和必要延迟测量中的未验证假设。
# 修改内容：逐事件记录 Redis 轮询、首次见帧、当前状态样本及帧与状态分发关系，并严格区分钟域。
# 修改时间：2026-09-16。
# 修改目的：避免把诱饵估计或过期估计送入正式目标上报，并实测各协同阶段的 FOV 读回值。
# 修改内容：上报队列增加真目标类别与半秒新鲜度门控，汇总 phase/FOV 计数。
# 修改时间：2026-09-16。
# 修改目的：让全部合格双机估计可记录，并按裁判一赫兹限制持续上报最新结果。
# 修改内容：桥接估计候选、替换协同阶段旧单机上报并分别统计候选、命令和限额落盘。
# 修改时间：2026-09-16。
# 修改目的：为实时双机定位补齐可离线审计的图像接收与状态取样时刻。
# 修改内容：记录 Redis 首次读取、本机单调时钟、上下文发布及姿态真值样本时间。
# 修改时间：2026-09-16。
# 修改目的：为双机图像定位提供可控的 UE 短测入口。
# 修改内容：接线专属 UE、Redis 新帧、协同上下文、输出约束和幂等收尾。
"""20 秒 UE 短测：仅在协同阶段评估双机 bbox 中心射线定位。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import redis

from competition.sdk.core.runner import ScenarioConfig
from competition.sdk.core.commands import report_target

from .control_test_runner import IdealPerceptionCoopDecoyRunner
from .dropout_capture import StudyRenderer
from .paths import OUTPUT_ROOT, RUNTIME_ROOT, SCENARIO_ROOT
from .personal_v1 import PersonalV1Agent
from .redis_runtime import RedisRuntime


def _session_value(value):
    """将协同会话转换为稳定的 JSON 值，避免将 Agent 对象泄漏给评估器。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_session_value(item) for item in value]
    return str(value)


def _json_value(value):
    """将引擎原始状态约束为可审计 JSON，不丢弃未知字段名。"""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return repr(value)


def _raw_field(raw, name, *, record_value=True):
    """同时保存字段覆盖情况和原值，缺失与空值保持可区分。"""
    present = isinstance(raw, dict) and name in raw
    value = raw.get(name) if present else None
    return {
        "present": present,
        "is_mapping": isinstance(value, dict),
        "keys": sorted(str(key) for key in value) if isinstance(value, dict) else [],
        "value_recorded": bool(present and record_value),
        "value": _json_value(value) if present and record_value else None,
    }


class FrameTimeProbe:
    """受字节和记录数双限的原始时间事件流。"""

    def __init__(self, output, max_records, max_output_bytes):
        self.path = Path(output) / "frame_time_probe.jsonl"
        self.summary_path = Path(output) / "frame_time_probe_summary.json"
        self.max_records = int(max_records)
        self.max_output_bytes = int(max_output_bytes)
        self._stream = self.path.open("x", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._counts = Counter()
        self._written_bytes = 0
        self._closed = False
        self.write(
            "probe_metadata",
            schema_version=1,
            clock_domains={
                "unix_s": "local_python_wall_clock",
                "monotonic_s": "local_python_monotonic_clock_same_process_only",
                "perf_counter_s": (
                    "local_python_high_resolution_monotonic_clock_same_process_only"),
                "world_sim_time": "engine_absolute_simulation_clock",
                "world_timestamp": "engine_state_timestamp_semantics_unverified",
                "source_sim_time": (
                    "redis_renderer_sim_time_semantics_under_test_not_verified_exposure_time"),
                "score_view_sim_time": "runner_relative_simulation_time_from_previous_tick",
            },
            arithmetic_rule=(
                "do_not_subtract_values_from_different_clock_domains_without_a_verified_mapping"),
            exposure_time_verified=False,
            atomic_frame_pose_binding=False,
            clock_resolution_s={
                "time": time.get_clock_info("time").resolution,
                "monotonic": time.get_clock_info("monotonic").resolution,
                "perf_counter": time.get_clock_info("perf_counter").resolution,
            },
        )

    def add_count(self, name, value=1):
        with self._lock:
            self._counts[str(name)] += int(value)

    def write(self, kind, **payload):
        record = {"schema_version": 1, "kind": str(kind), **payload}
        line = json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        size = len(line.encode("utf-8"))
        with self._lock:
            self._counts["events_seen"] += 1
            self._counts[f"{kind}_seen"] += 1
            if self._closed:
                self._counts["events_after_close"] += 1
                return False
            if self._counts["events_written"] >= self.max_records:
                self._counts["events_suppressed_record_limit"] += 1
                self._counts[f"{kind}_suppressed"] += 1
                return False
            if self._written_bytes + size > self.max_output_bytes:
                self._counts["events_suppressed_byte_limit"] += 1
                self._counts[f"{kind}_suppressed"] += 1
                return False
            self._stream.write(line)
            self._written_bytes += size
            self._counts["events_written"] += 1
            self._counts[f"{kind}_written"] += 1
            return True

    def close(self):
        with self._lock:
            if self._closed:
                return json.loads(self.summary_path.read_text(encoding="utf-8"))
            self._closed = True
            self._stream.close()
            summary = {
                "schema_version": 1,
                "path": self.path.name,
                "max_records": self.max_records,
                "max_output_bytes": self.max_output_bytes,
                "written_bytes": self._written_bytes,
                "counts": dict(sorted(self._counts.items())),
                "source_sim_time": (
                    "redis_renderer_sim_time_semantics_under_test_not_verified_exposure_time"),
                "cross_clock_subtraction_permitted": False,
                "atomic_frame_pose_binding_observed": False,
            }
            self.summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            return summary


class RedisFrameBridge:
    """读取每架无人机的最新唯一帧，不保存图片和全量帧日志。"""

    def __init__(self, uids, host, port, evaluator, probe):
        self.uids = tuple(uids)
        self.client = redis.Redis(
            host=host,
            port=port,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
        )
        self.evaluator = evaluator
        self.probe = probe
        self.contexts = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._seen = {}
        self._poll_indexes = defaultdict(int)
        self._duplicate_reads = Counter()
        self._report_candidates = []
        self._error = None

    @staticmethod
    def _clock_pair():
        return {
            "unix_s": time.time(),
            "monotonic_s": time.monotonic(),
            "perf_counter_s": time.perf_counter(),
        }

    @staticmethod
    def _key_text(key):
        return key.decode("utf-8", errors="backslashreplace")

    def _read_latest(self, uid, purpose):
        self._poll_indexes[uid] += 1
        poll_index = self._poll_indexes[uid]
        scan_started = self._clock_pair()
        keys = list(self.client.scan_iter(match=f"sync_camera:{uid}:frame:*", count=100))
        scan_completed = self._clock_pair()
        base = {
            "uid": str(uid),
            "poll_index": poll_index,
            "purpose": str(purpose),
            "scan_started_unix_s": scan_started["unix_s"],
            "scan_started_monotonic_s": scan_started["monotonic_s"],
            "scan_started_perf_counter_s": scan_started["perf_counter_s"],
            "scan_completed_unix_s": scan_completed["unix_s"],
            "scan_completed_monotonic_s": scan_completed["monotonic_s"],
            "scan_completed_perf_counter_s": scan_completed["perf_counter_s"],
            "key_count": len(keys),
        }
        if not keys:
            self.probe.add_count("bridge_poll_no_key")
            self.probe.write("bridge_poll", **base, outcome="no_key")
            return None
        key = max(keys, key=lambda item: int(item.rsplit(b":", 1)[1]))
        frame_no = int(key.rsplit(b":", 1)[1])
        redis_key = self._key_text(key)
        hmget_started = self._clock_pair()
        image, source_time, detections = self.client.hmget(
            key, "image", "sim_time", "detections")
        hmget_completed = self._clock_pair()
        base.update({
            "redis_key": redis_key,
            "frame_no": frame_no,
            "hmget_started_unix_s": hmget_started["unix_s"],
            "hmget_started_monotonic_s": hmget_started["monotonic_s"],
            "hmget_started_perf_counter_s": hmget_started["perf_counter_s"],
            "hmget_completed_unix_s": hmget_completed["unix_s"],
            "hmget_completed_monotonic_s": hmget_completed["monotonic_s"],
            "hmget_completed_perf_counter_s": hmget_completed["perf_counter_s"],
            "hash_field_presence": {
                "image": bool(image),
                "sim_time": source_time is not None,
                "detections": detections is not None,
            },
        })
        if not image or source_time is None:
            self.probe.add_count("bridge_poll_incomplete_hash")
            self.probe.write("bridge_poll", **base, outcome="incomplete_hash")
            return None
        source_sim_time_raw = source_time.decode(
            "utf-8", errors="backslashreplace") if isinstance(source_time, bytes) else str(source_time)
        source_sim_time = float(source_time)
        base["source_sim_time_raw"] = source_sim_time_raw
        return {
            "redis_key": redis_key,
            "frame_no": frame_no,
            "source_sim_time": source_sim_time,
            "source_sim_time_raw": source_sim_time_raw,
            "redis_read_unix_s": hmget_completed["unix_s"],
            "redis_read_monotonic_s": hmget_completed["monotonic_s"],
            "redis_read_perf_counter_s": hmget_completed["perf_counter_s"],
            "image": image,
            "detections": json.loads(detections) if detections else [],
        }, base

    def _classify_read(self, uid, frame, poll, *, delivery_eligible):
        signature = (frame["frame_no"], frame["source_sim_time"])
        previous = self._seen.get(uid)
        duplicate = signature == previous
        gap = None
        if previous is not None and not duplicate:
            gap = max(0, int(frame["frame_no"]) - int(previous[0]) - 1)
        if duplicate:
            self._duplicate_reads[(str(uid), signature)] += 1
            duplicate_index = self._duplicate_reads[(str(uid), signature)]
            outcome = "duplicate"
            self.probe.add_count("bridge_poll_duplicate")
        else:
            self._seen[uid] = signature
            duplicate_index = 0
            outcome = "new_frame" if delivery_eligible else "startup_existing_frame"
            self.probe.add_count("bridge_poll_new_signature")
            if gap:
                self.probe.add_count("frame_number_gap_total", gap)
                self.probe.add_count("frame_number_gap_events")
        poll_payload = dict(
            poll,
            outcome=outcome,
            source_sim_time=frame["source_sim_time"],
            previous_signature=(list(previous) if previous is not None else None),
            duplicate_read_index=duplicate_index,
            frame_number_gap_from_previous=gap,
        )
        events = [("bridge_poll", poll_payload)]
        if not duplicate:
            events.append(("frame_first_seen", {
                "uid": str(uid),
                "redis_key": frame["redis_key"],
                "frame_no": frame["frame_no"],
                "source_sim_time": frame["source_sim_time"],
                "source_sim_time_raw": frame["source_sim_time_raw"],
                "poll_index": poll["poll_index"],
                "purpose": poll["purpose"],
                "delivery_eligible": bool(delivery_eligible),
                "scan_started_unix_s": poll["scan_started_unix_s"],
                "scan_started_monotonic_s": poll["scan_started_monotonic_s"],
                "scan_started_perf_counter_s": poll["scan_started_perf_counter_s"],
                "scan_completed_unix_s": poll["scan_completed_unix_s"],
                "scan_completed_monotonic_s": poll["scan_completed_monotonic_s"],
                "scan_completed_perf_counter_s": poll["scan_completed_perf_counter_s"],
                "hmget_started_unix_s": poll["hmget_started_unix_s"],
                "hmget_started_monotonic_s": poll["hmget_started_monotonic_s"],
                "hmget_started_perf_counter_s": poll["hmget_started_perf_counter_s"],
                "hmget_completed_unix_s": poll["hmget_completed_unix_s"],
                "hmget_completed_monotonic_s": poll["hmget_completed_monotonic_s"],
                "hmget_completed_perf_counter_s": poll["hmget_completed_perf_counter_s"],
                "frame_number_gap_from_previous": gap,
                "source_sim_time_semantics": (
                    "redis_renderer_field_not_verified_exposure_time"),
            }))
        return not duplicate, events

    def _write_probe_events(self, events):
        for kind, payload in events:
            self.probe.write(kind, **payload)

    def start(self):
        # 记住启动前的残留帧，本轮只交付之后到达的新帧。
        for uid in self.uids:
            result = self._read_latest(uid, "startup_baseline")
            if result is not None:
                frame, poll = result
                _, events = self._classify_read(
                    uid, frame, poll, delivery_eligible=False)
                self._write_probe_events(events)
        self._thread.start()

    def update_context(self, uid, context):
        with self._lock:
            self.contexts[uid] = context

    def ensure_healthy(self):
        if self._error is not None:
            raise RuntimeError("实时 Redis 帧评估失败") from self._error

    def drain_report_candidates(self, reporter_uid):
        """取走分配给该主机的全部新估计；正式上报限速由 runner 负责。"""
        reporter_uid = str(reporter_uid)
        with self._lock:
            selected = [
                row for row in self._report_candidates
                if str(row["views"]["master"]["uid"]) == reporter_uid
            ]
            self._report_candidates = [
                row for row in self._report_candidates
                if str(row["views"]["master"]["uid"]) != reporter_uid
            ]
        return selected

    def _loop(self):
        while not self._stop.is_set():
            try:
                for uid in self.uids:
                    result = self._read_latest(uid, "delivery_loop")
                    if result is None:
                        continue
                    frame, poll = result
                    is_new, read_events = self._classify_read(
                        uid, frame, poll, delivery_eligible=True)
                    if not is_new:
                        self._write_probe_events(read_events)
                        continue
                    with self._lock:
                        context = self.contexts.get(uid)
                    dispatched = self._clock_pair()
                    dispatch_event = ("frame_context_dispatch", {
                        "uid": str(uid),
                        "redis_key": frame["redis_key"],
                        "frame_no": frame["frame_no"],
                        "source_sim_time": frame["source_sim_time"],
                        "dispatch_unix_s": dispatched["unix_s"],
                        "dispatch_monotonic_s": dispatched["monotonic_s"],
                        "dispatch_perf_counter_s": dispatched["perf_counter_s"],
                        "outcome": ("current_context_available" if context is not None
                                    else "no_current_context"),
                        "state_sample_id": (context.get("state_sample_id")
                                            if context is not None else None),
                        "runner_tick_index": (context.get("runner_tick_index")
                                              if context is not None else None),
                        "world_sim_time": (context.get("world_sim_time")
                                           if context is not None else None),
                        "world_timestamp": (context.get("world_timestamp")
                                            if context is not None else None),
                        "context_published_unix_s": (
                            context.get("context_published_unix_s")
                            if context is not None else None),
                        "context_published_monotonic_s": (
                            context.get("context_published_monotonic_s")
                            if context is not None else None),
                        "context_published_perf_counter_s": (
                            context.get("context_published_perf_counter_s")
                            if context is not None else None),
                        "atomic_frame_pose_binding": False,
                    })
                    if context is None:
                        self.probe.add_count("new_frames_without_current_context")
                        self._write_probe_events([*read_events, dispatch_event])
                        continue
                    context = dict(context)
                    world_time = context.get("world_sim_time")
                    delta = (float(world_time) - frame["source_sim_time"]
                             if world_time is not None else None)
                    context["pose_time_delta_s"] = delta
                    context["world_truth_time_delta_s"] = delta
                    try:
                        records = self.evaluator.observe_frame(uid, frame, context)
                    finally:
                        # 先让评估器接收帧，再序列化事件，避免探针写盘进入接收延迟。
                        self._write_probe_events([*read_events, dispatch_event])
                    if records:
                        with self._lock:
                            self._report_candidates.extend(records)
            except BaseException as exc:
                self._error = exc
                failed = self._clock_pair()
                self.probe.write(
                    "bridge_error",
                    captured_unix_s=failed["unix_s"],
                    captured_monotonic_s=failed["monotonic_s"],
                    captured_perf_counter_s=failed["perf_counter_s"],
                    error=repr(exc),
                )
                self._stop.set()
                return
            self._stop.wait(0.05)

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=5.0)
        self.client.close()
        self.ensure_healthy()


class PairedGeolocationLiveRunner(IdealPerceptionCoopDecoyRunner):
    """只增加定位旁路的 Runner，不向 Agent 返回图像或真值。"""

    MAX_FORMAL_REPORT_AGE_S = 0.5

    def __init__(self, cfg, output, runtime_root, evaluator, probe):
        super().__init__(cfg, PersonalV1Agent, log=lambda _message: None)
        self.output = Path(output)
        self.runtime_root = Path(runtime_root)
        self.evaluator = evaluator
        self.probe = probe
        self.renderer = None
        self.bridge = None
        self.world_context = {}
        self.live_summary = None
        self.probe_summary = None
        self.report_counts = Counter()
        self._pending_reports = {}
        self._last_report_sim_time = {}
        self.phase_fov_counts = Counter()
        self._report_written_bytes = 0
        self._report_stream = (
            self.output / "paired_geolocation_reports.jsonl").open(
                "x", encoding="utf-8", buffering=1)
        self._closed = False
        self._runner_tick_indexes = Counter()

    def _write_report_command(self, record, reporter_uid, sim_time):
        audit = {
            "schema_version": 1,
            "kind": "formal_report_target_command",
            "estimate_index": record["estimate_index"],
            "reporter_uid": str(reporter_uid),
            "target_id": record["target_id"],
            "sim_time": sim_time,
            "estimate": record["estimate"],
            "judge_acceptance": "unknown_until_evaluation_json",
        }
        line = json.dumps(
            audit, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        size = len(line.encode("utf-8"))
        if self.report_counts["report_records_written"] >= self.evaluator.max_records:
            self.report_counts["report_records_suppressed_record_limit"] += 1
            return
        if self._report_written_bytes + size > self.evaluator.max_output_bytes:
            self.report_counts["report_records_suppressed_byte_limit"] += 1
            return
        self._report_stream.write(line)
        self._report_written_bytes += size
        self.report_counts["report_records_written"] += 1

    def _append_paired_report(self, commands, entity_uid, coordinator, sim_time):
        if self.bridge is None:
            return commands
        candidates = self.bridge.drain_report_candidates(entity_uid)
        for record in candidates:
            self.report_counts["candidates_received"] += 1
            if record.get("class") != "TargetVehicle":
                self.report_counts["non_target_candidates_not_reported"] += 1
                continue
            key = str(record["target_id"])
            if key in self._pending_reports:
                self.report_counts["candidates_superseded_before_1hz"] += 1
            self._pending_reports[key] = record

        if coordinator.phase != coordinator.ACTIVE:
            for key, record in list(self._pending_reports.items()):
                if str(record["views"]["master"]["uid"]) == str(entity_uid):
                    self._pending_reports.pop(key)
                    self.report_counts["candidates_discarded_after_active"] += 1
            return commands
        if coordinator.role != coordinator.MASTER:
            return commands

        due = []
        for target_id, record in list(self._pending_reports.items()):
            if str(record["views"]["master"]["uid"]) != str(entity_uid):
                continue
            newest_source_time = max(
                float(view["source_sim_time"])
                for view in record["views"].values())
            if sim_time - newest_source_time > self.MAX_FORMAL_REPORT_AGE_S:
                self._pending_reports.pop(target_id, None)
                self.report_counts["stale_candidates_not_reported"] += 1
                continue
            last = self._last_report_sim_time.get(target_id)
            if last is None or sim_time - last >= 1.0:
                due.append((last if last is not None else -float("inf"), target_id, record))
        if not due:
            return commands
        _, target_id, record = min(due, key=lambda item: (item[0], item[1]))
        estimate = record["estimate"]
        commands.append(report_target(
            float(estimate["lat"]), float(estimate["lon"]), target_id))
        self._last_report_sim_time[target_id] = sim_time
        self._pending_reports.pop(target_id, None)
        self.report_counts["report_commands_emitted"] += 1
        self._write_report_command(record, entity_uid, sim_time)
        return commands

    def run(self):
        # 旁路和 UE 必须在本轮 Redis 退出前收尾，否则无法发送 UE shutdown。
        previous = Path.cwd()
        try:
            os.chdir(self.runtime_root)
            with RedisRuntime(
                    self.runtime_root, self.cfg.redis_host, self.cfg.redis_port):
                try:
                    result = self._run_with_idle_check()
                    if result.get("error"):
                        self.close(status="failed", error=str(result["error"]))
                    else:
                        self.close()
                    return result
                except BaseException as exc:
                    self.close(status="failed", error=repr(exc))
                    raise
        finally:
            os.chdir(previous)

    def _build_perception(self, uids):
        perception = super()._build_perception(uids)
        self.renderer = StudyRenderer(
            self.runtime_root,
            self.output,
            self.cfg.redis_host,
            self.cfg.redis_port,
            lambda _message: None,
        )
        self.renderer.start(self._scenario_cfg, uids)
        self.evaluator.start(uids)
        self.bridge = RedisFrameBridge(
            uids, self.cfg.redis_host, self.cfg.redis_port, self.evaluator, self.probe)
        self.bridge.start()
        return perception

    def _extract_truth(self, ws, uid):
        vehicle_truth = {}
        for target_uid, entity in list(ws.targets.items()) + list(ws.decoys.items()):
            vehicle_truth[str(target_uid)] = {
                "lat": entity.lat,
                "lon": entity.lon,
                "alt": entity.alt,
            }
        aircraft = ws.uavs.get(uid)
        attitude = None
        raw = {}
        if aircraft is not None:
            raw = aircraft.raw if isinstance(aircraft.raw, dict) else {}
            attitude = (raw.get("platform", {}) or {}).get("attitude")
        platform = raw.get("platform", {}) if isinstance(raw, dict) else {}
        platform = platform if isinstance(platform, dict) else {}
        gimbal = raw.get("gimbal_tracking", {}) if isinstance(raw, dict) else {}
        gimbal = gimbal if isinstance(gimbal, dict) else {}
        self._runner_tick_indexes[str(uid)] += 1
        runner_tick_index = self._runner_tick_indexes[str(uid)]
        captured = RedisFrameBridge._clock_pair()
        self.world_context[uid] = {
            "world_sim_time": ws.sim_time,
            "world_timestamp": getattr(ws, "timestamp", None),
            "runner_tick_index": runner_tick_index,
            "state_sample_id": f"{uid}:{runner_tick_index}",
            "world_state_observed_unix_s": captured["unix_s"],
            "world_state_observed_monotonic_s": captured["monotonic_s"],
            "world_state_observed_perf_counter_s": captured["perf_counter_s"],
            "aircraft_attitude": attitude,
            "raw_state_coverage": {
                "entity_keys": sorted(str(key) for key in raw),
                "platform": _raw_field(raw, "platform", record_value=False),
                "platform_position": _raw_field(platform, "position"),
                "platform_attitude": _raw_field(platform, "attitude"),
                "entity_heading": _raw_field(raw, "heading"),
                "gimbal_tracking": _raw_field(
                    raw, "gimbal_tracking", record_value=False),
                "gimbal_pose_fields": {
                    key: _raw_field(gimbal, key)
                    for key in ("pan_angle", "tilt_angle", "fov", "fov_deg")
                },
                "kinematics": _raw_field(raw, "kinematics", record_value=False),
            },
            # 真值只交给旁路评分，不进入 Agent 或定位算法输入。
            "truth_by_target_id": vehicle_truth,
        }
        return super()._extract_truth(ws, uid)

    def make_agent_for(self, entity_type, entity_uid, world_state):
        agent = super().make_agent_for(entity_type, entity_uid, world_state)
        decide = agent.decide

        def observed_decide(obs, dt):
            decide_entered = RedisFrameBridge._clock_pair()
            commands = decide(obs, dt)
            decide_completed = RedisFrameBridge._clock_pair()
            if self.bridge is not None:
                own = obs.self
                coordinator = agent._coordinator
                self.phase_fov_counts[
                    f"{coordinator.phase}|{float(own.gimbal_fov_deg):.6g}"] += 1
                world = self.world_context.get(entity_uid, {})
                context_published = RedisFrameBridge._clock_pair()
                score_view = getattr(obs.briefing, "score_view", None)
                pose = {
                    "lat": own.lat,
                    "lon": own.lon,
                    "alt": own.alt,
                    "heading_deg": own.heading_deg,
                    "gimbal_pan": own.gimbal_pan,
                    "gimbal_tilt": own.gimbal_tilt,
                    "gimbal_fov_deg": own.gimbal_fov_deg,
                }
                context = {
                    "active": coordinator.phase == coordinator.ACTIVE,
                    "phase": coordinator.phase,
                    "session_id": _session_value(coordinator.current_session),
                    "role": coordinator.role,
                    "partner_uid": coordinator.partner_uid,
                    "pose": pose,
                    "world_sim_time": world.get("world_sim_time"),
                    "world_timestamp": world.get("world_timestamp"),
                    "runner_tick_index": world.get("runner_tick_index"),
                    "state_sample_id": world.get("state_sample_id"),
                    "pose_sample_sim_time": world.get("world_sim_time"),
                    "truth_sample_sim_time": world.get("world_sim_time"),
                    "context_published_unix_s": context_published["unix_s"],
                    "context_published_monotonic_s": context_published["monotonic_s"],
                    "context_published_perf_counter_s": context_published["perf_counter_s"],
                    "time_alignment": "nearest_current_state_unverified",
                    "aircraft_attitude": world.get("aircraft_attitude"),
                    "truth_by_target_id": world.get("truth_by_target_id", {}),
                }
                # 先发布原有 current context，再序列化探针，避免探针对帧关联本身增添延迟。
                self.bridge.update_context(entity_uid, context)
                self.probe.write(
                    "state_sample",
                    uid=str(entity_uid),
                    state_sample_id=world.get("state_sample_id"),
                    runner_tick_index=world.get("runner_tick_index"),
                    dt_s=float(dt),
                    world_sim_time=world.get("world_sim_time"),
                    world_timestamp=world.get("world_timestamp"),
                    agent_t=float(agent._t),
                    score_view_sim_time=(getattr(score_view, "sim_time", None)
                                         if score_view is not None else None),
                    phase=coordinator.phase,
                    role=coordinator.role,
                    partner_uid=coordinator.partner_uid,
                    session_id=_session_value(coordinator.current_session),
                    world_state_observed_unix_s=world.get("world_state_observed_unix_s"),
                    world_state_observed_monotonic_s=world.get(
                        "world_state_observed_monotonic_s"),
                    world_state_observed_perf_counter_s=world.get(
                        "world_state_observed_perf_counter_s"),
                    decide_entered_unix_s=decide_entered["unix_s"],
                    decide_entered_monotonic_s=decide_entered["monotonic_s"],
                    decide_entered_perf_counter_s=decide_entered["perf_counter_s"],
                    decide_completed_unix_s=decide_completed["unix_s"],
                    decide_completed_monotonic_s=decide_completed["monotonic_s"],
                    decide_completed_perf_counter_s=decide_completed["perf_counter_s"],
                    context_published_unix_s=context_published["unix_s"],
                    context_published_monotonic_s=context_published["monotonic_s"],
                    context_published_perf_counter_s=context_published["perf_counter_s"],
                    current_observation_pose=pose,
                    current_world_aircraft_attitude=world.get("aircraft_attitude"),
                    raw_state_coverage=world.get("raw_state_coverage", {}),
                )
                self.bridge.ensure_healthy()
                if coordinator.phase == coordinator.ACTIVE:
                    legacy_count = sum(
                        getattr(command, "verb", None) == "agent.report"
                        for command in commands)
                    if legacy_count:
                        commands = [
                            command for command in commands
                            if getattr(command, "verb", None) != "agent.report"
                        ]
                        self.report_counts[
                            "legacy_report_commands_suppressed"] += legacy_count
                sim_time = world.get("world_sim_time")
                if sim_time is None:
                    sim_time = agent._t
                commands = self._append_paired_report(
                    commands, entity_uid, coordinator, float(sim_time))
            return commands

        agent.decide = observed_decide
        return agent

    def should_finish(self, agents):
        # 功能短测总是跑满 duration；未进入协同由摘要如实记录。
        return False

    def close(self, status="completed", error=None):
        if self._closed:
            return
        self._closed = True
        errors = []
        for name, resource in (
                ("bridge", self.bridge),
                ("evaluator", self.evaluator),
                ("renderer", self.renderer),
                ("probe", self.probe)):
            if resource is None:
                continue
            try:
                result = (resource.close(status=status, error=error)
                          if name == "evaluator" else resource.close())
                if name == "evaluator":
                    self.live_summary = result
                elif name == "probe":
                    self.probe_summary = result
            except BaseException as exc:
                errors.append(exc)
        self.report_counts["pending_reports_at_close"] = len(self._pending_reports)
        self.report_counts["report_written_bytes"] = self._report_written_bytes
        try:
            self._report_stream.close()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise errors[0]


def _output_path(parser, raw_output):
    output = Path(raw_output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        parser.error(f"--output 必须位于父项目输出根下：{OUTPUT_ROOT}")
    if output.exists():
        parser.error(f"输出目录已存在，拒绝覆盖：{output}")
    return output


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout", default=str(SCENARIO_ROOT / "static-decoys.json"))
    parser.add_argument("--runtime-root", default=str(RUNTIME_ROOT))
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--angle-threshold-deg", type=float, default=3.0)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--max-output-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--probe-max-records", type=int, default=20000)
    parser.add_argument("--probe-max-output-bytes", type=int, default=32 * 1024 * 1024)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    layout = Path(args.layout).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    output = _output_path(parser, args.output)
    if not layout.is_file():
        parser.error(f"场景文件不存在：{layout}")
    if not (runtime_root / "opensim-sim.exe").is_file():
        parser.error(f"运行底座缺少 opensim-sim.exe：{runtime_root}")
    if args.duration <= 0:
        parser.error("--duration 必须大于 0")
    if args.angle_threshold_deg <= 0:
        parser.error("--angle-threshold-deg 必须大于 0")
    if (args.max_records <= 0 or args.max_output_bytes <= 0
            or args.probe_max_records <= 0 or args.probe_max_output_bytes <= 0):
        parser.error("所有记录数和字节上限必须大于 0")

    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running",
        "mode": "paired_geolocation_live_ue_short_test",
        "layout": str(layout),
        "duration_s": args.duration,
        "seed": args.seed,
        "angle_threshold_deg": args.angle_threshold_deg,
        "max_records": args.max_records,
        "max_output_bytes": args.max_output_bytes,
        "probe_max_records": args.probe_max_records,
        "probe_max_output_bytes": args.probe_max_output_bytes,
        "time_alignment": "nearest_current_state_unverified",
        "frame_time_probe": {
            "path": "frame_time_probe.jsonl",
            "summary_path": "frame_time_probe_summary.json",
            "source_sim_time": "not_verified_exposure_time",
            "cross_clock_subtraction_permitted": False,
            "atomic_frame_pose_binding_observed": False,
        },
        "formal_reporting": {
            "policy": "latest_qualifying_estimate_per_target_at_most_1hz",
            "class_filter": "TargetVehicle",
            "class_source": "research_only_redis_detection_class",
            "max_estimate_age_s": PairedGeolocationLiveRunner.MAX_FORMAL_REPORT_AGE_S,
            "legacy_personal_v1_reports_suppressed_during_coop_active": True,
            "judge_acceptance_source": "evaluation_json_n_reports",
        },
        "redis": {"host": args.redis_host, "port": args.redis_port},
        "output": str(output),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip(),
        "argv": sys.argv,
    }
    run_path = output / "run.json"
    run_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.environ["OPENSIM_SIM_STDERR"] = str(output / "engine.log")

    # 延迟导入使 CLI 帮助可以在未合并核心评估分支时独立查看。
    from .paired_geolocation_live import LivePairedGeolocationEvaluator
    evaluator = LivePairedGeolocationEvaluator(
        output,
        angle_threshold_deg=args.angle_threshold_deg,
        max_records=args.max_records,
        max_output_bytes=args.max_output_bytes,
    )
    probe = FrameTimeProbe(
        output,
        max_records=args.probe_max_records,
        max_output_bytes=args.probe_max_output_bytes,
    )
    cfg = ScenarioConfig(
        "coop_decoy",
        str(layout),
        args.duration,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        output_dir=str(output),
        sim_binary=str(runtime_root / "opensim-sim.exe"),
        start_sim_flag=True,
        photo_mode="on",
        seed=args.seed,
        quiet=True,
    )
    runner = PairedGeolocationLiveRunner(
        cfg, output, runtime_root, evaluator, probe)
    failure = None
    try:
        result = runner.run()
        if result.get("error"):
            raise RuntimeError(result["error"])
        summary = runner.live_summary or {}
        counts = summary.get("counts", {})
        metadata.update(
            status=summary.get("status", "completed_with_no_estimate"),
            n_destroyed=result.get("n_destroyed", 0),
            live_counts={
                "frames_strict_cooperation": counts.get("frames_strict_cooperation", 0),
                "estimates_produced": counts.get("estimates_produced", 0),
                "records_written": counts.get("records_written", 0),
                **dict(sorted(runner.report_counts.items())),
            },
            observed_fov_by_phase=dict(sorted(runner.phase_fov_counts.items())),
            frame_time_probe_summary=runner.probe_summary,
        )
    except BaseException as exc:
        failure = exc
        metadata.update(status="failed", error=repr(exc))
    finally:
        try:
            runner.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
                metadata.update(status="failed", error=repr(exc))
        metadata["finished_unix_s"] = time.time()
        run_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if failure is not None:
        raise failure
    print(json.dumps({
        "status": metadata["status"],
        "output": str(output),
        "duration_s": args.duration,
        "counts": metadata.get("live_counts", {}),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
