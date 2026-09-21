# 修改时间：2026-09-21（后续协同审计证据）。
# 修改目的：让丢失超时退出和鲁棒静止完成可由有界运行轨迹直接复核。
# 修改内容：透出轨迹与静止拟合摘要，封顶确认指纹并省略周期样本中重复的 before 快照以延长 trace 覆盖时间。
# 修改时间：2026-09-21（静止判定审计）。
# 修改目的：让真实运行 trace 保留 V3 鲁棒静止判定的输入、拟合和连续新帧证据。
# 修改内容：采集协调器 stationary 快照，并在连续计数或 ready 变化时记录决策边沿。
# 修改时间：2026-09-21。
# 修改目的：为真实场景简化协同补齐不阻塞 Runner 热循环的时序证据。
# 修改内容：以内存有界状态边沿和零点五秒采样记录五帧、配对、门控、瞄准、结束与计数传播并自动离线审计。
# 修改时间：2026-09-20（正式场景与运动审计修复）。
# 修改目的：避免把静止诱饵控制实验误当成正式 coop_decoy 测评环境。
# 修改内容：强制使用官方动态场景和默认参数，并以裁判侧有界摘要记录真车与诱饵实际位移。
# 修改时间：2026-09-20（相机残留帧隔离）。
# 修改目的：避免上一轮 Redis 相机键在新 UE 出帧前被 V3 当成本轮首帧。
# 修改内容：记录启动时最新帧键并只在键变化后向 Agent 交付图片，不读取投影框元数据。
# 修改时间：2026-09-20（共享像素 worker 接线）。
# 修改目的：确保三机共用一个 YOLO-V2 模型和一个 latest-only 推理线程。
# 修改内容：由 Runner 创建共享 V3PerceptionWorker、注入各 Agent 并保存覆盖帧统计。
# 修改时间：2026-09-20。
# 修改目的：提供真实像素感知 PersonalV3 的单轮 UE 测评入口。
# 修改内容：固定四十八度视场、管理 Redis/UE/网页生命周期并保存有界运行摘要。
"""PersonalV3 单轮真实测评入口。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import re

from competition.sdk.core.perception import DetectionResolver, PhotoCache
from competition.sdk.core.runner import ScenarioConfig

from .dropout_capture import StudyRenderer
from .paths import OUTPUT_ROOT, PROJECT_ROOT, RUNTIME_ROOT, SIM_ROOT
from .personal_v3 import PersonalV3Agent
from .redis_runtime import RedisRuntime
from .sdk_compat import IdleCompatibleCoopDecoyRunner
from .v3_perception import V3PerceptionWorker, submit_observation


WEATHERS = (
    "Clear_Skies",
    "Partly_Cloudy",
    "Rain",
    "Foggy",
    "Snow_Light",
    "Sand_Dust_Calm",
)
DEFAULT_LAYOUT = SIM_ROOT / "competition/scenarios/coop_decoy/scenario.json"
_FRAME_NUMBER = re.compile(r"frame:(\d+)$")
_TRACE_SAMPLE_PERIOD_S = 0.5
_TRACE_DEFAULT_MAX_RECORDS = 12_000
_TRACE_DEFAULT_MAX_BYTES = 16 * 1024 * 1024


def _field(value, name, default=None):
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_json(value):
    """只保留运行证据需要的 JSON 安全值。"""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_safe_json(item) for item in sorted(value, key=repr)]
    if hasattr(value, "__dict__"):
        return {
            str(key): _safe_json(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _mapping(value):
    if callable(value):
        value = value()
    return value if isinstance(value, dict) else {}


def _deep_value(value, names):
    """从少量鸭子类型证据字典中寻找稳定字段。"""
    if not isinstance(value, dict):
        return None
    for name in names:
        if name in value and value[name] is not None:
            return value[name]
    for key in (
        "coordination", "control", "follower_guidance", "guidance",
        "runtime_evidence", "simple_control", "v3_simple_control", "perception",
    ):
        nested = value.get(key)
        found = _deep_value(nested, names)
        if found is not None:
            return found
    return None


def _observation_time(obs, agent):
    score = getattr(getattr(obs, "briefing", None), "score_view", None)
    if score is not None:
        return float(score.sim_time)
    return float(getattr(agent, "_t", 0.0))


def _perception_evidence(agent):
    snapshot = getattr(agent, "_last_snapshot", None)
    detection = _field(snapshot, "detection")
    return {
        "frame_id": _safe_json(_field(snapshot, "frame_id")),
        "source_sim_time": _safe_json(_field(snapshot, "source_sim_time")),
        "observed_sim_time": _safe_json(_field(snapshot, "observed_sim_time")),
        "class_name": _field(detection, "class_name"),
        "track_id": _safe_json(_field(detection, "track_id")),
        "confidence": _safe_json(
            _field(detection, "detector_confidence", _field(detection, "confidence"))
        ),
    }


def _control_evidence(agent):
    direct = {}
    for name in ("runtime_evidence", "_runtime_evidence"):
        try:
            direct.update(_mapping(getattr(agent, name, None)))
        except (TypeError, ValueError):
            continue
    for name in ("_simple_control", "simple_control", "_coop_control"):
        control = getattr(agent, name, None)
        if control is None:
            continue
        for evidence_name in ("as_evidence", "evidence", "summary"):
            try:
                direct.update(_mapping(getattr(control, evidence_name, None)))
            except (TypeError, ValueError):
                continue
    aliases = {
        "master_gate_m": ("master_gate_m", "uav_to_master_gate_m"),
        "target_gate_m": ("target_gate_m", "uav_to_target_gate_m"),
        "uav_distance_to_master_m": (
            "uav_distance_to_master_m", "distance_to_master_m", "partner_distance_m",
        ),
        "partner_distance_m": (
            "partner_distance_m", "uav_distance_to_master_m", "distance_to_master_m",
        ),
        "target_distance_m": ("target_distance_m", "distance_to_target_m"),
        "within_master_gate": ("within_master_gate",),
        "within_target_gate": ("within_target_gate",),
        "rendezvous_ready": ("rendezvous_ready",),
        "guidance_enabled": ("guidance_enabled",),
        "aiming_enabled": ("aiming_enabled",),
        "aim_target_lat": ("aim_target_lat",),
        "aim_target_lon": ("aim_target_lon",),
        "gimbal_pan_cmd_deg": ("gimbal_pan_cmd_deg",),
        "gimbal_tilt_cmd_deg": ("gimbal_tilt_cmd_deg",),
        "fly_to_lat": ("fly_to_lat",),
        "fly_to_lon": ("fly_to_lon",),
        "fly_to_alt_m": ("fly_to_alt_m", "fly_to_altitude_m"),
    }
    return {
        key: _safe_json(_deep_value(direct, names))
        for key, names in aliases.items()
        if _deep_value(direct, names) is not None
    }


def _agent_evidence(agent):
    coordinator = getattr(agent, "_coordinator", None)
    try:
        event_summary = _mapping(getattr(coordinator, "event_summary", None))
    except (TypeError, ValueError):
        event_summary = {}
    runtime = {}
    for name in ("runtime_evidence", "_runtime_evidence"):
        try:
            runtime.update(_mapping(getattr(agent, name, None)))
        except (TypeError, ValueError):
            continue
    sources = (event_summary, runtime)

    def value(names, fallback=None):
        for source in sources:
            found = _deep_value(source, names)
            if found is not None:
                return found
        for name in names:
            found = getattr(coordinator, name, None)
            if found is not None:
                return found
        return fallback

    session = value(("session", "current_session"))
    completed_sessions = value(("completed_sessions",), ())
    completed_count = value(("completed_count", "estimated_destroyed_count"))
    if completed_count is None:
        completed_count = len(completed_sessions or ())
    gate = getattr(agent, "_target_gate", None)
    confirmation = {
        "count": int(getattr(gate, "count", 0)),
        "required": int(getattr(gate, "required_frames", 0)),
        "ready": bool(getattr(gate, "ready", False)),
    }
    perception = _perception_evidence(agent)
    only_decoys = value(("only_decoys",))
    if only_decoys is not None:
        perception["only_decoys"] = bool(only_decoys)
    result = {
        "revision": _safe_json(value(("revision",))),
        "event": _safe_json(value(("event",))),
        "phase": _safe_json(value(("phase", "state"), getattr(agent, "_state", None))),
        "role": _safe_json(value(("role",), "NONE")),
        "session": _safe_json(session),
        "partner_uid": _safe_json(value(("partner_uid",))),
        "proposal_count": int(value(
            ("proposal_count", "master_sessions_started"), 0
        ) or 0),
        "follower_accept_count": int(value(
            ("follower_accept_count", "follower_sessions_started"), 0
        ) or 0),
        "finish_reason": _safe_json(value(("finish_reason",))),
        "decoy_only_count": int(value(("decoy_only_count",), 0) or 0),
        "decoy_only_required": int(value(("decoy_only_required",), 0) or 0),
        "last_completed_session": _safe_json(value(("last_completed_session",))),
        "stage_reason": _safe_json(value(("stage_reason",))),
        "completed_count": int(completed_count or 0),
        "completed_sessions": _safe_json(completed_sessions or ()),
        "master_position": _safe_json(value(("master_position",))),
        "follow_position": _safe_json(value(("follow_position",))),
        "stationary": _safe_json(value(("stationary",), {})),
        "confirmation": confirmation,
        "perception": perception,
    }
    for key in (
        "track_state", "track_last_seen_age_s", "master_lost_timeout_s",
        "track_predict_position", "stationary", "last_transition",
    ):
        found = _safe_json(value((key,)))
        if found is not None:
            result[key] = found
    control = _control_evidence(agent)
    result["control"] = control
    for key in (
        "master_gate_m", "target_gate_m", "uav_distance_to_master_m",
        "partner_distance_m", "target_distance_m", "within_master_gate",
        "within_target_gate", "rendezvous_ready", "guidance_enabled",
        "aiming_enabled", "aim_target_lat", "aim_target_lon",
        "gimbal_pan_cmd_deg", "gimbal_tilt_cmd_deg",
    ):
        found = control.get(key)
        if found is None:
            found = _safe_json(value((key,)))
        if found is not None:
            result[key] = found
    return result


def _command_evidence(commands):
    return [
        {
            "verb": str(getattr(command, "verb", "")),
            "params": _safe_json(getattr(command, "params", {})),
        }
        for command in commands
    ]


def _inbox_evidence(obs):
    return [
        {
            "sender_uid": str(getattr(message, "sender_uid", "")),
            "payload": _safe_json(getattr(message, "payload", None)),
            "recv_time": _safe_json(getattr(message, "recv_time", None)),
        }
        for message in getattr(obs, "comm_inbox", ())
    ]


def _state_fingerprint(state):
    stationary = state.get("stationary") or {}
    return (
        state.get("revision"), state.get("event"), state.get("phase"),
        state.get("role"), state.get("track_state"),
        json.dumps(state.get("session"), sort_keys=True),
        state.get("partner_uid"), state.get("proposal_count"),
        state.get("follower_accept_count"), state.get("finish_reason"),
        state.get("decoy_only_count"), state.get("decoy_only_required"),
        (state.get("perception") or {}).get("only_decoys"),
        state.get("completed_count"), state.get("rendezvous_ready"),
        state.get("within_master_gate"), state.get("within_target_gate"),
        state.get("guidance_enabled"), state.get("aiming_enabled"),
        stationary.get("stationary_consecutive_frames"), stationary.get("ready"),
    )


class CoordinationEvidenceRecorder:
    """热循环仅把有界 JSON 行留在内存，结束时一次写盘。"""

    def __init__(self, output, *, max_records, max_bytes):
        self.output = Path(output)
        self.max_records = int(max_records)
        self.max_bytes = int(max_bytes)
        self._lines = []
        self._bytes = 0
        self._attempted = 0
        self._dropped_limit = 0
        self._dropped_bytes = 0
        self._capture_errors = []
        self._last_sensor = {}
        self._last_decision_state = {}
        self._last_decision_time = {}
        self._closed = False

    def _append(self, row):
        self._attempted += 1
        if len(self._lines) >= self.max_records:
            self._dropped_limit += 1
            return
        encoded = (
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if self._bytes + len(encoded) > self.max_bytes:
            self._dropped_bytes += 1
            return
        self._lines.append(encoded)
        self._bytes += len(encoded)

    def capture_error(self, uid, stage, exc):
        if len(self._capture_errors) < 20:
            self._capture_errors.append({
                "uid": str(uid), "stage": str(stage), "error": repr(exc),
            })

    def record_sensor(self, uid, agent, obs):
        state = _agent_evidence(agent)
        confirmation = state["confirmation"]
        perception = state["perception"]
        confirmation_progress = min(
            int(confirmation["count"]), int(confirmation["required"])
        )
        fingerprint = (
            # 连续确认达到门槛后不再按无上限累计值写一条完整状态；类别、
            # track 和协同边沿变化仍会单独留下证据。
            confirmation_progress, confirmation["required"], confirmation["ready"],
            perception.get("class_name"), perception.get("track_id"),
            state.get("decoy_only_count"), state.get("decoy_only_required"),
            perception.get("only_decoys"),
        )
        if fingerprint == self._last_sensor.get(str(uid)):
            return
        self._last_sensor[str(uid)] = fingerprint
        self._append({
            "schema_version": 1,
            "kind": "perception",
            "uid": str(uid),
            "agent_time_s": _observation_time(obs, agent),
            "recorded_unix_s": time.time(),
            "basis": "agent_internal_and_formal_observation",
            "after": state,
        })

    def record_decision(self, uid, agent, obs, dt, commands, before, decide_wall_ms,
                        error=None):
        after = _agent_evidence(agent)
        now = float(getattr(agent, "_t", _observation_time(obs, agent)))
        uid = str(uid)
        fingerprint = _state_fingerprint(after)
        changed = fingerprint != self._last_decision_state.get(uid)
        periodic = now - self._last_decision_time.get(uid, -1e9) >= _TRACE_SAMPLE_PERIOD_S
        if not changed and not periodic and error is None:
            return
        self._last_decision_state[uid] = fingerprint
        self._last_decision_time[uid] = now
        own = obs.self
        self._append({
            "schema_version": 1,
            "kind": "decision",
            "capture_reason": "error" if error else ("state_change" if changed else "periodic_0_5s"),
            "uid": uid,
            "agent_time_s": now,
            "observation_time_s": _observation_time(obs, agent),
            "recorded_unix_s": time.time(),
            "dt_s": float(dt),
            "decide_wall_ms": float(decide_wall_ms),
            "basis": "agent_internal_and_formal_observation",
            # 周期样本的 before/after 通常完全相同；只在状态边沿和错误时
            # 保留 before，避免 600 秒有界 trace 过早耗尽字节预算。
            "before": before if changed or error is not None else None,
            "after": after,
            "self_pose": {
                "lat": _safe_json(own.lat), "lon": _safe_json(own.lon),
                "alt": _safe_json(own.alt),
                "heading_deg": _safe_json(own.heading_deg),
                "gimbal_pan_deg": _safe_json(own.gimbal_pan),
                "gimbal_tilt_deg": _safe_json(own.gimbal_tilt),
                "gimbal_fov_deg": _safe_json(own.gimbal_fov_deg),
            },
            "inbox": _inbox_evidence(obs),
            "commands": _command_evidence(commands),
            "error": error,
        })

    @property
    def summary(self):
        return {
            "schema_version": 1,
            "trace_path": "coordination_trace.jsonl",
            "summary_path": "coordination_trace_summary.json",
            "audit_path": "coordination_audit.json",
            "basis": "agent_internal_and_formal_observation_no_judge_world_state",
            "sample_period_s": _TRACE_SAMPLE_PERIOD_S,
            "max_records": self.max_records,
            "max_bytes": self.max_bytes,
            "records_attempted": self._attempted,
            "records_written": len(self._lines),
            "records_dropped_limit": self._dropped_limit,
            "records_dropped_byte_cap": self._dropped_bytes,
            "bytes_written": self._bytes,
            "capture_errors": list(self._capture_errors),
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        (self.output / "coordination_trace.jsonl").write_bytes(b"".join(self._lines))
        (self.output / "coordination_trace_summary.json").write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _scenario_profile(path: Path) -> dict:
    """读取正式场景静态契约；这些信息只用于启动校验和结果审计。"""
    raw = path.read_bytes()
    scenario = json.loads(raw.decode("utf-8-sig"))
    entities = scenario.get("entities", [])
    vehicles = []
    for entity in entities:
        entity_type = entity.get("type")
        if entity_type not in ("TargetVehicle", "ground_vehicle", "DecoyVehicle"):
            continue
        params = (
            entity.get("components", {})
            .get("trajectory", {})
            .get("params", {})
        )
        vehicles.append({
            "uid": str(entity.get("id") or entity.get("name") or ""),
            "type": entity_type,
            "speed_mps": float(params.get("speed", 0.0)),
            "speed_jitter_mps": float(params.get("speed_jitter", 0.0)),
        })
    decoys = [item for item in vehicles if item["type"] == "DecoyVehicle"]
    targets = [item for item in vehicles if item["type"] != "DecoyVehicle"]
    return {
        "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "runner": "competition.sdk.scenarios.coop_decoy.runner.CoopDecoyRunner",
        "uav_count": sum(entity.get("type") == "FixedWingUAV" for entity in entities),
        "target_count": len(targets),
        "decoy_count": len(decoys),
        "all_targets_commanded_to_move": bool(targets) and all(
            item["speed_mps"] > 0.0 for item in targets
        ),
        "all_decoys_commanded_to_move": bool(decoys) and all(
            item["speed_mps"] > 0.0 for item in decoys
        ),
        "target_speeds_mps": sorted({item["speed_mps"] for item in targets}),
        "decoy_speeds_mps": sorted({item["speed_mps"] for item in decoys}),
        "target_speed_jitter_mps": sorted({
            item["speed_jitter_mps"] for item in targets
        }),
        "decoy_speed_jitter_mps": sorted({
            item["speed_jitter_mps"] for item in decoys
        }),
        "weather": scenario.get("weather", {}).get("type"),
    }


def _horizontal_distance_m(first, current):
    """计算裁判侧审计位移，不把该结果暴露给 Agent。"""
    lat1, lon1 = first
    lat2, lon2 = current
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    north = math.radians(lat2 - lat1) * 6_378_137.0
    east = math.radians(lon2 - lon1) * 6_378_137.0 * math.cos(mean_lat)
    return math.hypot(east, north)


class FreshPhotoCache(PhotoCache):
    """只在本轮最新相机键越过启动基线后交付图片。"""

    def __init__(self, redis_client, uids):
        super().__init__(redis_client=redis_client, uids=uids)
        self._baseline = {str(uid): self._scan(str(uid)) for uid in uids}
        self._run_keys = {str(uid): set() for uid in uids}

    def _scan(self, uid):
        signatures = {}
        for key in self._redis.scan_iter(f"sync_camera:{uid}:frame:*"):
            value = key.decode() if isinstance(key, (bytes, bytearray)) else str(key)
            if _FRAME_NUMBER.search(value):
                signatures[value] = self._redis.hget(key, "sim_time")
        return signatures

    def _poll_once(self, uid):
        current = self._scan(uid)
        baseline = self._baseline.get(uid, {})
        self._run_keys[uid].update(
            key for key, source in current.items()
            if key not in baseline or source != baseline[key]
        )
        candidates = []
        for key in self._run_keys[uid]:
            if key not in current:
                continue
            source = current[key]
            try:
                source_time = float(source)
            except (TypeError, ValueError):
                continue
            candidates.append((source_time, key))
        if not candidates:
            return
        _, key = max(candidates)
        image = self._redis.hget(key, "image")
        if image is not None:
            self._cache[uid] = image


class AgentV3Runner(IdleCompatibleCoopDecoyRunner):
    """只向 Agent 注入公开观测与相机字节的真实像素 Runner。"""

    def __init__(self, cfg, output, runtime_root, weather, *, device="0",
                 detector_config=None, weights=None,
                 trace_max_records=_TRACE_DEFAULT_MAX_RECORDS,
                 trace_max_bytes=_TRACE_DEFAULT_MAX_BYTES, log=print):
        super().__init__(cfg, PersonalV3Agent, log=log)
        self.output = Path(output)
        self.runtime_root = Path(runtime_root).resolve()
        self.weather = str(weather)
        self.renderer = None
        self.photo_cache = None
        self.agents = {}
        self.prepared_scenario_sha256 = None
        self.device = str(device)
        self.detector_config = (
            None if detector_config is None else Path(detector_config).resolve()
        )
        self.weights = None if weights is None else Path(weights).resolve()
        self.sequence_id = f"{self.weather}-seed-{self.cfg.seed}"
        self.source_scenario_profile = _scenario_profile(Path(cfg.scenario_path))
        self._motion_audit = {}
        self.coordination_trace = CoordinationEvidenceRecorder(
            self.output,
            max_records=trace_max_records,
            max_bytes=trace_max_bytes,
        )
        self.perception_worker = None
        if not self.cfg.dry_run:
            detector_kwargs = {"device": self.device}
            if self.detector_config is not None:
                detector_kwargs["config"] = self.detector_config
            if self.weights is not None:
                detector_kwargs["weights"] = self.weights
            self.perception_worker = V3PerceptionWorker(
                detector_kwargs=detector_kwargs
            )
            self.perception_worker.wait_until_ready()

    def prepare_scenario(self):
        super().prepare_scenario()
        self._scenario_cfg.setdefault("weather", {})["type"] = self.weather
        self.cfg.weather = self.weather
        self._scenario_cfg.setdefault("simulation", {})["seed"] = self.cfg.seed
        for entity in self._scenario_cfg.get("entities", []):
            if entity.get("type") != "FixedWingUAV":
                continue
            gimbal = entity.setdefault("components", {}).setdefault(
                "gimbal_tracking", {}
            )
            gimbal.setdefault("params", {})["fov"] = PersonalV3Agent.SEARCH_FOV_DEG
        prepared = json.dumps(
            self._scenario_cfg,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        path = self.output / "prepared_scenario.json"
        path.write_text(prepared, encoding="utf-8")
        self.prepared_scenario_sha256 = hashlib.sha256(
            prepared.encode("utf-8")
        ).hexdigest()

    def _build_perception(self, uids):
        # 不构造 AccuracySimulator 或官方默认 YOLO；正式 detection 只能来自
        # PersonalV3.sensor() 对 obs.self.photo 的真实像素推理。
        if not self.cfg.dry_run:
            import redis

            for agent in self.agents.values():
                agent.set_perception_provider(
                    lambda obs, dt, worker=self.perception_worker,
                    sequence_id=self.sequence_id: submit_observation(
                        worker, obs, sequence_id=sequence_id
                    )
                )
            client = redis.Redis(
                host=self.cfg.redis_host,
                port=self.cfg.redis_port,
                socket_timeout=2,
            )
            self.photo_cache = FreshPhotoCache(redis_client=client, uids=uids)
            self.photo_cache.start()
            self.renderer = StudyRenderer(
                self.runtime_root,
                self.output,
                self.cfg.redis_host,
                self.cfg.redis_port,
                self.log,
            )
            self.renderer.start(self._scenario_cfg, uids)
        return self.photo_cache, DetectionResolver(default_detector=None)

    def make_agent_for(self, entity_type, entity_uid, world_state):
        agent = super().make_agent_for(entity_type, entity_uid, world_state)
        uid = str(entity_uid)
        self.agents[uid] = agent
        sensor = agent.sensor
        decide = agent.decide

        def recorded_sensor(obs, dt):
            result = sensor(obs, dt)
            try:
                self.coordination_trace.record_sensor(uid, agent, obs)
            except BaseException as exc:
                self.coordination_trace.capture_error(uid, "sensor", exc)
            return result

        def recorded_decide(obs, dt):
            try:
                before = _agent_evidence(agent)
            except BaseException as exc:
                before = {}
                self.coordination_trace.capture_error(uid, "before_decide", exc)
            started = time.perf_counter()
            try:
                commands = decide(obs, dt) or []
            except BaseException as exc:
                elapsed = (time.perf_counter() - started) * 1000.0
                try:
                    self.coordination_trace.record_decision(
                        uid, agent, obs, dt, (), before, elapsed, error=repr(exc)
                    )
                except BaseException as capture_exc:
                    self.coordination_trace.capture_error(uid, "decide_error", capture_exc)
                raise
            elapsed = (time.perf_counter() - started) * 1000.0
            try:
                self.coordination_trace.record_decision(
                    uid, agent, obs, dt, commands, before, elapsed
                )
            except BaseException as exc:
                self.coordination_trace.capture_error(uid, "after_decide", exc)
            return commands

        agent.sensor = recorded_sensor
        agent.decide = recorded_decide
        return agent

    def should_finish(self, agents):
        # 测评入口始终跑满命令行指定时长，阶段性短测才可相互比较。
        return False

    def _observe_scoring(self, evaluator, ws, sim_t0, destroyed, all_cmds=()):
        # 仅在裁判侧旁路记录车辆位移；Agent observation 和决策输入保持不变。
        relative_time = float(max(0.0, ws.sim_time - sim_t0))
        for vehicle_type, entities in (("target", ws.targets), ("decoy", ws.decoys)):
            for uid, entity in entities.items():
                current = (float(entity.lat), float(entity.lon))
                record = self._motion_audit.setdefault(str(uid), {
                    "type": vehicle_type,
                    "first_sim_time_s": relative_time,
                    "first_lat": current[0],
                    "first_lon": current[1],
                    "last_sim_time_s": relative_time,
                    "last_lat": current[0],
                    "last_lon": current[1],
                    "max_displacement_m": 0.0,
                })
                record["last_sim_time_s"] = relative_time
                record["last_lat"] = current[0]
                record["last_lon"] = current[1]
                record["max_displacement_m"] = max(
                    record["max_displacement_m"],
                    _horizontal_distance_m(
                        (record["first_lat"], record["first_lon"]), current
                    ),
                )
        return super()._observe_scoring(
            evaluator, ws, sim_t0, destroyed, all_cmds
        )

    def _motion_audit_summary(self):
        records = dict(sorted(self._motion_audit.items()))
        decoys = [item for item in records.values() if item["type"] == "decoy"]
        targets = [item for item in records.values() if item["type"] == "target"]
        return {
            "basis": "judge_side_world_state_not_exposed_to_agent",
            "moving_threshold_m": 1.0,
            "target_count_observed": len(targets),
            "decoy_count_observed": len(decoys),
            "targets_moved_over_threshold": sum(
                item["max_displacement_m"] > 1.0 for item in targets
            ),
            "decoys_moved_over_threshold": sum(
                item["max_displacement_m"] > 1.0 for item in decoys
            ),
            "vehicles": records,
        }

    def _close_owned_resources(self):
        errors = []
        for agent in self.agents.values():
            close = getattr(agent, "close", None)
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:
                errors.append(exc)
        if self.perception_worker is not None:
            try:
                self.perception_worker.close()
                (self.output / "perception_summary.json").write_text(
                    json.dumps(
                        self.perception_worker.stats,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    ) + "\n",
                    encoding="utf-8",
                )
            except BaseException as exc:
                errors.append(exc)
            self.perception_worker = None
        if self.renderer is not None:
            try:
                self.renderer.close()
            except BaseException as exc:
                errors.append(exc)
            self.renderer = None
        if self.photo_cache is not None:
            try:
                self.photo_cache.stop()
                close = getattr(self.photo_cache._redis, "close", None)
                if callable(close):
                    close()
            except BaseException as exc:
                errors.append(exc)
            self.photo_cache = None
        try:
            self.coordination_trace.close()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise errors[0]

    def run(self):
        previous = Path.cwd()
        result = None
        error = None
        try:
            os.chdir(self.runtime_root)
            redis_context = (
                nullcontext()
                if self.cfg.dry_run
                else RedisRuntime(
                    self.runtime_root, self.cfg.redis_host, self.cfg.redis_port
                )
            )
            with redis_context:
                with getattr(self, "visualization", nullcontext()):
                    try:
                        result = self._run_with_idle_check()
                    except BaseException as exc:
                        error = repr(exc)
                        raise
                    finally:
                        self._close_owned_resources()
            return result
        finally:
            os.chdir(previous)
            summaries = {
                uid: getattr(agent, "completion_summary", {})
                for uid, agent in self.agents.items()
            }
            payload = {
                "schema_version": 1,
                "status": "failed" if error else "completed",
                "error": error or (result or {}).get("error"),
                "evaluation": result,
                "agents": summaries,
                "weather": self.weather,
                "seed": self.cfg.seed,
                "duration_s": self.cfg.duration_s,
                "fov_deg": PersonalV3Agent.SEARCH_FOV_DEG,
                "prepared_scenario_sha256": self.prepared_scenario_sha256,
                "source_scenario": self.source_scenario_profile,
                "motion_audit": self._motion_audit_summary(),
                "coordination_evidence": self.coordination_trace.summary,
                "formal_inputs": [
                    "obs.self.photo",
                    "obs.self_pose_and_gimbal",
                    "obs.comm_inbox",
                    "obs.briefing.score_view.sim_time",
                ],
                "excluded_inputs": [
                    "ue_projected_bbox",
                    "world_target_truth",
                    "central_cross_uav_redis_bridge",
                ],
            }
            try:
                (self.output / "run.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
            except OSError:
                if error is None:
                    raise
            try:
                from .analyze_v3_coordination import analyze_run

                analyze_run(self.output)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.log(f"[coordination-audit] 离线审计失败：{exc!r}")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout", type=Path, default=DEFAULT_LAYOUT,
        help="正式入口只接受官方 competition/scenarios/coop_decoy/scenario.json",
    )
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--weather", choices=WEATHERS, default=None)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--device", default="0")
    parser.add_argument("--detector-config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--trace-max-records", type=int, default=_TRACE_DEFAULT_MAX_RECORDS,
        help="协同证据内存记录上限，默认 12000 条",
    )
    parser.add_argument(
        "--trace-max-bytes", type=int, default=_TRACE_DEFAULT_MAX_BYTES,
        help="协同证据 UTF-8 字节上限，默认 16 MiB",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualization-port", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    args.layout = args.layout.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.output = args.output.resolve()
    if not args.layout.is_file():
        parser.error(f"--layout 不存在：{args.layout}")
    if args.layout != DEFAULT_LAYOUT.resolve():
        parser.error(
            "正式 V3 入口只允许官方 coop_decoy 场景；"
            f"应为 {DEFAULT_LAYOUT.resolve()}"
        )
    try:
        source_profile = _scenario_profile(args.layout)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(f"官方场景读取失败：{exc}")
    if (
        source_profile["uav_count"] != 3
        or source_profile["target_count"] != 3
        or source_profile["decoy_count"] != 15
        or not source_profile["all_targets_commanded_to_move"]
        or not source_profile["all_decoys_commanded_to_move"]
    ):
        parser.error(f"官方场景实体或运动契约不满足：{source_profile}")
    if args.weather is None:
        args.weather = source_profile["weather"]
    if args.weather not in WEATHERS:
        parser.error(f"官方场景天气不受支持：{args.weather}")
    if not args.dry_run and not (args.runtime_root / "opensim-sim.exe").is_file():
        parser.error(f"运行底座缺少 opensim-sim.exe：{args.runtime_root}")
    if not _is_relative_to(args.output, OUTPUT_ROOT.resolve()):
        parser.error(f"--output 必须位于 {OUTPUT_ROOT.resolve()} 下")
    if args.output.exists():
        parser.error(f"输出目录已存在，拒绝覆盖：{args.output}")
    if args.duration <= 0:
        parser.error("--duration 必须大于 0")
    if args.seed < 0:
        parser.error("--seed 不能为负数")
    if args.trace_max_records <= 0:
        parser.error("--trace-max-records 必须大于 0")
    if args.trace_max_bytes <= 0:
        parser.error("--trace-max-bytes 必须大于 0")
    args.output.mkdir(parents=True)

    os.environ["HF2026_V3_DEVICE"] = str(args.device)
    if args.detector_config is not None:
        os.environ["HF2026_V3_DETECTOR_CONFIG"] = str(
            args.detector_config.resolve()
        )
    if args.weights is not None:
        os.environ["HF2026_V3_WEIGHTS"] = str(args.weights.resolve())

    if args.dry_run:
        preflight = {
            "schema_version": 1,
            "status": "preflight_completed",
            "starts_simulation": False,
            "layout": str(args.layout),
            "runtime_root": str(args.runtime_root),
            "weather": args.weather,
            "seed": args.seed,
            "duration_s": args.duration,
            "fov_deg": PersonalV3Agent.SEARCH_FOV_DEG,
            "coordination_trace": {
                "starts_simulation": False,
                "sample_period_s": _TRACE_SAMPLE_PERIOD_S,
                "max_records": args.trace_max_records,
                "max_bytes": args.trace_max_bytes,
            },
            "source_scenario": source_profile,
        }
        (args.output / "run.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=PROJECT_ROOT,
                        text=True,
                    ).strip(),
                    "argv": os.sys.argv,
                    "time_basis": (
                        "score_view_sim_time_not_verified_exposure_time_or_pose"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(preflight, ensure_ascii=False), flush=True)
        return 0

    cfg = ScenarioConfig(
        scenario_name="coop_decoy",
        scenario_path=str(args.layout),
        duration_s=args.duration,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        output_dir=str(args.output),
        sim_binary=str(args.runtime_root / "opensim-sim.exe"),
        start_sim_flag=not args.dry_run,
        dry_run=args.dry_run,
        quiet=False,
        seed=args.seed,
        run_mode="eval",
        photo_mode="on",
        weather=args.weather,
    )
    with (args.output / "run.log").open(
        "x", encoding="utf-8", buffering=1
    ) as stream:
        def log(message):
            stream.write(str(message) + "\n")
            print(message, flush=True)

        runner = AgentV3Runner(
            cfg,
            args.output,
            args.runtime_root,
            args.weather,
            device=args.device,
            detector_config=args.detector_config,
            weights=args.weights,
            trace_max_records=args.trace_max_records,
            trace_max_bytes=args.trace_max_bytes,
            log=log,
        )
        if args.visualize and not args.dry_run:
            from .web_visualization import WebVisualization

            runner.visualization = WebVisualization(
                args.redis_host,
                args.redis_port,
                args.output,
                args.visualization_port,
            )
        result = runner.run()
        if result and result.get("error"):
            raise RuntimeError(result["error"])

    metadata = {
        "schema_version": 1,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "argv": os.sys.argv,
        "time_basis": "score_view_sim_time_not_verified_exposure_time_or_pose",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
