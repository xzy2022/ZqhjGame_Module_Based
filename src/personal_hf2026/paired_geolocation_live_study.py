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
from collections import Counter
import json
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


class RedisFrameBridge:
    """读取每架无人机的最新唯一帧，不保存图片和全量帧日志。"""

    def __init__(self, uids, host, port, evaluator):
        self.uids = tuple(uids)
        self.client = redis.Redis(
            host=host,
            port=port,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
        )
        self.evaluator = evaluator
        self.contexts = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._seen = {}
        self._report_candidates = []
        self._error = None

    def _read_latest(self, uid):
        keys = list(self.client.scan_iter(match=f"sync_camera:{uid}:frame:*", count=100))
        if not keys:
            return None
        key = max(keys, key=lambda item: int(item.rsplit(b":", 1)[1]))
        image, source_time, detections = self.client.hmget(
            key, "image", "sim_time", "detections")
        redis_read_unix_s = time.time()
        redis_read_monotonic_s = time.monotonic()
        if not image or source_time is None:
            return None
        frame_no = int(key.rsplit(b":", 1)[1])
        return {
            "frame_no": frame_no,
            "source_sim_time": float(source_time),
            "redis_read_unix_s": redis_read_unix_s,
            "redis_read_monotonic_s": redis_read_monotonic_s,
            "image": image,
            "detections": json.loads(detections) if detections else [],
        }

    def start(self):
        # 记住启动前的残留帧，本轮只交付之后到达的新帧。
        for uid in self.uids:
            frame = self._read_latest(uid)
            if frame is not None:
                self._seen[uid] = (frame["frame_no"], frame["source_sim_time"])
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
                    frame = self._read_latest(uid)
                    if frame is None:
                        continue
                    signature = (frame["frame_no"], frame["source_sim_time"])
                    if signature == self._seen.get(uid):
                        continue
                    self._seen[uid] = signature
                    with self._lock:
                        context = self.contexts.get(uid)
                    if context is not None:
                        context = dict(context)
                        world_time = context.get("world_sim_time")
                        delta = (float(world_time) - frame["source_sim_time"]
                                 if world_time is not None else None)
                        context["pose_time_delta_s"] = delta
                        context["world_truth_time_delta_s"] = delta
                        records = self.evaluator.observe_frame(uid, frame, context)
                        if records:
                            with self._lock:
                                self._report_candidates.extend(records)
            except BaseException as exc:
                self._error = exc
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

    def __init__(self, cfg, output, runtime_root, evaluator):
        super().__init__(cfg, PersonalV1Agent, log=lambda _message: None)
        self.output = Path(output)
        self.runtime_root = Path(runtime_root)
        self.evaluator = evaluator
        self.renderer = None
        self.bridge = None
        self.world_context = {}
        self.live_summary = None
        self.report_counts = Counter()
        self._pending_reports = {}
        self._last_report_sim_time = {}
        self.phase_fov_counts = Counter()
        self._report_written_bytes = 0
        self._report_stream = (
            self.output / "paired_geolocation_reports.jsonl").open(
                "x", encoding="utf-8", buffering=1)
        self._closed = False

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
            uids, self.cfg.redis_host, self.cfg.redis_port, self.evaluator)
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
        if aircraft is not None:
            attitude = (aircraft.raw.get("platform", {}) or {}).get("attitude")
        self.world_context[uid] = {
            "world_sim_time": ws.sim_time,
            "aircraft_attitude": attitude,
            # 真值只交给旁路评分，不进入 Agent 或定位算法输入。
            "truth_by_target_id": vehicle_truth,
        }
        return super()._extract_truth(ws, uid)

    def make_agent_for(self, entity_type, entity_uid, world_state):
        agent = super().make_agent_for(entity_type, entity_uid, world_state)
        decide = agent.decide

        def observed_decide(obs, dt):
            commands = decide(obs, dt)
            if self.bridge is not None:
                own = obs.self
                coordinator = agent._coordinator
                self.phase_fov_counts[
                    f"{coordinator.phase}|{float(own.gimbal_fov_deg):.6g}"] += 1
                world = self.world_context.get(entity_uid, {})
                context = {
                    "active": coordinator.phase == coordinator.ACTIVE,
                    "phase": coordinator.phase,
                    "session_id": _session_value(coordinator.current_session),
                    "role": coordinator.role,
                    "partner_uid": coordinator.partner_uid,
                    "pose": {
                        "lat": own.lat,
                        "lon": own.lon,
                        "alt": own.alt,
                        "heading_deg": own.heading_deg,
                        "gimbal_pan": own.gimbal_pan,
                        "gimbal_tilt": own.gimbal_tilt,
                        "gimbal_fov_deg": own.gimbal_fov_deg,
                    },
                    "world_sim_time": world.get("world_sim_time"),
                    "pose_sample_sim_time": world.get("world_sim_time"),
                    "truth_sample_sim_time": world.get("world_sim_time"),
                    "context_published_unix_s": time.time(),
                    "context_published_monotonic_s": time.monotonic(),
                    "time_alignment": "nearest_current_state_unverified",
                    "aircraft_attitude": world.get("aircraft_attitude"),
                    "truth_by_target_id": world.get("truth_by_target_id", {}),
                }
                self.bridge.update_context(entity_uid, context)
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
                ("renderer", self.renderer)):
            if resource is None:
                continue
            try:
                result = (resource.close(status=status, error=error)
                          if name == "evaluator" else resource.close())
                if name == "evaluator":
                    self.live_summary = result
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
    if args.max_records <= 0 or args.max_output_bytes <= 0:
        parser.error("--max-records 和 --max-output-bytes 必须大于 0")

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
        "time_alignment": "nearest_current_state_unverified",
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
    runner = PairedGeolocationLiveRunner(cfg, output, runtime_root, evaluator)
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
