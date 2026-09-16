# 修改时间：2026-09-16。
# 修改目的：为双机图像定位提供可控的 UE 短测入口。
# 修改内容：接线专属 UE、Redis 新帧、协同上下文、输出约束和幂等收尾。
"""20 秒 UE 短测：仅在协同阶段评估双机 bbox 中心射线定位。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import redis

from competition.sdk.core.runner import ScenarioConfig

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
        self._error = None

    def _read_latest(self, uid):
        keys = list(self.client.scan_iter(match=f"sync_camera:{uid}:frame:*", count=100))
        if not keys:
            return None
        key = max(keys, key=lambda item: int(item.rsplit(b":", 1)[1]))
        image, source_time, detections = self.client.hmget(
            key, "image", "sim_time", "detections")
        if not image or source_time is None:
            return None
        frame_no = int(key.rsplit(b":", 1)[1])
        return {
            "frame_no": frame_no,
            "source_sim_time": float(source_time),
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
                        self.evaluator.observe_frame(uid, frame, context)
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

    def __init__(self, cfg, output, runtime_root, evaluator):
        super().__init__(cfg, PersonalV1Agent, log=lambda _message: None)
        self.output = Path(output)
        self.runtime_root = Path(runtime_root)
        self.evaluator = evaluator
        self.renderer = None
        self.bridge = None
        self.world_context = {}
        self.live_summary = None
        self._closed = False

    def run(self):
        # 旁路和 UE 必须在本轮 Redis 退出前收尾，否则无法发送 UE shutdown。
        previous = Path.cwd()
        try:
            os.chdir(self.runtime_root)
            with RedisRuntime(
                    self.runtime_root, self.cfg.redis_host, self.cfg.redis_port):
                try:
                    return self._run_with_idle_check()
                finally:
                    self.close()
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
                    "time_alignment": "nearest_current_state_unverified",
                    "aircraft_attitude": world.get("aircraft_attitude"),
                    "truth_by_target_id": world.get("truth_by_target_id", {}),
                }
                self.bridge.update_context(entity_uid, context)
                self.bridge.ensure_healthy()
            return commands

        agent.decide = observed_decide
        return agent

    def should_finish(self, agents):
        # 功能短测总是跑满 duration；未进入协同由摘要如实记录。
        return False

    def close(self):
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
                result = resource.close()
                if name == "evaluator":
                    self.live_summary = result
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
    if args.max_records <= 0:
        parser.error("--max-records 必须大于 0")

    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running",
        "mode": "paired_geolocation_live_ue_short_test",
        "layout": str(layout),
        "duration_s": args.duration,
        "seed": args.seed,
        "angle_threshold_deg": args.angle_threshold_deg,
        "max_records": args.max_records,
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
            },
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
