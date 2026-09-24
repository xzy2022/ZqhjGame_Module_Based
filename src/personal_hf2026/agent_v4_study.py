# 修改时间：2026-09-24。
# 修改目的：避免启动首拍尚无 score_view 时紧凑记录器触发异常。
# 修改内容：记录器取 Agent 已使用的本机累计时间。
# 修改时间：2026-09-24。
# 修改目的：在官方 Runner 生命周期中运行三架独立的 PersonalV4Agent。
# 修改内容：接入共享单帧感知、显式诊断、裁判侧审计和紧凑 Agent4 日志。
"""PersonalV4 单轮真实像素测评入口。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess

from competition.sdk.core.perception import DetectionResolver
from competition.sdk.core.runner import ScenarioConfig

from .agent_v3_study import DEFAULT_LAYOUT, WEATHERS, FreshPhotoCache, _scenario_profile
from .dropout_capture import StudyRenderer
from .paths import PROJECT_ROOT, RUNTIME_ROOT
from .personal_v4 import PersonalV4Agent
from .redis_runtime import RedisRuntime
from .sdk_compat import IdleCompatibleCoopDecoyRunner
from .v4_logging import V4Trace, V4VisualLog
from .v4_perception import V4PerceptionWorker, VisionDiagnosticV4, submit_observation


def _vision_mode(value):
    try:
        return VisionDiagnosticV4(value).mode
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


class AgentV4Runner(IdleCompatibleCoopDecoyRunner):
    def __init__(self, cfg, output, runtime_root, weather, *, device="0",
                 detector_config=None, weights=None, save_images=False,
                 detailed_log=False, vision_diagnostic="000", log=print):
        super().__init__(cfg, PersonalV4Agent, log=log)
        self.output = Path(output)
        self.runtime_root = Path(runtime_root).resolve()
        self.weather = str(weather)
        self.device = str(device)
        self.detector_config = detector_config
        self.weights = weights
        self.agents = {}
        self.photo_cache = None
        self.renderer = None
        self.prepared_scenario_sha256 = None
        self.source_scenario_profile = _scenario_profile(Path(cfg.scenario_path))
        self.vision_diagnostic = VisionDiagnosticV4(vision_diagnostic)
        self.visual_log = (V4VisualLog(self.output, save_images=save_images,
                                       detailed_log=detailed_log)
                           if save_images or detailed_log else None)
        self.trace = V4Trace(self.output, detailed_log=detailed_log)
        self.perception_worker = None
        if not cfg.dry_run:
            self.perception_worker = V4PerceptionWorker(
                device=device, config=detector_config, weights=weights,
                diagnostic=self.vision_diagnostic)
            self.perception_worker.wait_until_ready()

    def prepare_scenario(self):
        super().prepare_scenario()
        self._scenario_cfg.setdefault("weather", {})["type"] = self.weather
        self.cfg.weather = self.weather
        self._scenario_cfg.setdefault("simulation", {})["seed"] = self.cfg.seed
        for entity in self._scenario_cfg.get("entities", []):
            if entity.get("type") == "FixedWingUAV":
                gimbal = entity.setdefault("components", {}).setdefault("gimbal_tracking", {})
                gimbal.setdefault("params", {})["fov"] = PersonalV4Agent.SEARCH_FOV_DEG
        prepared = json.dumps(self._scenario_cfg, ensure_ascii=False, indent=2,
                              sort_keys=True, allow_nan=False) + "\n"
        (self.output / "prepared_scenario.json").write_text(prepared, encoding="utf-8")
        self.prepared_scenario_sha256 = hashlib.sha256(prepared.encode()).hexdigest()

    def _build_perception(self, uids):
        if not self.cfg.dry_run:
            import redis

            for uid, agent in self.agents.items():
                agent.set_perception_provider(
                    lambda obs, dt, uid=uid: self._submit(uid, obs))
            client = redis.Redis(host=self.cfg.redis_host, port=self.cfg.redis_port,
                                 socket_timeout=2)
            self.photo_cache = FreshPhotoCache(
                redis_client=client, uids=uids,
                capture_metadata=bool(self.visual_log) or self.vision_diagnostic.enabled)
            self.photo_cache.start()
            self.renderer = StudyRenderer(self.runtime_root, self.output,
                                          self.cfg.redis_host, self.cfg.redis_port, self.log)
            self.renderer.start(self._scenario_cfg, uids)
        return self.photo_cache, DetectionResolver(default_detector=None)

    def _submit(self, uid, obs):
        metadata = None
        if self.vision_diagnostic.enabled and self.photo_cache is not None:
            metadata = self.photo_cache.metadata_for(uid, obs.self.photo)
        return submit_observation(self.perception_worker, obs,
                                  diagnostic_metadata=metadata)

    def make_agent_for(self, entity_type, entity_uid, world_state):
        agent = super().make_agent_for(entity_type, entity_uid, world_state)
        uid = str(entity_uid)
        self.agents[uid] = agent
        original_sensor, original_decide = agent.sensor, agent.decide

        def sensor(obs, dt):
            result = original_sensor(obs, dt)
            if self.visual_log is not None:
                snapshot = agent._last_snapshot
                if snapshot is not None:
                    entry = self.photo_cache.frame_for(uid, snapshot.frame_id)
                    if entry is not None:
                        self.visual_log.record_processed_frame(uid, entry[0], entry[1], snapshot)
                    else:
                        self.visual_log.record_prediction(uid, snapshot)
            return result

        def decide(obs, dt):
            commands = original_decide(obs, dt) or []
            self.trace.record(uid, agent, agent._t, commands)
            return commands

        agent.sensor, agent.decide = sensor, decide
        return agent

    def should_finish(self, agents):
        return False

    def _observe_scoring(self, evaluator, ws, sim_t0, destroyed, all_cmds=()):
        if self.visual_log is not None:
            self.visual_log.record_truth(max(0.0, ws.sim_time - sim_t0), ws)
        return super()._observe_scoring(evaluator, ws, sim_t0, destroyed, all_cmds)

    def _close_resources(self):
        if self.perception_worker is not None:
            self.perception_worker.close()
            (self.output / "perception_summary.json").write_text(
                json.dumps(self.perception_worker.stats, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        if self.renderer is not None:
            self.renderer.close()
        if self.photo_cache is not None:
            self.photo_cache.stop()
            self.photo_cache._redis.close()
        if self.visual_log is not None:
            self.visual_log.close()
        self.trace.close()

    def run(self):
        previous = Path.cwd()
        result = None
        error = None
        try:
            os.chdir(self.runtime_root)
            redis_context = (nullcontext() if self.cfg.dry_run else
                             RedisRuntime(self.runtime_root, self.cfg.redis_host,
                                          self.cfg.redis_port))
            with redis_context:
                with getattr(self, "visualization", nullcontext()):
                    try:
                        result = self._run_with_idle_check()
                    except BaseException as exc:
                        error = repr(exc)
                        raise
                    finally:
                        self._close_resources()
            return result
        finally:
            os.chdir(previous)
            payload = {
                "schema_version": 1, "status": "failed" if error else "completed",
                "error": error or (result or {}).get("error"), "evaluation": result,
                "agents": {uid: agent.completion_summary for uid, agent in self.agents.items()},
                "weather": self.weather, "seed": self.cfg.seed,
                "duration_s": self.cfg.duration_s, "fov_deg": PersonalV4Agent.SEARCH_FOV_DEG,
                "prepared_scenario_sha256": self.prepared_scenario_sha256,
                "source_scenario": self.source_scenario_profile,
                "visual_logging": (self.visual_log.summary if self.visual_log else
                                   {"enabled": False}),
                "agent_v4_trace": self.trace.summary,
                "vision_diagnostic": self.vision_diagnostic.summary,
                "formal_inputs": ["obs.self.photo", "obs.self_pose_and_gimbal",
                                  "obs.comm_inbox", "obs.briefing.score_view.sim_time"],
                "diagnostic_oracle_inputs": (["redis_sync_camera_same_frame_ue_projected_boxes"]
                                             if self.vision_diagnostic.enabled else []),
            }
            (self.output / "run.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8")
            try:
                from .analyze_v4 import analyze_run
                analyze_run(self.output)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.log(f"[agent-v4] 离线审计失败：{exc!r}")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
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
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--detailed-log", action="store_true")
    parser.add_argument("--vision-diagnostic", type=_vision_mode, default="000")
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
    if args.layout != DEFAULT_LAYOUT.resolve() or not args.layout.is_file():
        parser.error(f"正式入口只允许官方 coop_decoy 场景：{DEFAULT_LAYOUT}")
    profile = _scenario_profile(args.layout)
    if (profile["uav_count"] != 3 or profile["target_count"] != 3
            or profile["decoy_count"] != 15
            or not profile["all_targets_commanded_to_move"]
            or not profile["all_decoys_commanded_to_move"]):
        parser.error("官方场景实体或运动契约不满足")
    if args.weather is None:
        args.weather = profile["weather"]
    if args.output.exists():
        parser.error(f"输出目录已存在：{args.output}")
    if args.duration <= 0 or args.seed < 0:
        parser.error("duration 必须大于 0 且 seed 不能为负")
    if not args.dry_run and not (args.runtime_root / "opensim-sim.exe").is_file():
        parser.error(f"运行底座缺少 opensim-sim.exe：{args.runtime_root}")
    args.output.mkdir(parents=True)
    if args.dry_run:
        payload = {"status": "preflight_completed", "starts_simulation": False,
                   "layout": str(args.layout), "runtime_root": str(args.runtime_root),
                   "vision_diagnostic": VisionDiagnosticV4(args.vision_diagnostic).summary}
        (args.output / "run.json").write_text(json.dumps(payload, ensure_ascii=False,
                                                          indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    cfg = ScenarioConfig(
        scenario_name="coop_decoy", scenario_path=str(args.layout),
        duration_s=args.duration, redis_host=args.redis_host, redis_port=args.redis_port,
        output_dir=str(args.output), sim_binary=str(args.runtime_root / "opensim-sim.exe"),
        start_sim_flag=True, dry_run=False, quiet=False, seed=args.seed,
        run_mode="eval", photo_mode="on", weather=args.weather)
    with (args.output / "run.log").open("x", encoding="utf-8", buffering=1) as stream:
        def log(message):
            stream.write(str(message) + "\n")
            print(message, flush=True)

        runner = AgentV4Runner(cfg, args.output, args.runtime_root, args.weather,
                               device=args.device, detector_config=args.detector_config,
                               weights=args.weights, save_images=args.save_images,
                               detailed_log=args.detailed_log,
                               vision_diagnostic=args.vision_diagnostic, log=log)
        if args.visualize:
            from .web_visualization import WebVisualization
            runner.visualization = WebVisualization(args.redis_host, args.redis_port,
                                                     args.output, args.visualization_port)
        result = runner.run()
        if result and result.get("error"):
            raise RuntimeError(result["error"])
    metadata = {"schema_version": 1,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                      cwd=PROJECT_ROOT, text=True).strip(),
                "argv": os.sys.argv,
                "time_basis": "score_view_sim_time_not_verified_exposure_time_or_pose"}
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
