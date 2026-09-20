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
import os
from pathlib import Path
import subprocess
import time
import re

from competition.sdk.core.perception import DetectionResolver, PhotoCache
from competition.sdk.core.runner import ScenarioConfig

from .dropout_capture import StudyRenderer
from .paths import OUTPUT_ROOT, PROJECT_ROOT, RUNTIME_ROOT, SCENARIO_ROOT
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
DEFAULT_LAYOUT = SCENARIO_ROOT / "static-decoys.json"
_FRAME_NUMBER = re.compile(r"frame:(\d+)$")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


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
                 detector_config=None, weights=None, log=print):
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
        self.agents[str(entity_uid)] = agent
        return agent

    def should_finish(self, agents):
        # 测评入口始终跑满命令行指定时长，阶段性短测才可相互比较。
        return False

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


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--weather", choices=WEATHERS, default="Clear_Skies")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--device", default="0")
    parser.add_argument("--detector-config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
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
