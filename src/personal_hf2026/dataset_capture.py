# 修改时间：2026-09-14
# 修改目的：让个人实验脱离官方仓库的后续修改并支持独立运行。
# 修改内容：统一模块、SDK、运行资源及输出路径并保留实验行为。
# 修改时间：2026-09-13
# 修改目的：按独立天气和真值身份连续采集原图，并支持双盘调度正常中止。
# 修改内容：新增专用单轮入口、准备模式、身份审计、资源与覆盖摘要及完整收尾。
"""多天气单轮原图采集；真值身份仅作采集辅助，不代表正式视觉能力。"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from .paths import PROJECT_ROOT, SIM_ROOT, RUNTIME_ROOT, OUTPUT_ROOT, SCENARIO_ROOT
import re
import shutil
import subprocess
import time

import psutil

from competition.sdk.core.perception import DetectionResolver
from competition.sdk.core.runner import ScenarioConfig
from competition.sdk.core.scenario_randomizer import randomize_scenario
from .capture_dataset import build_index, rows
from .control_test_runner import MultiTargetIdealDetector, _ground_distance_m
from .dropout_capture import StudyRenderer
from .oracle_identity import OracleIdentityAgent, identify_position, oracle_detections
from .timing_study import StudyRunner
from .visual_shadow_study import TimedPhotoCache


ROOT = PROJECT_ROOT
WEATHERS = ("Clear_Skies", "Partly_Cloudy", "Rain", "Foggy", "Snow_Light", "Sand_Dust_Calm")
GIB = 1073741824


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class DatasetCaptureRunner(StudyRunner):
    def __init__(self, cfg, output, runtime_root, log, stop_file=None, min_free_gib=20):
        super().__init__(cfg, OracleIdentityAgent, output, log)
        self.runtime_root = Path(runtime_root)
        self.requested_weather = cfg.weather
        self.stop_file = Path(stop_file) if stop_file else None
        self.min_free_bytes = min_free_gib * GIB
        self.stop_reason = None
        self.camera = None
        self.identity_by_uid = {}
        self.identities = (output / "identity.jsonl").open("w", encoding="utf-8", buffering=1)
        self.resources = (output / "resources.jsonl").open("w", encoding="utf-8", buffering=1)
        (output / "dataset").mkdir()
        self.deliveries = (output / "dataset/deliveries.jsonl").open("w", encoding="utf-8", buffering=1)
        self.delivered_frames = set()
        self.next_resource_check = 0
        self.image_progress = {}
        self.sim_progress = (None, time.monotonic())
        self.actual_duration_s = 0.0
        self.identity_counts = Counter()
        self.done_at = {}
        self.bound_identities = {}
        self.astar = {}
        self.scenario_prepared = False

    def prepare_scenario(self):
        if self.scenario_prepared:
            return
        super().prepare_scenario()
        # 两个父类沿用晴天默认值；专用入口在最终副本显式恢复请求天气。
        self._scenario_cfg.setdefault("weather", {})["type"] = self.requested_weather
        self.cfg.weather = self.requested_weather
        for ent in self._scenario_cfg["entities"]:
            if ent["type"] == "FixedWingUAV":
                ent.setdefault("components", {}).setdefault("gimbal_tracking", {}).setdefault("params", {})["fov"] = OracleIdentityAgent.SEARCH_FOV_DEG
        self.scenario_prepared = True
        self.log(f"[dataset] prepared weather={self.cfg.weather}; oracle_identity; ideal_positions")

    def materialize_scenario(self):
        if self.cfg.seed > 0:
            path = randomize_scenario(self.cfg.scenario_path, self.cfg.seed,
                                      self.scenario_name, out_dir=str(self.output))
            self._scenario_cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        self.prepare_scenario()
        path = self.output / "scenario_coop_decoy_prepared.json"
        write_json(path, self._scenario_cfg)
        self.cfg.scenario_path = str(path)

    def _start_engine(self):
        # 已在启动前生成并核查最终副本，避免父入口再次随机化或吞掉准备错误。
        from competition.sdk._vendored.sim_runner import start_sim
        return start_sim(self.cfg.sim_binary, self.cfg.scenario_path, log=self.log,
                         redis_host=self.cfg.redis_host, redis_port=self.cfg.redis_port,
                         stderr_file=str(self.output / "engine.log"))

    def _build_perception(self, uids):
        self.camera = TimedPhotoCache(uids, self.output, self.cfg.redis_host, self.cfg.redis_port, True)
        self.renderer = StudyRenderer(self.runtime_root, self.output, self.cfg.redis_host, self.cfg.redis_port, self.log)
        self.renderer.start(self._scenario_cfg, uids)
        self.camera.start()
        self.image_progress = {uid: (0, time.monotonic()) for uid in uids}
        self.sim_progress = (None, time.monotonic())
        return self.camera, DetectionResolver(default_detector=MultiTargetIdealDetector())

    def _extract_truth(self, ws, uid):
        self.world_times[uid] = ws.sim_time
        detections, audit = oracle_detections(ws, uid)
        self.identity_by_uid[uid] = audit
        return detections

    def make_agent_for(self, entity_type, entity_uid, world_state):
        agent = super().make_agent_for(entity_type, entity_uid, world_state)
        recorded = agent.decide

        def decide(obs, dt):
            packet = self.camera.delivered.get(entity_uid) if self.camera else None
            if packet and obs.briefing.score_view is not None:
                signature = (entity_uid, packet["frame_no"], packet["source_sim_time"])
                if signature not in self.delivered_frames:
                    agent_t = obs.briefing.score_view.sim_time
                    offset = self.world_times[entity_uid] - agent_t
                    self.deliveries.write(json.dumps(dict(uid=entity_uid, frame_no=packet["frame_no"],
                        source_sim_time=packet["source_sim_time"], source_t=packet["source_sim_time"] - offset,
                        observe_t=agent_t, observed_unix_s=time.time(),
                        frame_age_s=self.world_times[entity_uid] - packet["source_sim_time"])) + "\n")
                    self.delivered_frames.add(signature)
            try:
                commands = recorded(obs, dt)
            except Exception:
                self.stop_reason = "agent_error"
                raise
            audit = self.identity_by_uid.get(entity_uid, [])
            current = agent._track.position if abs(agent._track.last_seen - agent._t) < 1e-6 else None
            known = [(r["vehicle_id"], r["identity"], r["forwarded"]["target_lat"], r["forwarded"]["target_lon"])
                     for r in audit if r["vehicle_id"] and r["forwarded"]["detected"]]
            known = list(dict.fromkeys(known))
            accepted_uid, accepted_kind = identify_position(current, known, 0.001) if current else (None, None)
            error = current is not None and accepted_kind != "TargetVehicle"
            binding = (entity_uid, agent._track.epoch)
            previous_identity = self.bound_identities.get(binding)
            misbinding = bool(accepted_uid and previous_identity and accepted_uid != previous_identity)
            if accepted_uid:
                self.bound_identities[binding] = accepted_uid
            if misbinding:
                self.identity_counts["identity_misbindings"] += 1
                self.stop_reason = "identity_misbinding"
            if error:
                self.identity_counts["acceptance_errors"] += 1
                self.stop_reason = "identity_acceptance_error"
            self.identity_counts["observations"] += 1
            self.identity_counts["unknown_detections"] += sum(r["identity"] == "unknown" and r["forwarded"]["detected"] for r in audit)
            self.identity_counts["native_decoy_primary"] += bool(audit and audit[0]["identity"] == "DecoyVehicle")
            self.identities.write(json.dumps(dict(uid=entity_uid, t=agent._t,
                sim_time=self.world_times.get(entity_uid), state=agent._state,
                role=agent._coordinator.role, observations=audit,
                accepted_position=current, accepted_vehicle_id=accepted_uid,
                accepted_identity=accepted_kind, acceptance_error=error,
                previous_vehicle_id=previous_identity, identity_misbinding=misbinding,
                local_track_epoch=agent._track.epoch, primary_matches=agent._gimbal_lock.primary_matches)) + "\n")
            if agent.done_at is not None:
                self.done_at.setdefault(entity_uid, agent.done_at)
            return commands

        agent.decide = decide
        return agent

    def _observe_scoring(self, evaluator, ws, sim_t0, destroyed, all_cmds=()):
        self.actual_duration_s = max(0.0, ws.sim_time - sim_t0)
        super()._observe_scoring(evaluator, ws, sim_t0, destroyed, all_cmds)

    def should_finish(self, agents):
        super().should_finish(agents)
        # DONE 只记录，不提前结束完整采集；外部停止和空间保护都走正常收尾。
        if self.stop_file and self.stop_file.exists():
            self.stop_reason = self.stop_reason or "stop_file"
        if self.camera and self.camera.error_count:
            self.stop_reason = self.stop_reason or "camera_error"
        now = time.monotonic()
        if self.sim_progress[0] != self.actual_duration_s:
            self.sim_progress = (self.actual_duration_s, now)
        elif now - self.sim_progress[1] > 30.0:
            self.stop_reason = self.stop_reason or "no_simulation_progress"
        if now >= self.next_resource_check:
            self.next_resource_check = now + 5.0
            free = shutil.disk_usage(self.output).free
            if free <= self.min_free_bytes + 0.25 * GIB:
                self.stop_reason = self.stop_reason or "minimum_free_space"
            process = psutil.Process()
            memory = []
            for item in [process, *process.children(recursive=True)]:
                try:
                    memory.append(dict(pid=item.pid, name=item.name(), rss_bytes=item.memory_info().rss))
                except psutil.Error:
                    pass
            try:
                redis_memory = self.camera.redis.info("memory").get("used_memory") if self.camera else None
            except Exception:
                redis_memory = None
                self.stop_reason = self.stop_reason or "redis_disconnected"
            self.resources.write(json.dumps(dict(unix_s=time.time(), t=self.actual_duration_s,
                free_bytes=free, processes=memory, redis_used_memory_bytes=redis_memory)) + "\n")
            if self.camera and self.camera.dataset:
                received_counts = Counter(key[0] for key in self.camera.dataset.seen.copy())
                for uid, (count, last_change) in self.image_progress.items():
                    current_count = received_counts[uid]
                    if current_count != count:
                        self.image_progress[uid] = (current_count, now)
                    elif now - last_change > 30.0:
                        self.stop_reason = self.stop_reason or "no_new_images"
        return self.stop_reason is not None

    def should_finish_when_idle(self, agents):
        return self.should_finish(agents)

    def close(self):
        try:
            if self.camera:
                self.camera.stop()
        finally:
            try:
                if self.renderer:
                    self.renderer.close()
            finally:
                for stream in (self.trace, self.judge, self.identities, self.resources, self.deliveries):
                    stream.close()


def directory_bytes(output):
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def capture_statistics(output, done_at, astar):
    """仅作采集覆盖和间隔统计；投影框不声称人工确认可见或无遮挡。"""
    times, after_done = defaultdict(list), Counter()
    vehicle_ids, classes, events = Counter(), Counter(), Counter()
    last_seen, motion, fovs = {}, {}, Counter()
    image_bytes = 0
    missing_images = []
    verified_images = set()
    offset = None
    for row in rows(output / "judge.jsonl"):
        offset = row["sim_time"] - row["t"] if offset is None else offset
        for collection, kind in (("world_targets", "TargetVehicle"), ("world_decoys", "DecoyVehicle")):
            for uid, value in row[collection].items():
                pos = (value["lat"], value["lon"])
                item = motion.setdefault(uid, dict(identity=kind, first_position=pos, last_position=pos,
                    path_distance_m=0.0, max_displacement_m=0.0, stationary_observed_s=0.0,
                    previous_t=row["t"], astar=astar.get(uid, "unverified")))
                delta = _ground_distance_m(*item["last_position"], *pos)
                item["path_distance_m"] += delta
                item["max_displacement_m"] = max(item["max_displacement_m"], _ground_distance_m(*item["first_position"], *pos))
                if delta < 0.05:
                    item["stationary_observed_s"] += max(0.0, row["t"] - item["previous_t"])
                item.update(last_position=pos, previous_t=row["t"])
    for frame in rows(output / "dataset/frames.jsonl"):
        uid, source = frame["uid"], frame["source_sim_time"]
        times[uid].append(source)
        image_bytes += frame["image_bytes"]
        image_path = output / frame["image_path"]
        if not image_path.is_file() or image_path.stat().st_size != frame["image_bytes"]:
            missing_images.append(frame["image_path"])
        elif uid not in verified_images:
            from PIL import Image
            with Image.open(image_path) as image:
                image.verify()
            verified_images.add(uid)
        after_done[uid] += uid in done_at and offset is not None and source - offset >= done_at[uid]
        for obj in frame["ue_projected_objects"]:
            key = str(obj["target_id"])
            vehicle_ids[key] += 1
            classes[str(obj["target_type"])] += 1
            event_key = (uid, key)
            if source - last_seen.get(event_key, -1e30) > 1.0:
                events[key] += 1
            last_seen[event_key] = source
    for row in rows(output / "observations.jsonl"):
        fovs[str(row["self"]["gimbal_fov_deg"])] += 1
    by_uid = {}
    for uid, values in times.items():
        values.sort()
        gaps = sorted(b - a for a, b in zip(values, values[1:]))
        by_uid[uid] = dict(frames=len(values), interval_p50_s=gaps[len(gaps) // 2] if gaps else None,
            interval_p95_s=gaps[min(len(gaps) - 1, int(len(gaps) * .95))] if gaps else None,
            max_interval_s=max(gaps) if gaps else None,
            interruptions=[dict(start_source_s=a, end_source_s=b, duration_s=b-a)
                           for a, b in zip(values, values[1:]) if b-a > 1.5],
            first_source_s=values[0], last_source_s=values[-1],
            done_at_s=done_at.get(uid), frames_after_done=after_done[uid],
            fraction_after_done=after_done[uid] / len(values))
    return dict(image_bytes=image_bytes, frames_by_uid=by_uid, projected_vehicle_ids=vehicle_ids,
                projected_classes=classes, projected_visible_events=events,
                actual_fov_counts=fovs, vehicle_motion=motion, visibility="unverified",
                missing_images=missing_images, verified_representative_uids=sorted(verified_images))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", default=RUNTIME_ROOT, type=Path)
    p.add_argument("--weather", choices=WEATHERS, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--duration", type=int, choices=(20, 200), default=200)
    p.add_argument("--fov", type=float, default=48.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layout", type=Path, default=ROOT / "configs/scenarios/coop_decoy/scenario.json")
    p.add_argument("--prepare-only", action="store_true", help="生成真实场景、路线与元数据，不连接 Redis 或启动仿真")
    p.add_argument("--stop-file", type=Path)
    p.add_argument("--min-free-gib", type=float, default=20.0)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not 5 <= args.fov <= 50 or args.seed < 0 or args.min_free_gib < 0:
        p.error("FOV 需为 5～50 度，seed 和最小空间不能为负数")
    runtime, output, layout = args.runtime_root.resolve(), args.output.resolve(), args.layout.resolve()
    stop_file = args.stop_file.resolve() if args.stop_file else None
    if output.exists():
        if not output.is_dir() or any(item.name != "batch-child.log" for item in output.iterdir()):
            p.error("输出目录已有数据，禁止覆盖；仅允许批处理预建的 batch-child.log")
    else:
        output.mkdir(parents=True)
    OracleIdentityAgent.SEARCH_FOV_DEG = args.fov
    started = time.time()
    summary = dict(status="failed", stop_reason="initialization_failed", requested_duration_s=args.duration,
        actual_duration_s=0.0, total_bytes=0, image_bytes=0,
        control_mode="oracle_identity", perception="ideal_positions", visibility="unverified",
        weather=args.weather, seed=args.seed, fov=args.fov, output=str(output),
        started_unix_s=started, prepare_only=args.prepare_only)
    runner = None
    with (output / "run.log").open("w", encoding="utf-8", buffering=1) as stream:
        def log(message):
            message = str(message)
            stream.write(message + "\n")
            print(message, flush=True)
            if runner is not None:
                match = re.search(r"\[NAV\] (?:诱饵 )?(\d+) .*?(A\* 规划并直推完成|路线含 A\* 失败段)", message)
                if match:
                    runner.astar[match.group(1)] = "success" if "直推完成" in match.group(2) else "failed_stopped"
        try:
            if not (runtime / "opensim-sim.exe").is_file():
                raise FileNotFoundError(runtime / "opensim-sim.exe")
            source = output / "source"
            source.mkdir()
            hashes = {}
            for path in Path(__file__).parent.glob("*.py"):
                data = path.read_bytes()
                (source / path.name).write_bytes(data)
                hashes[path.name] = hashlib.sha256(data).hexdigest()
            capture_config = runtime / "ue-renderer/Windows/testwl/Content/Config/capture_config.json"
            capture_hash = None
            if capture_config.exists():
                data = capture_config.read_bytes()
                (output / "capture_config.json").write_bytes(data)
                capture_hash = hashlib.sha256(data).hexdigest()
            metadata = dict(summary, runtime_root=str(runtime), layout=str(layout), sources=hashes,
                capture_config_path=str(capture_config), capture_config_sha256=capture_hash,
                code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                image_policy="original_received_bytes_unique_new_frames_existing_cache_frequency",
                frame_time="source_sim_time_not_verified_exposure", twin_check="disabled_oracle_policy",
                requested_weather=args.weather, requested_fov_deg=args.fov)
            write_json(output / "metadata.json", metadata)
            cfg = ScenarioConfig("coop_decoy", str(layout), args.duration, output_dir=str(output),
                sim_binary=str(runtime / "opensim-sim.exe"), start_sim_flag=True,
                photo_mode="on", seed=args.seed, weather=args.weather)
            runner = DatasetCaptureRunner(cfg, output, runtime, log, stop_file, args.min_free_gib)
            runner.materialize_scenario()
            if args.prepare_only:
                summary.update(status="completed", stop_reason="prepare_only")
            elif shutil.disk_usage(output).free <= (args.min_free_gib + .25) * GIB:
                summary.update(status="aborted", stop_reason="minimum_free_space_before_start")
            elif stop_file and stop_file.exists():
                summary.update(status="aborted", stop_reason="stop_file_before_start")
            else:
                # 原生引擎的相对资源路径从 runtime 解析，代码仍来自当前独立工作树。
                os.chdir(runtime)
                result = runner.run()
                summary.update(actual_duration_s=runner.actual_duration_s, v1=runner.v1_summary,
                               identity=dict(runner.identity_counts), astar=runner.astar)
                if result.get("error"):
                    raise RuntimeError(result["error"])
                if runner.stop_reason:
                    summary.update(status="aborted" if runner.stop_reason in ("stop_file", "minimum_free_space") else "failed",
                                   stop_reason=runner.stop_reason)
                elif runner.actual_duration_s < args.duration - 1.0:
                    summary.update(status="failed", stop_reason="simulation_ended_early")
                else:
                    summary.update(status="completed", stop_reason="requested_duration_reached")
        except KeyboardInterrupt:
            summary.update(status="aborted", stop_reason="keyboard_interrupt")
        except Exception as exc:
            summary.update(status="failed", stop_reason=runner.stop_reason if runner and runner.stop_reason else "exception", error=repr(exc))
            log(f"[dataset] failed: {exc!r}")
        finally:
            if runner:
                summary["actual_duration_s"] = runner.actual_duration_s
                try:
                    runner.close()
                except Exception as exc:
                    summary.update(status="failed", stop_reason="close_failed", error=repr(exc))
            if (output / "dataset/frames.jsonl").exists():
                try:
                    if runner.camera and runner.camera.thread.is_alive():
                        raise RuntimeError("相机线程未退出，保留原始文件并拒绝生成不完整索引")
                    index = build_index(output)
                    summary.update(capture_statistics(output, runner.done_at, runner.astar))
                    summary["index"] = index
                    summary["frame_errors"] = runner.camera.error_count if runner.camera else 0
                    summary["identity"] = dict(runner.identity_counts)
                    summary["valid_pose_fraction"] = index["counts"]["with_source_pose"] / max(1, index["counts"]["frames"])
                    if summary["status"] == "completed" and any(not index["frames_by_uid"].get(uid) for uid in ("20001", "20002", "20003")):
                        summary.update(status="failed", stop_reason="missing_uav_images")
                    if summary["frame_errors"] or summary["missing_images"]:
                        summary.update(status="failed", stop_reason="frame_errors_or_missing_files")
                    moving = {kind: sum(v["identity"] == kind and v["path_distance_m"] >= 1.0
                              for v in summary["vehicle_motion"].values())
                              for kind in ("TargetVehicle", "DecoyVehicle")}
                    summary["moving_vehicles_by_class"] = moving
                    if args.duration == 200 and not all(moving.values()) and summary["status"] == "completed":
                        summary.update(status="failed", stop_reason="systemic_vehicle_stoppage")
                except Exception as exc:
                    summary.update(status="failed", stop_reason="index_failed", error=repr(exc))
            summary.update(ended_unix_s=time.time(), wall_duration_s=time.time() - started)
            log(f"[dataset] status={summary['status']}; stop_reason={summary['stop_reason']}")
            write_json(output / "summary.json", summary)
            # 自身 JSON 的体积参与预算，重复一次消除总字节数字长变化。
            for _ in range(2):
                summary["total_bytes"] = directory_bytes(output)
                write_json(output / "summary.json", summary)
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
