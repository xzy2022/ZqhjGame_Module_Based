# 修改时间：2026-09-14
# 修改目的：让个人实验脱离官方仓库的后续修改并支持独立运行。
# 修改内容：统一模块、SDK、运行资源及输出路径并保留实验行为。
# 修改时间：2026-09-13
# 修改目的：保存普查补扫实验使用的航线和覆盖实现。
# 修改内容：将survey_search模块加入实验源码快照。
# 修改时间：2026-09-13
# 修改目的：让横向巡视实验保留完整的搜索云台实现快照。
# 修改内容：将搜索云台模块加入每次实验的源码快照列表。
# 修改时间：2026-09-13
# 修改目的：保留条带搜索实验使用的航线管理源码。
# 修改内容：把 search_route.py 加入运行快照列表。
# 修改时间：2026-09-12
# 修改目的：保留本轮竞争方向导航实验的完整算法快照。
# 修改内容：将 competition_flight.py 加入实验输出的源码快照清单。
# 修改时间：2026-09-12
# 修改目的：控制排查日志体积，避免重复保存通信收件箱。
# 修改内容：只在截图实验中记录无人机位置、姿态、运动和原生云台字段。
# 修改时间：2026-09-12
# 修改目的：用图像和真实计算耗时排查协同主锁定中断。
# 修改内容：添加可选 UE 截图旁路，记录会话、原生云台状态及图像对时所需时间。
# 修改时间：2026-09-12
# 修改目的：让本轮实验完整保存新增的云台锁定控制源码。
# 修改内容：将 gimbal_lock.py 加入实验输出的源码快照列表。
"""三机协同计时实验：记录三架无人机的观测、通信和裁判数据。"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

from .paths import PROJECT_ROOT, SIM_ROOT, RUNTIME_ROOT, OUTPUT_ROOT, SCENARIO_ROOT
import subprocess
import time
import types

from competition.sdk.core.runner import ScenarioConfig
from competition.sdk.scenarios._astar_navigator import _load_routes, _build_waypoints
from competition.sdk._vendored.uav_target_map import UavDetection, resolve_uav_to_target
from competition.baselines.coop_distributed import _haversine_m
from .control_test_runner import IdealPerceptionCoopDecoyRunner


ROOT = PROJECT_ROOT
AGENT_PATH = "src/personal_hf2026/personal_v1.py"
FLEET = (("A", "20001"), ("B", "20002"), ("C", "20003"))


def _finite_dict(value):
    """将无穷大转换为 JSON null，方便其它语言直接读取日志。"""
    if isinstance(value, dict):
        return {key: _finite_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_dict(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _track_dict(snapshot):
    return _finite_dict(asdict(snapshot))


def load_agent(revision):
    # 历史源码只在实验进程中加载，不覆盖用户工作区。
    source = ((ROOT / AGENT_PATH).read_text(encoding="utf-8") if revision == "working"
              else subprocess.check_output(["git", "show", f"{revision}:{AGENT_PATH}"],
                                           cwd=ROOT).decode("utf-8"))
    module = types.ModuleType("timing_study_agent")
    module.__package__ = "personal_hf2026"
    exec(compile(source, f"{revision}/{AGENT_PATH}", "exec"), module.__dict__)
    return module.PersonalV1Agent, source


class StudyRunner(IdealPerceptionCoopDecoyRunner):
    def __init__(self, cfg, agent_cls, output, log, capture_dropouts=False):
        super().__init__(cfg, agent_cls, log=log)
        self.output = output
        self.trace = (output / "observations.jsonl").open("w", encoding="utf-8", buffering=1)
        self.judge = (output / "judge.jsonl").open("w", encoding="utf-8", buffering=1)
        self.progress = {}
        self.capture_dropouts = capture_dropouts
        self.capture = None
        self.renderer = None
        self.world_times = {}

    def _build_perception(self, uids):
        perception = super()._build_perception(uids)
        if self.capture_dropouts:
            from .dropout_capture import DropoutCapture, StudyRenderer
            self.renderer = StudyRenderer(RUNTIME_ROOT, self.output, self.cfg.redis_host,
                                          self.cfg.redis_port, self.log)
            self.renderer.start(self._scenario_cfg, uids)
            self.capture = DropoutCapture(self.output, uids, self.cfg.redis_host, self.cfg.redis_port)
            self.capture.start()
        return perception

    def _extract_truth(self, ws, uid):
        self.world_times[uid] = ws.sim_time
        return super()._extract_truth(ws, uid)

    def prepare_scenario(self):
        if self.cfg.seed > 0:
            # 非零种子沿用官方随机化流程，改变车辆布局和路线。
            super().prepare_scenario()
        else:
            # 零种子用于复查已保存布局，只恢复对应路线名。
            for ent in self._scenario_cfg["entities"]:
                typ = ent["type"]
                if typ not in ("TargetVehicle", "DecoyVehicle"):
                    continue
                filename = ("points.json" if typ == "TargetVehicle"
                            else "random_routes_20.json")
                params = ent["params"]
                pos = (params["initial_latitude"], params["initial_longitude"])
                matches = []
                for route in _load_routes(SIM_ROOT / "config" / filename):
                    wps = _build_waypoints(route)
                    if wps and _haversine_m(*pos, wps[0]["lat"], wps[0]["lon"]) < 0.1:
                        matches.append(route["Name"])
                if len(matches) != 1:
                    raise ValueError(f"路线起点无法唯一匹配：{ent['id']} {matches}")
                self._route_assignment[str(ent["id"])] = matches[0]
        self._scenario_cfg.setdefault("weather", {})["type"] = "Clear_Skies"
        self.cfg.weather = "Clear_Skies"
        self._scenario_cfg["simulation"]["seed"] = self.cfg.seed
        self._scenario_cfg["simulation"]["time_scale"] = 1
        (self.output / "routes.json").write_text(
            json.dumps(self._route_assignment, indent=2), encoding="utf-8")

    def make_agent_for(self, entity_type, entity_uid, world_state):
        agent = super().make_agent_for(entity_type, entity_uid, world_state)
        decide = agent.decide

        def recorded(obs, dt):
            observed_at = time.time()
            started = time.perf_counter()
            cmds = decide(obs, dt)
            decide_ms = (time.perf_counter() - started) * 1000
            own = obs.self
            target = agent._candidate
            local_track = agent._track.snapshot(agent._t, window_s=6.0)
            peer_track = agent._coordinator.peer_track.snapshot(agent._t)
            search_result = agent._search_filter.last_result
            row = {"uid": entity_uid, "t": agent._t, "dt": dt,
                   "sim_time": self.world_times.get(entity_uid),
                   "observed_unix_s": observed_at, "decide_wall_ms": decide_ms,
                   "session": agent._coordinator.current_session,
                   "state": agent._state, "candidate": target,
                   "last_seen": agent._last_seen,
                   "peer": agent._peer, "peer_received": agent._peer_received,
                   "coop": agent._coop_seconds, "gap": agent._gap,
                   "summary": agent.completion_summary,
                   "local_track": _track_dict(local_track),
                   "peer_track": _track_dict(peer_track),
                   "track_match": _finite_dict(asdict(agent._coordinator.last_match)),
                   "search_filter": {
                       "current_status": agent._search_filter.status,
                       "epoch": agent._search_filter.epoch,
                       "started_at": agent._search_filter.started_at,
                       "last_result": asdict(search_result),
                       "rejected_positions": agent._search_filter.rejected_positions,
                   },
                   "self": {"lat": own.lat, "lon": own.lon, "alt": own.alt,
                            "heading_deg": own.heading_deg,
                            "gimbal_pan": own.gimbal_pan, "gimbal_tilt": own.gimbal_tilt,
                            "gimbal_fov_deg": own.gimbal_fov_deg,
                            "detection": asdict(own.detection),
                            "detections": [asdict(item) for item in own.detections],
                            "comm_stats": asdict(own.comm_stats)},
                   "inbox": [asdict(m) for m in obs.comm_inbox],
                   "distance_to_candidate": _haversine_m(own.lat, own.lon, *target) if target else None,
                   "commands": [asdict(c) for c in cmds]}
            self.trace.write(json.dumps(row) + "\n")
            if self.capture:
                self.capture.observe(row)
            p = self.progress.setdefault(entity_uid, {"first_state": {}, "max_coop": 0})
            p["first_state"].setdefault(agent._state, agent._t)
            p["max_coop"] = max(p["max_coop"], agent._coop_seconds)
            return cmds

        agent.decide = recorded
        return agent

    def _observe_scoring(self, evaluator, ws, sim_t0, destroyed, all_cmds=()):
        super()._observe_scoring(evaluator, ws, sim_t0, destroyed, all_cmds)
        # 裁判真值只落盘用于事后评估，绝不传给 Agent。
        detections = []
        for uid, e in ws.uavs.items():
            d = e.raw.get("gimbal_tracking", {}).get("detection", {})
            pos = d.get("target_position") or {}
            detections.append(UavDetection(uid, bool(d.get("detected")),
                                           pos.get("latitude"), pos.get("longitude"),
                                           destroyed=uid in destroyed))
        matched = resolve_uav_to_target(detections,
                                       {uid: (e.lat, e.lon) for uid, e in ws.targets.items()},
                                       {uid: (e.lat, e.lon) for uid, e in ws.decoys.items()})
        row = {"t": ws.sim_time - sim_t0,
               "sim_time": ws.sim_time, "observed_unix_s": time.time(),
               "matches": {uid: asdict(m) for uid, m in matched.items()},
               "targets": {uid: asdict(s) for uid, s in evaluator.states.items()},
               "decoys": {uid: asdict(s) for uid, s in evaluator.decoy_states.items()},
                "world_targets": {
                    uid: {"lat": entity.lat, "lon": entity.lon,
                          "alt": entity.alt, "status": entity.status}
                    for uid, entity in ws.targets.items()
                },
                "world_decoys": {
                    uid: {"lat": entity.lat, "lon": entity.lon,
                          "alt": entity.alt, "status": entity.status}
                    for uid, entity in ws.decoys.items()
                }}
        if self.capture_dropouts:
            row["world_uavs"] = {
                uid: {"lat": entity.lat, "lon": entity.lon, "alt": entity.alt,
                      "heading": entity.heading, "status": entity.status,
                      "attitude": entity.raw.get("platform", {}).get("attitude"),
                      "kinematics": entity.raw.get("kinematics"),
                      "gimbal_tracking": entity.raw.get("gimbal_tracking")}
                for uid, entity in ws.uavs.items()}
        self.judge.write(json.dumps(row) + "\n")

    def should_finish(self, agents):
        fleet = [agents.get(uid) for _, uid in FLEET]
        self.v1_summary = {
            name: agent.completion_summary
            for (name, _), agent in zip(FLEET, fleet)
        }
        self.v1_summary.update(reason="duration_or_simulation_end", progress=self.progress)
        finished = [agent for agent in fleet if agent.finished]
        if finished and (len(finished) == len(fleet)
                         or max(agent._t for agent in fleet)
                         - min(agent.done_at for agent in finished) >= 3):
            self.v1_summary["reason"] = (
                "three_local_kills" if len(finished) == len(fleet)
                else "three_local_kills_delivery_timeout")
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default=str(SCENARIO_ROOT / "static-decoys.json"))
    parser.add_argument("--revision", default="working")
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capture-dropouts", action="store_true",
                        help="启动 UE，在后台保存三机协同中断前后的画面和源时间戳")
    args = parser.parse_args()
    if not Path(args.layout).is_file():
        raise FileNotFoundError(args.layout)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    agent_cls, source = load_agent(args.revision)
    (output / "agent_snapshot.py").write_text(source, encoding="utf-8")
    for name in ("coop_clock.py", "tracking.py", "coordination.py",
                 "survey_search.py",
                 "search_filter.py", "search_route.py", "search_gimbal.py", "gimbal_lock.py", "competition_flight.py",
                 "dropout_capture.py", "timing_study.py"):
        helper = PROJECT_ROOT / "src/personal_hf2026" / name
        if helper.exists():
            (output / f"{helper.stem}_snapshot.py").write_text(
                helper.read_text(encoding="utf-8"), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(
        {"revision": args.revision, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
         "layout": str(Path(args.layout).resolve()), "duration": args.duration,
         "seed": args.seed, "time_scale": 1,
         "capture_dropouts": args.capture_dropouts,
         "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip(),
         "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
         "fleet": [{"name": name, "uid": uid} for name, uid in FLEET],
         "argv": os.sys.argv}, indent=2),
        encoding="utf-8")
    os.environ["OPENSIM_SIM_STDERR"] = str(output / "engine.log")
    cfg = ScenarioConfig("coop_decoy", str(Path(args.layout).resolve()), args.duration,
                         output_dir=str(output), sim_binary=str(RUNTIME_ROOT / "opensim-sim.exe"),
                         start_sim_flag=True, photo_mode="off",
                         seed=args.seed)
    with (output / "run.log").open("w", encoding="utf-8", buffering=1) as log_file:
        runner = StudyRunner(cfg, agent_cls, output, lambda s: log_file.write(str(s) + "\n"),
                             capture_dropouts=args.capture_dropouts)
        try:
            result = runner.run()
            summary = dict(runner.v1_summary, n_destroyed=result.get("n_destroyed", 0))
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary))
        finally:
            if runner.capture:
                runner.capture.close()
            if runner.renderer:
                runner.renderer.close()
            runner.trace.close()
            runner.judge.close()


if __name__ == "__main__":
    main()
