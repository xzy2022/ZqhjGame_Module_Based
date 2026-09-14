# 修改时间：2026-09-14
# 修改目的：让个人实验脱离官方仓库的后续修改并支持独立运行。
# 修改内容：统一模块、SDK、运行资源及输出路径并保留实验行为。
# 修改时间：2026-09-12
# 修改目的：让单数检测表示引擎主锁定，同时继续向搜索算法提供理想多目标候选。
# 修改内容：将引擎原生检测置于首位，并在复数检测中合并去重后的 FOV 候选。
"""赛题二控制开发 Runner：同时输出引擎主锁定和理想多目标坐标。

该入口只用于个人控制实验。它从 Runner 内部 WorldState 计算多目标检测，
不添加漏检和位置噪声，不修改官方 train/eval 入口与评分逻辑。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .paths import PROJECT_ROOT, SIM_ROOT, RUNTIME_ROOT, OUTPUT_ROOT, SCENARIO_ROOT

from competition.sdk.cli import _load_agent_class, _resolve_redis_port
from competition.sdk.core.observation import Detection
from competition.sdk.core.perception import BaseDetector, DetectionResolver
from competition.sdk.core.runner import ScenarioConfig
from competition.sdk.core.world_state import WorldState
from .sdk_compat import IdleCompatibleCoopDecoyRunner as CoopDecoyRunner


_DEFAULT_SCENARIO = SCENARIO_ROOT / "static-decoys.json"


def _bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta = math.radians(lon2 - lon1)
    y = math.sin(delta) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(delta))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _ground_distance_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    delta = math.radians(lon2 - lon1)
    value = (math.sin(dphi / 2.0) ** 2
             + math.cos(phi1) * math.cos(phi2) * math.sin(delta / 2.0) ** 2)
    return 2.0 * 6371000.0 * math.asin(math.sqrt(value))


def _angular_offset_deg(camera_azimuth, camera_elevation,
                        target_azimuth, target_elevation):
    """计算目标方向与相机光轴的三维夹角。"""
    caz, cel = math.radians(camera_azimuth), math.radians(camera_elevation)
    taz, tel = math.radians(target_azimuth), math.radians(target_elevation)
    dot = (math.cos(cel) * math.cos(tel) * math.cos(caz - taz)
           + math.sin(cel) * math.sin(tel))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def visible_vehicle_detections(world_state, uid):
    """按无人机、云台和圆锥 FOV 返回全部理想车辆检测。"""
    me = world_state.entities.get(uid)
    if me is None:
        return []
    gimbal = me.raw.get("gimbal_tracking", {}) or {}
    pan = float(gimbal.get("pan_angle", 0.0))
    tilt = float(gimbal.get("tilt_angle", 0.0))
    fov = float(gimbal.get("fov", gimbal.get("fov_deg", 30.0)))
    camera_azimuth = (me.heading + pan) % 360.0
    half_fov = max(0.1, fov * 0.5)
    visible = []
    vehicles = list(world_state.targets.items()) + list(world_state.decoys.items())
    for vehicle_uid, vehicle in vehicles:
        if vehicle.status != "active" or (vehicle.lat == 0.0 and vehicle.lon == 0.0):
            continue
        ground = _ground_distance_m(me.lat, me.lon, vehicle.lat, vehicle.lon)
        target_azimuth = _bearing_deg(me.lat, me.lon, vehicle.lat, vehicle.lon)
        target_elevation = math.degrees(math.atan2(vehicle.alt - me.alt, max(ground, 1e-6)))
        offset = _angular_offset_deg(camera_azimuth, tilt,
                                     target_azimuth, target_elevation)
        if offset >= half_fov:
            continue
        azimuth_error = ((target_azimuth - camera_azimuth + 180.0) % 360.0) - 180.0
        visible.append((offset, vehicle_uid, Detection(
            detected=True,
            confidence=max(0.0, 1.0 - offset / half_fov),
            target_lat=vehicle.lat,
            target_lon=vehicle.lon,
            azimuth_error_deg=azimuth_error,
            # 测试感知不向 Agent 泄露真目标与诱饵的身份差异。
            target_type="ground_vehicle",
        )))
    visible.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in visible]


class MultiTargetIdealDetector(BaseDetector):
    """原样转发 Runner 生成的主锁定和无噪声多目标检测。"""

    def detect(self, obs, dt, truth_source=None):
        return list(truth_source or ())


class IdealPerceptionCoopDecoyRunner(CoopDecoyRunner):
    """Coop runner with an explicit, test-only ideal position detector."""

    def should_finish(self, agents):
        # 仅个人 v1 启用提前结束，其他基线保持原行为。
        from personal_hf2026.personal_v1 import PersonalV1Agent
        uids = (PersonalV1Agent.A, PersonalV1Agent.B, PersonalV1Agent.C)
        fleet = [agents.get(uid) for uid in uids]
        if not all(isinstance(agent, PersonalV1Agent) for agent in fleet):
            return False
        self.v1_summary = {
            name: agent.completion_summary
            for name, agent in zip(("A", "B", "C"), fleet)
        }
        self.v1_summary["reason"] = "duration_or_simulation_end"
        finished = [agent for agent in fleet if agent.finished]
        if not finished:
            return False
        # 给第三个完成事件的广播留出时间，随后正常收尾并生成评估文件。
        first_done = min(agent.done_at for agent in finished)
        if len(finished) == len(fleet) or max(agent._t for agent in fleet) - first_done >= 3.0:
            self.v1_summary["reason"] = (
                "three_local_kills" if len(finished) == len(fleet)
                else "three_local_kills_delivery_timeout")
            return True
        return False

    def prepare_scenario(self) -> None:
        # Keep all official route preparation, then change only the prepared
        # copy consumed by the engine. The input scenario file is untouched.
        super().prepare_scenario()
        self._scenario_cfg.setdefault("weather", {})["type"] = "Clear_Skies"
        self.cfg.weather = "Clear_Skies"
        self.log("[control-test] prepared scenario weather = Clear_Skies")

    def _build_perception(self, uids):
        detector = MultiTargetIdealDetector()
        self.log(
            "[control-test] ENGINE PRIMARY + MULTI-TARGET IDEAL PERCEPTION ENABLED "
            "(3D cone FOV candidates, noise=0m, weather=Clear_Skies)"
        )
        return None, DetectionResolver(default_detector=detector)

    def _extract_truth(self, ws: WorldState, uid: str):
        primary = super()._extract_truth(ws, uid)
        visible = visible_vehicle_detections(ws, uid)
        if (primary.detected and primary.target_lat is not None
                and primary.target_lon is not None):
            primary_position = primary.target_lat, primary.target_lon
            visible = [detection for detection in visible
                       if detection.target_lat is None
                       or detection.target_lon is None
                       or _ground_distance_m(
                           *primary_position,
                           detection.target_lat,
                           detection.target_lon) >= 1.0]
        # 首项始终保留引擎主锁定；未锁定时使用 false 占位，避免候选顶替单数检测。
        return [primary, *visible]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run coop_decoy with opt-in ideal perception for control-only "
            "development. Normal competition train/eval modes are unchanged."
        )
    )
    parser.add_argument(
        "--scenario-json",
        default=str(_DEFAULT_SCENARIO),
        help="scenario JSON path (default: static-decoys.json)",
    )
    parser.add_argument(
        "--agent",
        default="baselines.coop_distributed:CoopDistributedAgent",
        help="agent as module.path:ClassName",
    )
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "control-test-ideal"))
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=None)
    parser.add_argument("--sim-binary", default=str(RUNTIME_ROOT / "opensim-sim.exe"))
    parser.add_argument("--no-start-sim", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    scenario_path = str(Path(args.scenario_json).resolve())
    if not Path(scenario_path).is_file():
        raise FileNotFoundError(f"scenario JSON not found: {scenario_path}")

    redis_port = args.redis_port
    if redis_port is None:
        redis_port = _resolve_redis_port("coop_decoy", scenario_path)

    cfg = ScenarioConfig(
        scenario_name="coop_decoy",
        scenario_path=scenario_path,
        duration_s=args.duration,
        redis_host=args.redis_host,
        redis_port=redis_port,
        output_dir=args.output,
        sim_binary=args.sim_binary,
        start_sim_flag=not args.no_start_sim,
        dry_run=args.dry_run,
        quiet=args.quiet,
        seed=args.seed,
        run_mode="train",
        photo_mode="off",
        # These legal placeholder values satisfy ScenarioConfig. The subclass
        # above constructs the opt-in 1.0/0m detector directly.
        accuracy=0.9,
        noise_sigma_m=30.0,
        weather="Clear_Skies",
        max_detection_range_m=0.0,
        full_accuracy_range_m=0.0,
        extra={"control_test_perception": "multi_target_ideal"},
    )
    # Record the actual opt-in values on this one config instance as well.
    # ScenarioConfig's construction-time clamps remain unchanged for every
    # official entry point.
    cfg.accuracy = 1.0
    cfg.noise_sigma_m = 0.0
    agent_cls = _load_agent_class(args.agent)
    runner = IdealPerceptionCoopDecoyRunner(cfg, agent_cls)
    evaluation = runner.run()
    if hasattr(runner, "v1_summary"):
        # 摘要区分行为完成与评分器实际摧毁数量。
        summary = dict(runner.v1_summary, n_destroyed=evaluation.get("n_destroyed", 0))
        output = Path(args.output) / "v1_summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[control-test] v1 结果摘要：{output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
