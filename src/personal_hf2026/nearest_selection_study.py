# 修改时间：2026-09-14
# 修改目的：让专项锁定实验同样在指定运行底座中启动引擎。
# 修改内容：使用本仓库兼容 Runner 统一引擎工作目录。
# 修改时间：2026-09-14
# 修改目的：让个人实验脱离官方仓库的后续修改并支持独立运行。
# 修改内容：统一模块、SDK、运行资源及输出路径并保留实验行为。
# 修改时间：2026-09-13
# 修改目的：验证诱饵竞争车辆与误识别概率对原生主检测的影响。
# 修改内容：增加三组诱饵概率对照，并将诱饵真值纳入几何与检测分类统计。
# 修改时间：2026-09-12
# 修改目的：排除通用启动随机化和首帧地形高度未更新对几何实验的影响。
# 修改内容：禁用场景随机化，并使用静止目标当前地形高度校准诊断云台。
# 修改时间：2026-09-12
# 修改目的：独立验证原生检测是否先选择三维距离最近车辆，再检查视野。
# 修改内容：构建静止双车和单机定向盘旋实验，提供单车、双车及交换车辆编号的对照。
"""本地引擎诊断入口；使用已知静止目标位置控制云台，不作为参赛智能体。"""

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .paths import PROJECT_ROOT, SIM_ROOT, RUNTIME_ROOT, OUTPUT_ROOT, SCENARIO_ROOT
import subprocess

from competition.sdk.core.agent import Agent
from competition.sdk.core.commands import fly_to, point_gimbal, set_gimbal_fov, set_speed
from competition.sdk.core.runner import ScenarioConfig
from .sdk_compat import IdleCompatibleCoopDecoyRunner as CoopDecoyRunner
from .control_test_runner import _angular_offset_deg, _bearing_deg, _ground_distance_m


ROOT = PROJECT_ROOT
CENTER = (27.0, 125.0)
RADIUS_M = 160.0
FOV_DEG = 5.0


def offset_position(east_m, north_m):
    """仅用于布置小范围实验；实际距离统计使用球面距离和引擎真实高度。"""
    return (CENTER[0] + math.degrees(north_m / 6371000.0),
            CENTER[1] + math.degrees(east_m / (6371000.0 * math.cos(math.radians(CENTER[0])))))


def make_scenario(case):
    source = json.loads((ROOT / "configs/scenarios/coop_decoy/static-decoys.json").read_text(encoding="utf-8"))
    uav = copy.deepcopy(source["entities"][0])
    lat, lon = offset_position(0.0, RADIUS_M)
    uav["params"].update(initial_latitude=lat, initial_longitude=lon, initial_heading=90.0)
    uav["components"]["gimbal_tracking"]["params"].update(fov=FOV_DEG, auto_track=False)
    is_decoy = case.startswith("decoy-")
    if is_decoy:
        uav["components"]["gimbal_tracking"]["params"].update(
            misid_prob=float(case.split("-", 1)[1]), misid_seed=14)
    entities = [uav]
    aim_uid = "10002" if case == "swapped" else "10001"
    ids = [aim_uid] if case == "single" else [aim_uid, "30001" if is_decoy else ("10001" if case == "swapped" else "10002")]
    template = next(e for e in source["entities"] if e["type"] == "TargetVehicle")
    for index, uid in enumerate(ids):
        vehicle = copy.deepcopy(template)
        if is_decoy and index == 1:
            vehicle = copy.deepcopy(next(e for e in source["entities"] if e["type"] == "DecoyVehicle"))
        lat, lon = offset_position(-40.0 if index == 0 else 40.0, 0.0)
        vehicle.update(id=uid, name="aim_vehicle" if index == 0 else "other_vehicle")
        vehicle["params"].update(initial_latitude=lat, initial_longitude=lon)
        vehicle["components"]["trajectory"]["params"].update(
            speed=0.0, speed_jitter=0.0, waypoints=[])
        entities.append(vehicle)
    return {"config_version": "1.0", "simulation": dict(source["simulation"], seed=0, time_scale=1),
            "entities": entities, "weather": {"type": "Clear_Skies"}}, aim_uid


class FixedAimAgent(Agent):
    """只执行相同的圆周飞行和固定点瞄准，不读取检测结果作控制反馈。"""

    def __init__(self, my_uid, aim):
        super().__init__(my_uid)
        self.aim = aim

    def decide(self, obs, dt):
        own = obs.self
        lat, lon, alt = self.aim
        ground = _ground_distance_m(own.lat, own.lon, lat, lon)
        bearing = _bearing_deg(own.lat, own.lon, lat, lon)
        pan = (bearing - own.heading_deg + 180.0) % 360.0 - 180.0
        tilt = math.degrees(math.atan2(alt - own.alt, ground))
        return [fly_to(*CENTER, speed=22.0, loiter_radius=RADIUS_M, turn_direction="right"),
                point_gimbal(pan, tilt), set_gimbal_fov(FOV_DEG)]


class NearestSelectionRunner(CoopDecoyRunner):
    def __init__(self, cfg, aim_uid, output, log):
        super().__init__(cfg, FixedAimAgent, log=log)
        self.aim_uid = aim_uid
        self.output = output
        self.trace = (output / "probe.jsonl").open("w", encoding="utf-8", buffering=1)
        self.rows = []

    def prepare_scenario(self):
        # 保持人工布置的坐标，禁止父类随机路线准备改变实验几何条件。
        pass

    def inject_startup(self, client, first):
        # 不注入任何路线，并显式停车；静止性还会通过每帧真实位置复核。
        for uid in {**first.targets, **first.decoys}:
            client.publish(uid, set_speed(0.0))

    def make_agent_for(self, entity_type, entity_uid, world_state):
        target = world_state.targets[self.aim_uid]
        self.controller = FixedAimAgent(entity_uid, (target.lat, target.lon, target.alt))
        return self.controller

    def _extract_truth(self, ws, uid):
        # 本模块是已知静止目标的引擎诊断；首帧高度可能尚未贴合地形。
        target = ws.targets[self.aim_uid]
        self.controller.aim = target.lat, target.lon, target.alt
        return super()._extract_truth(ws, uid)

    def _observe_scoring(self, evaluator, ws, sim_t0, destroyed, all_cmds=()):
        super()._observe_scoring(evaluator, ws, sim_t0, destroyed, all_cmds)
        aircraft = ws.uavs["20001"]
        gimbal = aircraft.raw["gimbal_tracking"]
        camera_az = aircraft.heading + gimbal["pan_angle"]
        camera_el = gimbal["tilt_angle"]
        candidates = []
        for uid, target in {**ws.targets, **ws.decoys}.items():
            ground = _ground_distance_m(aircraft.lat, aircraft.lon, target.lat, target.lon)
            az = _bearing_deg(aircraft.lat, aircraft.lon, target.lat, target.lon)
            el = math.degrees(math.atan2(target.alt - aircraft.alt, ground))
            angle = _angular_offset_deg(camera_az, camera_el, az, el)
            candidates.append({"uid": uid, "is_decoy": uid in ws.decoys,
                               "lat": target.lat, "lon": target.lon, "alt": target.alt,
                               "distance_m": math.hypot(ground, aircraft.alt - target.alt),
                               "ground_distance_m": ground, "angle_deg": angle,
                               "inside_fov": angle < gimbal["fov"] / 2.0})
        candidates.sort(key=lambda item: item["distance_m"])
        nearest = candidates[0]
        aim = next(c for c in candidates if c["uid"] == self.aim_uid)
        raw = gimbal["detection"]
        native_uid = None
        if raw.get("detected"):
            position = raw.get("target_position", {})
            native_uid = min(candidates, key=lambda c: _ground_distance_m(
                position["latitude"], position["longitude"], c["lat"], c["lon"]))["uid"]
        gap = candidates[1]["distance_m"] - nearest["distance_m"] if len(candidates) == 2 else None
        elapsed = ws.sim_time - sim_t0
        # 舍弃启动调姿、距离几乎相等及光轴偏离的采样，避免边界误差制造结论。
        eligible = (elapsed >= 3.0 and aim["angle_deg"] < 0.5
                    and abs(gimbal["fov"] - FOV_DEG) < 0.001
                    and (gap is None or gap >= 1.0)
                    and all(c["uid"] == self.aim_uid or c["angle_deg"] > FOV_DEG / 2 + 0.5
                            for c in candidates))
        row = {"t": elapsed, "sim_time": ws.sim_time, "aim_uid": self.aim_uid,
               "aircraft": {"lat": aircraft.lat, "lon": aircraft.lon, "alt": aircraft.alt,
                            "heading": aircraft.heading},
               "gimbal": gimbal, "vehicles": candidates, "native_uid": native_uid,
               "nearest_uid": nearest["uid"], "distance_margin_m": gap,
               "predicted_uid": nearest["uid"] if nearest["inside_fov"] else None,
               "eligible": eligible, "commands": [asdict(c) for uid, c in all_cmds if uid == "20001"]}
        self.trace.write(json.dumps(row) + "\n")
        self.rows.append(row)

    def summarize(self):
        eligible = [r for r in self.rows if r["eligible"]]
        groups = defaultdict(Counter)
        for row in eligible:
            condition = "aim_nearer" if row["nearest_uid"] == self.aim_uid else "other_nearer_outside_fov"
            groups[condition]["samples"] += 1
            groups[condition]["detected_aim"] += row["native_uid"] == self.aim_uid
            groups[condition]["empty"] += row["native_uid"] is None
            groups[condition]["detected_other"] += row["native_uid"] is not None and row["native_uid"] != self.aim_uid
            groups[condition]["model_matches"] += row["native_uid"] == row["predicted_uid"]
        # 同时保存所有排名及原生输出变化，便于检查未纳入统计的交界采样。
        transitions = []
        previous = None
        for row in self.rows:
            state = row["nearest_uid"], row["native_uid"]
            if state != previous:
                transitions.append({key: row[key] for key in (
                    "t", "nearest_uid", "native_uid", "distance_margin_m", "eligible")})
                previous = state
        initial = self.rows[0]["vehicles"]
        movement = {}
        for origin in initial:
            track = [next(c for c in row["vehicles"] if c["uid"] == origin["uid"]) for row in self.rows]
            movement[origin["uid"]] = {
                "max_horizontal_displacement_m": max(_ground_distance_m(
                    origin["lat"], origin["lon"], c["lat"], c["lon"]) for c in track),
                "altitude_range_m": [min(c["alt"] for c in track), max(c["alt"] for c in track)]}
        summary = {"total_samples": len(self.rows), "eligible_samples": len(eligible),
                   "conditions": dict(groups), "transitions": transitions, "vehicle_movement": movement,
                   "fov_range_deg": [min(r["gimbal"]["fov"] for r in self.rows),
                                     max(r["gimbal"]["fov"] for r in self.rows)],
                   "model_match_rate": (sum(r["native_uid"] == r["predicted_uid"] for r in eligible)
                                        / len(eligible)) if eligible else None}
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("single", "double", "swapped", "all",
                                          "decoy-0", "decoy-0.5", "decoy-1", "decoy-all"), default="all")
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (output / "runner_snapshot.py").write_bytes(source)
    (output / "metadata.json").write_text(json.dumps({
        "duration_per_case_s": args.duration, "fov_deg": FOV_DEG, "orbit_radius_m": RADIUS_M,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "engine_sha256": hashlib.sha256((RUNTIME_ROOT / "opensim-sim.exe").read_bytes()).hexdigest(),
        "case": args.case, "decoy_misid_seed": 14,
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
    }, indent=2), encoding="utf-8")
    suite = {}
    cases = {"all": ("single", "double", "swapped"), "decoy-all": ("decoy-0", "decoy-0.5", "decoy-1")}
    for case in cases.get(args.case, (args.case,)):
        case_output = output / case
        case_output.mkdir()
        scenario, aim_uid = make_scenario(case)
        scenario_path = case_output / "scenario.json"
        scenario_path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
        cfg = ScenarioConfig("coop_decoy", str(scenario_path), args.duration,
                             output_dir=str(case_output), sim_binary=str(RUNTIME_ROOT / "opensim-sim.exe"),
                             start_sim_flag=True,
                             photo_mode="off", seed=0)
        with (case_output / "run.log").open("w", encoding="utf-8", buffering=1) as stream:
            def log(message):
                print(message, flush=True)
                stream.write(str(message) + "\n")
            runner = NearestSelectionRunner(cfg, aim_uid, case_output, log)
            try:
                runner.run()
                suite[case] = runner.summarize()
                print(json.dumps({"case": case, "conditions": suite[case]["conditions"]}), flush=True)
            finally:
                runner.trace.close()
        (output / "suite_summary.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
