# 修改时间：2026-09-16
# 修改目的：把控制阶跃与三目标搜索几何放进同一轮 60Hz 真实仿真实验。
# 修改内容：接入三目标专项场景、30度中心包络阶段及逐拍光轴覆盖证据。
# 修改时间：2026-09-16
# 修改目的：实测相机变焦和云台姿态命令从发出到实际状态到达及稳定所需的时间。
# 修改内容：增加单机短时阶跃 Runner、逐拍命令与观测记录，并生成机器摘要和中文报告。
"""相机 FOV 与云台姿态阶跃专项实验；仅用于本地真实仿真诊断。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from competition.sdk.core.agent import Agent
from competition.sdk.core.commands import point_gimbal, set_gimbal_fov, set_speed
from competition.sdk.core.runner import ScenarioConfig

from .control_test_runner import _angular_offset_deg, _bearing_deg, _ground_distance_m
from .paths import OUTPUT_ROOT, PROJECT_ROOT, RUNTIME_ROOT
from .sdk_compat import IdleCompatibleCoopDecoyRunner as CoopDecoyRunner


UAV_UID = "20001"
TARGET_UIDS = ("10001", "10002", "10003")
CAMERA_SCENARIO = PROJECT_ROOT / "configs/scenarios/camera_control_probe/scenario.json"


@dataclass(frozen=True)
class Stage:
    name: str
    start_s: float
    pan_deg: float
    tilt_deg: float
    fov_deg: float
    aim_uid: str | None = None


STAGES = (
    Stage("基准中心_FOV48", 0.0, 0.0, -45.0, 48.0),
    Stage("仅变焦_48到10", 2.0, 0.0, -45.0, 10.0),
    Stage("目标A", 3.0, 0.0, -31.0, 10.0, "10001"),
    Stage("目标B", 5.0, -19.231456, -50.500938, 10.0, "10002"),
    Stage("目标C", 7.0, 19.231456, -50.500938, 10.0, "10003"),
    Stage("目标A_第二轮", 9.0, 0.0, -31.0, 10.0, "10001"),
    Stage("目标B_第二轮", 11.0, -19.231456, -50.500938, 10.0, "10002"),
    Stage("目标C_第二轮", 13.0, 19.231456, -50.500938, 10.0, "10003"),
    Stage("三目标中心_FOV30", 15.0, 0.0, -45.0, 30.0),
    Stage("三目标中心_FOV48", 17.0, 0.0, -45.0, 48.0),
    Stage("中心复测_48到10", 19.0, 0.0, -45.0, 10.0),
)


def _stage_at(elapsed_s: float) -> tuple[int, Stage]:
    index = 0
    for candidate, stage in enumerate(STAGES):
        if elapsed_s + 1e-9 < stage.start_s:
            break
        index = candidate
    return index, STAGES[index]


def _angle_error_deg(actual: float, desired: float) -> float:
    return (actual - desired + 180.0) % 360.0 - 180.0


def _target_geometry(aircraft, target, actual: dict) -> dict:
    """按引擎真实位姿计算目标方向、光轴夹角和当前 FOV 覆盖。"""
    ground = _ground_distance_m(aircraft.lat, aircraft.lon, target.lat, target.lon)
    azimuth = _bearing_deg(aircraft.lat, aircraft.lon, target.lat, target.lon)
    elevation = math.degrees(math.atan2(target.alt - aircraft.alt, max(ground, 1e-6)))
    camera_azimuth = (aircraft.heading + actual["pan_deg"]) % 360.0
    axis_error = _angular_offset_deg(
        camera_azimuth, actual["tilt_deg"], azimuth, elevation)
    return {
        "azimuth_deg": azimuth,
        "relative_pan_deg": _angle_error_deg(azimuth, aircraft.heading),
        "elevation_deg": elevation,
        "ground_distance_m": ground,
        "axis_error_deg": axis_error,
        "inside_fov": axis_error < actual["fov_deg"] / 2.0,
    }


def _make_scenario() -> dict:
    """读取单机三静止目标场景；运行输出仍保存一份实际使用副本。"""
    return json.loads(CAMERA_SCENARIO.read_text(encoding="utf-8"))


class CameraStepAgent(Agent):
    """按固定时序重复发送云台和 FOV 命令，不使用检测结果反馈。"""

    def __init__(self, my_uid: str):
        super().__init__(my_uid)
        self.elapsed_s = 0.0
        self.stage_index = 0
        self.stage = STAGES[0]

    def reset(self):
        self.elapsed_s = 0.0
        self.stage_index = 0
        self.stage = STAGES[0]

    def decide(self, obs, dt):
        self.stage_index, self.stage = _stage_at(self.elapsed_s)
        self.elapsed_s += dt
        return [
            set_speed(0.0),
            point_gimbal(self.stage.pan_deg, self.stage.tilt_deg),
            set_gimbal_fov(self.stage.fov_deg),
        ]


class CameraControlRunner(CoopDecoyRunner):
    """复用正式 Runner 生命周期，只追加控制阶跃的原始测量旁路。"""

    def __init__(self, cfg, output: Path, log=print):
        super().__init__(cfg, CameraStepAgent, log=log)
        self.output = output
        self.trace = (output / "ticks.jsonl").open(
            "w", encoding="utf-8", buffering=1)
        self.rows: list[dict] = []
        self.controller: CameraStepAgent | None = None
        self.first_wall_monotonic: float | None = None

    def prepare_scenario(self):
        # 保留专项场景的人工布置，禁止通用随机选路改变实验条件。
        pass

    def inject_startup(self, client, first):
        # 重申单机和三车静止；逐拍真实速度仍会落盘，验证引擎是否接受零速。
        client.publish(UAV_UID, set_speed(0.0))
        for uid in TARGET_UIDS:
            client.publish(uid, set_speed(0.0))

    def make_agent_for(self, entity_type, entity_uid, world_state):
        self.controller = CameraStepAgent(entity_uid)
        return self.controller

    def _observe_scoring(self, evaluator, ws, sim_t0, destroyed, all_cmds=()):
        super()._observe_scoring(evaluator, ws, sim_t0, destroyed, all_cmds)
        if self.controller is None:
            return
        now_monotonic = time.perf_counter()
        if self.first_wall_monotonic is None:
            self.first_wall_monotonic = now_monotonic
        aircraft = ws.uavs[UAV_UID]
        gimbal = aircraft.raw.get("gimbal_tracking", {}) or {}
        actual = {
            "pan_deg": float(gimbal.get("pan_angle", 0.0)),
            "tilt_deg": float(gimbal.get("tilt_angle", 0.0)),
            "fov_deg": float(gimbal.get("fov", gimbal.get("fov_deg", 0.0))),
        }
        stage = self.controller.stage
        desired = {
            "pan_deg": stage.pan_deg,
            "tilt_deg": stage.tilt_deg,
            "fov_deg": stage.fov_deg,
        }
        errors = {
            "pan_deg": _angle_error_deg(actual["pan_deg"], desired["pan_deg"]),
            "tilt_deg": actual["tilt_deg"] - desired["tilt_deg"],
            "fov_deg": actual["fov_deg"] - desired["fov_deg"],
        }
        targets = {
            uid: _target_geometry(aircraft, ws.targets[uid], actual)
            for uid in TARGET_UIDS if uid in ws.targets
        }
        aim = targets.get(stage.aim_uid) if stage.aim_uid else None
        commands = [asdict(command) for uid, command in all_cmds if uid == UAV_UID]
        row = {
            "tick": len(self.rows),
            "sim_time": float(ws.sim_time),
            "sim_elapsed_s": float(ws.sim_time - sim_t0),
            "wall_time_unix_s": time.time(),
            "wall_time_monotonic_s": now_monotonic,
            "wall_elapsed_s": now_monotonic - self.first_wall_monotonic,
            "stage_index": self.controller.stage_index,
            "stage": stage.name,
            "stage_nominal_start_s": stage.start_s,
            "aim_uid": stage.aim_uid,
            "desired": desired,
            "actual": actual,
            "error": errors,
            "aircraft": {
                "lat": aircraft.lat,
                "lon": aircraft.lon,
                "alt": aircraft.alt,
                "heading_deg": aircraft.heading,
                "speed_mps": aircraft.speed,
            },
            "targets": targets,
            "aim_axis_error_deg": aim["axis_error_deg"] if aim else None,
            "all_targets_inside_fov": bool(targets) and all(
                target["inside_fov"] for target in targets.values()),
            "commands": commands,
        }
        self.trace.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows.append(row)


def _within(row: dict, tolerances: dict) -> bool:
    return all(abs(row["error"][axis]) <= tolerances[axis]
               for axis in ("pan_deg", "tilt_deg", "fov_deg"))


def _peak_rates(rows: list[dict]) -> dict:
    peaks = {"pan_dps": 0.0, "tilt_dps": 0.0, "fov_dps": 0.0}
    for before, after in zip(rows, rows[1:]):
        dt = after["sim_time"] - before["sim_time"]
        if dt <= 0:
            continue
        peaks["pan_dps"] = max(peaks["pan_dps"], abs(_angle_error_deg(
            after["actual"]["pan_deg"], before["actual"]["pan_deg"])) / dt)
        peaks["tilt_dps"] = max(peaks["tilt_dps"], abs(
            after["actual"]["tilt_deg"] - before["actual"]["tilt_deg"]) / dt)
        peaks["fov_dps"] = max(peaks["fov_dps"], abs(
            after["actual"]["fov_deg"] - before["actual"]["fov_deg"]) / dt)
    return peaks


def _summarize(rows: list[dict], tolerances: dict, stable_window_s: float) -> dict:
    stage_results = []
    for index, stage in enumerate(STAGES):
        samples = [row for row in rows if row["stage_index"] == index]
        if not samples:
            stage_results.append({"index": index, "name": stage.name, "samples": 0,
                                  "status": "not_observed"})
            continue
        started = samples[0]
        arrival = next((row for row in samples if _within(row, tolerances)), None)
        stable_enter = None
        stable_confirm = None
        run_start = None
        for row in samples:
            if _within(row, tolerances):
                run_start = run_start or row
                if row["sim_time"] - run_start["sim_time"] >= stable_window_s:
                    stable_enter, stable_confirm = run_start, row
                    break
            else:
                run_start = None
        previous = STAGES[index - 1] if index else stage
        changed_axes = [axis for axis, before, after in (
            ("pan", previous.pan_deg, stage.pan_deg),
            ("tilt", previous.tilt_deg, stage.tilt_deg),
            ("fov", previous.fov_deg, stage.fov_deg),
        ) if abs(after - before) > 1e-9]
        result = {
            "index": index,
            "name": stage.name,
            "samples": len(samples),
            "changed_axes": changed_axes,
            "command": {"pan_deg": stage.pan_deg, "tilt_deg": stage.tilt_deg,
                        "fov_deg": stage.fov_deg},
            "command_sim_time": started["sim_time"],
            "command_sim_elapsed_s": started["sim_elapsed_s"],
            "command_wall_time_unix_s": started["wall_time_unix_s"],
            "actual_when_commanded": started["actual"],
            "error_when_commanded": started["error"],
            "arrival_latency_sim_s": (arrival["sim_time"] - started["sim_time"]
                                      if arrival else None),
            "arrival_latency_wall_s": (arrival["wall_elapsed_s"] - started["wall_elapsed_s"]
                                       if arrival else None),
            "stable_enter_latency_sim_s": (
                stable_enter["sim_time"] - started["sim_time"] if stable_enter else None),
            "stable_confirm_latency_sim_s": (
                stable_confirm["sim_time"] - started["sim_time"] if stable_confirm else None),
            "stable_confirm_latency_wall_s": (
                stable_confirm["wall_elapsed_s"] - started["wall_elapsed_s"]
                if stable_confirm else None),
            "final_error": samples[-1]["error"],
            "peak_observed_rate": _peak_rates(samples),
            "aim_uid": stage.aim_uid,
            "aim_axis_error_deg": ({
                "min": min(row["aim_axis_error_deg"] for row in samples),
                "max": max(row["aim_axis_error_deg"] for row in samples),
                "final": samples[-1]["aim_axis_error_deg"],
            } if stage.aim_uid else None),
            "aim_centered_sample_ratio": (
                sum(row["aim_axis_error_deg"] <= 0.5 for row in samples) / len(samples)
                if stage.aim_uid else None),
            "all_targets_inside_fov_ratio": (
                sum(row["all_targets_inside_fov"] for row in samples) / len(samples)),
            "status": "stable" if stable_confirm else ("arrived" if arrival else "not_arrived"),
        }
        stage_results.append(result)
    transitions = [item for item in stage_results[1:] if item.get("samples")]
    stable = [item for item in transitions if item["status"] == "stable"]
    settle_values = [item["stable_confirm_latency_sim_s"] for item in stable]
    sample_intervals = [after["sim_time"] - before["sim_time"]
                        for before, after in zip(rows, rows[1:])
                        if after["sim_time"] > before["sim_time"]]
    first_aircraft = rows[0]["aircraft"] if rows else None
    max_aircraft_displacement = (max(_ground_distance_m(
        first_aircraft["lat"], first_aircraft["lon"],
        row["aircraft"]["lat"], row["aircraft"]["lon"]) for row in rows)
        if rows else None)
    return {
        "schema_version": 1,
        "measurement": "engine_observed_gimbal_step_response",
        "sample_count": len(rows),
        "sim_elapsed_range_s": ([rows[0]["sim_elapsed_s"], rows[-1]["sim_elapsed_s"]]
                                if rows else None),
        "wall_elapsed_range_s": ([rows[0]["wall_elapsed_s"], rows[-1]["wall_elapsed_s"]]
                                 if rows else None),
        "tolerances_deg": tolerances,
        "stable_window_s": stable_window_s,
        "sample_interval_sim_s": ({
            "min": min(sample_intervals),
            "median": statistics.median(sample_intervals),
            "max": max(sample_intervals),
        } if sample_intervals else None),
        "aircraft_motion": ({
            "max_horizontal_displacement_m": max_aircraft_displacement,
            "speed_range_mps": [min(row["aircraft"]["speed_mps"] for row in rows),
                                max(row["aircraft"]["speed_mps"] for row in rows)],
            "heading_range_deg": [min(row["aircraft"]["heading_deg"] for row in rows),
                                  max(row["aircraft"]["heading_deg"] for row in rows)],
        } if rows else None),
        "stages": stage_results,
        "transition_count": len(transitions),
        "stable_transition_count": len(stable),
        "all_transitions_stable": len(stable) == len(transitions) and bool(transitions),
        "camera_only_min_confirmed_dwell_s": max(settle_values) if settle_values else None,
        "scope_limit": (
            "只测命令到引擎观测姿态/FOV的动态；不包含图像到达、检测重捕获、目标运动或识别耗时。"
        ),
    }


def _fmt(value) -> str:
    return "未到达" if value is None else f"{value:.3f}"


def _write_report(output: Path, summary: dict) -> None:
    lines = [
        "# 相机变焦与云台姿态控制实测",
        "",
        f"- 逐拍样本：{summary['sample_count']}",
        f"- 稳定判据：pan/tilt/FOV 误差分别不超过 "
        f"{summary['tolerances_deg']['pan_deg']}/"
        f"{summary['tolerances_deg']['tilt_deg']}/"
        f"{summary['tolerances_deg']['fov_deg']} 度，并连续保持 "
        f"{summary['stable_window_s']} 秒",
        f"- 阶跃稳定：{summary['stable_transition_count']}/"
        f"{summary['transition_count']}",
        "",
        "| 阶段 | 改变量 | 到达延迟(s) | 稳定进入(s) | 稳定确认(s) | 状态 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in summary["stages"]:
        if not item.get("samples"):
            lines.append(f"| {item['name']} | - | - | - | - | {item['status']} |")
            continue
        lines.append(
            f"| {item['name']} | {','.join(item['changed_axes']) or '基准'} | "
            f"{_fmt(item['arrival_latency_sim_s'])} | "
            f"{_fmt(item['stable_enter_latency_sim_s'])} | "
            f"{_fmt(item['stable_confirm_latency_sim_s'])} | {item['status']} |"
        )
    lines.extend(["", "## 对搜索策略的含义", ""])
    dwell = summary["camera_only_min_confirmed_dwell_s"]
    if summary["all_transitions_stable"]:
        lines.append(
            f"本轮所有阶跃均稳定；仅考虑相机执行动态，循环切换 10° FOV 时，"
            f"每个观察点至少预留 {_fmt(dwell)} 秒，才覆盖本轮最慢的稳定确认。"
        )
    else:
        lines.append(
            "本轮存在未稳定阶跃，不能据此确认 10° FOV 多目标循环的最小驻留时间。"
        )
    center_30 = next((item for item in summary["stages"]
                      if item["name"] == "三目标中心_FOV30"), None)
    if center_30:
        lines.append(
            f"30°中心阶段三目标同时位于引擎几何 FOV 的样本比例为 "
            f"{center_30['all_targets_inside_fov_ratio']:.1%}。"
        )
    lines.extend([
        "",
        "该结论不包含截图链路、视觉检测和目标重捕获延迟；是否采用 10° 循环或 30° "
        "联合包络，还需要把这些耗时叠加到本报告的相机执行时间上。",
        "",
        f"> 范围限制：{summary['scope_limit']}",
        "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0,
                        help="仿真时长（秒），默认 20；完整固定阶跃序列至少需要 20 秒")
    parser.add_argument("--output", default=str(OUTPUT_ROOT / "camera-control-study"),
                        help="新输出目录；必须位于项目外的 ../output 下")
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--sim-binary", default=str(RUNTIME_ROOT / "opensim-sim.exe"))
    parser.add_argument("--pan-tolerance", type=float, default=0.2)
    parser.add_argument("--tilt-tolerance", type=float, default=0.2)
    parser.add_argument("--fov-tolerance", type=float, default=0.1)
    parser.add_argument("--stable-window", type=float, default=0.3)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.duration < 20.0:
        parser.error("--duration 至少为 20 秒，才能覆盖完整阶跃序列")
    if min(args.pan_tolerance, args.tilt_tolerance,
           args.fov_tolerance, args.stable_window) <= 0:
        parser.error("误差阈值和稳定窗口必须大于 0")
    output = Path(args.output).resolve()
    allowed_output = OUTPUT_ROOT.resolve()
    try:
        output.relative_to(allowed_output)
    except ValueError:
        parser.error(f"--output 必须位于 {allowed_output} 下")
    output.mkdir(parents=True, exist_ok=False)

    scenario = _make_scenario()
    scenario_path = output / "scenario.json"
    scenario_path.write_text(json.dumps(scenario, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    source = Path(__file__).read_bytes()
    (output / "runner_snapshot.py").write_bytes(source)
    metadata = {
        "duration_s": args.duration,
        "stages": [asdict(stage) for stage in STAGES],
        "scenario_source": str(CAMERA_SCENARIO),
        "control_rate_hz": 60.0,
        "runner_sha256": hashlib.sha256(source).hexdigest(),
        "engine_sha256": hashlib.sha256(Path(args.sim_binary).read_bytes()).hexdigest(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=PROJECT_ROOT).decode().strip(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip(),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = ScenarioConfig(
        "coop_decoy", str(scenario_path), args.duration,
        control_rate_hz=60.0,
        redis_host=args.redis_host, redis_port=args.redis_port,
        output_dir=str(output), sim_binary=str(Path(args.sim_binary).resolve()),
        start_sim_flag=True, photo_mode="off", seed=0, quiet=args.quiet,
    )
    with (output / "run.log").open("w", encoding="utf-8", buffering=1) as stream:
        def log(message):
            print(message, flush=True)
            stream.write(str(message) + "\n")

        runner = CameraControlRunner(cfg, output, log=log)
        try:
            runner.run()
        finally:
            runner.trace.close()
        tolerances = {
            "pan_deg": args.pan_tolerance,
            "tilt_deg": args.tilt_tolerance,
            "fov_deg": args.fov_tolerance,
        }
        summary = _summarize(runner.rows, tolerances, args.stable_window)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_report(output, summary)
        log(f"[camera-control] 逐拍记录：{output / 'ticks.jsonl'}")
        log(f"[camera-control] 摘要：{output / 'summary.json'}")
        log(f"[camera-control] 报告：{output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
