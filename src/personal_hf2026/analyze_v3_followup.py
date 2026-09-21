# 修改时间：2026-09-21（静止帧审计修复）。
# 修改目的：避免把二点五秒拟合窗口点数误当成跨窗口累计的七帧静止确认数。
# 修改内容：分别验证拟合最小点数和 ACTIVE 连续静止帧数，并要求完成帧处于确认阶段。
# 修改时间：2026-09-21。
# 修改目的：为 V3 MASTER 丢失退出和鲁棒静止完成提供有界、可复用的离线验收入口。
# 修改内容：审计预测瞄准、五秒退出、H=0 鲁棒拟合和不同视觉帧，并把裁判销毁仅作旁路交叉核对。
"""离线审计 V3 后续协同修复的 A/B 运行证据。"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
import math
from pathlib import Path
import re


MAX_TRACE_BYTES = 32 * 1024 * 1024
MAX_TRACE_RECORDS = 25_000
MAX_TRACE_LINE_BYTES = 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_RUNS = 64
AIM_TOLERANCE_M = 2.0
EXIT_MARGIN_S = 0.75
STATIC_WINDOW_S = (2.0, 3.5)
_HALT = re.compile(
    r"target\s+(?P<uid>\S+)\s+destroyed.*halted at\s*"
    r"\((?P<lat>[-+0-9.eE]+),\s*(?P<lon>[-+0-9.eE]+)\)"
)


def _json(path: Path, default=None, byte_cap=MAX_JSON_BYTES):
    if not path.is_file():
        return default
    with path.open("rb") as stream:
        raw = stream.read(byte_cap + 1)
    if len(raw) > byte_cap:
        raise ValueError(f"JSON 超过 {byte_cap} 字节上限：{path}")
    return json.loads(raw.decode("utf-8-sig"))


def _trace(path: Path, byte_cap: int, record_cap: int):
    rows, invalid, used, truncated = [], [], 0, False
    if not path.is_file():
        return rows, {"file_bytes": None, "bytes_read": 0, "records_read": 0,
                      "invalid_lines": [], "truncated_by_reader": False}
    with path.open("rb") as stream:
        line_number = 0
        while len(rows) < record_cap:
            remaining = byte_cap - used
            if remaining <= 0:
                truncated = True
                break
            line_limit = min(remaining, MAX_TRACE_LINE_BYTES)
            raw = stream.readline(line_limit + 1)
            if not raw:
                break
            line_number += 1
            if len(raw) > line_limit:
                truncated = True
                break
            used += len(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if len(invalid) < 20:
                    invalid.append({"line": line_number, "error": str(exc)})
                continue
            if isinstance(row, dict):
                rows.append(row)
        if len(rows) >= record_cap and stream.read(1):
            truncated = True
    return rows, {
        "file_bytes": path.stat().st_size, "bytes_read": used,
        "records_read": len(rows), "invalid_lines": invalid,
        "truncated_by_reader": truncated,
    }


def _num(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _time(row):
    return next((_num(row.get(key)) for key in (
        "agent_time_s", "observation_time_s", "recorded_time_s"
    ) if _num(row.get(key)) is not None), 0.0)


def _state(row, side="after"):
    value = row.get(side)
    return value if isinstance(value, dict) else {}


def _session(value):
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return tuple(str(item) for item in value[:3])


def _position(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    lat, lon = _num(value[0]), _num(value[1])
    return None if lat is None or lon is None else (lat, lon)


def _distance(first, second):
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * 6_378_137.0 * math.asin(min(1.0, math.sqrt(value)))


def _transition(state):
    last = state.get("last_transition")
    last = last if isinstance(last, dict) else {}
    return (state.get("event") or last.get("event"), last.get("reason"),
            _session(last.get("session") or state.get("session")),
            _num(last.get("at_s")))


def _check(status, conclusion, evidence=None):
    return {"status": status, "conclusion": conclusion,
            "evidence": evidence or {}}


def _status(checks):
    statuses = [item["status"] for item in checks.values()]
    if "failed" in statuses:
        return "failed"
    if statuses and all(item == "passed" for item in statuses):
        return "passed"
    if all(item == "insufficient_evidence" for item in statuses):
        return "insufficient_evidence"
    if all(item in {"not_observed", "insufficient_evidence"} for item in statuses):
        return "not_observed"
    return "inconclusive"


def _previous_master(decisions, index, uid, session):
    before = _state(decisions[index], "before")
    if before.get("role") == "MASTER" and _session(before.get("session")) == session:
        return before
    for row in reversed(decisions[:index]):
        state = _state(row)
        if (str(row.get("uid")) == uid and state.get("role") == "MASTER"
                and _session(state.get("session")) == session):
            return state
    return {}


def _a_checks(decisions):
    fields = any(
        all(key in _state(row) for key in (
            "track_state", "track_last_seen_age_s", "master_lost_timeout_s"
        )) for row in decisions
    )
    coasting, aim_violations = [], []
    overdue, exits, seen_exits = {}, {}, set()
    for index, row in enumerate(decisions):
        state, uid = _state(row), str(row.get("uid"))
        active_master = state.get("role") == "MASTER" and state.get("phase") == "COOP_ACTIVE"
        if active_master and state.get("track_state") == "COASTING":
            predicted = _position(state.get("track_predict_position"))
            aimed = _position((state.get("aim_target_lat"), state.get("aim_target_lon")))
            error = None if predicted is None or aimed is None else _distance(predicted, aimed)
            sample = {"uid": uid, "time_s": _time(row),
                      "session": list(_session(state.get("session")) or ()),
                      "last_seen_age_s": _num(state.get("track_last_seen_age_s")),
                      "aim_error_m": error}
            coasting.append(sample)
            if error is None or error > AIM_TOLERANCE_M:
                aim_violations.append(sample)
        age, timeout = (_num(state.get("track_last_seen_age_s")),
                        _num(state.get("master_lost_timeout_s")))
        session = _session(state.get("session"))
        if active_master and session and age is not None and timeout is not None and age >= timeout:
            overdue[(uid, session)] = max(age, overdue.get((uid, session), 0.0))
        event, reason, ended_session, transition_at = _transition(state)
        if event != "session_cancelled" or reason != "master_track_timeout":
            continue
        exit_key = (uid, ended_session, transition_at)
        if exit_key in seen_exits:
            continue
        seen_exits.add(exit_key)
        previous = _previous_master(decisions, index, uid, ended_session)
        age = (_num(state.get("track_last_seen_age_s"))
               if _num(state.get("track_last_seen_age_s")) is not None
               else _num(previous.get("track_last_seen_age_s")))
        timeout = (_num(state.get("master_lost_timeout_s"))
                   if _num(state.get("master_lost_timeout_s")) is not None
                   else _num(previous.get("master_lost_timeout_s")))
        lag = None if age is None or timeout is None else age - timeout
        passed = bool(
            ended_session and age is not None and timeout is not None
            # before 是退出前一拍；允许其在阈值下方一个采样裕量内。
            and -EXIT_MARGIN_S <= lag <= EXIT_MARGIN_S
            and state.get("phase") == "SEARCH"
            and str(state.get("role") or "NONE") in {"NONE", "SEARCH"}
        )
        exits[(uid, ended_session)] = {
            "uid": uid, "time_s": _time(row), "session": list(ended_session or ()),
            "transition_at_s": transition_at,
            "last_seen_age_s": age, "timeout_s": timeout,
            "exit_lag_s": lag, "direct_search": state.get("phase") == "SEARCH",
            "passed": passed,
        }
    missing_exits = [
        {"uid": uid, "session": list(session), "max_age_s": age}
        for (uid, session), age in overdue.items() if (uid, session) not in exits
    ]
    if not fields:
        unavailable = _check(
            "insufficient_evidence",
            "旧 trace 缺少轨迹状态、last_seen_age 和五秒阈值。",
        )
        return unavailable, unavailable
    aim_evidence = {
        "sample_count": len(coasting), "tolerance_m": AIM_TOLERANCE_M,
        "max_error_m": max((item["aim_error_m"] for item in coasting
                            if item["aim_error_m"] is not None), default=None),
        "violations": aim_violations[:20],
    }
    if not coasting:
        aim_check = _check("not_observed", "本轮没有 MASTER/COASTING 样本。", aim_evidence)
    elif aim_violations:
        aim_check = _check("failed", "COASTING 未持续瞄准 predict_position(now)。", aim_evidence)
    else:
        aim_check = _check("passed", "COASTING 均瞄准 predict_position(now)。", aim_evidence)
    exit_evidence = {"exit_margin_s": EXIT_MARGIN_S,
                     "exit_cases": list(exits.values()),
                     "timed_out_without_exit": missing_exits}
    if missing_exits or any(not item["passed"] for item in exits.values()):
        exit_check = _check("failed", "MASTER 超时后滞留或未直接回 SEARCH。", exit_evidence)
    elif exits:
        exit_check = _check("passed", "MASTER 在约五秒阈值后直接回到 SEARCH。", exit_evidence)
    else:
        exit_check = _check("not_observed", "本轮没有触发 MASTER 丢失超时。", exit_evidence)
    return aim_check, exit_check


def _stationary(row):
    for state in (_state(row), _state(row, "before")):
        value = state.get("stationary")
        if isinstance(value, dict):
            return value
    return None


def _b_checks(decisions):
    fields = any(_stationary(row) is not None for row in decisions)
    cases, seen = [], set()
    for row in decisions:
        state = _state(row)
        event, reason, session, transition_at = _transition(state)
        if event != "coordination_finished" or reason != "master_static":
            continue
        key = (str(row.get("uid")), session, transition_at)
        if key in seen:
            continue
        seen.add(key)
        item = _stationary(row) or {}
        east, north = _num(item.get("east_velocity_mps")), _num(item.get("north_velocity_mps"))
        speed, threshold = _num(item.get("speed_mps")), _num(item.get("speed_threshold_mps"))
        span = _num(item.get("window_span_s"))
        try:
            distinct = int(item.get("distinct_frame_count"))
            fit_min_points = int(item.get("fit_min_points"))
            consecutive = int(item.get("stationary_consecutive_frames"))
            required = int(item.get("required_stationary_frames"))
        except (TypeError, ValueError):
            distinct = fit_min_points = consecutive = required = None
        fit_ok = bool(
            _position(item.get("position_h0")) and span is not None
            and STATIC_WINDOW_S[0] <= span <= STATIC_WINDOW_S[1]
            and None not in (east, north, speed, threshold) and speed <= threshold
            and abs(speed - math.hypot(east, north)) <= 0.05
        )
        frames_ok = bool(
            item.get("frame_id") is not None and _num(item.get("source_sim_time")) is not None
            and None not in (distinct, fit_min_points, consecutive, required)
            and distinct >= fit_min_points and consecutive >= required
            and item.get("confirmation_enabled") is True
            and item.get("ready") is True
        )
        cases.append({
            "uid": str(row.get("uid")), "time_s": _time(row),
            "session": list(session or ()), "stationary": item,
            "fit_passed": fit_ok, "distinct_frames_passed": frames_ok,
            "direct_search": state.get("phase") == "SEARCH",
        })
    ready_without_edge = [
        {"uid": str(row.get("uid")), "time_s": _time(row)}
        for row in decisions
        if isinstance(_state(row).get("stationary"), dict)
        and _state(row)["stationary"].get("ready") is True
        and _state(row).get("role") == "MASTER"
        and _state(row).get("phase") == "COOP_ACTIVE"
    ] if fields and not cases else []
    if not fields:
        unavailable = _check(
            "insufficient_evidence",
            "旧 trace 缺少逐帧 H=0 鲁棒拟合和不同视觉帧计数。",
        )
        return unavailable, unavailable, unavailable
    if not cases:
        status = "failed" if ready_without_edge else "not_observed"
        result = _check(status, "静止已 ready 却未完成。" if ready_without_edge
                        else "本轮没有触发 master_static。",
                        {"ready_without_completion": ready_without_edge[:20]})
        return result, result, result
    fit_ok = all(item["fit_passed"] for item in cases)
    frames_ok = all(item["distinct_frames_passed"] for item in cases)
    edge_ok = all(item["direct_search"] for item in cases)
    return (
        _check("passed" if fit_ok else "failed",
               "master_static 有合格的 2–3 秒 H=0 鲁棒拟合。" if fit_ok
               else "至少一个 master_static 缺少合格的 H=0 鲁棒拟合。",
               {"cases": cases}),
        _check("passed" if frames_ok else "failed",
               "master_static 由足够多不同视觉帧支持。" if frames_ok
               else "至少一个 master_static 的不同视觉帧证据不足。",
               {"cases": cases}),
        _check("passed" if edge_ok else "failed",
               "master_static 后直接回到 SEARCH。" if edge_ok
               else "至少一个 master_static 未直接回到 SEARCH。",
               {"cases": cases}),
    )


def _judge(run, directory):
    evaluation = run.get("evaluation") or {}
    if not evaluation:
        path = next(directory.glob("*.evaluation.json"), None)
        evaluation = _json(path, {}) if path is not None else {}
    transitions, previous = [], 0.0
    for row in evaluation.get("score_timeline") or ():
        current = _num(row.get("completion_rate"))
        if current is not None and current > previous:
            transitions.append({"time_s": _num(row.get("sim_time")),
                                "completion_rate": current})
            previous = current
    halted = []
    log = directory / "run.log"
    if log.is_file() and log.stat().st_size <= 256 * 1024:
        for line in log.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            match = _HALT.search(line)
            if match:
                halted.append({"uid": match.group("uid"), "position": [
                    float(match.group("lat")), float(match.group("lon"))
                ]})
    return {
        "basis": "judge_side_corroboration_not_agent_input",
        "n_destroyed": evaluation.get("n_destroyed"),
        "completion_rate_transitions": transitions,
        "halted_target_log_records": halted,
    }


def _legacy(decisions, run, judge):
    terminal = []
    for uid, agent in sorted((run.get("agents") or {}).items()):
        runtime = agent.get("v3_runtime_evidence") or {}
        if not (runtime.get("role") == "MASTER" and runtime.get("phase") == "COOP_ACTIVE"
                and agent.get("track_state") == "LOST"
                and agent.get("coop_lock_valid") is False):
            continue
        session = _session(runtime.get("session"))
        matching = [row for row in decisions
                    if str(row.get("uid")) == str(uid)
                    and _session(_state(row).get("session")) == session
                    and _state(row).get("role") == "MASTER"
                    and _state(row).get("phase") == "COOP_ACTIVE"]
        real = [row for row in matching
                if (_state(row).get("perception") or {}).get("class_name") == "real_vehicle"]
        final_non_real = []
        for row in reversed(matching):
            if (_state(row).get("perception") or {}).get("class_name") == "real_vehicle":
                break
            final_non_real.append(row)
        final_non_real.reverse()
        terminal.append({
            "uid": str(uid), "session": list(session or ()),
            "active_trace_start_s": _time(matching[0]) if matching else None,
            "active_trace_end_s": _time(matching[-1]) if matching else None,
            "last_sampled_real_vehicle_time_s": _time(real[-1]) if real else None,
            "final_non_real_window": None if not final_non_real else {
                "start_s": _time(final_non_real[0]), "end_s": _time(final_non_real[-1]),
                "sample_count": len(final_non_real),
                "source_start_s": _num((_state(final_non_real[0]).get("perception") or {}).get("source_sim_time")),
                "source_end_s": _num((_state(final_non_real[-1]).get("perception") or {}).get("source_sim_time")),
            },
            "terminal_track_state": "LOST", "terminal_coop_lock_valid": False,
            "terminal_follow_position": runtime.get("follow_position"),
            "terminal_target_position_h0": agent.get("v3_target_position"),
            "terminal_frame_id": agent.get("v3_perception_frame_id"),
            "loss_onset_s": None,
            "loss_onset_limit": "旧 trace 未记录 track_state/last_seen_age，不能反推首次跟丢时刻。",
        })
    windows = []
    for transition in judge["completion_rate_transitions"]:
        destroyed_at = transition["time_s"]
        groups = {}
        for row in decisions:
            state = _state(row)
            if state.get("role") == "MASTER" and state.get("phase") == "COOP_ACTIVE":
                groups.setdefault((str(row.get("uid")), _session(state.get("session"))), []).append(row)
        for (uid, session), rows in groups.items():
            before = [row for row in rows if _time(row) <= destroyed_at]
            after = [row for row in rows if _time(row) >= destroyed_at]
            if not before or not after or destroyed_at - _time(before[-1]) > 0.75:
                continue
            frames, sources, positions, classes, tracks = [], [], [], Counter(), Counter()
            for row in after:
                state, perception = _state(row), _state(row).get("perception") or {}
                if perception.get("frame_id") is not None:
                    frames.append(str(perception["frame_id"]))
                if _num(perception.get("source_sim_time")) is not None:
                    sources.append(_num(perception["source_sim_time"]))
                if perception.get("class_name") is not None:
                    classes[str(perception["class_name"])] += 1
                if perception.get("track_id") is not None:
                    tracks[str(perception["track_id"])] += 1
                if _position(state.get("follow_position")):
                    positions.append(_position(state["follow_position"]))
            first = positions[0] if positions else None
            terminal_same = any(item["uid"] == uid and item["session"] == list(session or ())
                                for item in terminal)
            windows.append({
                "judge_destroyed_time_s": destroyed_at, "uid": uid,
                "session": list(session or ()), "active_before_s": _time(before[-1]),
                "active_through_s": _time(after[-1]), "terminal_same_session": terminal_same,
                "distinct_sampled_frames": len(set(frames)),
                "source_time_start_s": min(sources) if sources else None,
                "source_time_end_s": max(sources) if sources else None,
                "class_counts": dict(classes), "track_id_counts": dict(tracks),
                "follow_position": first,
                "follow_position_spread_m": max((_distance(first, point) for point in positions), default=None) if first else None,
                "raw_h0_history_available": any(_stationary(row) for row in after),
                "position_limit": "follow_position 是锁存/广播位置，不能替代逐帧 H=0 原始位置。",
            })
    return {"terminal_active_lost_masters": terminal,
            "post_judge_destroy_active_windows": windows}


def analyze_run(directory: Path, byte_cap=MAX_TRACE_BYTES, record_cap=MAX_TRACE_RECORDS):
    directory = directory.resolve()
    rows, integrity = _trace(directory / "coordination_trace.jsonl", byte_cap, record_cap)
    decisions = sorted((row for row in rows if row.get("kind") == "decision"),
                       key=lambda row: (_time(row), str(row.get("uid"))))
    summary = _json(directory / "coordination_trace_summary.json", {}) or {}
    run = _json(directory / "run.json", {}) or {}
    aim, timeout = _a_checks(decisions)
    fit, frames, edge = _b_checks(decisions)
    checks = {
        "a_coasting_predict_aim": aim, "a_master_timeout_exit": timeout,
        "b_robust_h0_stationary_fit": fit, "b_distinct_visual_frames": frames,
        "b_static_completion_edge": edge,
    }
    judge = _judge(run, directory)
    integrity["source_summary"] = summary
    integrity["trace_loss"] = bool(
        integrity["invalid_lines"] or integrity["truncated_by_reader"]
        or summary.get("records_dropped_byte_cap")
        or summary.get("records_dropped_limit") or summary.get("capture_errors")
    )
    if integrity["trace_loss"]:
        for item in checks.values():
            if item["status"] == "passed":
                item["status"] = "inconclusive"
                item["conclusion"] += "；但 trace 不完整，不能作为整轮通过结论。"
                item["evidence"]["trace_loss"] = True
    run_status = _status(checks)
    if integrity["trace_loss"] and run_status == "passed":
        run_status = "inconclusive"
    return {
        "schema_version": 1, "status": run_status,
        "input_dir": str(directory), "checks": checks,
        "trace_integrity": integrity,
        "legacy_baseline_diagnosis": _legacy(decisions, run, judge),
        "judge_cross_check": judge,
        "limitations": [
            "A/B 通过判定只使用 Agent 本机 trace；裁判销毁只定位旧故障窗口。",
            "source_sim_time 不是已验证的相机曝光时刻。",
            "旧 follow_position 不是逐帧 H=0 原始位置。",
        ],
    }


def analyze_input(directory: Path, max_runs=MAX_RUNS,
                  byte_cap=MAX_TRACE_BYTES, record_cap=MAX_TRACE_RECORDS):
    directory = directory.resolve()
    if (directory / "coordination_trace.jsonl").is_file():
        return analyze_run(directory, byte_cap, record_cap)
    root = directory / "runs"
    if not root.is_dir():
        raise ValueError("输入既不是单轮输出，也不包含 runs 子目录")
    run_dirs = list(islice((
        path for path in root.iterdir()
        if path.is_dir() and (path / "run.json").is_file()
    ), max_runs + 1))
    if len(run_dirs) > max_runs:
        raise ValueError(f"run 数超过 --max-runs={max_runs}")
    run_dirs.sort()
    audits = [analyze_run(path, byte_cap, record_cap) for path in run_dirs]
    aggregate = {}
    for name in (audits[0]["checks"] if audits else ()):
        statuses = [item["checks"][name]["status"] for item in audits]
        observed = [item for item in statuses if item not in {
            "not_observed", "insufficient_evidence"
        }]
        lossless_pass = any(
            item["checks"][name]["status"] == "passed"
            and not item["trace_integrity"]["trace_loss"]
            for item in audits
        )
        lossy_pass = any(
            item["checks"][name]["status"] == "passed"
            and item["trace_integrity"]["trace_loss"]
            for item in audits
        )
        status = ("failed" if "failed" in observed else "passed" if lossless_pass
                  else "inconclusive" if lossy_pass
                  else "insufficient_evidence" if "insufficient_evidence" in statuses
                  else "not_observed")
        aggregate[name] = {"status": status, "status_counts": dict(Counter(statuses)),
                           "passed_runs": [Path(item["input_dir"]).name for item in audits
                                           if item["checks"][name]["status"] == "passed"],
                           "failed_runs": [Path(item["input_dir"]).name for item in audits
                                           if item["checks"][name]["status"] == "failed"]}
    stuck = [{"run": Path(item["input_dir"]).name, **entry} for item in audits
             for entry in item["legacy_baseline_diagnosis"]["terminal_active_lost_masters"]]
    windows = [{"run": Path(item["input_dir"]).name, **entry} for item in audits
               for entry in item["legacy_baseline_diagnosis"]["post_judge_destroy_active_windows"]]
    return {
        "schema_version": 1, "status": _status(aggregate),
        "input_dir": str(directory), "run_count": len(audits), "checks": aggregate,
        "baseline_summary": {
            "terminal_active_lost_master_count": len(stuck),
            "terminal_active_lost_masters": stuck,
            "post_judge_destroy_active_window_count": len(windows),
            "post_judge_destroy_active_windows": windows,
            "trace_loss_run_count": sum(item["trace_integrity"]["trace_loss"] for item in audits),
        },
        "runs": audits,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-runs", type=int, default=MAX_RUNS)
    parser.add_argument("--max-trace-bytes", type=int, default=MAX_TRACE_BYTES)
    parser.add_argument("--max-trace-records", type=int, default=MAX_TRACE_RECORDS)
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f"--input 不是目录：{args.input}")
    if min(args.max_runs, args.max_trace_bytes, args.max_trace_records) <= 0:
        parser.error("所有 max 参数必须大于零")
    try:
        audit = analyze_input(args.input, args.max_runs,
                              args.max_trace_bytes, args.max_trace_records)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"], "output": str(output),
        "run_count": audit.get("run_count", 1),
        "checks": {name: item["status"] for name, item in audit["checks"].items()},
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
