# 修改时间：2026-09-21。
# 修改目的：让真实场景简化协同的关键行为可以由有界运行轨迹逐项复核。
# 修改内容：新增五帧、选从机、距离门控、持续瞄准、结束回搜索和计数传播的离线审计。
"""离线审计 PersonalV3 简化协同运行证据。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_MASTER_GATE_M = 220.0
EXPECTED_TARGET_GATE_M = 200.0
SAMPLE_GAP_LIMIT_S = 0.75
POSITION_JOIN_LIMIT_S = 0.75


def _read_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_trace(path: Path):
    rows = []
    invalid_lines = []
    if not path.is_file():
        return rows, invalid_lines
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_lines.append({"line": line_number, "error": str(exc)})
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                invalid_lines.append({"line": line_number, "error": "row_not_object"})
    return rows, invalid_lines


def _check(status, conclusion, evidence=None):
    return {
        "status": status,
        "conclusion": conclusion,
        "evidence": evidence or {},
    }


def _session_key(value):
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _state(row, side="after"):
    value = row.get(side)
    return value if isinstance(value, dict) else {}


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _time(row):
    for key in ("agent_time_s", "observation_time_s", "recorded_time_s"):
        value = _number(row.get(key))
        if value is not None:
            return value
    return 0.0


def _role(state):
    return str(state.get("role") or "NONE").upper()


def _phase(state):
    return str(state.get("phase") or state.get("state") or "UNKNOWN").upper()


def _completed_sessions(state):
    values = state.get("completed_sessions") or []
    return {_session_key(value) for value in values if value is not None}


def _completion_count(state):
    for key in ("completed_count", "estimated_destroyed_count"):
        value = state.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return len(_completed_sessions(state))


def _pose(row):
    value = row.get("self_pose")
    if not isinstance(value, dict):
        return None
    lat = _number(value.get("lat"))
    lon = _number(value.get("lon"))
    return None if lat is None or lon is None else (lat, lon)


def _distance_m(first, second):
    lat1, lon1 = first
    lat2, lon2 = second
    mean_lat = math.radians((lat1 + lat2) * 0.5)
    north = math.radians(lat2 - lat1) * 6_378_137.0
    east = math.radians(lon2 - lon1) * 6_378_137.0 * math.cos(mean_lat)
    return math.hypot(east, north)


def _nearest_row(rows, uid, when):
    candidates = [
        row for row in rows
        if row.get("uid") == uid and _pose(row) is not None
        and abs(_time(row) - when) <= POSITION_JOIN_LIMIT_S
    ]
    return min(candidates, key=lambda item: abs(_time(item) - when), default=None)


def _master_starts(decisions):
    """优先识别 proposal_started；旧轨迹回退到 MASTER 新会话边沿。"""
    starts = []
    for row in decisions:
        before = _state(row, "before")
        after = _state(row)
        session = _session_key(after.get("session"))
        if session is None:
            continue
        event = str(after.get("event") or "")
        new_session = _session_key(before.get("session")) != session
        proposal_advanced = int(after.get("proposal_count") or 0) > int(
            before.get("proposal_count") or 0
        )
        legacy_master_edge = (
            _role(after) == "MASTER"
            and (_role(before) != "MASTER" or new_session)
        )
        if event == "proposal_started" or proposal_advanced or legacy_master_edge:
            starts.append(row)
    return starts


def _pair_accepts(decisions):
    """识别 MASTER 收到 FOLLOWER 接受的边沿。"""
    accepts = []
    for row in decisions:
        before = _state(row, "before")
        after = _state(row)
        event = str(after.get("event") or "")
        partner = after.get("partner_uid")
        active_master = (
            _role(after) == "MASTER"
            and _phase(after) in {"COOP_INIT", "COOP_ACTIVE", "INIT", "ACTIVE"}
            and after.get("session") is not None
            and partner is not None
        )
        partner_edge = partner != before.get("partner_uid")
        if event == "follower_accept_received" or (active_master and partner_edge):
            accepts.append(row)
    return accepts


def _five_frame_check(perceptions, starts):
    if not starts:
        return _check("not_observed", "轨迹中没有 MASTER 发起会话。")
    cases = []
    all_passed = True
    for start in starts:
        uid = start.get("uid")
        when = _time(start)
        candidates = [
            row for row in perceptions
            if row.get("uid") == uid and _time(row) <= when
            and when - _time(row) <= 10.0
        ]
        ready = [
            row for row in candidates
            if bool((_state(row).get("confirmation") or {}).get("ready"))
        ]
        terminal = ready[-1] if ready else None
        confirmation = (_state(terminal).get("confirmation") or {}) if terminal else {}
        required = int(confirmation.get("required") or 0)
        count = int(confirmation.get("count") or 0)
        if terminal is None:
            distinct = []
        else:
            prior = [row for row in candidates if _time(row) <= _time(terminal)]
            streak = []
            for row in reversed(prior):
                item = _state(row).get("confirmation") or {}
                item_count = int(item.get("count") or 0)
                if item_count <= 0:
                    break
                streak.append(row)
                if len(streak) >= max(required, 1):
                    break
            distinct = sorted({
                (_state(row).get("perception") or {}).get("frame_id")
                for row in streak
                if (_state(row).get("perception") or {}).get("frame_id") is not None
            })
        passed = required >= 5 and count >= required and len(distinct) >= required
        all_passed = all_passed and passed
        cases.append({
            "uid": uid,
            "session": _state(start).get("session"),
            "master_start_s": when,
            "required_frames": required,
            "observed_count": count,
            "distinct_frame_count": len(distinct),
            "ready_time_s": None if terminal is None else _time(terminal),
            "passed": passed,
        })
    return _check(
        "passed" if all_passed else "failed",
        "每次 MASTER 发起前均记录到至少五个不同像素帧的连续确认。"
        if all_passed else "至少一次 MASTER 发起缺少五个不同像素帧的确认链。",
        {"cases": cases},
    )


def _master_start_check(starts):
    if not starts:
        return _check("not_observed", "未观察到 MASTER 发起会话。")
    events = [{
        "uid": row.get("uid"),
        "time_s": _time(row),
        "session": _state(row).get("session"),
        "partner_uid": _state(row).get("partner_uid"),
        "event": _state(row).get("event"),
    } for row in starts]
    return _check("passed", "观察到 MASTER 会话发起边沿。", {"events": events})


def _nearest_follower_check(decisions, accepts):
    if not accepts:
        return _check("not_observed", "未观察到可核对的 MASTER/FOLLOWER 配对。")
    uids = sorted({str(row.get("uid")) for row in decisions if row.get("uid")})
    cases = []
    conclusive = []
    for start in accepts:
        state = _state(start)
        master_uid = str(start.get("uid"))
        partner_uid = state.get("partner_uid")
        when = _time(start)
        master_row = _nearest_row(decisions, master_uid, when)
        master_pose = _pose(master_row) if master_row else None
        distances = {}
        eligible = []
        for uid in uids:
            if uid == master_uid:
                continue
            peer = _nearest_row(decisions, uid, when)
            peer_pose = _pose(peer) if peer else None
            if master_pose is not None and peer_pose is not None:
                distances[uid] = _distance_m(master_pose, peer_pose)
            peer_state = _state(peer) if peer else {}
            if uid == str(partner_uid) or (
                _role(peer_state) == "NONE" and _phase(peer_state) in {"SEARCH", "HOLD_TARGET"}
            ):
                eligible.append(uid)
        selected_distance = distances.get(str(partner_uid))
        available_distances = {
            uid: distances[uid] for uid in eligible if uid in distances
        }
        nearest = min(available_distances.values(), default=None)
        passed = (
            partner_uid is not None and selected_distance is not None and nearest is not None
            and selected_distance <= nearest + 25.0
        )
        conclusive.append(nearest is not None and selected_distance is not None)
        cases.append({
            "master_uid": master_uid,
            "partner_uid": partner_uid,
            "session": state.get("session"),
            "time_s": when,
            "eligible_distances_m": available_distances,
            "selected_distance_m": selected_distance,
            "nearest_distance_m": nearest,
            "join_tolerance_s": POSITION_JOIN_LIMIT_S,
            "distance_tolerance_m": 25.0,
            "basis": "offline_combination_of_each_agents_own_pose",
            "passed": passed,
        })
    if not all(conclusive):
        return _check("inconclusive", "配对已出现，但缺少同一时段的三机自身位置采样。", {"cases": cases})
    passed = all(case["passed"] for case in cases)
    return _check(
        "passed" if passed else "failed",
        "被接受的 FOLLOWER 是采样误差范围内最近的可用无人机。"
        if passed else "至少一次配对未选择采样位置下最近的可用无人机。",
        {"cases": cases},
    )


def _distance_gate_check(decisions):
    follower_rows = [row for row in decisions if _role(_state(row)) == "FOLLOWER"]
    measured = []
    for row in follower_rows:
        state = _state(row)
        master_distance = _number(state.get("uav_distance_to_master_m"))
        if master_distance is None:
            master_distance = _number(state.get("partner_distance_m"))
        target_distance = _number(state.get("target_distance_m"))
        ready = state.get("rendezvous_ready")
        if master_distance is None or target_distance is None or ready is None:
            continue
        measured.append((row, master_distance, target_distance, bool(ready)))
    if not measured:
        return _check("not_observed", "FOLLOWER 轨迹中没有两级距离及门控结果。")
    threshold_values = {
        (
            _number(_state(row).get("master_gate_m")),
            _number(_state(row).get("target_gate_m")),
        )
        for row, _, _, _ in measured
    }
    thresholds_confirmed = (EXPECTED_MASTER_GATE_M, EXPECTED_TARGET_GATE_M) in threshold_values
    blocked = []
    ready_rows = []
    violations = []
    for row, master_distance, target_distance, ready in measured:
        inside = (
            master_distance <= EXPECTED_MASTER_GATE_M
            and target_distance <= EXPECTED_TARGET_GATE_M
        )
        evidence = {
            "uid": row.get("uid"),
            "time_s": _time(row),
            "master_distance_m": master_distance,
            "target_distance_m": target_distance,
            "rendezvous_ready": ready,
        }
        if ready and inside:
            ready_rows.append(evidence)
        elif not ready and not inside:
            blocked.append(evidence)
        else:
            violations.append(evidence)
    evidence = {
        "expected_master_gate_m": EXPECTED_MASTER_GATE_M,
        "expected_target_gate_m": EXPECTED_TARGET_GATE_M,
        "thresholds_confirmed_in_runtime_state": thresholds_confirmed,
        "blocked_sample_count": len(blocked),
        "ready_sample_count": len(ready_rows),
        "violations": violations[:20],
        "first_blocked_sample": blocked[0] if blocked else None,
        "first_ready_sample": ready_rows[0] if ready_rows else None,
    }
    if violations:
        return _check("failed", "观察到两级距离门控与 220m/200m 条件不一致。", evidence)
    if thresholds_confirmed and blocked and ready_rows:
        return _check("passed", "门外保持未就绪，进入 220m/200m 双门后转为就绪。", evidence)
    return _check("inconclusive", "门控样本存在，但阈值元数据或门内外两侧证据不完整。", evidence)


def _has_aim_command(row):
    for command in row.get("commands") or []:
        if str(command.get("verb")) == "component.gimbal_tracking.set_orientation":
            return True
    return False


def _continuous_aim_check(decisions, trace_loss):
    required = []
    follower_aim_latched = set()
    for row in decisions:
        state = _state(row)
        role = _role(state)
        phase = _phase(state)
        if role == "MASTER" and phase not in {"SEARCH", "MISSION_DONE", "DONE"}:
            required.append(row)
        elif role == "FOLLOWER":
            key = (str(row.get("uid")), _session_key(state.get("session")))
            if bool(state.get("rendezvous_ready")):
                follower_aim_latched.add(key)
            if key in follower_aim_latched:
                required.append(row)
    if not required:
        return _check("not_observed", "未观察到需要持续瞄准的协同采样区间。")
    missing = []
    by_uid = {}
    for row in required:
        state = _state(row)
        aiming = state.get("aiming_enabled")
        commanded = _has_aim_command(row)
        if aiming is not True or not commanded:
            missing.append({
                "uid": row.get("uid"), "time_s": _time(row),
                "aiming_enabled": aiming, "point_gimbal_command": commanded,
            })
        group = "|".join((
            str(row.get("uid")),
            _session_key(state.get("session")) or "none",
            _role(state),
        ))
        by_uid.setdefault(group, []).append(_time(row))
    max_gaps = {
        uid: max((right - left for left, right in zip(times, times[1:])), default=0.0)
        for uid, times in by_uid.items()
    }
    gap_violations = {uid: gap for uid, gap in max_gaps.items() if gap > SAMPLE_GAP_LIMIT_S}
    evidence = {
        "required_sample_count": len(required),
        "missing_or_disabled_samples": missing[:20],
        "max_sample_gap_s_by_uid": max_gaps,
        "sample_gap_limit_s": SAMPLE_GAP_LIMIT_S,
        "trace_loss": trace_loss,
    }
    if missing:
        return _check("failed", "协同采样中出现未启用瞄准或未发送云台姿态命令。", evidence)
    if trace_loss or gap_violations:
        return _check("inconclusive", "已记录样本均持续瞄准，但轨迹丢失或采样间隔过大。", evidence)
    return _check("passed", "MASTER 及门控后的 FOLLOWER 在全部 0.5s 有界样本中持续瞄准。", evidence)


def _finish_reason_check(decisions):
    completions = []
    for row in decisions:
        before = _state(row, "before")
        after = _state(row)
        if _completion_count(after) <= _completion_count(before):
            continue
        if _role(before) != "MASTER":
            continue
        reason = str(after.get("finish_reason") or "")
        normalized = reason.lower()
        static_reason = any(token in normalized for token in (
            "static", "stationary", "stopped", "静止",
        ))
        decoy_reason = any(token in normalized for token in ("decoy", "诱饵"))
        decoy_count = int(after.get("decoy_only_count") or before.get("decoy_only_count") or 0)
        decoy_required = int(
            after.get("decoy_only_required") or before.get("decoy_only_required") or 0
        )
        only_decoys = (after.get("perception") or {}).get("only_decoys")
        if only_decoys is None:
            only_decoys = (before.get("perception") or {}).get("only_decoys")
        decoy_threshold_met = (
            decoy_reason and decoy_required > 0 and decoy_count >= decoy_required
        )
        allowed = static_reason or decoy_threshold_met
        completions.append({
            "uid": row.get("uid"),
            "time_s": _time(row),
            "session": after.get("last_completed_session") or before.get("session"),
            "finish_reason": reason or None,
            "decoy_only_count": decoy_count,
            "decoy_only_required": decoy_required,
            "only_decoys": only_decoys,
            "decoy_threshold_met": decoy_threshold_met,
            "allowed": allowed,
        })
    if not completions:
        return _check("not_observed", "未观察到完成计数增加。")
    passed = all(item["allowed"] for item in completions)
    return _check(
        "passed" if passed else "failed",
        "所有完成均明确由 MASTER 静止判定或连续只见诱饵触发。"
        if passed else "至少一次完成缺少静止/连续诱饵结束原因。",
        {"completions": completions},
    )


def _return_and_propagation_checks(decisions, accepts):
    pair_by_session = {}
    for row in accepts:
        state = _state(row)
        session = _session_key(state.get("session"))
        if session is not None and state.get("partner_uid") is not None:
            pair_by_session[session] = (str(row.get("uid")), str(state["partner_uid"]))

    completion_sessions = set()
    for row in decisions:
        before = _state(row, "before")
        after = _state(row)
        if _completion_count(after) > _completion_count(before):
            completion_sessions.update(_completed_sessions(after) - _completed_sessions(before))
            previous = _session_key(before.get("session"))
            if previous is not None:
                completion_sessions.add(previous)
    if not completion_sessions:
        none = _check("not_observed", "未观察到可核对的完成会话。")
        return none, none

    return_cases = []
    propagation_cases = []
    all_uids = sorted({str(row.get("uid")) for row in decisions if row.get("uid")})
    for session in sorted(completion_sessions):
        pair = pair_by_session.get(session)
        returned = {}
        if pair is not None:
            for uid in pair:
                matching = [
                    row for row in decisions if str(row.get("uid")) == uid
                    and _session_key(_state(row, "before").get("session")) == session
                    and _phase(_state(row)) == "SEARCH"
                ]
                returned[uid] = None if not matching else _time(matching[0])
        return_cases.append({
            "session_key": session,
            "pair": pair,
            "search_return_time_s": returned,
            "passed": pair is not None and all(value is not None for value in returned.values()),
        })

        propagated = {}
        for uid in all_uids:
            matching = [
                row for row in decisions if str(row.get("uid")) == uid
                and session in _completed_sessions(_state(row))
            ]
            propagated[uid] = None if not matching else {
                "time_s": _time(matching[0]),
                "completed_count": _completion_count(_state(matching[0])),
            }
        propagation_cases.append({
            "session_key": session,
            "received_by_uid": propagated,
            "passed": bool(all_uids) and all(value is not None for value in propagated.values()),
        })

    return_passed = all(item["passed"] for item in return_cases)
    propagation_passed = all(item["passed"] for item in propagation_cases)
    return (
        _check(
            "passed" if return_passed else "failed",
            "每个完成会话的 MASTER 和 FOLLOWER 均回到 SEARCH。"
            if return_passed else "至少一个完成会话缺少双机回到 SEARCH 的证据。",
            {"cases": return_cases},
        ),
        _check(
            "passed" if propagation_passed else "failed",
            "完成会话已传播到全部 Agent 的完成集合。"
            if propagation_passed else "至少一个完成会话未传播到全部 Agent。",
            {"cases": propagation_cases},
        ),
    )


def analyze_run(input_dir: Path, output_path: Path | None = None):
    input_dir = Path(input_dir).resolve()
    output_path = (Path(output_path).resolve() if output_path is not None
                   else input_dir / "coordination_audit.json")
    trace_path = input_dir / "coordination_trace.jsonl"
    trace_summary = _read_json(input_dir / "coordination_trace_summary.json", {}) or {}
    run = _read_json(input_dir / "run.json", {}) or {}
    perception = _read_json(input_dir / "perception_summary.json", {}) or {}
    evaluation_files = sorted(input_dir.glob("*.evaluation.json"))
    rows, invalid_lines = _read_trace(trace_path)

    inventory = {
        "observations_jsonl": (input_dir / "observations.jsonl").is_file(),
        "run_json": (input_dir / "run.json").is_file(),
        "perception_summary_json": (input_dir / "perception_summary.json").is_file(),
        "evaluation_json_count": len(evaluation_files),
        "coordination_trace_jsonl": trace_path.is_file(),
        "coordination_trace_summary_json": bool(trace_summary),
    }
    if not rows:
        checks = {
            name: _check("not_observed", "缺少 coordination_trace.jsonl，结束快照不能证明事件时序。")
            for name in (
                "five_frame_confirmation", "master_initiated",
                "nearest_follower_accepted", "distance_gates_220m_200m",
                "continuous_aiming", "master_finish_reason",
                "both_returned_to_search", "completion_count_propagated",
            )
        }
        audit = {
            "schema_version": 1,
            "status": "insufficient_evidence",
            "input_dir": str(input_dir),
            "artifact_inventory": inventory,
            "trace_integrity": {
                "row_count": 0,
                "invalid_lines": invalid_lines,
                "summary": trace_summary,
            },
            "checks": checks,
            "evaluation_cross_check": {
                "n_destroyed": (run.get("evaluation") or {}).get("n_destroyed"),
                "perception_worker_error": perception.get("worker_error"),
            },
            "limitations": [
                "run.json 的 Agent 字段是结束时快照，不能重建协同边沿。",
                "perception_summary.json 只有 worker 总量，不能证明五个连续真车帧。",
                "evaluation 的 score_timeline 不含 Agent 角色、门控、瞄准或结束原因。",
            ],
        }
        output_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return audit

    decisions = [row for row in rows if row.get("kind") == "decision"]
    perceptions = [row for row in rows if row.get("kind") == "perception"]
    decisions.sort(key=lambda row: (_time(row), str(row.get("uid"))))
    perceptions.sort(key=lambda row: (_time(row), str(row.get("uid"))))
    starts = _master_starts(decisions)
    accepts = _pair_accepts(decisions)
    trace_loss = bool(
        invalid_lines
        or trace_summary.get("records_dropped_byte_cap")
        or trace_summary.get("records_dropped_limit")
        or trace_summary.get("capture_errors")
    )
    returned, propagated = _return_and_propagation_checks(decisions, accepts)
    checks = {
        "five_frame_confirmation": _five_frame_check(perceptions, starts),
        "master_initiated": _master_start_check(starts),
        "nearest_follower_accepted": _nearest_follower_check(decisions, accepts),
        "distance_gates_220m_200m": _distance_gate_check(decisions),
        "continuous_aiming": _continuous_aim_check(decisions, trace_loss),
        "master_finish_reason": _finish_reason_check(decisions),
        "both_returned_to_search": returned,
        "completion_count_propagated": propagated,
    }
    statuses = [item["status"] for item in checks.values()]
    if "failed" in statuses:
        status = "failed"
    elif statuses and all(item == "passed" for item in statuses) and not trace_loss:
        status = "passed"
    elif all(item == "not_observed" for item in statuses):
        status = "insufficient_evidence"
    else:
        status = "inconclusive"

    evaluation = run.get("evaluation") or {}
    final_counts = {
        uid: _completion_count(_state(row))
        for uid in sorted({str(row.get("uid")) for row in decisions if row.get("uid")})
        for row in [next(
            item for item in reversed(decisions) if str(item.get("uid")) == uid
        )]
    }
    audit = {
        "schema_version": 1,
        "status": status,
        "input_dir": str(input_dir),
        "artifact_inventory": inventory,
        "trace_integrity": {
            "row_count": len(rows),
            "decision_row_count": len(decisions),
            "perception_row_count": len(perceptions),
            "invalid_lines": invalid_lines,
            "trace_loss": trace_loss,
            "summary": trace_summary,
        },
        "checks": checks,
        "evaluation_cross_check": {
            "n_destroyed": evaluation.get("n_destroyed"),
            "agent_completed_count_by_uid": final_counts,
            "perception_worker_error": perception.get("worker_error"),
            "basis": "judge_evaluation_is_corroboration_not_agent_input",
        },
        "limitations": [
            "持续瞄准结论基于状态边沿与每 0.5 秒有界采样，不是相机曝光时刻证明。",
            "最近 FOLLOWER 使用各 Agent 自身位置的离线合并，未向任何 Agent 提供跨机真值。",
            "source_sim_time 仅是像素帧来源时间，不等同于已验证曝光时间。",
        ],
    }
    output_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return audit


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="单轮 V3 输出目录")
    parser.add_argument("--output", type=Path, default=None, help="审计 JSON；默认写入输入目录")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f"--input 不是目录：{args.input}")
    output = args.output.resolve() if args.output is not None else args.input.resolve() / "coordination_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = analyze_run(args.input, output)
    print(json.dumps({
        "status": audit["status"],
        "output": str(output),
        "checks": {name: item["status"] for name, item in audit["checks"].items()},
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
