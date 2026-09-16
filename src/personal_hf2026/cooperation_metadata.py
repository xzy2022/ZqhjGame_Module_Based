# 修改时间：2026-09-16。
# 修改目的：让正式采集索引在不复制整批样本的情况下生成双视角配对。
# 修改内容：记录协同状态、对齐相机帧，并按主从互指和时间差生成严格双视角候选。
"""协同逐帧元数据；仅消费调用方显式传入的状态，不连接 UE 或 Redis。"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy


ACTIVE_PHASE = "COOP_ACTIVE"
MASTER = "MASTER"
FOLLOWER = "FOLLOWER"
VALID_ALIGNMENTS = ("exact", "stable_bracket")
VALID_TARGET_SOURCES = ("oracle_identity_offline", "algorithm_internal")

COOPERATION_CONTRACT = {
    "version": 1,
    "alignment": {
        "exact": "frame source_sim_time 与一次已记录 observation sim_time 在容差内相等",
        "stable_bracket": "相邻 observation 的协同状态一致且间隔不超过上限",
        "unavailable": "无法证明时 alignment 为 null，其余未知状态为 null",
    },
    "field_sources": {
        "active": "由算法内部 phase、role、session_id、partner_uid 严格派生",
        "phase": "agent._coordinator.phase",
        "session_id": "agent._coordinator.current_session",
        "role": "agent._coordinator.role",
        "partner_uid": "agent._coordinator.partner_uid",
        "coop_seconds": "agent._coordinator.coop_seconds；本机算法计时",
        "accepted_target_id": "oracle_identity 离线位置关联；不暴露给 Agent",
        "dual_view_candidate": "同会话、互为主从、伙伴互指、同目标且时间差不超过阈值",
    },
}


def _session_list(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(item is None for item in value):
        return None
    master_uid, start_tick, track_epoch = value
    try:
        return [str(master_uid), int(start_tick), int(track_epoch)]
    except (TypeError, ValueError):
        return None


def _finite_nonnegative(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0 or number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _strict_active(phase, role, session_id, partner_uid):
    return bool(phase == ACTIVE_PHASE and role in (MASTER, FOLLOWER)
                and session_id is not None and partner_uid is not None)


def _normalized_state(value):
    phase = value.get("phase")
    role = value.get("role")
    session_id = _session_list(value.get("session_id"))
    partner_uid = value.get("partner_uid")
    partner_uid = str(partner_uid) if partner_uid is not None else None
    target_id = value.get("accepted_target_id")
    target_id = str(target_id) if target_id is not None else None
    target_source = value.get("accepted_target_source")
    if target_id is not None and target_source not in VALID_TARGET_SOURCES:
        target_id = None
        target_source = None
    return {
        "phase": str(phase) if phase is not None else None,
        "session_id": session_id,
        "role": str(role) if role is not None else None,
        "partner_uid": partner_uid,
        "coop_seconds": _finite_nonnegative(value.get("coop_seconds")),
        "accepted_target_id": target_id,
        "accepted_target_source": target_source,
        "active": _strict_active(phase, role, session_id, partner_uid),
    }


def capture_agent_cooperation(agent, *, accepted_target_id=None,
                              accepted_target_source=None):
    """在 ``agent.decide`` 返回后读取本机算法状态；不会猜测车辆身份。"""
    coordinator = agent._coordinator
    if accepted_target_id is not None and accepted_target_source not in VALID_TARGET_SOURCES:
        raise ValueError("accepted_target_id 非空时必须声明可追溯的 accepted_target_source")
    return _normalized_state({
        "phase": coordinator.phase,
        "session_id": coordinator.current_session,
        "role": coordinator.role,
        "partner_uid": coordinator.partner_uid,
        "coop_seconds": coordinator.coop_seconds,
        "accepted_target_id": accepted_target_id,
        "accepted_target_source": accepted_target_source,
    })


def cooperation_state_from_observation(row, identity_row=None):
    """读取新格式快照，也兼容历史 observations/identity 两条真实日志。"""
    if row.get("cooperation_state") is not None:
        value = dict(row["cooperation_state"])
    else:
        summary = row.get("summary") or {}
        value = {
            "phase": row.get("state"),
            "session_id": row.get("session"),
            "role": row.get("role", summary.get("role")),
            "partner_uid": row.get("partner_uid", summary.get("partner_uid")),
            "coop_seconds": row.get("coop", summary.get("estimated_coop_seconds")),
        }
    if identity_row is not None and identity_row.get("accepted_identity") == "TargetVehicle":
        value["accepted_target_id"] = identity_row.get("accepted_vehicle_id")
        value["accepted_target_source"] = "oracle_identity_offline"
    return _normalized_state(value)


def _fingerprint(state):
    return (state["active"], state["phase"], tuple(state["session_id"] or ()),
            state["role"], state["partner_uid"], state["accepted_target_id"],
            state["accepted_target_source"])


def _frame_value(state, alignment):
    return {
        "alignment": alignment,
        "active": state["active"],
        "phase": state["phase"],
        "session_id": state["session_id"],
        "role": state["role"],
        "partner_uid": state["partner_uid"],
        "coop_seconds": state["coop_seconds"],
        "accepted_target_id": state["accepted_target_id"],
        "dual_view_candidate": False,
    }


def _unavailable_frame_value():
    return {"alignment": None, "active": None, "phase": None, "session_id": None,
            "role": None, "partner_uid": None, "coop_seconds": None,
            "accepted_target_id": None, "dual_view_candidate": False}


def align_frame_cooperation(source_sim_time, before, after, *, exact_tolerance_s=1e-5,
                            max_bracket_gap_s=0.25):
    """把一帧对齐到相邻 observation；证据不足时返回显式未知值。"""
    exact = []
    for row in (before, after):
        if row is None or row.get("sim_time") is None:
            continue
        if abs(float(row["sim_time"]) - float(source_sim_time)) <= exact_tolerance_s:
            exact.append(row)
    if exact:
        state = cooperation_state_from_observation(exact[0], exact[0].get("identity"))
        return _frame_value(state, "exact")

    if before is None or after is None:
        return _unavailable_frame_value()
    before_t, after_t = before.get("sim_time"), after.get("sim_time")
    if before_t is None or after_t is None:
        return _unavailable_frame_value()
    before_t, after_t, source_sim_time = float(before_t), float(after_t), float(source_sim_time)
    if not (before_t < source_sim_time < after_t) or after_t - before_t > max_bracket_gap_s:
        return _unavailable_frame_value()

    left = cooperation_state_from_observation(before, before.get("identity"))
    right = cooperation_state_from_observation(after, after.get("identity"))
    if _fingerprint(left) != _fingerprint(right):
        return _unavailable_frame_value()
    left_seconds, right_seconds = left["coop_seconds"], right["coop_seconds"]
    if left_seconds is None or right_seconds is None or right_seconds + 1e-9 < left_seconds:
        return _unavailable_frame_value()
    return _frame_value(left, "stable_bracket")


def _frame_ref(sample):
    return {"uid": str(sample["uid"]), "frame_no": sample.get("frame_no"),
            "source_sim_time": sample.get("source_sim_time"),
            "image_path": sample.get("image_path"),
            "coop_seconds": sample["cooperation"].get("coop_seconds")}


def _pairable(sample):
    cooperation = sample.get("cooperation") or {}
    return bool(cooperation.get("alignment") in VALID_ALIGNMENTS
                and cooperation.get("active") is True
                and cooperation.get("session_id") is not None
                and cooperation.get("role") in (MASTER, FOLLOWER)
                and cooperation.get("partner_uid") is not None
                and cooperation.get("accepted_target_id") is not None
                and sample.get("source_sim_time") is not None)


def annotate_dual_view_candidates(samples, *, max_time_delta_s=0.1, copy_samples=True):
    """双向标记严格主从帧；正式索引可用原地模式控制峰值内存。"""
    annotated = deepcopy(list(samples)) if copy_samples else list(samples)
    groups = defaultdict(lambda: {MASTER: [], FOLLOWER: []})
    for sample in annotated:
        if not isinstance(sample.get("cooperation"), dict):
            sample["cooperation"] = _unavailable_frame_value()
        cooperation = sample["cooperation"]
        cooperation["dual_view_candidate"] = False
        if not _pairable(sample):
            continue
        uid = str(sample["uid"])
        partner_uid = str(cooperation["partner_uid"])
        role = cooperation["role"]
        session_id = tuple(cooperation["session_id"])
        target_id = str(cooperation["accepted_target_id"])
        if role == MASTER:
            master_uid, follower_uid = uid, partner_uid
        else:
            master_uid, follower_uid = partner_uid, uid
        if session_id[0] != master_uid:
            continue
        groups[(session_id, target_id, master_uid, follower_uid)][role].append(sample)

    pairs = []
    for (session_id, target_id, master_uid, follower_uid), by_role in sorted(groups.items()):
        masters = sorted(by_role[MASTER], key=lambda row: float(row["source_sim_time"]))
        followers = sorted(by_role[FOLLOWER], key=lambda row: float(row["source_sim_time"]))
        master_times = [float(row["source_sim_time"]) for row in masters]
        used_master_indexes = set()
        for follower in followers:
            follower_coop = follower["cooperation"]
            if str(follower["uid"]) != follower_uid or str(follower_coop["partner_uid"]) != master_uid:
                continue
            follower_t = float(follower["source_sim_time"])
            index = bisect_left(master_times, follower_t)
            choices = [i for i in (index - 1, index)
                       if 0 <= i < len(masters) and i not in used_master_indexes]
            if not choices:
                continue
            index = min(choices, key=lambda i: abs(master_times[i] - follower_t))
            master = masters[index]
            master_coop = master["cooperation"]
            delta = follower_t - master_times[index]
            if abs(delta) > max_time_delta_s:
                continue
            if str(master["uid"]) != master_uid or str(master_coop["partner_uid"]) != follower_uid:
                continue
            used_master_indexes.add(index)
            master_coop["dual_view_candidate"] = True
            follower_coop["dual_view_candidate"] = True
            pairs.append({
                "session_id": list(session_id),
                "accepted_target_id": target_id,
                "source_time_delta_s": delta,
                "master": _frame_ref(master),
                "follower": _frame_ref(follower),
                "criteria": {"reciprocal_roles_and_partners": True,
                             "same_session": True,
                             "same_accepted_target": True,
                             "max_time_delta_s": max_time_delta_s},
            })
    return annotated, pairs
