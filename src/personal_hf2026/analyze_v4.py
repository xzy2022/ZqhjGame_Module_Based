# 修改时间：2026-09-28。
# 修改目的：从正式协同逐拍日志量化安全间距、目标半径和视觉可见性。
# 修改内容：汇总安全状态、同时和单侧入场距离、分离事件及云台角度范围。
# 修改时间：2026-09-26。
# 修改目的：从紧凑日志直接量化三机搜索及接管的出现次数。
# 修改内容：按规划器模式统计控制采样并单列 TAKEOVER 采样数。
# 修改时间：2026-09-24。
# 修改目的：让 Agent4 的紧凑运行轨迹可直接离线复核。
# 修改内容：按业务事件、状态和视觉诊断干预汇总审计结果。
"""汇总 Agent4 实际运行日志。"""
from __future__ import annotations

from collections import Counter
from bisect import bisect_left
import argparse
import json
from pathlib import Path

from .v3_simple_control import ground_distance_m


def _rows(path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def analyze_run(output):
    output = Path(output)
    events = Counter()
    ready_distances = []
    states = Counter()
    search_modes = Counter()
    samples = 0
    safety_samples = {mode: [] for mode in ("NORMAL", "SEPARATE", "EMERGENCY")}
    safety_distances = []
    safety_radii = []
    radius_rejoin_samples = 0
    safety_episodes = []
    active_episode = {}
    gimbal_pan = []
    gimbal_tilt = []
    track_positions = {"MASTER": [], "FOLLOWER": []}
    all_positions = {}
    track_rows = []
    first_time = last_time = None
    for row in _rows(output / "agent_v4_trace.jsonl"):
        if row.get("kind") == "event":
            events[row.get("event", "unknown")] += 1
            if row.get("event") == "formation_ready" and row.get("pair_distance_m") is not None:
                ready_distances.append(row["pair_distance_m"])
        elif row.get("kind") == "control_sample":
            states[row.get("state", "unknown")] += 1
            mode = row.get("search_plan", {}).get("mode")
            if mode is not None:
                search_modes[mode] += 1
            samples += 1
            if row.get("own_position") is not None:
                all_positions.setdefault(row["uid"], []).append(
                    (row["time"], row["own_position"]))
            if row.get("state") != "COOP_TRACK" and row["uid"] in active_episode:
                episode = active_episode.pop(row["uid"])
                episode["exit_time_s"] = row["time"]
                episode["duration_s"] = row["time"] - episode["entry_time_s"]
                episode["ended_without_normal"] = True
                safety_episodes.append(episode)
            if row.get("state") == "COOP_TRACK":
                track_rows.append(row)
                guidance = row.get("coop_guidance") or {}
                radius_rejoin_samples += int(bool(guidance.get("radius_rejoin")))
                mode = guidance.get("safety_mode")
                if mode in safety_samples:
                    safety_samples[mode].append(row)
                distance = guidance.get("pair_distance_m")
                radius = guidance.get("rough_radius_m")
                if distance is not None and not guidance.get("safety_peer_stale"):
                    safety_distances.append(distance)
                if radius is not None:
                    safety_radii.append(radius)
                for key, values in (("gimbal_actual_pan", gimbal_pan),
                                    ("gimbal_actual_tilt", gimbal_tilt)):
                    if row.get(key) is not None:
                        values.append(row[key])
                uid = row["uid"]
                role = guidance.get("role")
                if role in track_positions and row.get("own_position") is not None:
                    track_positions[role].append((row["time"], row["own_position"]))
                episode = active_episode.get(uid)
                if mode in ("SEPARATE", "EMERGENCY"):
                    if episode is None:
                        episode = {"uid": uid, "entry_time_s": row["time"],
                                   "entry_distance_m": distance,
                                   "minimum_distance_m": distance}
                        active_episode[uid] = episode
                    if distance is not None:
                        episode["minimum_distance_m"] = min(
                            episode["minimum_distance_m"], distance)
                elif episode is not None:
                    episode["recovery_time_s"] = row["time"]
                    episode["recovery_distance_m"] = distance
                    episode["duration_s"] = row["time"] - episode["entry_time_s"]
                    safety_episodes.append(episode)
                    del active_episode[uid]
        time = row.get("time")
        if time is not None:
            first_time = time if first_time is None else min(first_time, time)
            last_time = time if last_time is None else max(last_time, time)
    predictions = 0
    changed = 0
    for row in _rows(output / "visual_predictions.jsonl"):
        predictions += 1
        changed += int(bool(row.get("vision_diagnostic", {}).get("changed")))
    follower_positions = sorted(track_positions["FOLLOWER"])
    follower_times = [item[0] for item in follower_positions]
    paired_distances = []
    for time, position in track_positions["MASTER"]:
        index = bisect_left(follower_times, time)
        neighbors = follower_positions[max(0, index - 1):index + 1]
        if neighbors:
            peer_time, peer_position = min(neighbors, key=lambda item: abs(item[0] - time))
            if abs(peer_time - time) <= 0.25:
                paired_distances.append(ground_distance_m(position, peer_position))
    all_times = {uid: [item[0] for item in rows]
                 for uid, rows in all_positions.items()}
    nearest_other_distances = []
    for row in track_rows:
        distances = []
        for uid, positions in all_positions.items():
            if uid == row["uid"]:
                continue
            times = all_times[uid]
            index = bisect_left(times, row["time"])
            neighbors = positions[max(0, index - 1):index + 1]
            if neighbors:
                time, position = min(neighbors, key=lambda item: abs(item[0] - row["time"]))
                if abs(time - row["time"]) <= 0.25:
                    distances.append(ground_distance_m(row["own_position"], position))
        if distances:
            nearest_other_distances.append(min(distances))
    result = {"schema_version": 1, "event_counts": dict(events),
              "state_sample_counts": dict(states), "control_samples": samples,
              "search_mode_sample_counts": dict(search_modes),
              "takeover_samples": search_modes["TAKEOVER"],
              "first_time_s": first_time, "last_time_s": last_time,
              "prediction_frames": predictions, "diagnostic_changed_frames": changed}
    result["pair_safety"] = {
        "pair_distance_m": {"minimum": min(safety_distances, default=None),
                            "p10": _percentile(safety_distances, 0.1),
                            "median": _percentile(safety_distances, 0.5),
                            "below_200_samples": sum(value < 200.0 for value in safety_distances)},
        "formation_ready_distance_m": {
            "minimum": min(ready_distances, default=None),
            "below_200_events": sum(value < 200.0 for value in ready_distances),
            "all": ready_distances},
        "paired_own_position_distance_m": {
            "samples": len(paired_distances),
            "minimum": min(paired_distances, default=None),
            "p10": _percentile(paired_distances, 0.1),
            "median": _percentile(paired_distances, 0.5),
            "below_200_samples": sum(value < 200.0 for value in paired_distances)},
        "nearest_other_own_position_distance_m": {
            "samples": len(nearest_other_distances),
            "minimum": min(nearest_other_distances, default=None),
            "p10": _percentile(nearest_other_distances, 0.1),
            "median": _percentile(nearest_other_distances, 0.5),
            "below_200_samples": sum(value < 200.0 for value in nearest_other_distances)},
        "rough_radius_m": {"median": _percentile(safety_radii, 0.5),
                           "p90": _percentile(safety_radii, 0.9),
                           "maximum": max(safety_radii, default=None)},
        "radius_rejoin_samples": radius_rejoin_samples,
        "modes": {mode: {"samples": len(rows),
                          "visual_available_ratio": (sum(bool(row.get("visual_target_available"))
                                                        for row in rows) / len(rows)
                                                    if rows else None)}
                  for mode, rows in safety_samples.items()},
        "separation_triggers": len(safety_episodes) + len(active_episode),
        "separation_episodes": safety_episodes + list(active_episode.values()),
        "gimbal_pan_range_deg": [min(gimbal_pan), max(gimbal_pan)] if gimbal_pan else None,
        "gimbal_tilt_range_deg": [min(gimbal_tilt), max(gimbal_tilt)] if gimbal_tilt else None,
        "gimbal_near_limit_samples": {
            "pan_abs_ge_175": sum(abs(value) >= 175.0 for value in gimbal_pan),
            "tilt_le_minus_89_5": sum(value <= -89.5 for value in gimbal_tilt)},
    }
    (output / "agent_v4_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(analyze_run(parser.parse_args(argv).output), ensure_ascii=False))


if __name__ == "__main__":
    main()
