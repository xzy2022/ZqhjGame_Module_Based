# 修改时间：2026-09-16。
# 修改目的：在 Windows 上优先使用高分辨率 QueryPerformanceCounter 计算同进程延迟。
# 修改内容：分析器优先读取 perf_counter_s，并仅在旧产物缺失时回退 monotonic_s。
# 修改时间：2026-09-16。
# 修改目的：让汇总分析严格匹配在线探针最终字段并正确识别零值姿态与日志截断。
# 修改内容：补齐状态单调钟、重复轮询字段别名，按无人机计算轮询周期并读取探针摘要。
# 修改时间：2026-09-16。
# 修改目的：区分仿真钟差、同主机单调钟延迟和跨进程墙钟近似，避免误判曝光延迟。
# 修改内容：兼容 frame_time_probe 四类事件并输出帧步长、管线分段、字段覆盖、截断与证据缺口。
# 修改时间：2026-09-16。
# 修改目的：避免把门控前陈旧候选的等待时间误解为合格双机估计的处理延迟。
# 修改内容：用预测帧标识关联时间日志，并分别汇总全部候选与最终成功估计。
# 修改时间：2026-09-16。
# 修改目的：从真实双机定位产物离线汇总图像新鲜度、双机时间差和姿态取样错配。
# 修改内容：优先读取候选配对时间日志，并兼容旧产物中仅有成功预测记录的有限证据。
"""离线判别双机定位时间字段语义和可观测延迟分量。"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    if not path.is_file():
        return rows, invalid
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                invalid += 1
    return rows, invalid


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_number(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value: Any = row
        for part in name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        number = _number(value)
        if number is not None:
            return number
    return None


def _metric(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(value for value in values if math.isfinite(value))

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "mean": fmean(ordered) if ordered else None,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1] if ordered else None,
    }


def _difference(later: float | None, earlier: float | None) -> float | None:
    return later - earlier if later is not None and earlier is not None else None


def _append(bucket: dict[str, list[float]], key: str, value: Any) -> None:
    number = _number(value)
    if number is not None:
        bucket[key].append(number)


def _frame_key(row: Mapping[str, Any]) -> tuple[str, int, float] | None:
    uid = row.get("uid")
    frame_no = _number(row.get("frame_no"))
    source = _number(row.get("source_sim_time"))
    if uid is None or frame_no is None or source is None:
        return None
    return str(uid), int(frame_no), source


def _pair_key(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    views = row.get("views", {})
    if not isinstance(views, Mapping) or len(views) != 2:
        return None
    try:
        frames = tuple(sorted(
            (str(view["uid"]), int(view["frame_no"]))
            for view in views.values() if isinstance(view, Mapping)))
        return (*frames, str(row["target_id"])) if len(frames) == 2 else None
    except (KeyError, TypeError, ValueError):
        return None


def _timing_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        _append(metrics, "pair.source_time_delta_s", row.get("source_time_delta_s"))
        _append(metrics, "pair.redis_read_wall_delta_s", row.get("redis_read_wall_delta_s"))
        _append(metrics, "pair.evaluator_receive_wall_delta_s",
                row.get("evaluator_receive_wall_delta_s"))
        views = row.get("views", {})
        if not isinstance(views, Mapping):
            continue
        for view in views.values():
            if not isinstance(view, Mapping):
                continue
            _append(metrics, "view.world_minus_source_sim_s",
                    view.get("frame_age_sim_s", view.get("pose_time_delta_s")))
            _append(metrics, "view.pose_sample_minus_source_sim_s",
                    view.get("pose_time_mismatch_s", view.get("pose_time_delta_s")))
            _append(metrics, "view.truth_sample_minus_source_sim_s",
                    view.get("truth_time_mismatch_s", view.get("world_truth_time_delta_s")))
            _append(metrics, "view.redis_read_to_evaluator_monotonic_s",
                    view.get("redis_read_to_evaluator_s"))
            _append(metrics, "view.context_publish_to_redis_read_monotonic_s",
                    view.get("context_to_redis_read_wall_s"))
            _append(metrics, "view.redis_read_to_pair_match_monotonic_s",
                    view.get("redis_read_to_pair_match_s"))
    return metrics


def _event_kind(row: Mapping[str, Any]) -> str:
    return str(row.get("kind", row.get("event", "unknown")))


def _index_by_frame(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int, float], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = _frame_key(row)
        if key is not None:
            result[key] = row
    return result


def _match_frame(index: Mapping[tuple[str, int, float], Mapping[str, Any]],
                 row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    key = _frame_key(row)
    if key in index:
        return index[key]
    if key is None:
        return None
    uid, frame_no, source = key
    for candidate_key, candidate in index.items():
        if candidate_key[0] == uid and candidate_key[1] == frame_no \
                and abs(candidate_key[2] - source) <= 1e-6:
            return candidate
    return None


def _all_named_values(value: Any, result: dict[str, list[Any]]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            result[str(key).lower()].append(child)
            _all_named_values(child, result)
    elif isinstance(value, list):
        for child in value:
            _all_named_values(child, result)


def _field_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aliases = {
        "aircraft_yaw": {"yaw", "heading", "heading_deg", "aircraft_yaw_deg"},
        "aircraft_roll": {"roll", "roll_deg", "aircraft_roll_deg"},
        "aircraft_pitch": {"pitch", "pitch_deg", "aircraft_pitch_deg"},
        "gimbal_pan": {"gimbal_pan", "pan", "pan_deg"},
        "gimbal_tilt": {"gimbal_tilt", "tilt", "tilt_deg"},
        "gimbal_fov": {"gimbal_fov", "gimbal_fov_deg", "fov", "fov_deg"},
        "world_timestamp": {"world_timestamp"},
        "t_world_camera": {"t_world_camera", "world_camera_transform"},
        "fx": {"fx"}, "fy": {"fy"}, "cx": {"cx"}, "cy": {"cy"},
        "distortion": {"distortion", "distortion_coefficients", "dist_coeffs"},
    }
    counts = Counter()
    for row in rows:
        values: dict[str, list[Any]] = defaultdict(list)
        _all_named_values(row, values)
        for label, names in aliases.items():
            if any(any(value is not None for value in values.get(name, []))
                   for name in names):
                counts[label] += 1
    total = len(rows)
    return {
        "state_sample_rows": total,
        "fields": {
            label: {"count": counts[label], "ratio": counts[label] / total if total else None}
            for label in aliases
        },
    }


def _fit_affine(points: list[tuple[float, float]]) -> tuple[float, float, list[float]] | None:
    """拟合 host_monotonic = slope * world_sim_time + intercept。"""
    if len(points) < 2:
        return None
    mean_x = fmean(point[0] for point in points)
    mean_y = fmean(point[1] for point in points)
    variance = sum((point[0] - mean_x) ** 2 for point in points)
    if variance <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    intercept = mean_y - slope * mean_x
    residuals = [y - (slope * x + intercept) for x, y in points]
    return slope, intercept, residuals


def _segmented_map(points: list[tuple[float, float]], source: float) -> tuple[float, bool] | None:
    """在相邻状态点间线性映射；区间外使用最近一段并标记外插。"""
    if len(points) < 2:
        return None
    positions = [point[0] for point in points]
    upper = bisect.bisect_left(positions, source)
    extrapolated = upper == 0 or upper == len(points)
    if upper == 0:
        left, right = points[0], points[1]
    elif upper == len(points):
        left, right = points[-2], points[-1]
    else:
        left, right = points[upper - 1], points[upper]
    if right[0] == left[0]:
        return None
    fraction = (source - left[0]) / (right[0] - left[0])
    return left[1] + fraction * (right[1] - left[1]), extrapolated


def _local_mapping_residuals(points: list[tuple[float, float]]) -> list[float]:
    """用相邻前后状态点预测中间点，估计分段映射的局部残差。"""
    residuals = []
    for index in range(1, len(points) - 1):
        left, actual, right = points[index - 1], points[index], points[index + 1]
        if right[0] == left[0]:
            continue
        fraction = (actual[0] - left[0]) / (right[0] - left[0])
        predicted = left[1] + fraction * (right[1] - left[1])
        residuals.append(actual[1] - predicted)
    return residuals


def _clock_mapping(state_samples: list[dict[str, Any]],
                   first_seen: list[dict[str, Any]]) -> dict[str, Any]:
    state_by_uid: dict[str, dict[float, float]] = defaultdict(dict)
    for row in state_samples:
        uid = row.get("uid")
        world = _number(row.get("world_sim_time"))
        captured = _first_number(row, (
            "world_state_observed_perf_counter_s",
            "world_state_observed_monotonic_s", "captured_monotonic_s"))
        if uid is not None and world is not None and captured is not None:
            state_by_uid[str(uid)][world] = captured
    first_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in first_seen:
        if row.get("uid") is not None:
            first_by_uid[str(row["uid"])].append(row)

    by_uid = {}
    all_estimates: list[float] = []
    all_residuals: list[float] = []
    all_local_residuals: list[float] = []
    total_extrapolated = 0
    total_mapped = 0
    for uid in sorted(set(state_by_uid) | set(first_by_uid)):
        points = sorted(state_by_uid.get(uid, {}).items())
        fit = _fit_affine(points)
        residuals = fit[2] if fit is not None else []
        local_residuals = _local_mapping_residuals(points)
        estimates: list[float] = []
        extrapolated = 0
        unmapped = 0
        for row in first_by_uid.get(uid, []):
            source = _number(row.get("source_sim_time"))
            first_mono = _first_number(row, (
                "hmget_completed_perf_counter_s",
                "hmget_completed_monotonic_s", "first_seen_monotonic_s"))
            mapped = _segmented_map(points, source) if source is not None else None
            if mapped is None or first_mono is None:
                unmapped += 1
                continue
            mapped_mono, is_extrapolated = mapped
            estimates.append(first_mono - mapped_mono)
            extrapolated += int(is_extrapolated)
        all_estimates.extend(estimates)
        all_residuals.extend(residuals)
        all_local_residuals.extend(local_residuals)
        total_extrapolated += extrapolated
        total_mapped += len(estimates)
        by_uid[uid] = {
            "state_mapping_points": len(points),
            "world_sim_time_range": (
                {"first": points[0][0], "last": points[-1][0]} if points else {}),
            "captured_monotonic_range": (
                {"first": points[0][1], "last": points[-1][1]} if points else {}),
            "affine_fit": ({
                "host_monotonic_per_sim_second": fit[0],
                "intercept_s": fit[1],
                "residual_s": _metric(residuals),
            } if fit is not None else None),
            "segmented_mapping_local_residual_s": _metric(local_residuals),
            "renderer_source_time_to_python_first_seen_conditional_estimate_s": _metric(estimates),
            "mapped_frames": len(estimates),
            "extrapolated_frames": extrapolated,
            "unmapped_frames": unmapped,
        }
    return {
        "method": "同一 uid 的 state_sample 建立 world_sim_time 到同进程高分辨率单调钟的相邻点分段线性映射；区间外沿最近一段外插。新产物优先 perf_counter_s，旧产物回退 monotonic_s。",
        "meaning": "renderer source time 到 Python 首见的条件估计；前提是 source_sim_time 与 world_sim_time 同域且映射在局部有效。不是曝光到 Redis 延迟。",
        "renderer_source_time_to_python_first_seen_conditional_estimate_s": _metric(all_estimates),
        "affine_fit_residual_s": _metric(all_residuals),
        "segmented_mapping_local_residual_s": _metric(all_local_residuals),
        "mapped_frames": total_mapped,
        "extrapolated_frames": total_extrapolated,
        "extrapolated_ratio": total_extrapolated / total_mapped if total_mapped else None,
        "by_uid": by_uid,
    }


def _probe_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = Counter(_event_kind(row) for row in rows)
    metadata = next(
        (row for row in rows if _event_kind(row) == "probe_metadata"), {})
    first_seen = [row for row in rows if _event_kind(row) == "frame_first_seen"]
    state_samples = [row for row in rows if _event_kind(row) == "state_sample"]
    dispatches = [row for row in rows if _event_kind(row) == "frame_context_dispatch"]
    polls = [row for row in rows if _event_kind(row) == "bridge_poll"]
    metrics: dict[str, list[float]] = defaultdict(list)
    poll_outcomes = Counter(str(row.get("outcome", "unknown")) for row in polls)
    positive_duplicate_polls = 0

    for row in first_seen:
        scan_start = _first_number(row, (
            "scan_started_perf_counter_s", "scan_started_monotonic_s"))
        scan_done = _first_number(row, (
            "scan_completed_perf_counter_s", "scan_completed_monotonic_s"))
        hmget_start = _first_number(row, (
            "hmget_started_perf_counter_s", "hmget_started_monotonic_s"))
        hmget_done = _first_number(row, (
            "hmget_completed_perf_counter_s",
            "hmget_completed_monotonic_s", "first_seen_monotonic_s"))
        for key, value in (
            ("scan_monotonic_s", _difference(scan_done, scan_start)),
            ("hmget_monotonic_s", _difference(hmget_done, hmget_start)),
            ("scan_start_to_first_seen_monotonic_s", _difference(hmget_done, scan_start)),
        ):
            _append(metrics, key, value)

    poll_times_by_uid: dict[str, list[float]] = defaultdict(list)
    for row in polls:
        poll_time = _first_number(row, (
            "poll_started_perf_counter_s", "scan_started_perf_counter_s",
            "poll_started_monotonic_s", "scan_started_monotonic_s",
            "captured_monotonic_s", "monotonic_s"))
        if poll_time is not None:
            poll_times_by_uid[str(row.get("uid", "unknown"))].append(poll_time)
        _append(metrics, "bridge_poll_key_count", row.get("key_count"))
        selected = row.get("selected")
        if isinstance(selected, (list, tuple, dict, set)):
            _append(metrics, "bridge_poll_selected_count", len(selected))
        else:
            _append(metrics, "bridge_poll_selected_count", selected)
        duplicate_index = _first_number(
            row, ("duplicate_read_index", "duplicate_index"))
        if duplicate_index is not None:
            _append(metrics, "bridge_poll_duplicate_index", duplicate_index)
            positive_duplicate_polls += int(duplicate_index > 0)
    for poll_times in poll_times_by_uid.values():
        ordered = sorted(poll_times)
        for before, after in zip(ordered, ordered[1:]):
            _append(metrics, "bridge_poll_interval_monotonic_s", after - before)

    first_index = _index_by_frame(first_seen)
    state_index = {
        str(row["state_sample_id"]): row
        for row in state_samples if row.get("state_sample_id") is not None
    }
    wall_approximations: list[float] = []
    matched_dispatches = 0
    matched_states = 0
    for row in dispatches:
        seen = _match_frame(first_index, row)
        if seen is not None:
            matched_dispatches += 1
            first_mono = _first_number(seen, (
                "hmget_completed_perf_counter_s",
                "hmget_completed_monotonic_s", "first_seen_monotonic_s"))
            dispatch_mono = _first_number(row, (
                "dispatch_perf_counter_s", "dispatch_monotonic_s"))
            _append(metrics, "redis_first_seen_to_context_dispatch_monotonic_s",
                    _difference(dispatch_mono, first_mono))
            first_unix = _first_number(seen, ("hmget_completed_unix_s", "first_seen_unix_s"))
            dispatch_unix = _first_number(row, ("dispatch_unix_s",))
            wall_delta = _difference(dispatch_unix, first_unix)
            if wall_delta is not None:
                wall_approximations.append(wall_delta)
        state_id = row.get("state_sample_id")
        state = state_index.get(str(state_id)) if state_id is not None else None
        source = _number(row.get("source_sim_time"))
        if state is not None:
            matched_states += 1
            world = _number(state.get("world_sim_time"))
            _append(metrics, "dispatch_state_world_minus_source_sim_s",
                    _difference(world, source))
            captured_mono = _first_number(state, (
                "world_state_observed_perf_counter_s",
                "world_state_observed_monotonic_s", "captured_monotonic_s"))
            dispatch_mono = _first_number(row, (
                "dispatch_perf_counter_s", "dispatch_monotonic_s"))
            _append(metrics, "state_capture_to_context_dispatch_monotonic_s",
                    _difference(dispatch_mono, captured_mono))

    first_seen_mono = sorted(value for value in (
        _first_number(row, (
            "hmget_completed_perf_counter_s",
            "hmget_completed_monotonic_s", "first_seen_monotonic_s"))
        for row in first_seen) if value is not None)
    state_mono = sorted(value for value in (
        _first_number(row, (
            "world_state_observed_perf_counter_s",
            "world_state_observed_monotonic_s", "captured_monotonic_s"))
        for row in state_samples)
                        if value is not None)
    return {
        "event_counts": dict(sorted(kinds.items())),
        "clock_metadata": {
            "clock_domains": metadata.get("clock_domains", {}),
            "clock_resolution_s": metadata.get("clock_resolution_s", {}),
            "analysis_priority": "perf_counter_s_then_legacy_monotonic_s",
        },
        "bridge_poll": {
            "outcomes": dict(sorted(poll_outcomes.items())),
            "positive_duplicate_index_rows": positive_duplicate_polls,
        },
        "metrics": {key: _metric(values) for key, values in sorted(metrics.items())},
        "wall_clock_approximations": {
            "first_seen_to_dispatch_s": _metric(wall_approximations),
            "interpretation": "仅作跨进程或单调钟缺失时的近似；受系统校时和记录顺序影响。",
        },
        "linkage": {
            "dispatch_rows": len(dispatches),
            "dispatch_matched_to_first_seen": matched_dispatches,
            "dispatch_matched_to_state_sample": matched_states,
        },
        "clock_mapping": _clock_mapping(state_samples, first_seen),
        "coverage": {
            "first_seen_monotonic_s": ({
                "first": first_seen_mono[0], "last": first_seen_mono[-1],
                "span_s": first_seen_mono[-1] - first_seen_mono[0],
            } if first_seen_mono else {}),
            "state_sample_monotonic_s": ({
                "first": state_mono[0], "last": state_mono[-1],
                "span_s": state_mono[-1] - state_mono[0],
            } if state_mono else {}),
        },
        "state_field_coverage": _field_coverage(state_samples),
    }


def _frame_sequence(probe_rows: list[dict[str, Any]],
                    timing_rows: list[dict[str, Any]],
                    prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    probe_frames = [row for row in probe_rows if _event_kind(row) == "frame_first_seen"]
    source_kind = "frame_first_seen"
    frames: list[Mapping[str, Any]] = probe_frames
    if not frames:
        source_kind = "candidate_pair_timing_unique_views"
        frames = []
        for row in timing_rows:
            views = row.get("views", {})
            if isinstance(views, Mapping):
                frames.extend(view for view in views.values() if isinstance(view, Mapping))
    if not frames:
        source_kind = "prediction_unique_views"
        frames = []
        for row in prediction_rows:
            views = row.get("views", {})
            if isinstance(views, Mapping):
                frames.extend(view for view in views.values() if isinstance(view, Mapping))

    unique: dict[tuple[str, int, float], Mapping[str, Any]] = {}
    frame_sources: dict[tuple[str, int], set[float]] = defaultdict(set)
    for row in frames:
        key = _frame_key(row)
        if key is not None:
            unique[key] = row
            frame_sources[(key[0], key[1])].add(key[2])
    by_uid: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for uid, frame_no, source in unique:
        by_uid[uid].append((frame_no, source))

    result_by_uid = {}
    for uid, values in sorted(by_uid.items()):
        ordered = sorted(values)
        frame_steps: list[float] = []
        source_steps: list[float] = []
        source_per_frame: list[float] = []
        gaps = 0
        missing_frames = 0
        nonpositive_source = 0
        for (frame_a, source_a), (frame_b, source_b) in zip(ordered, ordered[1:]):
            frame_delta = frame_b - frame_a
            source_delta = source_b - source_a
            frame_steps.append(float(frame_delta))
            source_steps.append(source_delta)
            if frame_delta > 0:
                source_per_frame.append(source_delta / frame_delta)
            if frame_delta > 1:
                gaps += 1
                missing_frames += frame_delta - 1
            if source_delta <= 0:
                nonpositive_source += 1
        result_by_uid[uid] = {
            "unique_frames": len(ordered),
            "frame_no": {"first": ordered[0][0], "last": ordered[-1][0]} if ordered else {},
            "source_sim_time": {"first": ordered[0][1], "last": ordered[-1][1]} if ordered else {},
            "frame_no_step": _metric(frame_steps),
            "source_sim_time_step_s": _metric(source_steps),
            "source_sim_time_per_frame_s": _metric(source_per_frame),
            "observed_frame_number_gaps": gaps,
            "missing_frame_numbers_inside_observed_span": missing_frames,
            "nonpositive_source_time_steps": nonpositive_source,
        }
    return {
        "source_kind": source_kind,
        "selection_warning": (
            None if source_kind == "frame_first_seen" else
            "该序列只含进入配对/预测日志的帧，步长和缺号受选择与记录上限影响。"),
        "duplicate_frame_no_with_conflicting_source_time": sum(
            len(values) > 1 for values in frame_sources.values()),
        "by_uid": result_by_uid,
    }


def _value_range(rows: Iterable[Mapping[str, Any]], names: Sequence[str]) -> dict[str, float]:
    values = sorted(value for value in (_first_number(row, names) for row in rows)
                    if value is not None)
    if not values:
        return {}
    return {"first": values[0], "last": values[-1], "span": values[-1] - values[0]}


def _truncation(summary: Mapping[str, Any], timing_rows: list[dict[str, Any]],
                prediction_rows: list[dict[str, Any]],
                probe_rows: list[dict[str, Any]],
                probe_summary: Mapping[str, Any]) -> dict[str, Any]:
    counts = summary.get("counts", {}) if isinstance(summary.get("counts"), Mapping) else {}
    limits = summary.get("limits", {}) if isinstance(summary.get("limits"), Mapping) else {}
    suppressed = int(_number(counts.get("timing_records_suppressed_record_limit")) or 0)
    suppressed += int(_number(counts.get("timing_records_suppressed_byte_limit")) or 0)
    timing_written = int(_number(counts.get("timing_records_written")) or len(timing_rows))
    candidates = int(_number(counts.get("candidate_pairs_seen")) or 0)
    max_records = int(_number(limits.get("max_records")) or 0)
    timing_truncated = suppressed > 0 or (max_records > 0 and timing_written >= max_records
                                         and candidates > timing_written)
    probe_counts = (probe_summary.get("counts", {})
                    if isinstance(probe_summary.get("counts"), Mapping) else {})
    probe_suppressed = sum(
        int(_number(probe_counts.get(name)) or 0)
        for name in ("events_suppressed_record_limit", "events_suppressed_byte_limit"))
    return {
        "timing_stream_truncated": timing_truncated,
        "timing_rows_loaded": len(timing_rows),
        "prediction_rows_loaded": len(prediction_rows),
        "probe_rows_loaded": len(probe_rows),
        "timing_records_suppressed_reported": suppressed,
        "candidate_pairs_seen_reported": candidates or None,
        "log_coverage": {
            "timing_matched_unix_s": _value_range(timing_rows, ("matched_unix_s",)),
            "prediction_estimate_index": _value_range(prediction_rows, ("estimate_index",)),
            "probe_monotonic_s": _value_range(probe_rows, (
                "hmget_completed_perf_counter_s", "dispatch_perf_counter_s",
                "world_state_observed_perf_counter_s",
                "hmget_completed_monotonic_s", "dispatch_monotonic_s",
                "world_state_observed_monotonic_s", "captured_monotonic_s",
                "scan_started_monotonic_s")),
        },
        "probe_stream_truncation": {
            "status": ("truncated" if probe_suppressed else
                       "complete_within_configured_limits" if probe_summary else
                       "unknown_summary_missing" if probe_rows else
                       "not_applicable_probe_missing"),
            "events_written": int(
                _number(probe_counts.get("events_written")) or len(probe_rows)),
            "events_suppressed": probe_suppressed,
            "max_records": _number(probe_summary.get("max_records")),
            "max_output_bytes": _number(probe_summary.get("max_output_bytes")),
            "written_bytes": _number(probe_summary.get("written_bytes")),
        },
        "consequence": (
            "时间分布只覆盖日志上限前的候选；不能外推为全部成功估计或全程分布。"
            if timing_truncated else None),
    }


def analyze(run_root: Path) -> dict[str, Any]:
    timing_path = run_root / "paired_geolocation_timing.jsonl"
    prediction_path = run_root / "paired_geolocation_predictions.jsonl"
    probe_path = run_root / "frame_time_probe.jsonl"
    probe_summary_path = run_root / "frame_time_probe_summary.json"
    summary_path = run_root / "paired_geolocation_summary.json"
    timing_rows, timing_invalid = _read_rows(timing_path)
    prediction_rows, prediction_invalid = _read_rows(prediction_path)
    probe_rows, probe_invalid = _read_rows(probe_path)
    summary = _read_object(summary_path)
    probe_summary = _read_object(probe_summary_path)

    accepted_keys = {_pair_key(row) for row in prediction_rows}
    accepted_timing = [row for row in timing_rows if _pair_key(row) in accepted_keys]
    timing_metrics = _timing_metrics(timing_rows if timing_rows else prediction_rows)
    accepted_metrics = _timing_metrics(accepted_timing if timing_rows else prediction_rows)
    probe = _probe_analysis(probe_rows)
    truncation = _truncation(
        summary, timing_rows, prediction_rows, probe_rows, probe_summary)

    same_sim_count = _metric(timing_metrics.get("view.world_minus_source_sim_s", []))["count"]
    probe_sim_count = probe["metrics"].get(
        "dispatch_state_world_minus_source_sim_s", {"count": 0})["count"]
    monotonic_count = sum(
        metric["count"] for name, metric in probe["metrics"].items()
        if "monotonic" in name)
    monotonic_count += sum(
        len(values) for name, values in timing_metrics.items() if "monotonic" in name)

    evidence_gaps = [
        "source_sim_time 没有已验证的渲染/曝光定义，也没有与图像原子绑定的生产端曝光时间戳。",
        "缺少生产端曝光/渲染时刻后，曝光到 Redis 首见延迟不可识别。",
    ]
    if not probe_rows:
        evidence_gaps.append("缺少 frame_time_probe.jsonl，不能分解 Redis 扫描/HMGET/首次见帧到上下文派发。")
    coverage = probe["state_field_coverage"]["fields"]
    if coverage["t_world_camera"]["count"] == 0:
        evidence_gaps.append("没有逐帧原子绑定的 T_world_camera。")
    if coverage["aircraft_roll"]["count"] == 0 or coverage["aircraft_pitch"]["count"] == 0:
        evidence_gaps.append("没有证实每个相关帧均具有完整机体 roll/pitch/yaw。")
    if any(coverage[field]["count"] == 0 for field in ("fx", "fy", "cx", "cy")):
        evidence_gaps.append("没有逐帧完整内参 fx/fy/cx/cy；FOV 不能替代完整内参。")
    if truncation["timing_stream_truncated"]:
        evidence_gaps.append("paired_geolocation_timing.jsonl 已截断，成功估计的完整时序覆盖不足。")

    return {
        "schema_version": 2,
        "run_root": str(run_root),
        "inputs": {
            "paired_geolocation_timing": {"path": str(timing_path), "rows": len(timing_rows),
                                           "invalid_rows": timing_invalid},
            "paired_geolocation_predictions": {"path": str(prediction_path),
                                                "rows": len(prediction_rows),
                                                "invalid_rows": prediction_invalid},
            "frame_time_probe": {"path": str(probe_path), "rows": len(probe_rows),
                                 "invalid_rows": probe_invalid},
            "frame_time_probe_summary": {
                "path": str(probe_summary_path), "available": bool(probe_summary)},
            "paired_geolocation_summary": {"path": str(summary_path),
                                           "available": bool(summary)},
        },
        "frame_sequence": _frame_sequence(probe_rows, timing_rows, prediction_rows),
        "same_simulation_clock": {
            "world_minus_source_metrics": _metric(
                timing_metrics.get("view.world_minus_source_sim_s", [])),
            "probe_dispatch_state_world_minus_source_metrics": probe["metrics"].get(
                "dispatch_state_world_minus_source_sim_s", _metric([])),
            "judgement": (
                "已测得同一运行内 world_sim_time-source_sim_time 的相对差；这只支持比较仿真钟，不能证明 source_sim_time 是曝光时刻。"
                if same_sim_count or probe_sim_count else
                "evidence_gap：没有可关联的 world_sim_time 与 source_sim_time 样本。"),
        },
        "same_host_monotonic_pipeline": {
            "probe_metrics": {name: metric for name, metric in probe["metrics"].items()
                              if "monotonic" in name},
            "paired_timing_metrics": {name: _metric(values) for name, values in timing_metrics.items()
                                      if "monotonic" in name},
            "judgement": (
                "可测的是 Redis 读取/探针首见之后的本机处理分段，不包含未知的曝光到 Redis 段。"
                if monotonic_count else
                "evidence_gap：没有同主机 monotonic 起止时间或已计算分段。"),
        },
        "cross_process_wall_clock": probe["wall_clock_approximations"],
        "pair_and_pose_alignment": {
            "candidate_metrics": {name: _metric(values) for name, values in timing_metrics.items()
                                  if name.startswith("pair.") or "sample_minus_source" in name},
            "accepted_metrics": {name: _metric(values) for name, values in accepted_metrics.items()
                                 if name.startswith("pair.") or "sample_minus_source" in name},
            "accepted_timing_rows": len(accepted_timing) if timing_rows else len(prediction_rows),
            "prediction_rows": len(prediction_rows),
            "accepted_timing_coverage": (
                len(accepted_timing) / len(prediction_rows)
                if timing_rows and prediction_rows else None),
        },
        "probe": probe,
        "truncation": truncation,
        "identifiability": {
            "source_sim_time_exposure_semantics": {
                "status": "unresolved",
                "reason": "只能观测 Redis 字段及其与 world_sim_time 的关系，不能由接收侧日志反推字段是否代表曝光/渲染时刻。",
            },
            "exposure_or_render_to_redis_first_seen_latency": {
                "status": "not_identifiable",
                "reason": "缺少经验证的生产端曝光/渲染时间戳与跨钟映射。",
            },
            "redis_observation_pipeline": {
                "status": "measured_after_first_observation" if monotonic_count else "not_measured",
                "reason": "同主机 monotonic 仅覆盖扫描、HMGET、派发、评估和配对中实际存在的时间点。",
            },
            "wall_clock_cross_process": {
                "status": "approximation_only",
                "reason": "unix wall clock 可能受校时和调钟影响，只用于跨进程近似，不替代 monotonic 延迟。",
            },
            "atomic_frame_pose_binding": {
                "status": "unverified",
                "reason": "state_sample_id 关联能证明软件派发引用，仍不等于渲染器在曝光时原子提供 T_world_camera。",
            },
        },
        "evidence_gaps": evidence_gaps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.run_root.resolve())
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
