# 修改时间：2026-09-22。
# 修改目的：提供不接入 V3 控制的单机横移视差离线审计入口。
# 修改内容：从 Agent 可见预测日志拟合 bbox 底边中心射线，并将 runner 真值严格限制为报告旁注。
"""单机横移视差静止判定的离线审计。

分类只读取 ``visual_predictions.jsonl`` 的本机位姿、云台朝向、图像尺寸及
``real_vehicle`` 框的底边中心。``visual_frames.jsonl`` 的 UE 投影框是 runner
侧审计资料：只为已完成的窗口附加 ``audit_truth``，从不参与候选、拟合或分类。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class Ray:
    """一条由 Agent 可见数据构造的目标框底边中心射线。"""

    time_s: float
    uid: str
    track_id: str
    frame_id: str
    center: tuple[float, float, float]
    direction: tuple[float, float, float]
    time_basis: str
    audit_truth: str | None


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: JSON 无法解析：{exc}") from exc
            if isinstance(row, dict):
                yield row


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x - y for x, y in zip(a, b))


def _scale(vector: tuple[float, float, float], value: float) -> tuple[float, float, float]:
    return tuple(value * item for item in vector)


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    length = _norm(vector)
    return tuple(item / length for item in vector) if length > 1e-9 else None


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * percent / 100.0
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (position - low)


def _iou(a: list[Any], b: list[Any]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    try:
        left, top = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
        right, bottom = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    except (TypeError, ValueError):
        return 0.0
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def _audit_by_frame(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """返回 runner 侧 frame 映射；调用方只能在报告阶段读取它。"""
    path = run_dir / "visual_frames.jsonl"
    if not path.is_file():
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for row in _jsonl(path):
        frame_id, objects = row.get("frame_id"), row.get("ue_projected_objects")
        if isinstance(frame_id, str) and isinstance(objects, list):
            result[frame_id] = [obj for obj in objects if isinstance(obj, dict)]
    return result


def _audit_id(objects: list[dict[str, Any]], bbox: list[Any]) -> str | None:
    best, best_iou = None, 0.0
    for obj in objects:
        target_id, candidate = obj.get("target_id"), obj.get("bbox")
        if not isinstance(target_id, str) or not isinstance(candidate, list):
            continue
        overlap = _iou(bbox, candidate)
        if overlap > best_iou:
            best, best_iou = target_id, overlap
    return best if best_iou >= 0.25 else None


def _center(pose: dict[str, Any], origin: tuple[float, float, float]) -> tuple[float, float, float] | None:
    lat, lon, alt = (_number(pose.get(key)) for key in ("lat", "lon", "alt"))
    if None in (lat, lon, alt):
        return None
    origin_lat, origin_lon, origin_alt = origin
    north = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    east = math.radians(lon - origin_lon) * EARTH_RADIUS_M * math.cos(math.radians((lat + origin_lat) / 2.0))
    return east, north, alt - origin_alt


def _bottom_ray(pose: dict[str, Any], bbox: list[Any], image_size: list[Any]) -> tuple[float, float, float] | None:
    """采用当前 V3 的水平 FOV、heading+pan、零 roll 射线约定。"""
    if len(bbox) != 4 or len(image_size) != 2:
        return None
    values = [_number(pose.get(key)) for key in ("heading_deg", "gimbal_pan", "gimbal_tilt", "gimbal_fov_deg")]
    width, height = _number(image_size[0]), _number(image_size[1])
    heading, pan, tilt, fov = values
    if None in (heading, pan, tilt, fov, width, height) or width <= 1 or height <= 1 or not 1 < fov < 179:
        return None
    try:
        u, v = (float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3])
    except (TypeError, ValueError):
        return None
    focal = width / (2.0 * math.tan(math.radians(fov / 2.0)))
    x, y = (u - (width - 1.0) / 2.0) / focal, (v - (height - 1.0) / 2.0) / focal
    yaw, pitch = math.radians(heading + pan), math.radians(tilt)
    forward = (math.cos(pitch) * math.sin(yaw), math.cos(pitch) * math.cos(yaw), math.sin(pitch))
    right = (math.cos(yaw), -math.sin(yaw), 0.0)
    down = (math.sin(pitch) * math.sin(yaw), math.sin(pitch) * math.cos(yaw), -math.cos(pitch))
    return _unit(tuple(forward[i] + x * right[i] + y * down[i] for i in range(3)))


def _read_observations(run_dir: Path, use_audit: bool) -> tuple[list[Ray], dict[str, int], Counter[str]]:
    path = run_dir / "visual_predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 Agent 预测日志：{path}")
    audit = _audit_by_frame(run_dir) if use_audit else {}
    raw: list[tuple[float, str, str, str, dict[str, Any], list[Any], list[Any], str, str | None]] = []
    skipped = Counter()
    for row in _jsonl(path):
        detection = row.get("detection")
        if not isinstance(detection, dict) or detection.get("class_name") != "real_vehicle":
            skipped["not_reported_real_vehicle"] += 1
            continue
        time_s = _number(row.get("observed_sim_time"))
        uid, frame_id, pose = row.get("uid"), row.get("frame_id"), row.get("source_pose")
        track_id, bbox, image_size = detection.get("track_id"), detection.get("bbox_xyxy"), row.get("image_size")
        if time_s is None or not isinstance(uid, str) or not isinstance(frame_id, str) or track_id is None or not isinstance(pose, dict) or not isinstance(bbox, list) or not isinstance(image_size, list):
            skipped["invalid_agent_geometry"] += 1
            continue
        raw.append((time_s, uid, str(track_id), frame_id, pose, bbox, image_size, str(row.get("source_time_basis", "missing")), _audit_id(audit.get(frame_id, []), bbox)))
    if not raw:
        return [], dict(skipped), Counter()
    origin = tuple(_number(raw[0][4].get(key)) for key in ("lat", "lon", "alt"))
    if any(value is None for value in origin):
        raise ValueError("首个有效目标框缺少 Agent 位姿，无法建立 ENU 原点。")
    rays, bases = [], Counter()
    for time_s, uid, track_id, frame_id, pose, bbox, image_size, basis, truth in raw:
        center, direction = _center(pose, origin), _bottom_ray(pose, bbox, image_size)
        if center is None or direction is None:
            skipped["invalid_agent_geometry"] += 1
            continue
        rays.append(Ray(time_s, uid, track_id, frame_id, center, direction, basis, truth))
        bases[basis] += 1
    return sorted(rays, key=lambda row: (row.uid, row.track_id, row.time_s)), dict(skipped), bases


def _solve(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float] | None:
    rows = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-8:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for row in range(3):
            if row != column:
                multiplier = rows[row][column]
                rows[row] = [value - multiplier * base for value, base in zip(rows[row], rows[column])]
    return tuple(rows[row][3] for row in range(3))


def _fit(window: list[Ray]) -> tuple[tuple[float, float, float] | None, list[float], list[float]]:
    matrix, vector = [[0.0] * 3 for _ in range(3)], [0.0] * 3
    for ray in window:
        projection = [[(1.0 if i == j else 0.0) - ray.direction[i] * ray.direction[j] for j in range(3)] for i in range(3)]
        for i in range(3):
            for j in range(3):
                matrix[i][j] += projection[i][j]
            vector[i] += sum(projection[i][j] * ray.center[j] for j in range(3))
    point = _solve(matrix, vector)
    if point is None:
        return None, [], []
    residuals, reprojection = [], []
    for ray in window:
        offset = _sub(point, ray.center)
        distance = _norm(offset)
        miss = _norm(_sub(offset, _scale(ray.direction, _dot(offset, ray.direction))))
        residuals.append(miss)
        reprojection.append(math.degrees(math.asin(min(1.0, miss / distance))) if distance > 1e-6 else 90.0)
    return point, residuals, reprojection


def _geometry(window: list[Ray]) -> tuple[float, float]:
    baseline = _sub(window[-1].center, window[0].center)
    average = _unit(tuple(sum(ray.direction[i] for ray in window) for i in range(3)))
    if average is None:
        return 0.0, 0.0
    transverse = _sub(baseline, _scale(average, _dot(baseline, average)))
    angles = [math.degrees(math.acos(max(-1.0, min(1.0, _dot(first.direction, second.direction))))) for index, first in enumerate(window) for second in window[index + 1:]]
    return _norm(transverse), max(angles, default=0.0)


def _segments(rays: list[Ray], gap_s: float) -> Iterable[list[Ray]]:
    segment: list[Ray] = []
    previous: Ray | None = None
    for ray in rays:
        if previous is not None and ((ray.uid, ray.track_id) != (previous.uid, previous.track_id) or ray.time_s - previous.time_s > gap_s):
            if segment:
                yield segment
            segment = []
        segment.append(ray)
        previous = ray
    if segment:
        yield segment


def _windows(rays: list[Ray], args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    for segment in _segments(rays, args.max_detection_gap_s):
        start = 0
        while start < len(segment):
            chosen = None
            for end in range(start + args.min_rays - 1, len(segment)):
                window = segment[start:end + 1]
                span = window[-1].time_s - window[0].time_s
                if span > args.max_span_s:
                    break
                if span < args.min_span_s:
                    continue
                baseline, angle = _geometry(window)
                if args.min_transverse_baseline_m <= baseline <= args.max_transverse_baseline_m:
                    chosen = window, baseline, angle
                    break
            if chosen is None:
                start += 1
                continue
            window, baseline, angle = chosen
            point, residuals, reprojection = _fit(window)
            median, p95 = _percentile(residuals, 50), _percentile(residuals, 95)
            median_deg, p95_deg = _percentile(reprojection, 50), _percentile(reprojection, 95)
            valid = point is not None and angle >= args.min_ray_angle_deg
            moving = bool(valid and median is not None and p95 is not None and median_deg is not None and p95_deg is not None and median >= args.moving_median_residual_m and p95 >= args.moving_p95_residual_m and median_deg >= args.moving_median_reprojection_deg and p95_deg >= args.moving_p95_reprojection_deg)
            truth_ids = sorted({ray.audit_truth for ray in window if ray.audit_truth})
            results.append({
                "uid": window[0].uid, "track_id": window[0].track_id, "start_s": window[0].time_s, "end_s": window[-1].time_s,
                "span_s": window[-1].time_s - window[0].time_s, "ray_count": len(window), "transverse_baseline_m": baseline,
                "max_ray_angle_deg": angle, "fit_point_enu_m": point, "median_ray_residual_m": median, "p95_ray_residual_m": p95,
                "median_reprojection_deg": median_deg, "p95_reprojection_deg": p95_deg, "classification": "moving" if moving else "stationary_or_uncertain",
                "fit_status": "valid" if valid else "insufficient_ray_angle", "audit_truth_ids": truth_ids,
                "audit_truth_coverage": sum(ray.audit_truth is not None for ray in window) / len(window),
                "algorithm_input": "agent_visible_pose_orientation_bbox_bottom_center_only",
            })
            start = end + 1
    return results


def _report(run_dir: Path, rays: list[Ray], skipped: dict[str, int], bases: Counter[str], windows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    focus = [row for time_s in args.focus_time for row in windows if row["start_s"] - 3.0 <= time_s <= row["end_s"] + 3.0]
    return {
        "schema_version": 2,
        "purpose": "single_uav_lateral_parallax_static_motion_audit",
        "run_dir": str(run_dir),
        "agent_boundary": "classification uses only visual_predictions source_pose, source orientation, image_size and real_vehicle bbox bottom-center; visual_frames UE truth is report-only.",
        "exposure_limit": "source_time_basis is recorded but not verified camera exposure time; results must not be treated as exposure-synchronized ground truth.",
        "parameters": {key: getattr(args, key) for key in ("min_span_s", "max_span_s", "min_rays", "min_transverse_baseline_m", "max_transverse_baseline_m", "min_ray_angle_deg", "moving_median_residual_m", "moving_p95_residual_m", "moving_median_reprojection_deg", "moving_p95_reprojection_deg")},
        "input": {"reported_target_observations": len(rays), "skipped": skipped, "source_time_basis_counts": dict(sorted(bases.items()))},
        "summary": {"candidate_windows": len(windows), "classification_counts": dict(Counter(row["classification"] for row in windows))},
        "focus_windows": focus,
        "windows": windows,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]["classification_counts"]
    lines = ["# 单机横移视差静止判定审计", "", f"- Agent 目标报告帧：{report['input']['reported_target_observations']}；横移窗口：{report['summary']['candidate_windows']}。", f"- 分类：moving={summary.get('moving', 0)}，stationary_or_uncertain={summary.get('stationary_or_uncertain', 0)}。", "- 判定未读取 runner 真值；`audit_truth_ids` 与覆盖率只供事后核验。", f"- 时间限制：{report['exposure_limit']}", "", "## 关注时刻窗口", "", "| UAV | track | 时间 s | 横向基线 m | P50/P95 残差 m | P50/P95 重投影 deg | 结果 | runner 审计 ID |", "| --- | ---: | --- | ---: | --- | --- | --- | --- |"]
    for row in report["focus_windows"]:
        lines.append("| {uid} | {track_id} | {start_s:.3f}-{end_s:.3f} | {transverse_baseline_m:.1f} | {median_ray_residual_m:.2f}/{p95_ray_residual_m:.2f} | {median_reprojection_deg:.3f}/{p95_reprojection_deg:.3f} | {classification} | {audit} |".format(**row, audit=", ".join(row["audit_truth_ids"]) or "无"))
    if not report["focus_windows"]:
        lines.append("| - | - | 未覆盖指定时刻 | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用 Agent 可见日志进行单机横移视差静止/运动离线审计。")
    parser.add_argument("run_dir", type=Path, help="含 visual_predictions.jsonl 的 V3 运行目录")
    parser.add_argument("--output", type=Path, required=True, help="新的审计输出目录，不能已存在")
    parser.add_argument("--focus-time", type=float, action="append", default=[], help="要在 Markdown 汇总的仿真时刻，可重复")
    parser.add_argument("--no-audit-truth", action="store_true", help="完全不读取 runner 侧 visual_frames")
    parser.add_argument("--min-span-s", type=float, default=1.4)
    parser.add_argument("--max-span-s", type=float, default=2.7)
    parser.add_argument("--max-detection-gap-s", type=float, default=0.45)
    parser.add_argument("--min-rays", type=int, default=5)
    parser.add_argument("--min-transverse-baseline-m", type=float, default=30.0)
    parser.add_argument("--max-transverse-baseline-m", type=float, default=60.0)
    parser.add_argument("--min-ray-angle-deg", type=float, default=1.0)
    parser.add_argument("--moving-median-residual-m", type=float, default=2.7)
    parser.add_argument("--moving-p95-residual-m", type=float, default=5.5)
    parser.add_argument("--moving-median-reprojection-deg", type=float, default=0.32)
    parser.add_argument("--moving-p95-reprojection-deg", type=float, default=0.65)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{args.output}")
    if args.min_span_s <= 0 or args.max_span_s < args.min_span_s or args.min_rays < 2:
        raise ValueError("时间窗口或最少射线参数无效。")
    rays, skipped, bases = _read_observations(args.run_dir, not args.no_audit_truth)
    report = _report(args.run_dir, rays, skipped, bases, _windows(rays, args), args)
    args.output.mkdir(parents=True)
    (args.output / "static_motion_parallax.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
