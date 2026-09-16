# 修改时间：2026-09-16。
# 修改目的：用真实双机配对数据审计地理定位可观测性与相机射线坐标约定。
# 修改内容：统计配对、基线、视线夹角和框中心，并用真值仅离线选择坐标约定及评分无真值输入的双射线三角化。
"""审计双机配对数据的可观测性、射线约定和三角化误差。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METRES_PER_DEGREE = 111_320.0
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """逐行读取 JSONL，避免把图像或未知规模文本整体载入内存。"""
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: JSON 无法解析") from error


def finite_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "quantiles": {}}
    quantiles = np.quantile(array, QUANTILES)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "quantiles": {
            f"p{int(round(q * 100)):02d}": float(value)
            for q, value in zip(QUANTILES, quantiles)
        },
    }


def interval_counts(values: Iterable[float], edges: tuple[float, ...]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values], dtype=float)
    array = array[np.isfinite(array)]
    result: dict[str, Any] = {}
    lower = -math.inf
    for upper in (*edges, math.inf):
        if math.isinf(lower):
            label = f"lt_{upper:g}"
        elif math.isinf(upper):
            label = f"ge_{lower:g}"
        else:
            label = f"ge_{lower:g}_lt_{upper:g}"
        count = int(np.count_nonzero((array >= lower) & (array < upper)))
        result[label] = {
            "count": count,
            "fraction": float(count / array.size) if array.size else None,
        }
        lower = upper
    return result


def local_enu(position: dict[str, Any], origin: dict[str, float]) -> np.ndarray:
    east = ((float(position["lon"]) - origin["lon"]) * METRES_PER_DEGREE
            * math.cos(math.radians(origin["lat"])))
    north = (float(position["lat"]) - origin["lat"]) * METRES_PER_DEGREE
    return np.asarray([east, north, float(position["alt"])], dtype=float)


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("零长度或非有限向量")
    return vector / norm


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(normalized(first), normalized(second)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def camera_basis(pose: dict[str, Any], yaw_mode: str) -> tuple[np.ndarray, ...]:
    heading = float(pose["heading_deg"])
    pan = float(pose["gimbal_pan"])
    if yaw_mode == "heading_plus_pan":
        yaw_deg = heading + pan
    elif yaw_mode == "heading_minus_pan":
        yaw_deg = heading - pan
    elif yaw_mode == "world_pan":
        yaw_deg = pan
    elif yaw_mode == "neg_heading_plus_pan":
        yaw_deg = -heading + pan
    else:
        raise ValueError(f"未知 yaw_mode：{yaw_mode}")
    yaw = math.radians(yaw_deg)
    pitch = math.radians(float(pose["gimbal_tilt"]))
    forward = np.asarray([
        math.cos(pitch) * math.sin(yaw),
        math.cos(pitch) * math.cos(yaw),
        math.sin(pitch),
    ])
    right = np.asarray([math.cos(yaw), -math.sin(yaw), 0.0])
    down = np.asarray([
        math.sin(pitch) * math.sin(yaw),
        math.sin(pitch) * math.cos(yaw),
        -math.cos(pitch),
    ])
    return forward, right, down


def transform_pixel(x: float, y: float, transform: str) -> tuple[float, float]:
    transforms = {
        "u_right_v_down": (x, y),
        "u_left_v_down": (-x, y),
        "u_right_v_up": (x, -y),
        "u_left_v_up": (-x, -y),
        "swap_u_v": (y, x),
        "swap_u_minus_v": (y, -x),
        "swap_minus_u_v": (-y, x),
        "swap_minus_u_minus_v": (-y, -x),
    }
    if transform not in transforms:
        raise ValueError(f"未知像素变换：{transform}")
    return transforms[transform]


def pixel_ray(side: dict[str, Any], fov_axis: str, yaw_mode: str,
              pixel_transform: str) -> np.ndarray:
    bbox = [float(value) for value in side["bbox_xyxy"]]
    u = (bbox[0] + bbox[2]) / 2.0
    v = (bbox[1] + bbox[3]) / 2.0
    width = float(side["width"])
    height = float(side["height"])
    fov = math.radians(float(side["source_pose"]["gimbal_fov_deg"]))
    focal_extent = width if fov_axis == "horizontal" else height
    focal = (focal_extent / 2.0) / math.tan(fov / 2.0)
    x = (u - (width - 1.0) / 2.0) / focal
    y = (v - (height - 1.0) / 2.0) / focal
    x, y = transform_pixel(x, y, pixel_transform)
    forward, right, down = camera_basis(side["source_pose"], yaw_mode)
    return normalized(forward + x * right + y * down)


def candidate_definitions() -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for fov_axis in ("horizontal", "vertical"):
        for transform in (
                "u_right_v_down", "u_left_v_down", "u_right_v_up", "u_left_v_up",
                "swap_u_v", "swap_u_minus_v", "swap_minus_u_v",
                "swap_minus_u_minus_v"):
            candidates.append({
                "fov_axis": fov_axis,
                "yaw_mode": "heading_plus_pan",
                "pixel_transform": transform,
            })
    for yaw_mode in ("heading_minus_pan", "world_pan", "neg_heading_plus_pan"):
        candidates.append({
            "fov_axis": "horizontal",
            "yaw_mode": yaw_mode,
            "pixel_transform": "u_right_v_down",
        })
    return candidates


def candidate_name(candidate: dict[str, str]) -> str:
    return "/".join((candidate["fov_axis"], candidate["yaw_mode"],
                     candidate["pixel_transform"]))


def truth_position(pair: dict[str, Any]) -> dict[str, float]:
    canonical = pair["ground_truth"][pair["ground_truth"]["canonical_side"]]
    return {key: float(canonical[key]) for key in ("lat", "lon", "alt")}


def closest_ray_parameters(camera_a: np.ndarray, ray_a: np.ndarray,
                           camera_b: np.ndarray, ray_b: np.ndarray) -> tuple[float, float]:
    matrix = np.column_stack((ray_a, -ray_b))
    parameters, _, _, _ = np.linalg.lstsq(matrix, camera_b - camera_a, rcond=None)
    return float(parameters[0]), float(parameters[1])


def triangulate(camera_a: np.ndarray, ray_a: np.ndarray,
                camera_b: np.ndarray, ray_b: np.ndarray) -> dict[str, Any]:
    matrices = [np.eye(3) - np.outer(ray, ray) for ray in (ray_a, ray_b)]
    design = np.vstack(matrices)
    target = np.concatenate((matrices[0] @ camera_a, matrices[1] @ camera_b))
    point, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    distance_a, distance_b = closest_ray_parameters(camera_a, ray_a, camera_b, ray_b)
    closest_a = camera_a + distance_a * ray_a
    closest_b = camera_b + distance_b * ray_b
    condition = (float(singular_values[0] / singular_values[-1])
                 if singular_values[-1] > 0.0 else math.inf)
    return {
        "point": point,
        "rank": int(rank),
        "condition_number": condition,
        "ray_gap_m": float(np.linalg.norm(closest_a - closest_b)),
        "distance_along_master_m": distance_a,
        "distance_along_follower_m": distance_b,
        "forward": distance_a > 0.0 and distance_b > 0.0,
    }


def run_label(run_name: str) -> str:
    prefix = "seed01-"
    suffix = "-150s-fov30"
    if run_name.startswith(prefix) and run_name.endswith(suffix):
        return run_name[len(prefix):-len(suffix)]
    return run_name


def bbox_center_metrics(side: dict[str, Any]) -> dict[str, float]:
    x1, y1, x2, y2 = map(float, side["bbox_xyxy"])
    width, height = float(side["width"]), float(side["height"])
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return {
        "center_x_px": center_x,
        "center_y_px": center_y,
        "center_x_normalized": (center_x - (width - 1.0) / 2.0) / (width / 2.0),
        "center_y_normalized": (center_y - (height - 1.0) / 2.0) / (height / 2.0),
        "width_px": x2 - x1,
        "height_px": y2 - y1,
    }


def markdown_quantiles(summary: dict[str, Any], digits: int = 3) -> str:
    q = summary.get("quantiles", {})
    return (f"n={summary.get('count', 0)}, P05={q.get('p05', math.nan):.{digits}f}, "
            f"P50={q.get('p50', math.nan):.{digits}f}, "
            f"P95={q.get('p95', math.nan):.{digits}f}, "
            f"范围=[{q.get('p00', math.nan):.{digits}f}, {q.get('p100', math.nan):.{digits}f}]")


def optional_number(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def build_report(audit: dict[str, Any]) -> str:
    totals = audit["totals"]
    observability = audit["observability"]
    chosen = audit["coordinate_convention_selection"]["selected"]
    triangulation = audit["triangulation_with_selected_convention"]
    gated = triangulation["quality_slices_by_estimated_ray_angle_deg"]["ge_3"]
    lines = [
        "# 双机协同目标经纬高估计：数据与可观测性审计",
        "",
        "## 结论",
        "",
        (f"- 数据集有 {totals['runs']} 个天气 run，但只有 {totals['runs_with_pairs']} 个含有效双机对；"
         f"共 {totals['pairs']} 对、{totals['images']} 张图、{totals['sessions']} 个 run 内 session、"
         f"{totals['targets']} 个全局 target_id。Rain 为 0 对，因此不能称为六天气定位覆盖。"),
        (f"- 双机水平基线：{markdown_quantiles(observability['horizontal_baseline_m'])} m；"
         f"GT 视线夹角：{markdown_quantiles(observability['gt_view_angle_deg'])}°。"
         "`1/sin(夹角)` 越大，深度/高度误差越容易被像素误差放大。"),
        (f"- 离线候选约定中角残差最小的是 `{chosen['name']}`："
         f"单视图 GT 角残差 P50={chosen['angular_residual_deg']['quantiles']['p50']:.4f}°、"
         f"P95={chosen['angular_residual_deg']['quantiles']['p95']:.4f}°。"
         "GT 只用于这一步离线选约定和评分，没有进入三角化输入。"),
        (f"- 用该约定从两架无人机位姿和两个 bbox 中心直接三角化，"
         f"有效正向射线 {triangulation['forward_pairs']}/{triangulation['pairs']} 对；"
         f"水平误差 P50={triangulation['horizontal_error_m']['quantiles']['p50']:.3f} m、"
         f"P95={triangulation['horizontal_error_m']['quantiles']['p95']:.3f} m，"
         f"高度误差绝对值 P50={triangulation['absolute_altitude_error_m']['quantiles']['p50']:.3f} m、"
         f"P95={triangulation['absolute_altitude_error_m']['quantiles']['p95']:.3f} m。"),
        (f"- 仅用估计器自身可见的射线夹角作质量门：夹角 ≥3° 时保留 {gated['pairs']} 对，"
         f"水平误差 P95={gated['horizontal_error_m']['quantiles']['p95']:.3f} m、"
         f"高度绝对误差 P95={gated['absolute_altitude_error_m']['quantiles']['p95']:.3f} m；"
         f"全量最差高度误差为 {triangulation['absolute_altitude_error_m']['quantiles']['p100']:.3f} m。"
         "这说明小夹角样本应拒绝或累计更大基线，而不是强制输出高度。"),
        "",
        "## 天气与实体覆盖",
        "",
        "| run | 天气 | source/accepted/rejected pair | session | target | 基线 P50 m | GT 夹角 P50° | GT 高度范围 m | 主/从 UID |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in audit["runs"]:
        baseline = run["observability"]["horizontal_baseline_m"]["quantiles"]
        angle = run["observability"]["gt_view_angle_deg"]["quantiles"]
        altitude = run["observability"]["ground_truth_altitude_m"]["quantiles"]
        lines.append(
            f"| {run['run']} | {run['weather']} | {run['source_pairs']} / {run['pairs']} / "
            f"{run['rejected_pairs']} | {run['sessions']} | {run['targets']} | "
            f"{optional_number(baseline.get('p50'))} | {optional_number(angle.get('p50'))} | "
            f"{optional_number(altitude.get('p00'))}～{optional_number(altitude.get('p100'))} | "
            f"{', '.join(run['uids']) or '-'} |")
    lines.extend([
        "",
        "## bbox 中心分布",
        "",
        "归一化坐标以图像中心为 0，半幅宽/半幅高为 1；因此 -1/+1 近似对应左右或上下边界。",
        "",
        "| 角色 | x 像素 | y 像素 | x 归一化 | y 归一化 |",
        "|---|---|---|---|---|",
    ])
    for role in ("master", "follower", "all"):
        stats = audit["bbox_centers"][role]
        lines.append(
            f"| {role} | {markdown_quantiles(stats['center_x_px'])} | "
            f"{markdown_quantiles(stats['center_y_px'])} | "
            f"{markdown_quantiles(stats['center_x_normalized'])} | "
            f"{markdown_quantiles(stats['center_y_normalized'])} |")
    lines.extend([
        "",
        "## 坐标约定候选（按单视图 GT 角残差 P50 排序）",
        "",
        "| 候选 | 角残差 P50 / P95（度） | 说明 |",
        "|---|---:|---|",
    ])
    for candidate in audit["coordinate_convention_selection"]["ranked_candidates"]:
        residual = candidate["angular_residual_deg"]["quantiles"]
        note = "选定" if candidate["name"] == chosen["name"] else "候选"
        lines.append(
            f"| `{candidate['name']}` | {residual['p50']:.4f} / {residual['p95']:.4f} | {note} |")
    lines.extend([
        "",
        "## 字段语义与证据边界",
        "",
        "- `bbox_xyxy` 来自采集时 `audit_boxes -> ue_projected_objects[].ue_projected_bbox`，为图像左上原点的 xyxy 投影框；本审计使用框几何中心，不把框当人工可见性标注。",
        "- `source_pose` 是按照片 `source_sim_time` 在观测历史中 exact/插值得到的 `lat/lon/alt/heading_deg/gimbal_pan/gimbal_tilt/gimbal_fov_deg`；`source_sim_time` 仍不是已验证曝光时刻。",
        "- `heading_deg` 来自官方实体 heading；`gimbal_pan/tilt/fov` 来自 `raw.gimbal_tracking`。采集脚本的既有射线实现使用 `heading + pan`、ENU、图像 u 向右/v 向下、零相机 roll。",
        "- 本数据中 pan、tilt、roll、pitch 基本不变化（实际唯一值见 audit.json），因此它可以区分图像轴符号、FOV 轴和是否使用 heading，却不能独立验证 pan 正负号、机体/云台外参、roll/pitch 轴约定或动态曝光对齐。",
        "- 标定文件只是在“FOV 为水平、方形像素、主点居中”假设下推导；畸变和相机外参未知。`aircraft_attitude.coordinate_convention_status=unverified`。",
        "- GT 车辆高度仅作离线评分。最终三角化函数输入只有两侧 `source_pose + bbox + 图像尺寸/FOV`，不读取 target_id、天气或 ground_truth。",
        "- 源码追踪：`visual_shadow_study.py` 从 Redis 同一帧哈希读取 image/sim_time/detections；`capture_dataset.py` 将 detections 原样记为 xyxy，并按 source_sim_time 对齐 source_pose；`visual_geometry.py` 给出既有 ENU 射线公式；`camera_metadata.py` 明确内参推导及未验证项。",
        "",
        "## 风险",
        "",
        "- bbox 中心未必等于车辆真值锚点（可能更接近可见包围盒中心），会形成系统性高度偏差。",
        "- 同一连续 session 的大量相邻帧不是独立样本；本报告是该固定 seed/路线上的开发证据，不是泛化置信区间。",
        "- Rain 没有双机对；所有 run 均为 FOV30、近似 500 m 高度、云台近乎正下，因此尚未覆盖倾斜视角、动态 pan/tilt 和显著机体姿态。",
        "- 曝光时间、完整内参、畸变与外参未验证前，不能把本离线米级结果宣传为线上绝对定位精度。",
        "",
        "机器可读全量结果见 `audit.json`。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if not (dataset / "manifest.json").is_file():
        parser.error(f"数据集缺少 manifest.json：{dataset}")
    output.mkdir(parents=True, exist_ok=False)

    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8-sig"))
    manifest_runs = {str(item["run"]): item for item in manifest["runs"]}
    pair_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    run_names = [str(item["run"]) for item in manifest["runs"]]
    for run_name in run_names:
        path = dataset / run_name / "pairs.jsonl"
        rows = list(read_jsonl(path)) if path.is_file() else []
        pair_rows.extend(rows)
        sessions = {json.dumps(row["session_id"], ensure_ascii=False) for row in rows}
        targets = {str(row["target_id"]) for row in rows}
        uids = sorted({str(row[role]["uid"]) for row in rows for role in ("master", "follower")})
        run_rows.append({
            "run": run_name,
            "weather": run_label(run_name),
            "source_pairs": int(manifest_runs[run_name]["source_pairs"]),
            "pairs": len(rows),
            "rejected_pairs": int(manifest_runs[run_name]["rejected_pairs"]),
            "rejection_reasons": manifest_runs[run_name].get("rejection_reasons", {}),
            "sessions": len(sessions),
            "targets": len(targets),
            "target_ids": sorted(targets),
            "uids": uids,
            "pair_count_by_target": dict(sorted(Counter(
                str(row["target_id"]) for row in rows).items())),
        })

    bbox_values = {
        role: defaultdict(list) for role in ("master", "follower", "all")
    }
    baselines: list[float] = []
    gt_view_angles: list[float] = []
    identifiability: list[float] = []
    gt_altitudes: list[float] = []
    source_time_deltas: list[float] = []
    field_values: dict[str, list[float]] = defaultdict(list)
    candidate_residuals: dict[str, list[float]] = defaultdict(list)
    candidates = candidate_definitions()
    per_run_metrics: dict[str, dict[str, Any]] = {}
    for run_name in run_names:
        per_run_metrics[run_name] = {
            "bbox": {role: defaultdict(list) for role in ("master", "follower", "all")},
            "baseline": [],
            "view_angle": [],
            "identifiability": [],
            "gt_altitude": [],
        }

    for pair in pair_rows:
        run_metric = per_run_metrics[str(pair["run"])]
        truth = truth_position(pair)
        origin = {"lat": truth["lat"], "lon": truth["lon"]}
        target = local_enu(truth, origin)
        cameras = {}
        for role in ("master", "follower"):
            side = pair[role]
            metrics = bbox_center_metrics(side)
            for key, value in metrics.items():
                bbox_values[role][key].append(value)
                bbox_values["all"][key].append(value)
                run_metric["bbox"][role][key].append(value)
                run_metric["bbox"]["all"][key].append(value)
            cameras[role] = local_enu(side["source_pose"], origin)
            for key in ("alt", "heading_deg", "gimbal_pan", "gimbal_tilt", "gimbal_fov_deg"):
                field_values[f"source_pose.{key}"].append(float(side["source_pose"][key]))
            attitude = side["aircraft_attitude"]
            for key in ("roll", "pitch", "yaw"):
                field_values[f"aircraft_attitude.{key}"].append(float(attitude[key]))
            truth_vector = target - cameras[role]
            for candidate in candidates:
                ray = pixel_ray(side, **candidate)
                candidate_residuals[candidate_name(candidate)].append(
                    angle_degrees(ray, truth_vector))
        baseline = float(np.linalg.norm(cameras["master"][:2] - cameras["follower"][:2]))
        view_angle = angle_degrees(target - cameras["master"], target - cameras["follower"])
        baselines.append(baseline)
        gt_view_angles.append(view_angle)
        sine = abs(math.sin(math.radians(view_angle)))
        amplification = 1.0 / sine if sine > 1e-12 else math.inf
        identifiability.append(amplification)
        gt_altitudes.append(truth["alt"])
        source_time_deltas.append(float(pair["quality"]["source_time_delta_s"]))
        run_metric["baseline"].append(baseline)
        run_metric["view_angle"].append(view_angle)
        run_metric["identifiability"].append(amplification)
        run_metric["gt_altitude"].append(truth["alt"])

    for run_row in run_rows:
        metrics = per_run_metrics[run_row["run"]]
        run_row["bbox_centers"] = {
            role: {key: finite_summary(values) for key, values in role_metrics.items()}
            for role, role_metrics in metrics["bbox"].items()
        }
        run_row["observability"] = {
            "horizontal_baseline_m": finite_summary(metrics["baseline"]),
            "gt_view_angle_deg": finite_summary(metrics["view_angle"]),
            "depth_error_amplification_proxy_1_over_sin_angle": finite_summary(
                metrics["identifiability"]),
            "ground_truth_altitude_m": finite_summary(metrics["gt_altitude"]),
        }

    ranked_candidates = []
    for candidate in candidates:
        name = candidate_name(candidate)
        ranked_candidates.append({
            "name": name,
            **candidate,
            "angular_residual_deg": finite_summary(candidate_residuals[name]),
        })
    # Python 排序稳定；完全相同的候选保留 candidate_definitions 中的先验顺序。
    # 这使本数据无法区分的 ±pan 并列时，优先采用已有采集实现 heading+pan。
    ranked_candidates.sort(key=lambda item: (
        item["angular_residual_deg"]["quantiles"]["p50"],
        item["angular_residual_deg"]["quantiles"]["p95"],
    ))
    selected = ranked_candidates[0]
    selected_arguments = {key: selected[key]
                          for key in ("fov_axis", "yaw_mode", "pixel_transform")}

    horizontal_errors: list[float] = []
    altitude_errors: list[float] = []
    absolute_altitude_errors: list[float] = []
    point_errors: list[float] = []
    ray_gaps: list[float] = []
    conditions: list[float] = []
    estimated_ray_angles: list[float] = []
    forward_pairs = 0
    rank_counts: Counter[str] = Counter()
    by_run_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    quality_slices: dict[float, dict[str, list[float]]] = {
        threshold: defaultdict(list) for threshold in (1.0, 3.0, 5.0, 10.0)
    }
    for pair in pair_rows:
        truth = truth_position(pair)
        origin = {"lat": truth["lat"], "lon": truth["lon"]}
        target = local_enu(truth, origin)
        camera_master = local_enu(pair["master"]["source_pose"], origin)
        camera_follower = local_enu(pair["follower"]["source_pose"], origin)
        ray_master = pixel_ray(pair["master"], **selected_arguments)
        ray_follower = pixel_ray(pair["follower"], **selected_arguments)
        fit = triangulate(camera_master, ray_master, camera_follower, ray_follower)
        estimate = fit["point"]
        horizontal_error = float(np.linalg.norm((estimate - target)[:2]))
        altitude_error = float(estimate[2] - target[2])
        point_error = float(np.linalg.norm(estimate - target))
        horizontal_errors.append(horizontal_error)
        altitude_errors.append(altitude_error)
        absolute_altitude_errors.append(abs(altitude_error))
        point_errors.append(point_error)
        ray_gaps.append(fit["ray_gap_m"])
        conditions.append(fit["condition_number"])
        estimated_angle = angle_degrees(ray_master, ray_follower)
        estimated_ray_angles.append(estimated_angle)
        forward_pairs += int(fit["forward"])
        rank_counts[str(fit["rank"])] += 1
        bucket = by_run_errors[str(pair["run"])]
        bucket["horizontal_error_m"].append(horizontal_error)
        bucket["altitude_error_m"].append(altitude_error)
        bucket["absolute_altitude_error_m"].append(abs(altitude_error))
        for threshold, slice_metrics in quality_slices.items():
            if estimated_angle >= threshold:
                slice_metrics["horizontal_error_m"].append(horizontal_error)
                slice_metrics["altitude_error_m"].append(altitude_error)
                slice_metrics["absolute_altitude_error_m"].append(abs(altitude_error))
                slice_metrics["three_dimensional_error_m"].append(point_error)
                slice_metrics["ray_gap_m"].append(fit["ray_gap_m"])

    all_sessions = {(str(row["run"]), json.dumps(row["session_id"], ensure_ascii=False))
                    for row in pair_rows}
    all_targets = {str(row["target_id"]) for row in pair_rows}
    audit: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "method_boundary": {
            "ground_truth_use": "offline coordinate-convention selection and scoring only",
            "triangulation_inputs": [
                "master/follower source_pose", "master/follower bbox_xyxy",
                "image width/height", "gimbal_fov_deg",
            ],
            "forbidden_as_estimator_input": [
                "ground_truth", "target_id", "run/weather label", "image path",
            ],
        },
        "totals": {
            "runs": len(run_rows),
            "runs_with_pairs": sum(row["pairs"] > 0 for row in run_rows),
            "pairs": len(pair_rows),
            "images": len(pair_rows) * 2,
            "sessions": len(all_sessions),
            "targets": len(all_targets),
            "target_ids": sorted(all_targets),
        },
        "runs": run_rows,
        "bbox_centers": {
            role: {key: finite_summary(values) for key, values in metrics.items()}
            for role, metrics in bbox_values.items()
        },
        "observability": {
            "horizontal_baseline_m": finite_summary(baselines),
            "horizontal_baseline_bins_m": interval_counts(baselines, (1.0, 5.0, 10.0, 20.0, 50.0)),
            "gt_view_angle_deg": finite_summary(gt_view_angles),
            "gt_view_angle_bins_deg": interval_counts(gt_view_angles, (1.0, 3.0, 5.0, 10.0)),
            "depth_error_amplification_proxy_1_over_sin_angle": finite_summary(identifiability),
            "ground_truth_altitude_m": finite_summary(gt_altitudes),
            "paired_source_time_delta_s": finite_summary(source_time_deltas),
        },
        "recorded_field_distributions": {
            key: {**finite_summary(values), "unique_rounded_1e9": len({round(v, 9) for v in values})}
            for key, values in sorted(field_values.items())
        },
        "source_trace": [
            {
                "file": "src/personal_hf2026/visual_shadow_study.py",
                "lines": "66-75, 94-99",
                "evidence": "Redis 同一 frame key 读取 image/sim_time/detections，detections 仅写审计数据",
            },
            {
                "file": "src/personal_hf2026/capture_dataset.py",
                "lines": "33, 52-72, 160-189",
                "evidence": "bbox 作为 xyxy 保存，source_pose 按 source_sim_time exact 或插值对齐",
            },
            {
                "file": "src/personal_hf2026/visual_geometry.py",
                "lines": "31-67",
                "evidence": "既有水平 FOV、heading+pan、ENU、u 右/v 下、零 roll 射线公式",
            },
            {
                "file": "src/personal_hf2026/camera_metadata.py",
                "lines": "71-126",
                "evidence": "针孔内参推导公式及 FOV 轴、主点、像素比例、畸变、外参未验证声明",
            },
            {
                "file": "competition/sdk/core/isolation.py (official simulator checkout)",
                "lines": "88-97",
                "evidence": "heading 来自 entity heading，云台角和 FOV 来自 raw.gimbal_tracking",
            },
        ],
        "coordinate_convention_selection": {
            "selection_metric": "minimum median single-view angular residual to GT line of sight",
            "ground_truth_role": "offline selection/scoring only; not triangulation input",
            "selected": selected,
            "ranked_candidates": ranked_candidates,
            "identifiability_note": (
                "All recorded gimbal_pan values are zero in this dataset, so heading+pan and "
                "heading-pan are observationally indistinguishable here."),
        },
        "triangulation_with_selected_convention": {
            "algorithm": "least-squares intersection of two 3D rays using sum(I-ddT)",
            "pairs": len(pair_rows),
            "forward_pairs": forward_pairs,
            "rank_counts": dict(sorted(rank_counts.items())),
            "horizontal_error_m": finite_summary(horizontal_errors),
            "altitude_error_m_signed": finite_summary(altitude_errors),
            "absolute_altitude_error_m": finite_summary(absolute_altitude_errors),
            "three_dimensional_error_m": finite_summary(point_errors),
            "ray_gap_m": finite_summary(ray_gaps),
            "condition_number": finite_summary(conditions),
            "estimated_ray_angle_deg": finite_summary(estimated_ray_angles),
            "quality_slices_by_estimated_ray_angle_deg": {
                f"ge_{threshold:g}": {
                    "pairs": len(metrics["horizontal_error_m"]),
                    **{key: finite_summary(values) for key, values in metrics.items()},
                }
                for threshold, metrics in quality_slices.items()
            },
            "by_run": {
                run: {key: finite_summary(values) for key, values in metrics.items()}
                for run, metrics in sorted(by_run_errors.items())
            },
        },
        "evidence_limits": [
            "visibility_annotation=unannotated and projected bbox is not a human visibility label",
            "exposure_time_verified=false",
            "FOV axis, principal point, square pixels, distortion and extrinsics are unverified",
            "aircraft attitude coordinate convention is unverified",
            "bbox center may not equal the target geodetic anchor",
            "adjacent frames within a session are correlated",
            "rain run contains zero accepted pairs",
        ],
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (output / "REPORT.md").write_text(build_report(audit), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "pairs": len(pair_rows),
        "selected_convention": selected["name"],
        "angular_residual_p50_deg": selected["angular_residual_deg"]["quantiles"]["p50"],
        "horizontal_error_p50_m": audit["triangulation_with_selected_convention"]
            ["horizontal_error_m"]["quantiles"]["p50"],
        "absolute_altitude_error_p50_m": audit["triangulation_with_selected_convention"]
            ["absolute_altitude_error_m"]["quantiles"]["p50"],
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
