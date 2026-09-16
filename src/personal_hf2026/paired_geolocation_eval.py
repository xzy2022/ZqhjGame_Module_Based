# 修改时间：2026-09-16。
# 修改目的：为低夹角或无前向交会的双机样本增加不读取测试真值的稳健回退。
# 修改内容：新增一度射线夹角门控、双单机先验投影中点回退、覆盖率及逐对胜率统计。
# 修改时间：2026-09-16。
# 修改目的：用真实双机配对数据对照单机高度先验投影与双机射线三角化。
# 修改内容：实现按 run 隔离的高度先验、离线回放、逐样本预测与水平/高度/三维误差汇总。
"""双机目标经纬度高度估计离线对照实验。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


EARTH_RADIUS_M = 6_378_137.0


def jsonl_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(value, factor):
    return tuple(x * factor for x in value)


def norm(value):
    return math.sqrt(dot(value, value))


def normalized(value):
    length = norm(value)
    if length <= 0:
        raise ValueError("零长度射线")
    return scale(value, 1.0 / length)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class LocalFrame:
    """小范围 WGS84 经纬度的局部 ENU 近似。"""

    def __init__(self, lat_deg, lon_deg, alt_m=0.0):
        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.alt_m = float(alt_m)
        self.lat_scale = math.pi * EARTH_RADIUS_M / 180.0
        self.lon_scale = self.lat_scale * math.cos(math.radians(self.lat_deg))

    def to_enu(self, value):
        return (
            (float(value["lon"]) - self.lon_deg) * self.lon_scale,
            (float(value["lat"]) - self.lat_deg) * self.lat_scale,
            float(value["alt"]) - self.alt_m,
        )

    def to_geodetic(self, value):
        return {
            "lat": self.lat_deg + value[1] / self.lat_scale,
            "lon": self.lon_deg + value[0] / self.lon_scale,
            "alt": self.alt_m + value[2],
        }


def bbox_center(side):
    x1, y1, x2, y2 = side["bbox_xyxy"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def pixel_ray(side):
    """沿用现有 V2 约定：水平 FOV、机头相对 pan、零相机 roll。"""
    pose = side["source_pose"]
    u, v = bbox_center(side)
    width, height = float(side["width"]), float(side["height"])
    focal = width / (2.0 * math.tan(math.radians(float(pose["gimbal_fov_deg"])) / 2.0))
    x = (u - (width - 1.0) / 2.0) / focal
    y = (v - (height - 1.0) / 2.0) / focal
    yaw = math.radians(float(pose["heading_deg"]) + float(pose["gimbal_pan"]))
    pitch = math.radians(float(pose["gimbal_tilt"]))
    forward = (
        math.cos(pitch) * math.sin(yaw),
        math.cos(pitch) * math.cos(yaw),
        math.sin(pitch),
    )
    right = (math.cos(yaw), -math.sin(yaw), 0.0)
    down = (
        math.sin(pitch) * math.sin(yaw),
        math.sin(pitch) * math.cos(yaw),
        -math.cos(pitch),
    )
    return normalized(add(add(forward, scale(right, x)), scale(down, y)))


def single_height_projection(origin, direction, height_m):
    if abs(direction[2]) < 1e-9:
        return None
    distance = (height_m - origin[2]) / direction[2]
    if distance <= 0:
        return None
    return add(origin, scale(direction, distance))


def closest_ray_midpoint(origin_a, direction_a, origin_b, direction_b):
    """返回两条前向射线最短连线的中点及纯几何质量量。"""
    offset = subtract(origin_a, origin_b)
    a = dot(direction_a, direction_a)
    b = dot(direction_a, direction_b)
    c = dot(direction_b, direction_b)
    d = dot(direction_a, offset)
    e = dot(direction_b, offset)
    denominator = a * c - b * b
    if denominator <= 1e-12:
        return None
    distance_a = (b * e - c * d) / denominator
    distance_b = (a * e - b * d) / denominator
    if distance_a <= 0 or distance_b <= 0:
        return None
    point_a = add(origin_a, scale(direction_a, distance_a))
    point_b = add(origin_b, scale(direction_b, distance_b))
    midpoint = scale(add(point_a, point_b), 0.5)
    cosine = max(-1.0, min(1.0, abs(dot(direction_a, direction_b))))
    return {
        "point": midpoint,
        "ray_separation_m": norm(subtract(point_a, point_b)),
        "ray_angle_deg": math.degrees(math.acos(cosine)),
        "distance_a_m": distance_a,
        "distance_b_m": distance_b,
    }


def error_values(prediction, truth, frame):
    predicted_enu = frame.to_enu(prediction)
    truth_enu = frame.to_enu(truth)
    delta = subtract(predicted_enu, truth_enu)
    horizontal = math.hypot(delta[0], delta[1])
    altitude = abs(delta[2])
    return {
        "horizontal_m": horizontal,
        "altitude_m": altitude,
        "three_dimensional_m": math.hypot(horizontal, altitude),
    }


def summarize_errors(rows):
    result = {"count": len(rows)}
    for field in ("horizontal_m", "altitude_m", "three_dimensional_m"):
        values = [row[field] for row in rows]
        result[field] = {
            "mean": statistics.fmean(values) if values else None,
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": max(values) if values else None,
        }
    return result


def load_runs(dataset_root, run_names=None, max_pairs=None):
    selected = set(run_names or [])
    result = {}
    for run_root in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if selected and run_root.name not in selected:
            continue
        rows = []
        for row in jsonl_rows(run_root / "pairs.jsonl"):
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
        result[run_root.name] = rows
    missing = selected - result.keys()
    if missing:
        raise FileNotFoundError(f"未找到 run：{sorted(missing)}")
    return result


def leave_one_run_height_priors(runs, fallback_height_m):
    """每个测试 run 只用其他 run 的 GT 高度形成一个固定标量先验。"""
    heights_by_run = {
        run: [float(row["ground_truth"]["master"]["alt"]) for row in rows]
        for run, rows in runs.items()
    }
    priors = {}
    for held_out in runs:
        calibration = [height for run, heights in heights_by_run.items()
                       if run != held_out for height in heights]
        priors[held_out] = statistics.median(calibration) if calibration else fallback_height_m
    return priors


def evaluate_row(row, height_prior_m, minimum_ray_angle_deg):
    master, follower = row["master"], row["follower"]
    origin_lat = (float(master["source_pose"]["lat"]) +
                  float(follower["source_pose"]["lat"])) / 2.0
    origin_lon = (float(master["source_pose"]["lon"]) +
                  float(follower["source_pose"]["lon"])) / 2.0
    frame = LocalFrame(origin_lat, origin_lon)
    master_origin = frame.to_enu(master["source_pose"])
    follower_origin = frame.to_enu(follower["source_pose"])
    master_ray, follower_ray = pixel_ray(master), pixel_ray(follower)
    predictions = {}
    prior_points = {}
    for method, origin, direction in (
        ("single_master_height_prior", master_origin, master_ray),
        ("single_follower_height_prior", follower_origin, follower_ray),
    ):
        point = single_height_projection(origin, direction, height_prior_m)
        if point is not None:
            prior_points[method] = point
            predictions[method] = frame.to_geodetic(point)
    triangulated = closest_ray_midpoint(
        master_origin, master_ray, follower_origin, follower_ray)
    if triangulated is not None:
        predictions["dual_ray"] = frame.to_geodetic(triangulated["point"])
    use_raw_dual = bool(
        triangulated is not None and
        triangulated["ray_angle_deg"] >= minimum_ray_angle_deg)
    if use_raw_dual:
        robust_point = triangulated["point"]
        robust_source = "dual_ray"
    elif len(prior_points) == 2:
        robust_point = scale(add(
            prior_points["single_master_height_prior"],
            prior_points["single_follower_height_prior"]), 0.5)
        robust_source = "height_prior_fallback"
    else:
        robust_point = None
        robust_source = "unavailable"
    if robust_point is not None:
        predictions["dual_robust"] = frame.to_geodetic(robust_point)

    truth = row["ground_truth"]["master"]
    errors = {method: error_values(prediction, truth, frame)
              for method, prediction in predictions.items()}
    quality = {
        "baseline_m": norm(subtract(master_origin, follower_origin)),
        "source_time_delta_s": row["quality"]["source_time_delta_s"],
        "minimum_ray_angle_deg": minimum_ray_angle_deg,
        "robust_source": robust_source,
    }
    if triangulated is not None:
        quality.update({key: triangulated[key] for key in (
            "ray_separation_m", "ray_angle_deg", "distance_a_m", "distance_b_m")})
    return {
        "pair_id": row["pair_id"],
        "run": row["run"],
        "session_id": row["session_id"],
        "target_id": row["target_id"],
        "height_prior_m": height_prior_m,
        "predictions": predictions,
        "errors": errors,
        "geometry_quality": quality,
    }


def aggregate(results):
    by_run_method = defaultdict(lambda: defaultdict(list))
    all_method = defaultdict(list)
    for row in results:
        for method, errors in row["errors"].items():
            by_run_method[row["run"]][method].append(errors)
            all_method[method].append(errors)
    per_run = {
        run: {method: summarize_errors(errors) for method, errors in methods.items()}
        for run, methods in by_run_method.items()
    }
    overall = {method: summarize_errors(errors) for method, errors in all_method.items()}
    single_errors = (all_method["single_master_height_prior"] +
                     all_method["single_follower_height_prior"])
    overall["single_both_views_height_prior"] = summarize_errors(single_errors)
    return per_run, overall


def quality_summary(results):
    triangulated = [row["geometry_quality"] for row in results
                    if "ray_angle_deg" in row["geometry_quality"]]
    source_counts = defaultdict(int)
    for row in results:
        source_counts[row["geometry_quality"]["robust_source"]] += 1
    output = {
        "attempted_pairs": len(results),
        "triangulated_pairs": len(triangulated),
        "raw_dual_rejected_pairs": len(results) - len(triangulated),
        "raw_dual_coverage": len(triangulated) / len(results) if results else None,
        "robust_source_counts": dict(source_counts),
        "robust_coverage": (
            (len(results) - source_counts["unavailable"]) / len(results)
            if results else None),
    }
    for field in ("baseline_m", "ray_angle_deg", "ray_separation_m"):
        values = [row[field] for row in triangulated]
        output[field] = {
            "median": percentile(values, 0.5),
            "p05": percentile(values, 0.05),
            "p95": percentile(values, 0.95),
            "max": max(values) if values else None,
        }
    return output


def paired_comparison(results):
    """在同一 pair 内比较稳健双机与两侧单机误差均值。"""
    output = {}
    for field in ("horizontal_m", "altitude_m", "three_dimensional_m"):
        gains = []
        wins = 0
        for row in results:
            errors = row["errors"]
            if not all(method in errors for method in (
                    "single_master_height_prior", "single_follower_height_prior",
                    "dual_robust")):
                continue
            single_mean = (
                errors["single_master_height_prior"][field] +
                errors["single_follower_height_prior"][field]) / 2.0
            dual_error = errors["dual_robust"][field]
            gains.append(single_mean - dual_error)
            wins += dual_error < single_mean
        output[field] = {
            "compared_pairs": len(gains),
            "dual_win_rate": wins / len(gains) if gains else None,
            "median_gain_m": percentile(gains, 0.5),
            "mean_gain_m": statistics.fmean(gains) if gains else None,
        }
    return output


def markdown_report(summary):
    def metric(method, field, key="median"):
        value = summary["overall"].get(method, {}).get(field, {}).get(key)
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        "# 双机协同经纬度高度估计对照结果",
        "",
        "本报告来自数据集离线真实回放，不是逻辑测试，也没有启动 UE。",
        "",
        "## 主要方法",
        "",
        "- 单机：框中心射线与固定高度平面求交；该高度由其他 run 的 GT 中位数给出。",
        "- 双机：两条框中心射线的最近点中点；不读取目标高度先验。",
        "- 稳健双机：夹角小于门限或无前向交会时，回退到两侧单机先验投影的中点。",
        "- 相机约定固定为水平 FOV、主点居中、方形像素、零畸变、零外参偏移、机头相对云台 pan。",
        "",
        "## 总体误差（米）",
        "",
        "| 方法 | 水平中位数 | 水平 P95 | 高度中位数 | 高度 P95 | 三维中位数 |",
        "|---|---:|---:|---:|---:|---:|",
        ("| 单机两视角合并 | "
         f"{metric('single_both_views_height_prior', 'horizontal_m')} | "
         f"{metric('single_both_views_height_prior', 'horizontal_m', 'p95')} | "
         f"{metric('single_both_views_height_prior', 'altitude_m')} | "
         f"{metric('single_both_views_height_prior', 'altitude_m', 'p95')} | "
         f"{metric('single_both_views_height_prior', 'three_dimensional_m')} |"),
        ("| 双机射线 | "
         f"{metric('dual_ray', 'horizontal_m')} | "
         f"{metric('dual_ray', 'horizontal_m', 'p95')} | "
         f"{metric('dual_ray', 'altitude_m')} | "
         f"{metric('dual_ray', 'altitude_m', 'p95')} | "
         f"{metric('dual_ray', 'three_dimensional_m')} |"),
        ("| 稳健双机 | "
         f"{metric('dual_robust', 'horizontal_m')} | "
         f"{metric('dual_robust', 'horizontal_m', 'p95')} | "
         f"{metric('dual_robust', 'altitude_m')} | "
         f"{metric('dual_robust', 'altitude_m', 'p95')} | "
         f"{metric('dual_robust', 'three_dimensional_m')} |"),
        "",
        "## 覆盖率",
        "",
        (f"- 原始双机三角化：{summary['quality']['triangulated_pairs']}/"
         f"{summary['quality']['attempted_pairs']}，覆盖率 "
         f"{summary['quality']['raw_dual_coverage']:.2%}，拒绝 "
         f"{summary['quality']['raw_dual_rejected_pairs']} 对。"),
        (f"- 稳健双机：覆盖率 {summary['quality']['robust_coverage']:.2%}；"
         f"来源计数 {summary['quality']['robust_source_counts']}。"),
        "",
        "## 同 pair 胜率",
        "",
        (f"- 水平误差：双机优于两侧单机误差均值的比例为 "
         f"{summary['paired_comparison']['horizontal_m']['dual_win_rate']:.2%}。"),
        (f"- 高度误差：双机胜率 "
         f"{summary['paired_comparison']['altitude_m']['dual_win_rate']:.2%}，"
         f"误差中位改善 {summary['paired_comparison']['altitude_m']['median_gain_m']:.3f} m。"),
        (f"- 三维误差：双机胜率 "
         f"{summary['paired_comparison']['three_dimensional_m']['dual_win_rate']:.2%}，"
         f"误差中位改善 "
         f"{summary['paired_comparison']['three_dimensional_m']['median_gain_m']:.3f} m。"),
        "",
        "## 证据边界",
        "",
        "- GT 只参与评分，以及从其他 run 构造单机固定高度先验；预测函数不接收当前样本 GT。",
        "- 各天气 run 独立回放；留一 run 高度先验避免读取被评分 run 的真值。",
        "- 数据只有一个 seed、一个 target，且不同天气重复同一场景，不能据此证明跨路线泛化。",
        "- 相机完整内外参、畸变和曝光时刻未验证，因此结果是当前假设下的离线几何上限。",
        "- rain run 没有合格 pair；无样本不代表该天气通过测试。",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-names", nargs="*")
    parser.add_argument("--max-pairs", type=int,
                        help="每个 run 最多回放多少对；小样本实际运行时使用")
    parser.add_argument("--fallback-height-m", type=float, default=170.0,
                        help="只有一个非空 run 时使用的固定高度先验")
    parser.add_argument("--minimum-ray-angle-deg", type=float, default=1.0,
                        help="低于该夹角时稳健双机回退到高度先验，默认一度")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_root}")
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    runs = load_runs(dataset_root, args.run_names, args.max_pairs)
    priors = leave_one_run_height_priors(runs, args.fallback_height_m)
    results = []
    for run, rows in runs.items():
        results.extend(evaluate_row(row, priors[run], args.minimum_ray_angle_deg)
                       for row in rows)
    per_run, overall = aggregate(results)
    summary = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output": str(output),
        "max_pairs_per_run": args.max_pairs,
        "pairs_by_run": {run: len(rows) for run, rows in runs.items()},
        "height_prior": {
            "mode": "leave_one_run_out_ground_truth_median",
            "values_m": priors,
            "fallback_height_m": args.fallback_height_m,
            "current_test_row_ground_truth_used": False,
            "calibration_scope": "all_other_runs_only",
        },
        "robust_gate": {
            "minimum_ray_angle_deg": args.minimum_ray_angle_deg,
            "selection_uses_ground_truth": False,
            "fallback": "midpoint_of_two_single_height_prior_projections",
        },
        "camera_model": (
            "horizontal_fov_center_principal_square_pixel_zero_distortion_"
            "zero_camera_offset_heading_plus_pan_zero_roll"),
        "quality": quality_summary(results),
        "paired_comparison": paired_comparison(results),
        "per_run": per_run,
        "overall": overall,
        "limitations": [
            "camera_intrinsics_extrinsics_and_distortion_unverified",
            "source_sim_time_not_verified_exposure_time",
            "visibility_annotation_unannotated",
            "one_seed_one_target_repeated_scenario_across_weather",
            "rain_run_has_zero_accepted_pairs",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (output / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
