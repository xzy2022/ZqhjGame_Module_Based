# 修改时间：2026-09-16。
# 修改目的：区分退化小夹角中的正深度解与后方交点解。
# 修改内容：先求两射线深度再应用夹角门限，并由几何值生成显式质量状态。
# 修改时间：2026-09-16。
# 修改目的：让被质量门拒绝的双机样本仍保留可复核几何量。
# 修改内容：在小夹角、平行和后方交点失败记录中写入夹角、基线与深度诊断。
# 修改时间：2026-09-16。
# 修改目的：修正纯估计函数抽取后的异常处理结构。
# 修改内容：将样本异常捕获保留在评测包装函数内部。
# 修改时间：2026-09-16。
# 修改目的：保留可复用估计入口并让质量拒绝原因可审计。
# 修改内容：拆出纯双机估计函数并显式记录正深度和小夹角状态。
# 修改时间：2026-09-16。
# 修改目的：验证严格双机配对数据上的目标经纬度与高度三角测量能力。
# 修改内容：实现有限外参约定校准、双射线最近点估计及独立留出集分层评测。
"""双机 bbox 中心射线三角测量离线实验。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class Convention:
    name: str
    x_sign: int
    y_sign: int
    pan_sign: int
    yaw_offset_deg: int


CONVENTIONS = tuple(
    Convention(
        name=f"x{x_sign:+d}_y{y_sign:+d}_pan{pan_sign:+d}_yaw{yaw_offset:+d}",
        x_sign=x_sign,
        y_sign=y_sign,
        pan_sign=pan_sign,
        yaw_offset_deg=yaw_offset,
    )
    for x_sign in (1, -1)
    for y_sign in (1, -1)
    for pan_sign in (1, -1)
    for yaw_offset in (0, 180)
)


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    normal = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return (
        (normal + alt_m) * cos_lat * math.cos(lon),
        (normal + alt_m) * cos_lat * math.sin(lon),
        (normal * (1.0 - WGS84_E2) + alt_m) * sin_lat,
    )


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    lon = math.atan2(y, x)
    horizontal = math.hypot(x, y)
    lat = math.atan2(z, horizontal * (1.0 - WGS84_E2))
    alt = 0.0
    for _ in range(8):
        sin_lat = math.sin(lat)
        normal = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = horizontal / max(math.cos(lat), 1e-12) - normal
        lat = math.atan2(z, horizontal * (1.0 - WGS84_E2 * normal / (normal + alt)))
    sin_lat = math.sin(lat)
    normal = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = horizontal / max(math.cos(lat), 1e-12) - normal
    return math.degrees(lat), math.degrees(lon), alt


class LocalFrame:
    def __init__(self, lat_deg: float, lon_deg: float, alt_m: float):
        self.lat = math.radians(lat_deg)
        self.lon = math.radians(lon_deg)
        self.origin_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        sin_lat, cos_lat = math.sin(self.lat), math.cos(self.lat)
        sin_lon, cos_lon = math.sin(self.lon), math.cos(self.lon)
        self.east = (-sin_lon, cos_lon, 0.0)
        self.north = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
        self.up = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)

    def to_enu(self, lat_deg: float, lon_deg: float, alt_m: float) -> tuple[float, float, float]:
        ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        delta = tuple(ecef[i] - self.origin_ecef[i] for i in range(3))
        return tuple(sum(delta[i] * axis[i] for i in range(3)) for axis in (self.east, self.north, self.up))

    def to_geodetic(self, east: float, north: float, up: float) -> tuple[float, float, float]:
        ecef = tuple(
            self.origin_ecef[i] + east * self.east[i] + north * self.north[i] + up * self.up[i]
            for i in range(3)
        )
        return ecef_to_geodetic(*ecef)


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = norm(vector)
    return tuple(value / length for value in vector)


def camera_ray(side: dict, calibration: dict, convention: Convention) -> tuple[float, float, float]:
    bbox = side["bbox_xyxy"]
    u = (float(bbox[0]) + float(bbox[2])) / 2.0
    v = (float(bbox[1]) + float(bbox[3])) / 2.0
    intrinsics = calibration["intrinsics"]
    x = convention.x_sign * (u - float(intrinsics["cx_px"])) / float(intrinsics["fx_px"])
    y = convention.y_sign * (v - float(intrinsics["cy_px"])) / float(intrinsics["fy_px"])

    pose = side["source_pose"]
    attitude = side["aircraft_attitude"]
    yaw = math.radians(
        float(attitude["yaw"])
        + convention.pan_sign * float(pose["gimbal_pan"])
        + convention.yaw_offset_deg
    )
    tilt = math.radians(float(pose["gimbal_tilt"]))
    forward = (math.cos(tilt) * math.sin(yaw), math.cos(tilt) * math.cos(yaw), math.sin(tilt))
    right = (math.cos(yaw), -math.sin(yaw), 0.0)
    image_down = (math.sin(tilt) * math.sin(yaw), math.sin(tilt) * math.cos(yaw), -math.cos(tilt))
    return normalize(tuple(forward[i] + x * right[i] + y * image_down[i] for i in range(3)))


def triangulate(
    first_origin: tuple[float, float, float],
    first_ray: tuple[float, float, float],
    second_origin: tuple[float, float, float],
    second_ray: tuple[float, float, float],
    min_convergence_deg: float,
) -> tuple[dict | None, str | None]:
    ray_dot = max(-1.0, min(1.0, dot(first_ray, second_ray)))
    convergence_deg = math.degrees(math.acos(abs(ray_dot)))
    diagnostics = {
        "convergence_angle_deg": convergence_deg,
        "baseline_3d_m": norm(tuple(first_origin[i] - second_origin[i] for i in range(3))),
        "baseline_horizontal_m": math.hypot(first_origin[0] - second_origin[0], first_origin[1] - second_origin[1]),
    }
    offset = tuple(first_origin[i] - second_origin[i] for i in range(3))
    denominator = 1.0 - ray_dot * ray_dot
    if denominator < 1e-12:
        return diagnostics, "parallel_rays"
    first_depth = (ray_dot * dot(second_ray, offset) - dot(first_ray, offset)) / denominator
    second_depth = (dot(second_ray, offset) - ray_dot * dot(first_ray, offset)) / denominator
    diagnostics.update(first_depth_m=first_depth, second_depth_m=second_depth)
    if first_depth <= 0.0 or second_depth <= 0.0:
        return diagnostics, "intersection_behind_camera"
    if convergence_deg < min_convergence_deg:
        return diagnostics, "convergence_too_small"

    first_point = tuple(first_origin[i] + first_depth * first_ray[i] for i in range(3))
    second_point = tuple(second_origin[i] + second_depth * second_ray[i] for i in range(3))
    estimate = tuple((first_point[i] + second_point[i]) / 2.0 for i in range(3))
    gap = norm(tuple(first_point[i] - second_point[i] for i in range(3)))
    diagnostics.update({
        "estimate_enu_m": estimate,
        "ray_gap_m": gap,
    })
    return diagnostics, None


def iter_jsonl(path: Path, limit: int | None) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if limit is not None and index >= limit:
                return
            if line.strip():
                yield json.loads(line)


def load_dataset(dataset: Path, run_names: list[str], limit_per_run: int | None) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    calibrations: dict[str, dict] = {}
    for run_name in run_names:
        run_root = dataset / run_name
        pairs_path = run_root / "pairs.jsonl"
        if not pairs_path.is_file():
            raise FileNotFoundError(f"缺少配对文件：{pairs_path}")
        calibrations[run_name] = json.loads((run_root / "camera_calibration.json").read_text(encoding="utf-8"))
        rows.extend(iter_jsonl(pairs_path, limit_per_run))
    return rows, calibrations


def make_local_frame(rows: list[dict]) -> LocalFrame:
    if not rows:
        raise ValueError("所选轮次没有配对样本")
    pose = rows[0]["master"]["source_pose"]
    return LocalFrame(float(pose["lat"]), float(pose["lon"]), float(pose["alt"]))


def evaluate_row(
    row: dict,
    calibration: dict,
    local_frame: LocalFrame,
    convention: Convention,
    split: str,
    min_convergence_deg: float,
) -> dict:
    result = {
        "pair_id": row["pair_id"],
        "run": row["run"],
        "weather": weather_from_run(row["run"]),
        "split": split,
        "status": "failed",
        "failure_reason": None,
        "convention": convention.name,
    }
    try:
        estimate_result = estimate_pair(
            row,
            calibration,
            local_frame,
            convention,
            min_convergence_deg,
        )
        if estimate_result["status"] != "ok":
            result["failure_reason"] = estimate_result["failure_reason"]
            result["quality"] = estimate_result["quality"]
            result["geometry"] = estimate_result["geometry"]
            return result

        estimate_geo = estimate_result["estimate"]
        estimate_enu = estimate_result["estimate_enu_m"]
        truth = row["ground_truth"][row["ground_truth"]["canonical_side"]]
        truth_enu = local_frame.to_enu(float(truth["lat"]), float(truth["lon"]), float(truth["alt"]))
        error = tuple(estimate_enu[i] - truth_enu[i] for i in range(3))
        result.update(
            status="ok",
            estimate=estimate_geo,
            ground_truth={"lat": truth["lat"], "lon": truth["lon"], "alt": truth["alt"]},
            error={
                "east_m": error[0],
                "north_m": error[1],
                "vertical_signed_m": error[2],
                "horizontal_m": math.hypot(error[0], error[1]),
                "vertical_abs_m": abs(error[2]),
                "three_d_m": norm(error),
            },
            geometry=estimate_result["geometry"],
            quality=estimate_result["quality"],
        )
        return result
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        result["failure_reason"] = f"invalid_sample:{type(exc).__name__}"
        return result


def estimate_pair(
    row: dict,
    calibration: dict,
    local_frame: LocalFrame,
    convention: Convention,
    min_convergence_deg: float = 0.25,
) -> dict:
    """只使用双侧框、相机假设和飞行器/云台位姿估计目标坐标。"""
    origins = []
    rays = []
    for side_name in ("master", "follower"):
        side = row[side_name]
        pose = side["source_pose"]
        origins.append(local_frame.to_enu(float(pose["lat"]), float(pose["lon"]), float(pose["alt"])))
        rays.append(camera_ray(side, calibration, convention))
    geometry, failure = triangulate(origins[0], rays[0], origins[1], rays[1], min_convergence_deg)
    if failure:
        first_depth = geometry.get("first_depth_m")
        second_depth = geometry.get("second_depth_m")
        return {
            "status": "failed",
            "failure_reason": failure,
            "geometry": geometry,
            "quality": {
                "positive_depth": (
                    None if first_depth is None or second_depth is None
                    else first_depth > 0.0 and second_depth > 0.0
                ),
                "convergence_accepted": geometry["convergence_angle_deg"] >= min_convergence_deg,
            },
        }
    estimate_enu = geometry.pop("estimate_enu_m")
    latitude, longitude, altitude = local_frame.to_geodetic(*estimate_enu)
    return {
        "status": "ok",
        "failure_reason": None,
        "estimate": {"lat": latitude, "lon": longitude, "alt": altitude},
        "estimate_enu_m": estimate_enu,
        "geometry": geometry,
        "quality": {
            "positive_depth": True,
            "convergence_accepted": True,
        },
    }


def metric_summary(rows: list[dict]) -> dict:
    successes = [row for row in rows if row["status"] == "ok"]
    failures = [row for row in rows if row["status"] != "ok"]
    failure_reasons: dict[str, int] = defaultdict(int)
    for row in failures:
        failure_reasons[str(row.get("failure_reason"))] += 1

    def describe(key: str) -> dict:
        values = [float(row["error"][key]) for row in successes]
        return {
            "mean": mean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p75": percentile(values, 0.75),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    signed_vertical = [float(row["error"]["vertical_signed_m"]) for row in successes]
    return {
        "samples": len(rows),
        "successes": len(successes),
        "failures": len(failures),
        "failure_rate": len(failures) / len(rows) if rows else None,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "horizontal_error_m": describe("horizontal_m"),
        "vertical_abs_error_m": describe("vertical_abs_m"),
        "three_d_error_m": describe("three_d_m"),
        "vertical_signed_bias_m": mean(signed_vertical) if signed_vertical else None,
        "ray_gap_m": geometry_summary(successes, "ray_gap_m"),
        "convergence_angle_deg": geometry_summary(successes, "convergence_angle_deg"),
        "baseline_horizontal_m": geometry_summary(successes, "baseline_horizontal_m"),
    }


def geometry_summary(rows: list[dict], key: str) -> dict:
    values = [float(row["geometry"][key]) for row in rows if row["status"] == "ok"]
    return {
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def group_summary(rows: list[dict], key_function) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key_function(row)].append(row)
    return {name: metric_summary(group_rows) for name, group_rows in sorted(groups.items())}


def numeric_bin(value: float, boundaries: tuple[float, ...], unit: str) -> str:
    lower = 0.0
    for upper in boundaries:
        if value < upper:
            return f"[{lower:g},{upper:g}){unit}"
        lower = upper
    return f"[{lower:g},inf){unit}"


def weather_from_run(run_name: str) -> str:
    prefix = "seed01-"
    suffix = "-150s-fov30"
    if run_name.startswith(prefix) and run_name.endswith(suffix):
        return run_name[len(prefix):-len(suffix)]
    return run_name


def parse_run_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def available_runs(dataset: Path) -> list[str]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    return [item["run"] for item in manifest["runs"] if int(item.get("accepted_pairs", 0)) > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description="双机 bbox 中心射线三角测量离线评测")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-runs", help="逗号分隔；默认前两个非空轮次")
    parser.add_argument("--heldout-runs", help="逗号分隔；默认其余非空轮次")
    parser.add_argument("--limit-per-run", type=int)
    parser.add_argument("--min-convergence-deg", type=float, default=0.25)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}")
    all_runs = available_runs(dataset)
    calibration_runs = parse_run_list(args.calibration_runs) or all_runs[:2]
    heldout_runs = parse_run_list(args.heldout_runs) or [name for name in all_runs if name not in calibration_runs]
    if set(calibration_runs) & set(heldout_runs):
        raise ValueError("校准轮次和留出轮次不能重叠")
    if not calibration_runs or not heldout_runs:
        raise ValueError("必须同时提供非空校准轮次和独立留出轮次")

    calibration_rows, calibration_by_run = load_dataset(dataset, calibration_runs, args.limit_per_run)
    heldout_rows, heldout_by_run = load_dataset(dataset, heldout_runs, args.limit_per_run)
    local_frame = make_local_frame(calibration_rows)

    score_rows = []
    convention_outputs: dict[str, list[dict]] = {}
    for convention in CONVENTIONS:
        evaluated = [
            evaluate_row(
                row,
                calibration_by_run[row["run"]],
                local_frame,
                convention,
                "calibration",
                args.min_convergence_deg,
            )
            for row in calibration_rows
        ]
        convention_outputs[convention.name] = evaluated
        metrics = metric_summary(evaluated)
        median_3d = metrics["three_d_error_m"]["p50"]
        p95_3d = metrics["three_d_error_m"]["p95"]
        score_rows.append({
            "convention": convention.name,
            "metrics": metrics,
            "selection_key": [
                metrics["failure_rate"] if metrics["failure_rate"] is not None else 1.0,
                median_3d if median_3d is not None else 1e300,
                p95_3d if p95_3d is not None else 1e300,
            ],
        })
    score_rows.sort(key=lambda row: tuple(row["selection_key"]))
    selected = next(item for item in CONVENTIONS if item.name == score_rows[0]["convention"])
    selected_calibration = convention_outputs[selected.name]
    heldout = [
        evaluate_row(
            row,
            heldout_by_run[row["run"]],
            local_frame,
            selected,
            "heldout",
            args.min_convergence_deg,
        )
        for row in heldout_rows
    ]

    report = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "bbox_center_two_ray_closest_points",
        "dataset": str(dataset),
        "output": str(output),
        "split": {
            "unit": "whole_run_weather",
            "calibration_runs": calibration_runs,
            "heldout_runs": heldout_runs,
            "calibration_samples": len(calibration_rows),
            "heldout_samples": len(heldout_rows),
            "limit_per_run": args.limit_per_run,
        },
        "input_contract": {
            "estimation_inputs": ["bbox_xyxy", "derived_camera_intrinsics", "uav_lat_lon_alt", "aircraft_yaw", "gimbal_pan_tilt"],
            "ground_truth_usage": "calibration convention selection and offline scoring only",
            "camera_origin_assumption": "uav_gnss_position_without_lever_arm",
            "aircraft_roll_pitch_usage": "not_applied_dataset_values_are_zero",
        },
        "selected_convention": selected.__dict__,
        "candidate_selection": score_rows,
        "calibration_metrics": metric_summary(selected_calibration),
        "heldout_metrics": metric_summary(heldout),
        "heldout_by_weather": group_summary(heldout, lambda row: row["weather"]),
        "heldout_by_convergence": group_summary(
            heldout,
            lambda row: "failed" if row["status"] != "ok" else numeric_bin(row["geometry"]["convergence_angle_deg"], (1.0, 2.0, 5.0, 10.0, 20.0), "deg"),
        ),
        "heldout_by_ray_gap": group_summary(
            heldout,
            lambda row: "failed" if row["status"] != "ok" else numeric_bin(row["geometry"]["ray_gap_m"], (1.0, 3.0, 10.0, 30.0), "m"),
        ),
        "limitations": [
            "camera intrinsics and extrinsics are derived under unverified assumptions",
            "bbox centers come from UE projection labels rather than detector output",
            "visibility and exposure timestamps are unverified",
            "candidate convention selection is valid only for the stated calibration runs",
        ],
    }

    output.mkdir(parents=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected_calibration + heldout:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    run_summary = {
        "status": "completed",
        "created_at": report["created_at"],
        "dataset": str(dataset),
        "output": str(output),
        "selected_convention": selected.name,
        "heldout_samples": len(heldout),
        "heldout_failure_rate": report["heldout_metrics"]["failure_rate"],
        "heldout_horizontal_p50_m": report["heldout_metrics"]["horizontal_error_m"]["p50"],
        "heldout_vertical_p50_m": report["heldout_metrics"]["vertical_abs_error_m"]["p50"],
        "heldout_three_d_p50_m": report["heldout_metrics"]["three_d_error_m"]["p50"],
    }
    (output / "run.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
