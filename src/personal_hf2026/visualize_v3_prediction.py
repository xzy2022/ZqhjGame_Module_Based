# 修改时间：2026-09-21。
# 修改目的：使位置审计图聚焦轨迹关系，避免对象编号和高度文字遮挡曲线。
# 修改内容：取消真值轨迹的对象编号与高度标注，保留线条、点位和图例。
# 修改时间：2026-09-21。
# 修改目的：让位置审计图同时展示本机轨迹，便于判断相机投影与飞行位置关系。
# 修改内容：将同一时段各已处理帧的本机坐标以紫色三角形绘制到东北米制坐标。
# 修改时间：2026-09-21。
# 修改目的：避免稀疏轨迹被无标记线段误读为只有端点或连续运动。
# 修改内容：真值或预测轨迹少于五个实际采样点时，为每个点绘制显式标记。
# 修改时间：2026-09-21。
# 修改目的：让 V3 单局日志可按时段检视地面真值与像素推测位置的对应关系。
# 修改内容：绘制米制东北坐标、目标与诱饵轨迹、估高标记及最近同类配对虚线。
"""绘制 --detailed-log 产生的 V3 目标/诱饵真值和预测位置。"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _rows(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{number} 不是合法 JSONL") from exc


def _time_range(value: str) -> tuple[float, float]:
    try:
        start, end = (float(part) for part in value.split("-", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--time-range 格式应为 80-100") from exc
    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        raise argparse.ArgumentTypeError("--time-range 必须为有限且递增的秒数范围")
    return start, end


def _ue_id(item):
    for key in ("target_id", "uid", "id", "object_id", "name"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return None


def _meters(lat, lon, origin):
    origin_lat, origin_lon = origin
    north = math.radians(float(lat) - origin_lat) * 6_378_137.0
    east = (math.radians(float(lon) - origin_lon) * 6_378_137.0
            * math.cos(math.radians(origin_lat)))
    return east, north


def _nearest_truth(truth_times, truth_rows, timestamp):
    index = bisect.bisect_left(truth_times, timestamp)
    candidates = [item for item in (index - 1, index) if 0 <= item < len(truth_times)]
    return truth_rows[min(candidates, key=lambda item: abs(truth_times[item] - timestamp))] if candidates else None


def render(run: Path, uid: str, time_range: tuple[float, float], link_every_s: float) -> dict:
    frames_path, predictions_path, truth_path = (run / "visual_frames.jsonl",
                                                 run / "visual_predictions.jsonl",
                                                 run / "visual_truth.jsonl")
    if not all(path.is_file() for path in (frames_path, predictions_path, truth_path)):
        raise FileNotFoundError("需要 --detailed-log 产生 visual_frames.jsonl、visual_predictions.jsonl 和 visual_truth.jsonl")
    start, end = time_range
    frames = [row for row in _rows(frames_path)
              if str(row.get("uid")) == str(uid)
              and start <= float(row.get("observed_sim_time_s", -1e30)) <= end]
    predictions = {(str(row.get("uid")), str(row.get("frame_id"))): row
                   for row in _rows(predictions_path)}
    truth_rows = sorted(_rows(truth_path), key=lambda row: float(row.get("sim_time_s", -1e30)))
    truth_times = [float(row.get("sim_time_s", -1e30)) for row in truth_rows]
    actual, predicted, links = defaultdict(list), defaultdict(list), []
    all_points = []
    own_positions = []
    selected_for_link = -1e30
    frames_without_visible_ids = 0
    for frame in frames:
        timestamp = float(frame["observed_sim_time_s"])
        own_pose = frame.get("own_pose")
        if isinstance(own_pose, dict):
            try:
                own_lat, own_lon = float(own_pose["lat"]), float(own_pose["lon"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if math.isfinite(own_lat) and math.isfinite(own_lon):
                    own_positions.append({"time_s": timestamp, "lat": own_lat, "lon": own_lon})
                    all_points.append((own_lat, own_lon))
        truth = _nearest_truth(truth_times, truth_rows, timestamp)
        visible_ids = {_ue_id(item) for item in frame.get("ue_projected_objects", [])
                       if isinstance(item, dict) and _ue_id(item) is not None}
        if not visible_ids:
            frames_without_visible_ids += 1
        true_now = []
        if truth is not None:
            for item in truth.get("entities", []):
                if str(item.get("uid")) not in visible_ids:
                    continue
                if item.get("lat") is None or item.get("lon") is None:
                    continue
                row = {"time_s": timestamp, **item}
                actual[(str(item.get("category")), str(item.get("uid")))].append(row)
                true_now.append(row)
                all_points.append((float(item["lat"]), float(item["lon"])))
        prediction = predictions.get((str(uid), str(frame.get("frame_id"))))
        predicted_now = []
        if prediction is not None:
            for item in prediction.get("objects", []):
                point = item.get("ground_point_h0") if isinstance(item, dict) else None
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                category = "target" if item.get("class_name") == "real_vehicle" else "decoy"
                row = {"time_s": timestamp, "category": category,
                       "track_id": item.get("track_id"), "lat": float(point[0]),
                       "lon": float(point[1]), "alt": float(point[2]) if len(point) > 2 else 0.0}
                predicted[(category, str(item.get("track_id")))].append(row)
                predicted_now.append(row)
                all_points.append((row["lat"], row["lon"]))
        if timestamp - selected_for_link >= link_every_s:
            selected_for_link = timestamp
            for category in ("target", "decoy"):
                left = [item for item in true_now if item["category"] == category]
                right = [item for item in predicted_now if item["category"] == category]
                for item in left:
                    if not right:
                        continue
                    match = min(right, key=lambda other: (float(other["lat"]) - float(item["lat"])) ** 2
                                + (float(other["lon"]) - float(item["lon"])) ** 2)
                    links.append((item, match))
    if not all_points:
        raise ValueError("指定时段没有同时可绘制的真值或预测位置；请检查无人机 ID、时段或 UE 框元数据")
    origin = (sum(point[0] for point in all_points) / len(all_points),
              sum(point[1] for point in all_points) / len(all_points))
    figure, axis = plt.subplots(figsize=(11, 9), constrained_layout=True)
    styles = {"target": ("#8b0000", "-"), "decoy": ("#ff8c00", "-")}
    for (category, _object_id), values in actual.items():
        xy = [_meters(item["lat"], item["lon"], origin) for item in values]
        axis.plot([item[0] for item in xy], [item[1] for item in xy],
                  color=styles[category][0], linestyle=styles[category][1], linewidth=1.8)
        if len(values) < 5:
            axis.scatter([item[0] for item in xy], [item[1] for item in xy],
                         s=28, marker="o", color=styles[category][0],
                         edgecolors="white", linewidths=0.5, zorder=4)
    prediction_styles = {"target": "#90ee90", "decoy": "#87ceeb"}
    for (category, _track_id), values in predicted.items():
        xy = [_meters(item["lat"], item["lon"], origin) for item in values]
        axis.plot([item[0] for item in xy], [item[1] for item in xy],
                  color=prediction_styles[category], linestyle="--", linewidth=1.4)
        if len(values) < 5:
            axis.scatter([item[0] for item in xy], [item[1] for item in xy],
                         s=42, marker="x", color=prediction_styles[category],
                         linewidths=1.4, zorder=4)
    if own_positions:
        own_xy = [_meters(item["lat"], item["lon"], origin) for item in own_positions]
        axis.scatter([item[0] for item in own_xy], [item[1] for item in own_xy],
                     s=28, marker="^", color="#800080", edgecolors="white",
                     linewidths=0.35, alpha=0.9, zorder=3)
    for truth_item, prediction_item in links:
        first, second = _meters(truth_item["lat"], truth_item["lon"], origin), _meters(
            prediction_item["lat"], prediction_item["lon"], origin)
        axis.plot([first[0], second[0]], [first[1], second[1]], "k--", linewidth=0.65, alpha=0.65)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("East (m)")
    axis.set_ylabel("North (m)")
    axis.set_title(f"V3 prediction audit: UAV {uid}, {start:g}-{end:g}s")
    axis.grid(True, alpha=0.25)
    axis.legend(handles=[
        Line2D([0], [0], color="#8b0000", lw=2, label="ground truth target"),
        Line2D([0], [0], color="#ff8c00", lw=2, label="ground truth decoy"),
        Line2D([0], [0], color="#90ee90", lw=2, ls="--", label="predicted target"),
        Line2D([0], [0], color="#87ceeb", lw=2, ls="--", label="predicted decoy"),
        Line2D([0], [0], color="black", lw=1, ls="--", label="nearest same-class link"),
        Line2D([0], [0], color="#800080", marker="^", markeredgecolor="white",
               markersize=7, linestyle="None", label="own UAV position"),
    ], loc="best")
    output = run / "visual_prediction" / f"uav_{uid}_{start:g}-{end:g}s.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {"frames_in_range": len(frames), "actual_tracks": len(actual),
            "predicted_tracks": len(predicted), "links": len(links),
            "own_position_samples": len(own_positions),
            "frames_without_visible_ids": frames_without_visible_ids,
            "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="单局输出目录")
    parser.add_argument("--time-range", type=_time_range, required=True, help="仿真秒范围，如 80-100")
    parser.add_argument("--uav-id", required=True, help="无人机 UID")
    parser.add_argument("--link-every-s", type=float, default=2.0, help="黑色配对虚线最小间隔（秒）")
    args = parser.parse_args(argv)
    if args.link_every_s <= 0:
        parser.error("--link-every-s 必须大于 0")
    run = args.run.resolve()
    if not run.is_dir():
        parser.error(f"--run 不存在：{run}")
    print(json.dumps(render(run, args.uav_id, args.time_range, args.link_every_s), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
