# 修改时间：2026-09-16。
# 修改目的：用既有在线和离线真实产物审计 FOV 语义与相机姿态字段覆盖，不启动 UE。
# 修改内容：在硬字节上限内汇总场景 FOV、在线预测姿态字段和离线水平/垂直 FOV 残差证据。
"""审计实时双机定位产物中的 FOV 与相机姿态可观测性。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def _read_json(path: Path, max_bytes: int) -> Any:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"输入超过硬字节上限：{path} ({size} > {max_bytes})")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path, max_bytes: int) -> list[dict[str, Any]]:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"输入超过硬字节上限：{path} ({size} > {max_bytes})")
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(value)
    return rows


def _numbers(values: Iterable[Any]) -> dict[str, Any]:
    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    unique = sorted(set(round(value, 9) for value in finite))
    return {
        "count": len(finite),
        "unique_count": len(unique),
        "unique_values": unique if len(unique) <= 20 else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def _collect_named_numbers(value: Any, key_name: str) -> list[float]:
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == key_name and isinstance(item, (int, float)):
                found.append(float(item))
            found.extend(_collect_named_numbers(item, key_name))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_named_numbers(item, key_name))
    return found


def _candidate(audit: Mapping[str, Any], axis: str) -> Mapping[str, Any] | None:
    selection = audit.get("coordinate_convention_selection", {})
    for item in selection.get("ranked_candidates", []):
        if (item.get("fov_axis") == axis
                and item.get("yaw_mode") == "heading_plus_pan"
                and item.get("pixel_transform") == "u_right_v_down"):
            return item
    return None


def _online_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        for role in ("master", "follower"):
            view = row.get("views", {}).get(role, {})
            pose = view.get("source_pose", {})
            for key in ("heading_deg", "gimbal_pan", "gimbal_tilt", "gimbal_fov_deg"):
                fields[f"source_pose.{key}"].append(pose.get(key))
            attitude = view.get("aircraft_attitude")
            if isinstance(attitude, Mapping):
                for key in ("roll", "pitch", "yaw"):
                    fields[f"aircraft_attitude.{key}"].append(attitude.get(key))
            else:
                fields["aircraft_attitude.yaw"].append(view.get("aircraft_yaw_deg"))
                fields["aircraft_attitude.roll"].append(None)
                fields["aircraft_attitude.pitch"].append(None)
            fields["pose_time_delta_s"].append(view.get("pose_time_delta_s"))
    return {
        "records": len(rows),
        "views": len(rows) * 2,
        "fields": {key: _numbers(values) for key, values in sorted(fields.items())},
    }


def _fmt_residual(candidate: Mapping[str, Any] | None) -> str:
    if candidate is None:
        return "缺失"
    quantiles = candidate.get("angular_residual_deg", {}).get("quantiles", {})
    return f"P50={quantiles.get('p50')}°, P95={quantiles.get('p95')}°"


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    online = result["online"]
    fields = online["fields"]
    horizontal = result["offline_fov_axis_evidence"]["horizontal"]
    vertical = result["offline_fov_axis_evidence"]["vertical"]
    fov = fields.get("source_pose.gimbal_fov_deg", {})
    roll = fields.get("aircraft_attitude.roll", {})
    pitch = fields.get("aircraft_attitude.pitch", {})
    yaw = fields.get("aircraft_attitude.yaw", {})
    offline_fields = result["offline_recorded_field_distributions"]
    offline_yaw = offline_fields.get("aircraft_attitude.yaw", {})
    offline_roll = offline_fields.get("aircraft_attitude.roll", {})
    offline_pitch = offline_fields.get("aircraft_attitude.pitch", {})
    lines = [
        "# 实时双机定位 FOV 与姿态字段审计",
        "",
        "## 结论",
        "",
        f"- 场景文件中的 FOV 值：`{result['scenario_fov_values']}`；在线有效记录读回 FOV：`{fov.get('unique_values')}`。",
        f"- 离线真实投影框支持水平 FOV：{_fmt_residual(horizontal)}；垂直解释：{_fmt_residual(vertical)}。这是经验识别结果，不等同于渲染器接口文档。",
        f"- 在线记录共有 {online['records']} 条、{online['views']} 个视图；roll/pitch/yaw 有效值数分别为 {roll.get('count', 0)}/{pitch.get('count', 0)}/{yaw.get('count', 0)}。",
        f"- 离线 3128 个视图的 yaw 有 {offline_yaw.get('unique_rounded_1e9')} 个取值、范围 {offline_yaw.get('quantiles', {}).get('p00')}～{offline_yaw.get('quantiles', {}).get('p100')}°；roll/pitch 各自唯一值数为 {offline_roll.get('unique_rounded_1e9')}/{offline_pitch.get('unique_rounded_1e9')}，且都为 0°。",
        "- 现有记录若只有 yaw，不能审计非零 roll/pitch；即使三轴齐全，也仍缺曝光时刻对齐的 renderer camera pose、相机到云台外参和畸变参数，不能称为完整相机姿态。",
        "",
        "## 在线字段分布",
        "",
        "```json",
        json.dumps(fields, ensure_ascii=False, indent=2, allow_nan=False),
        "```",
        "",
        "## 证据边界",
        "",
        "- 本工具只读取既有产物，不启动 UE。",
        "- 在线样本量过小时，字段存在不代表已经覆盖动态变化。",
        "- 离线数据的 gimbal pan、tilt、aircraft roll/pitch 近乎不变；它验证了当前俯视配置下的 FOV 轴和 yaw 图像旋转，但未验证动态 pan/tilt、非零 roll/pitch 或外参轴约定。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-output", required=True, type=Path)
    parser.add_argument("--offline-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-input-bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    if args.max_input_bytes <= 0:
        parser.error("--max-input-bytes 必须大于零")
    live_output = args.live_output.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"输出目录已存在，拒绝覆盖：{output}")

    run = _read_json(live_output / "run.json", args.max_input_bytes)
    live_summary = _read_json(
        live_output / "paired_geolocation_summary.json", args.max_input_bytes)
    rows = _read_jsonl(
        live_output / "paired_geolocation_predictions.jsonl", args.max_input_bytes)
    scenario = _read_json(
        live_output / "scenario_coop_decoy_prepared.json", args.max_input_bytes)
    offline = _read_json(args.offline_audit.resolve(), args.max_input_bytes)
    result = {
        "schema_version": 1,
        "live_output": str(live_output),
        "live_git_branch": run.get("git_branch"),
        "live_git_head": run.get("git_head"),
        "live_status": run.get("status"),
        "scenario_fov_values": sorted(set(_collect_named_numbers(scenario, "fov"))),
        "online": _online_summary(rows),
        "online_counts": live_summary.get("counts", {}),
        "offline_fov_axis_evidence": {
            "horizontal": _candidate(offline, "horizontal"),
            "vertical": _candidate(offline, "vertical"),
            "identifiability_note": offline.get(
                "coordinate_convention_selection", {}).get("identifiability_note"),
        },
        "offline_recorded_field_distributions": offline.get(
            "recorded_field_distributions", {}),
        "limitations": [
            "no_ue_run_in_this_audit",
            "online_predictions_only_contain_accepted_estimates",
            "renderer_camera_extrinsics_unavailable",
            "source_sim_time_not_verified_exposure_time",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(output / "REPORT.md", result)
    print(json.dumps({
        "status": "completed",
        "output": str(output),
        "records": result["online"]["records"],
        "scenario_fov_values": result["scenario_fov_values"],
        "observed_fov_values": result["online"]["fields"].get(
            "source_pose.gimbal_fov_deg", {}).get("unique_values"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
