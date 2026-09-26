# 修改时间：2026-09-26。
# 修改目的：从紧凑日志直接量化三机搜索及接管的出现次数。
# 修改内容：按规划器模式统计控制采样并单列 TAKEOVER 采样数。
# 修改时间：2026-09-24。
# 修改目的：让 Agent4 的紧凑运行轨迹可直接离线复核。
# 修改内容：按业务事件、状态和视觉诊断干预汇总审计结果。
"""汇总 Agent4 实际运行日志。"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path


def _rows(path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def analyze_run(output):
    output = Path(output)
    events = Counter()
    states = Counter()
    search_modes = Counter()
    samples = 0
    first_time = last_time = None
    for row in _rows(output / "agent_v4_trace.jsonl"):
        if row.get("kind") == "event":
            events[row.get("event", "unknown")] += 1
        elif row.get("kind") == "control_sample":
            states[row.get("state", "unknown")] += 1
            mode = row.get("search_plan", {}).get("mode")
            if mode is not None:
                search_modes[mode] += 1
            samples += 1
        time = row.get("time")
        if time is not None:
            first_time = time if first_time is None else min(first_time, time)
            last_time = time if last_time is None else max(last_time, time)
    predictions = 0
    changed = 0
    for row in _rows(output / "visual_predictions.jsonl"):
        predictions += 1
        changed += int(bool(row.get("vision_diagnostic", {}).get("changed")))
    result = {"schema_version": 1, "event_counts": dict(events),
              "state_sample_counts": dict(states), "control_samples": samples,
              "search_mode_sample_counts": dict(search_modes),
              "takeover_samples": search_modes["TAKEOVER"],
              "first_time_s": first_time, "last_time_s": last_time,
              "prediction_frames": predictions, "diagnostic_changed_frames": changed}
    (output / "agent_v4_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(analyze_run(parser.parse_args(argv).output), ensure_ascii=False))


if __name__ == "__main__":
    main()
