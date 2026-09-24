# 修改时间：2026-09-24。
# 修改目的：无详细日志时避免新视觉帧事件在内存里无限累积。
# 修改内容：关闭 trace 输出时仍及时取走 Agent 内部事件队列。
# 修改时间：2026-09-24。
# 修改目的：让运行结束后的摘要仍显示已写出的 Agent4 trace 路径。
# 修改内容：单独保存启用标记而不以关闭后的文件句柄判断。
# 修改时间：2026-09-24。
# 修改目的：保留 V3 的外围视觉审计能力并避免每帧复制巨大业务快照。
# 修改内容：新增诊断前后 YOLO 行和关键事件加半秒采样的紧凑 Agent4 trace。
"""Agent4 视觉与事件日志；真值只由 Runner 调用 record_truth。"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .v3_logging import V3VisualLog


class V4VisualLog(V3VisualLog):
    def record_prediction(self, uid, snapshot):
        if self._predictions is None or snapshot is None:
            return
        key = (str(uid), str(snapshot.frame_id))
        if key in self._seen_predictions:
            return
        self._seen_predictions.add(key)
        row = {
            "schema_version": 1, "kind": "yolo_prediction", "uid": str(uid),
            "frame_id": snapshot.frame_id,
            "source_sim_time": snapshot.source_sim_time,
            "observed_sim_time": snapshot.observed_sim_time,
            "source_time_basis": snapshot.source_time_basis,
            "image_size": snapshot.image_size,
            "source_pose": snapshot.source_pose,
            "inference_wall_ms": snapshot.inference_wall_ms,
            "raw_yolo_objects": [asdict(item) for item in snapshot.raw_yolo_objects],
            "effective_yolo_objects": [asdict(item) for item in snapshot.effective_yolo_objects],
            "vision_diagnostic": dict(snapshot.vision_diagnostic),
            "error": snapshot.error,
        }
        self._write(self._predictions, row)
        self._counts["prediction_rows"] += 1


class V4Trace:
    def __init__(self, output, *, detailed_log=False):
        self.output = Path(output)
        self.enabled = bool(detailed_log)
        self.stream = ((self.output / "agent_v4_trace.jsonl").open("x", encoding="utf-8", buffering=1)
                       if detailed_log else None)
        self.rows = 0
        self.last_sample = {}

    def _write(self, row):
        if self.stream is None:
            return
        self.stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False) + "\n")
        self.rows += 1

    def record(self, uid, agent, now, commands):
        control = agent.control
        if self.stream is None:
            control.pop_events()
            return
        for event in control.pop_events():
            self._write({"kind": "event", **event})
        if now - self.last_sample.get(str(uid), -1e9) < 0.5:
            return
        self.last_sample[str(uid)] = now
        entity = control.entity.current
        self._write({
            "kind": "control_sample", "time": now, "uid": str(uid),
            "state": control.state, "frame_id": control.last_frame_id,
            "entity_id": entity.entity_id if entity else None,
            "entity_visible": entity.visible if entity else False,
            "entity_bbox": entity.bbox_xyxy if entity else None,
            "entity_missing_s": entity.missing_s if entity else None,
            "entity_lost": entity.lost if entity else None,
            "entity_observed_frames": entity.observed_frames if entity else None,
            "motion_decision": control.motion.decision,
            "gimbal_pixel_error_x": control.gimbal.last_error[0],
            "gimbal_pixel_error_y": control.gimbal.last_error[1],
            "gimbal_command_pan": control.gimbal.pan,
            "gimbal_command_tilt": control.gimbal.tilt,
            "gimbal_deadzone_hit": control.gimbal.deadzone_hit,
            "rough_position": control.rough.position,
            "flight_command": [dict(verb=command.verb, params=command.params)
                               for command in commands if command.verb == "set_destination"],
        })

    @property
    def summary(self):
        return {"enabled": self.enabled, "rows": self.rows,
                "path": "agent_v4_trace.jsonl" if self.enabled else None}

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
