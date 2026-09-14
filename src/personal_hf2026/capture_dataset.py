# 修改时间：2026-09-14。
# 修改目的：将个人算法迁移为独立 Git 仓库中的可安装模块。
# 修改内容：复制现有算法并调整包导入及模型路径，保持算法逻辑不变。
# 修改时间：2026-09-13（采集后核验）
# 修改目的：明确标记启动阶段照片源时间领先观测快照的记录。
# 修改内容：在离线索引标记帧龄状态，并记录索引生成源码哈希。
# 修改时间：2026-09-13
# 修改目的：按需保存仿真原图，并明确区分原始记录、时间插值和未标注字段。
# 修改内容：实现唯一帧落盘及按源时钟关联观测和裁判参考的离线索引。
"""采集辅助模块；不会把裁判身份、投影框或高程传给 Agent。"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
from pathlib import Path

from PIL import Image

from .visual_geometry import aligned_sample


POSE_KEYS = ("lat", "lon", "alt", "heading_deg", "gimbal_pan", "gimbal_tilt", "gimbal_fov_deg")


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class DatasetRecorder:
    """由相机线程调用；帧龄、模型结果和 Agent 是否消费均不影响保存。"""
    def __init__(self, output):
        self.output = Path(output)
        self.root = self.output / "dataset"
        self.root.mkdir(exist_ok=True)
        self.log = (self.root / "frames.jsonl").open("w", encoding="utf-8", buffering=1)
        self.seen = set()

    def record(self, uid, raw):
        key = (uid, raw["frame_no"], raw["source_sim_time"])
        if key in self.seen:
            return
        data = raw["image"]
        with Image.open(BytesIO(data)) as image:
            width, height, fmt = image.width, image.height, image.format
        directory = self.root / "images" / uid
        directory.mkdir(parents=True, exist_ok=True)
        suffix = ".jpg" if fmt == "JPEG" else ".png"
        # 加入源时间，避免帧号重启时覆盖；不同新帧即使图像相同也保留。
        path = directory / f'{raw["frame_no"]}_{round(raw["source_sim_time"] * 1000000)}{suffix}'
        path.write_bytes(data)
        row = dict(uid=uid, frame_no=raw["frame_no"], source_sim_time=raw["source_sim_time"],
                   received_unix_s=raw["received_unix_s"], image_path=path.relative_to(self.output).as_posix(),
                   image_sha256=hashlib.sha256(data).hexdigest(), image_bytes=len(data),
                   width=width, height=height, image_format=fmt,
                   exposure_time_verified=False, visibility=None, visibility_annotation="unannotated",
                   ue_projected_objects=[dict(target_id=b.get("target_id"), target_type=b.get("class"),
                       ue_projected_bbox=b.get("bbox"), visibility=None,
                       visibility_annotation="unannotated") for b in raw["audit_boxes"]])
        self.log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.seen.add(key)

    def close(self):
        self.log.close()


def observation(row, line):
    """保留两侧真实检测；不插值类别，也不声称候选一定在图片内。"""
    return dict(line=line, sim_time=row["sim_time"], agent_t=row["t"],
                observed_unix_s=row["observed_unix_s"], state=row["state"], own=row["self"],
                candidate=row["candidate"], local_track=row["local_track"],
                search_filter=row["search_filter"])


def bracket(history, times, source):
    i = bisect_left(times, source)
    # UE 的时间字段可能只保留六位小数。
    for index in (i, i - 1):
        if 0 <= index < len(times) and abs(times[index] - source) <= 1e-5:
            return history[index], history[index]
    return (history[i - 1] if i > 0 else None,
            history[i] if i < len(history) else None)


def build_index(output):
    """仿真结束后执行；以世界源时钟匹配，避免首拍 Agent 计时偏移。"""
    output = Path(output)
    histories = defaultdict(list)
    for line, row in enumerate(rows(output / "observations.jsonl"), 1):
        if row["sim_time"] is not None:
            histories[row["uid"]].append(observation(row, line))
    for history in histories.values():
        history.sort(key=lambda row: row["sim_time"])
    times = {uid: [r["sim_time"] for r in h] for uid, h in histories.items()}
    judges = list(rows(output / "judge.jsonl"))
    judge_times = [r["sim_time"] for r in judges]
    offset = judges[0]["sim_time"] - judges[0]["t"] if judges else None
    deliveries = {(r["uid"], r["frame_no"], r["source_sim_time"]): r
                  for r in rows(output / "dataset/deliveries.jsonl")}
    counts, by_uid, alignments, dimensions = Counter(), Counter(), Counter(), Counter()
    ranges = {}
    with (output / "dataset/samples.jsonl").open("w", encoding="utf-8") as stream:
        for frame in rows(output / "dataset/frames.jsonl"):
            uid, source = frame["uid"], frame["source_sim_time"]
            before, after = bracket(histories[uid], times.get(uid, []), source)
            sample, alignment = None, "time_not_bracketed"
            if before is not None and after is not None:
                endpoints = [before] if before is after else [before, after]
                sample, alignment = aligned_sample([dict(t=r["sim_time"],
                    own={k: r["own"][k] for k in POSE_KEYS}, tracks={}) for r in endpoints], source)
            reference = None
            if judges and judge_times[0] - 1e-5 <= source <= judge_times[-1] + 1e-5:
                i = min(bisect_left(judge_times, source), len(judges) - 1)
                i = min((i, max(0, i - 1)), key=lambda n: abs(judge_times[n] - source))
                judge = judges[i]
                reference = dict(line=i + 1, sim_time=judge["sim_time"],
                    time_delta_s=judge["sim_time"] - source, label_source="judge_offline_only",
                    objects=[dict(target_id=key, target_type=kind, **value,
                        destroyed=judge["targets"].get(key, {}).get("destroyed"))
                        for collection, kind in (("world_targets", "TargetVehicle"), ("world_decoys", "DecoyVehicle"))
                        for key, value in judge[collection].items()])
            delivery = deliveries.get((uid, frame["frame_no"], source))
            if delivery:
                age = delivery["frame_age_s"]
                delivery = dict(delivery, time_status=("source_ahead_of_observation" if age < 0
                    else "stale" if age > 1.5 else "within_age_window"))
            row = dict(frame, source_t=source - offset if offset is not None else None,
                       alignment=alignment, source_pose=sample["own"] if sample else None,
                       pose_gap_s=after["sim_time"] - before["sim_time"] if before and after else None,
                       observation_before=before, observation_after=after,
                       first_agent_delivery=delivery, reference=reference)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            counts["frames"] += 1
            counts["with_source_pose"] += sample is not None
            counts["with_reference"] += reference is not None
            counts["delivered_to_agent"] += delivery is not None
            counts["stale_at_first_delivery"] += bool(delivery and delivery["frame_age_s"] > 1.5)
            counts["source_ahead_at_first_delivery"] += bool(delivery and delivery["frame_age_s"] < 0)
            counts["with_ue_projected_objects"] += bool(frame["ue_projected_objects"])
            counts["ue_projected_objects"] += len(frame["ue_projected_objects"])
            by_uid[uid] += 1
            alignments[alignment] += 1
            dimensions[f'{frame["width"]}x{frame["height"]}'] += 1
            if row["source_t"] is not None:
                span = ranges.setdefault(uid, [row["source_t"], row["source_t"]])
                span[0], span[1] = min(span[0], row["source_t"]), max(span[1], row["source_t"])
    result = dict(counts=counts, frames_by_uid=by_uid, alignments=alignments,
                  dimensions=dimensions, source_t_ranges=ranges,
                  recorded_duration_s=judges[-1]["t"] if judges else None,
                  visibility_annotation="unannotated", exposure_time_verified=False,
                  index_builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="unique_frames_received_by_cache_not_every_rendered_frame")
    (output / "dataset/summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="仅从已有采集日志重建索引，不运行仿真或视觉模型")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_index(args.output), ensure_ascii=False, indent=2))
