# 修改时间：2026-09-18
# 修改目的：迁移 FrontierPipeline 间接依赖的原始公共辅助模块。
# 修改内容：保留队友交付源码，供检测框工具完成导入闭包。
"""共享路径、原始帧索引与 YOLO 像素坐标转换；不使用地理真值做推理。"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT.parent / "dataset-portable-20260914"
WEATHERS = {
    "Clear_Skies": "晴天", "Partly_Cloudy": "局部多云", "Rain": "雨天",
    "Foggy": "雾天", "Snow_Light": "小雪", "Sand_Dust_Calm": "静风沙尘",
}
CATEGORIES = ("target_only", "decoy_only", "mixed", "no_vehicle")


def configure_runtime():
    """将 Ultralytics 配置与离线实验保存在本项目，禁用外部实验服务回调。"""
    config=ROOT/".ultralytics"
    config.mkdir(parents=True,exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"]=str(config)
    os.environ["YOLO_AUTOINSTALL"]="false"
    from ultralytics.utils import SETTINGS
    changes={key:False for key in ('sync','clearml','comet','dvc','dvclive','hub','mlflow','neptune','raytune','wb','wandb') if key in SETTINGS}
    changes.update(weights_dir=str(ROOT/'weights'),runs_dir=str(ROOT/'runs'),datasets_dir=str(ROOT/'prepared'))
    SETTINGS.update(changes)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_samples(dataset: Path):
    """只沿 frames.csv 查找原图，避免把 previews/crops/sequence_images 重复当训练样本。"""
    dataset = dataset.resolve()
    samples = []
    for index in sorted(dataset.glob("*/*/frames.csv")):
        run = index.parent
        with index.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                row.update(weather=run.parent.name, run=run.name)
                row["key"] = f"{row['weather']}__{row['sample_id']}"
                for key in ("width", "height", "object_count", "sequence_index", "frame_no"):
                    row[key] = int(row[key])
                row["source_t"] = float(row["source_t"])
                row["image"] = str((run / row["image_path"]).resolve())
                row["label"] = str((run / row["detection_label_path"]).resolve())
                row["relative_image"] = (run / row["image_path"]).relative_to(dataset).as_posix()
                samples.append(row)
    if not samples:
        raise FileNotFoundError(f"未找到 */*/frames.csv：{dataset}")
    if len({r["key"] for r in samples}) != len(samples):
        raise ValueError("原始索引存在重复 sample key")
    return samples


def read_boxes(label: Path, width: int, height: int):
    """读单类 YOLO 标签，返回原图坐标 [left, top, right, bottom]。"""
    boxes = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"非法标签 {label}:{line_no}")
        cls, cx, cy, bw, bh = values
        if cls != 0 or bw <= 0 or bh <= 0:
            raise ValueError(f"期望 class=0、正宽高：{label}:{line_no}")
        xyxy = [(cx-bw/2)*width, (cy-bh/2)*height,
                (cx+bw/2)*width, (cy+bh/2)*height]
        if xyxy[0] < -1e-4 or xyxy[1] < -1e-4 or xyxy[2] > width+1e-4 or xyxy[3] > height+1e-4:
            raise ValueError(f"框越界：{label}:{line_no}: {xyxy}")
        boxes.append([max(0.,xyxy[0]), max(0.,xyxy[1]),
                      min(float(width),xyxy[2]), min(float(height),xyxy[3])])
    return boxes
