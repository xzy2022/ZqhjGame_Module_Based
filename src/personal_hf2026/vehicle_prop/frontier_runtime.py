# 修改时间：2026-09-18
# 修改目的：让队友交付的固定检测流水线复用本项目指定虚拟环境。
# 修改内容：固定 Ultralytics 版本并将运行缓存和生成目录约束到外置 output。
"""固定 YOLO26 运行时，避免静默切换到不兼容的 Ultralytics 版本。"""

from __future__ import annotations

import os

from personal_hf2026.paths import OUTPUT_ROOT, PROJECT_ROOT

VERSION = "8.4.154"
CONFIG_PATH = PROJECT_ROOT / "configs/detectors/vehicle_prop/vehicle_frontier.json"


def activate():
    """激活并核对迁移算法验证过的 Ultralytics 运行时。"""
    runtime_root = OUTPUT_ROOT / "vehicle_prop_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_root / "ultralytics_config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_VERBOSE"] = "false"

    import ultralytics

    if ultralytics.__version__ != VERSION:
        raise RuntimeError(
            f"FrontierPipeline 需要 ultralytics=={VERSION}，当前为 {ultralytics.__version__}"
        )
    from ultralytics.utils import SETTINGS

    changes = {
        key: False
        for key in (
            "sync",
            "clearml",
            "comet",
            "dvc",
            "dvclive",
            "hub",
            "mlflow",
            "neptune",
            "raytune",
            "wb",
            "wandb",
        )
        if key in SETTINGS
    }
    changes.update(
        weights_dir=str(PROJECT_ROOT / "assets/vehicle_prop"),
        runs_dir=str(runtime_root / "runs"),
        datasets_dir=str(runtime_root / "datasets"),
    )
    SETTINGS.update(changes)
    return ultralytics
