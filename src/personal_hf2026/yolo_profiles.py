# 修改时间：2026-09-19。
# 修改目的：明确区分旧模型三档与新 V2，并统一在线和离线的模型资源身份。
# 修改内容：提供大小写敏感的兼容别名、独立 V2 配置和权重解析及实际配置与源码哈希。
"""无需导入 GPU 库即可解析 YOLO 评估档位。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .paths import PROJECT_ROOT


YOLO_PROFILES = ("V1-v1", "V1-v2", "V1-v3", "V2")
PROFILE_ALIASES = {"v1": "V1-v1", "v2": "V1-v2", "v3": "V1-v3"}
DEFAULT_CONFIG = PROJECT_ROOT / "configs/detectors/vehicle_prop/vehicle_frontier.json"
DEFAULT_V2_CONFIG = PROJECT_ROOT / "configs/detectors/vehicle_prop_v2/vehicle_realtime.json"
DEFAULT_TRT_ENGINE = Path(
    "D:/Workspace/00_MyRepo/red_m_competiton/output/personal_v2/"
    "yolo-offline-accel/trt-fp16-build-20260918-214857-153/yolo26s_two_class.engine"
)
DEFAULT_V2_ENGINE = Path(
    "D:/Workspace/00_MyRepo/red_m_competiton/output/personal_v2/"
    "yolo-v2-online-20260919/engine/yolo26s_fovmix_blend50_fp16.engine"
)


def canonical_profile(value):
    """小写 v2 永远表示旧模型无翻转档；只有大写 V2 表示新模型。"""
    canonical = PROFILE_ALIASES.get(value, value)
    if canonical not in YOLO_PROFILES:
        raise ValueError(f"未知 YOLO 档位：{value}；支持 {', '.join(YOLO_PROFILES)} 和旧别名 v1/v2/v3")
    return canonical


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_resources(profile, config=None, trt_engine=None, *, v2_config=None,
                      v2_engine=None, weights=None, tracker_high=None, image_rotation_deg=0):
    """冻结本次推理实际使用的配置、权重和算法源码；不创建检测器。"""
    profile = canonical_profile(profile)
    config = Path(
        (v2_config or DEFAULT_V2_CONFIG) if profile == "V2" else (config or DEFAULT_CONFIG)
    ).resolve()
    settings = json.loads(config.read_text(encoding="utf-8"))
    configured_weights = (PROJECT_ROOT / settings["weights"]).resolve()
    if weights is not None:
        selected_weights = Path(weights).resolve()
    elif profile == "V2":
        selected_weights = Path(v2_engine or DEFAULT_V2_ENGINE).resolve()
    elif profile == "V1-v3":
        selected_weights = Path(trt_engine or DEFAULT_TRT_ENGINE).resolve()
    else:
        selected_weights = configured_weights
    if profile == "V1-v3" and selected_weights.suffix.lower() != ".engine":
        raise ValueError("V1-v3 必须使用独立的旧模型 .engine 权重")
    if profile == "V1-v3" and settings["detector"]["imgsz"] != 2048:
        raise ValueError("V1-v3 保持旧模型 2048 输入配置")
    if profile == "V2" and list(settings["detector"]["shape"]) != [1152, 1536]:
        raise ValueError("V2 固定输入配置必须为 [1152, 1536]（高、宽）")
    weights_sha256 = file_sha256(selected_weights)
    expected = settings.get("sha256") or settings["detector"].get("expected_sha256")
    if selected_weights == configured_weights and expected and weights_sha256 != expected:
        raise ValueError("权重 SHA256 与所选配置不一致")
    multiplier = 1.2 if profile == "V1-v3" else 1.0
    high = float(tracker_high) if tracker_high is not None else float(settings["tracker"]["high"]) * multiplier
    flip = profile == "V1-v1"
    effective_config = json.loads(json.dumps(settings))
    effective_config["weights"] = str(selected_weights)
    effective_config["sha256"] = weights_sha256
    effective_config["tracker"]["high"] = high
    if profile == "V2":
        effective_config["detector"]["expected_sha256"] = weights_sha256
        effective_config["backend"] = "tensorrt" if selected_weights.suffix.lower() == ".engine" else "pytorch"
    else:
        effective_config["detector"]["flip"] = flip
        effective_config["views"] = "flip" if flip else "original"
    package = "vehicle_prop_v2" if profile == "V2" else "vehicle_prop"
    source_files = [Path(__file__), *(PROJECT_ROOT / "src/personal_hf2026" / package).glob("*.py")]
    if profile == "V2":
        source_files.append(PROJECT_ROOT / "src/personal_hf2026/vehicle_prop/__init__.py")
    return {
        "profile": profile,
        "config_path": str(config),
        "config_sha256": file_sha256(config),
        "weights_path": str(selected_weights),
        "weights_sha256": weights_sha256,
        "configured_weights_path": str(configured_weights),
        "configured_weights_sha256": expected,
        "model_format": "tensorrt_engine" if selected_weights.suffix.lower() == ".engine" else "pytorch_pt",
        "effective_config": effective_config,
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in sorted(source_files)},
        "effective_options": {
            "flip_enabled": flip,
            "tracker_high_confidence_threshold": high,
            "tracker_low_confidence_threshold": float(settings["tracker"]["low"]),
            "unknown_class_confidence_threshold": float(settings["unknown_threshold"]),
            "tracker_high_multiplier": multiplier,
            "image_rotation_deg": image_rotation_deg,
            "input_shape_hw": settings["detector"].get("shape"),
            "imgsz": settings["detector"].get("imgsz"),
        },
    }


def detector_kwargs(resources):
    """把冻结资源转换为两版检测器共同支持的参数。"""
    options = resources["effective_options"]
    return {
        "config": resources["config_path"],
        "profile": resources["profile"],
        "weights": resources["weights_path"],
        "weights_sha256": resources["weights_sha256"],
        "flip": options["flip_enabled"],
        "tracker_high": options["tracker_high_confidence_threshold"],
        "image_rotation_deg": options.get("image_rotation_deg", 0),
    }
