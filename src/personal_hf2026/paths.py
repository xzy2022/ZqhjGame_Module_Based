# 修改时间：2026-09-14（迁移收尾）
# 修改目的：使内外层工作目录启动时使用同一外置输出目录。
# 修改内容：将默认输出根定位到官方运行底座的同级 output。
# 修改时间：2026-09-14
# 修改目的：明确独立算法仓库、官方 SDK 和运行底座的边界。
# 修改内容：集中提供环境可覆盖的根目录及场景和输出目录。
"""统一路径；命令行入口在导入算法前设置环境变量。"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = Path(os.environ.get("HF2026_SIM_ROOT", PROJECT_ROOT.parent)).resolve()
RUNTIME_ROOT = Path(os.environ.get("HF2026_RUNTIME_ROOT", SIM_ROOT)).resolve()
OUTPUT_ROOT = Path(os.environ.get("HF2026_OUTPUT_ROOT", RUNTIME_ROOT.parent / "output")).resolve()
SCENARIO_ROOT = PROJECT_ROOT / "configs/scenarios/coop_decoy"

