# 修改时间：2026-09-14
# 修改目的：让批处理父子进程分别保留实际导入来源证据。
# 修改内容：支持审计文件路径中的字面量 {pid} 替换为当前进程编号。
# 修改时间：2026-09-14
# 修改目的：从独立仓库调用指定官方 SDK 并分离二进制运行目录。
# 修改内容：设置模块搜索路径、继承环境及引擎位置后运行个人模块。
"""run.cmd [--sim-root SDK目录] [--runtime-root 发行目录] 模块 [模块参数]。"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", default=os.environ.get("HF2026_SIM_ROOT", str(project.parent)))
    parser.add_argument("--runtime-root", default=os.environ.get("HF2026_RUNTIME_ROOT"))
    parser.add_argument("module", help="个人模块短名，如 control_test_runner / visual_shadow_study")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sdk = Path(args.sim_root).resolve()
    runtime = Path(args.runtime_root).resolve() if args.runtime_root else sdk
    if not (sdk / "competition/sdk/core/runner.py").is_file():
        parser.error(f"未找到官方 SDK：{sdk}，请用 --sim-root 指定官方发行目录")
    os.environ["HF2026_SIM_ROOT"] = str(sdk)
    os.environ["HF2026_RUNTIME_ROOT"] = str(runtime)
    os.environ["OPENSIM_SIM_BIN"] = str(runtime / "opensim-sim.exe")
    # 官方嵌入式 Python 可忽略 PYTHONPATH，因此本进程和每个子进程均显式引导。
    paths = [str(project / "src"), str(sdk), str(sdk / "competition")]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    module = args.module if args.module.startswith("personal_hf2026.") else f"personal_hf2026.{args.module}"
    sys.argv = [module, *args.args]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        audit = os.environ.get("HF2026_IMPORT_AUDIT")
        if audit:
            destination = Path(audit.replace("{pid}", str(os.getpid()))).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            modules = {name: str(Path(item.__file__).resolve())
                       for name, item in sys.modules.copy().items()
                       if name.startswith(("competition.", "personal_hf2026."))
                       and getattr(item, "__file__", None)}
            destination.write_text(json.dumps({"project_root": str(project),
                "sim_root": str(sdk), "runtime_root": str(runtime), "module": module,
                "modules": modules}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
