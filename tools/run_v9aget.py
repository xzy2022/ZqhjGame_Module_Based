# 修改时间：2026-09-28。
# 修改目的：为 v9aget 提供与原版官方 CLI 一致的本地启动参数入口。
# 修改内容：将原版 --photo 开关映射为当前 SDK 的相机模式，并透传场景、种子和天气配置路径。
"""本地启动适配器；提交包仅包含 v9aget.py 等赛方允许的文件。"""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT.parent
SUBMISSION = PROJECT / "submission" / "v9aget_赛题二"


def official_arguments(arguments: list[str]) -> list[str]:
    """保留原版参数，只适配已改名的相机开关。"""
    if any(arg == "--photo-mode" or arg.startswith("--photo-mode=")
           for arg in arguments):
        raise ValueError("此入口使用原版 --photo 开关，请勿传入 --photo-mode")
    enabled = "--photo" in arguments
    forwarded = [arg for arg in arguments if arg != "--photo"]
    return ["run", *forwarded, "--photo-mode", "on" if enabled else "off"]


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if "--help" in arguments or "-h" in arguments:
        print("用法：run_v9aget.py --scenario coop_decoy --agent v9aget:V9aget "
              "[--mode eval] [--photo] [--seed N] [--scenario-json PATH] "
              "[--duration SECONDS] [--output PATH]")
        print("其余参数透传给官方 competition run；天气由场景 JSON 的 weather.type 决定。")
        return 0
    # Agent 仅通过官方模块加载；场景、天气和种子均由 Runner 处理。
    sys.path[:0] = [str(SUBMISSION), str(SDK_ROOT)]
    from competition.sdk.cli import main as official_main

    return official_main(official_arguments(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
