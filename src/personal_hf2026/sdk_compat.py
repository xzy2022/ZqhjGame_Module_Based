# 修改时间：2026-09-14
# 修改目的：在基准官方 SDK 上保留采集停更时的正常中止与资源释放。
# 修改内容：仅给本仓库 Runner 子类的空转分支增加检查，不修改官方源码或类。
"""兼容 6fc47ca 的空转循环；评分、控制循环和收尾沿用所选 SDK。"""
import ast
import inspect
import os
from pathlib import Path
import textwrap

from competition.sdk.core.runner import RunnerBase
from competition.sdk.scenarios.coop_decoy.runner import CoopDecoyRunner
from .paths import RUNTIME_ROOT


def _idle_compatible_run():
    if hasattr(RunnerBase, "should_finish_when_idle"):
        return RunnerBase.run
    tree = ast.parse(textwrap.dedent(inspect.getsource(RunnerBase.run)))
    branches = [node for node in ast.walk(tree)
                if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "should_idle"]
    if len(branches) != 1:
        raise RuntimeError("所选官方 SDK 的空转分支不符合基准接口，请使用指定版本")
    # 插入正常 break，继续执行官方 evaluation 和 finally，不用异常中断代替收尾。
    branches[0].body[:0] = ast.parse("if self.should_finish_when_idle(agents):\n    break").body
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, "<personal_hf2026.sdk_compat>", "exec"),
         RunnerBase.run.__globals__, namespace)
    return namespace["run"]


class IdleCompatibleCoopDecoyRunner(CoopDecoyRunner):
    _run_with_idle_check = _idle_compatible_run()

    def __init__(self, cfg, agent_cls, log=print):
        # 在切换二进制工作目录前固定用户提供的相对路径。
        cfg.scenario_path = str(Path(cfg.scenario_path).resolve())
        cfg.output_dir = str(Path(cfg.output_dir).resolve())
        if cfg.sim_binary:
            cfg.sim_binary = str(Path(cfg.sim_binary).resolve())
        super().__init__(cfg, agent_cls, log=log)

    def run(self):
        previous = Path.cwd()
        runtime = Path(getattr(self, "runtime_root", RUNTIME_ROOT)).resolve()
        try:
            # 引擎及其原生资源使用发行目录；算法和 SDK 已通过绝对路径导入。
            os.chdir(runtime)
            return self._run_with_idle_check()
        finally:
            os.chdir(previous)

    def should_finish_when_idle(self, agents):
        return False
