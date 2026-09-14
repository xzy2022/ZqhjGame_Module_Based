# 修改时间：2026-09-14（Redis 端口一致性修复）
# 修改目的：让引擎和个人控制器使用命令行指定的同一 Redis 地址。
# 修改内容：仅在本轮待写出的场景副本同步 Redis host 和 port。
# 修改时间：2026-09-14（迁移收尾）
# 修改目的：使个人实验在回退后的官方底座上保留正常结束和可靠引擎就绪。
# 修改内容：在个人 Runner 补齐结束接口、引擎就绪与自有 Redis 和可选网页生命周期。
# 修改时间：2026-09-14
# 修改目的：在基准官方 SDK 上保留采集停更时的正常中止与资源释放。
# 修改内容：仅给本仓库 Runner 子类的空转分支增加检查，不修改官方源码或类。
"""兼容旧版运行底座；评分、观测和控制节拍仍沿用所选官方 SDK。"""
import ast
from contextlib import nullcontext
import inspect
import os
from pathlib import Path
import textwrap
import time
import types

from competition.sdk._vendored import sim_runner
from competition.sdk.core.runner import RunnerBase
from competition.sdk.scenarios.coop_decoy.runner import CoopDecoyRunner
from .paths import RUNTIME_ROOT
from .redis_runtime import RedisRuntime


def _idle_compatible_run():
    if hasattr(RunnerBase, "should_finish_when_idle") and hasattr(RunnerBase, "should_finish"):
        return RunnerBase.run
    tree = ast.parse(textwrap.dedent(inspect.getsource(RunnerBase.run)))
    branches = [node for node in ast.walk(tree)
                if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "should_idle"]
    if len(branches) != 1:
        raise RuntimeError("所选官方 SDK 的空转分支不符合基准接口，请使用指定版本")
    # 插入正常 break，继续执行官方 evaluation 和 finally，不用异常中断代替收尾。
    if not hasattr(RunnerBase, "should_finish_when_idle"):
        branches[0].body[:0] = ast.parse("if self.should_finish_when_idle(agents):\n    break").body
    if not hasattr(RunnerBase, "should_finish"):
        # 在控制循环保存本拍状态后插入结束检查，保留官方已有 deadline 节拍。
        positions = [(node, index) for node in ast.walk(tree) if isinstance(node, ast.While)
                     for index, statement in enumerate(node.body)
                     if isinstance(statement, ast.Assign) and ast.unparse(statement) == "last = ws"]
        if len(positions) != 1:
            raise RuntimeError("所选官方 SDK 的控制循环不符合基准接口")
        loop, index = positions[0]
        loop.body[index + 1:index + 1] = ast.parse("if self.should_finish(agents):\n    break").body
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, "<personal_hf2026.sdk_compat>", "exec"),
         RunnerBase.run.__globals__, namespace)
    return namespace["run"]


def start_sim(sim_binary, scenario, **kwargs):
    # 新版底座已按启动前订阅数判断就绪，直接使用官方实现。
    if hasattr(sim_runner, "_get_subscriber_count"):
        return sim_runner.start_sim(sim_binary, scenario, **kwargs)
    import redis
    client = redis.Redis(host=kwargs.get("redis_host", "127.0.0.1"),
                         port=kwargs.get("redis_port", 6379), socket_timeout=1.0)
    channel = kwargs.get("ready_channel", "sim:commands")
    timeout = kwargs.get("ready_timeout", sim_runner.DEFAULT_READY_TIMEOUT)
    log = kwargs.get("log") or print
    proc = None
    try:
        baseline = int(client.execute_command("PUBSUB", "NUMSUB", channel)[-1])
        # 保留官方进程创建与日志，只跳过旧版会误认已有订阅者的等待。
        proc = sim_runner.start_sim(sim_binary, scenario, **dict(kwargs, ready_timeout=0))
        if proc is None or timeout <= 0:
            return proc
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and proc.poll() is None:
            count = int(client.execute_command("PUBSUB", "NUMSUB", channel)[-1])
            if count > baseline:
                log(f"[run] opensim-sim ready ({channel} subscribers {baseline} -> {count})")
                return proc
            time.sleep(0.2)
        raise RuntimeError("引擎未在就绪期限内新增命令订阅，请检查引擎日志")
    except BaseException:
        sim_runner.stop_sim(proc)
        raise
    finally:
        client.close()


def _compatible_start_engine():
    original = RunnerBase._start_engine
    # 为个人子类复制函数绑定，不改官方模块全局或共享 Runner 类。
    return types.FunctionType(original.__code__, dict(original.__globals__, start_sim=start_sim),
                              original.__name__, original.__defaults__, original.__closure__)


class IdleCompatibleCoopDecoyRunner(CoopDecoyRunner):
    _run_with_idle_check = _idle_compatible_run()
    _start_engine = _compatible_start_engine()

    def __init__(self, cfg, agent_cls, log=print):
        # 在切换二进制工作目录前固定用户提供的相对路径。
        cfg.scenario_path = str(Path(cfg.scenario_path).resolve())
        cfg.output_dir = str(Path(cfg.output_dir).resolve())
        if cfg.sim_binary:
            cfg.sim_binary = str(Path(cfg.sim_binary).resolve())
        super().__init__(cfg, agent_cls, log=log)

    def prepare_scenario(self):
        super().prepare_scenario()
        # 官方启动流程随后将此内存副本写入 output，不改输入场景文件。
        simulation = self._scenario_cfg.setdefault("simulation", {})
        simulation.update(redis_host=self.cfg.redis_host, redis_port=self.cfg.redis_port)

    def run(self):
        previous = Path.cwd()
        runtime = Path(getattr(self, "runtime_root", RUNTIME_ROOT)).resolve()
        try:
            # 引擎及其原生资源使用发行目录；算法和 SDK 已通过绝对路径导入。
            os.chdir(runtime)
            redis_context = (nullcontext() if self.cfg.dry_run else
                             RedisRuntime(runtime, self.cfg.redis_host, self.cfg.redis_port))
            with redis_context:
                with getattr(self, "visualization", nullcontext()):
                    return self._run_with_idle_check()
        finally:
            os.chdir(previous)

    def should_finish_when_idle(self, agents):
        return False

    def should_finish(self, agents):
        return False
