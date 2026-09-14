# 修改时间：2026-09-14。
# 修改目的：让完整官方底座上的新队友无需手动启动 Redis 即可运行个人实验。
# 修改内容：复用可达服务或创建本轮本地进程，退出时只关闭自身创建的服务。
"""个人实验使用的 Redis 生命周期，不接管已运行的 Redis。"""
import subprocess
import time
from pathlib import Path

import redis


class RedisRuntime:
    def __init__(self, runtime_root, host, port):
        self.runtime_root = Path(runtime_root)
        self.host, self.port = host, port
        self.process = None
        self.client = redis.Redis(host=host, port=port, socket_timeout=0.3,
                                  socket_connect_timeout=0.3)

    def _ready(self):
        try:
            return bool(self.client.ping())
        except redis.ConnectionError:
            return False
        except redis.TimeoutError:
            return False

    def __enter__(self):
        try:
            if self._ready():
                return self
            if self.host not in ("127.0.0.1", "localhost", "::1"):
                raise RuntimeError(f"远端 Redis 不可连接：{self.host}:{self.port}")
            binary = self.runtime_root / "bin/redis-server.exe"
            if not binary.is_file():
                raise FileNotFoundError(binary)
            self.process = subprocess.Popen(
                [str(binary), "--bind", self.host, "--port", str(self.port),
                 "--save", "", "--appendonly", "no"],
                cwd=self.runtime_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + 10.0
            while self.process.poll() is None and time.monotonic() < deadline:
                if self._ready():
                    print(f"[runtime] 本轮 Redis 已启动：{self.host}:{self.port}", flush=True)
                    return self
                time.sleep(0.1)
            raise RuntimeError(f"本轮 Redis 启动失败：{self.host}:{self.port}")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        self.client.close()
        if self.process is not None and self.process.poll() is None:
            # 不发送全局 shutdown，始终只终止本上下文创建的进程。
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
