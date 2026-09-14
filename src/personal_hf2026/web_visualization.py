# 修改时间：2026-09-14
# 修改目的：让命令行实验可选择打开官方网页实时观测界面。
# 修改内容：管理独立只读网页服务和空闲端口，并在实验结束时释放自有进程。
"""复用官方发行包前端；不启动仿真、Redis 或 UE。"""
import json
import os
from pathlib import Path
import subprocess
import time
import webbrowser

from .paths import PROJECT_ROOT, RUNTIME_ROOT


class WebVisualization:
    def __init__(self, redis_host, redis_port, output_dir, port=0):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.output_dir = Path(output_dir).resolve()
        self.port = port
        self.process = None
        self.log_file = None

    def __enter__(self):
        runtime = RUNTIME_ROOT.resolve()
        required = [runtime / "bin/node.exe", runtime / "frontend/index.html",
                    runtime / "visualization/dist-bridge/bridge/server.js"]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"网页可视化所需的官方发行资源不存在：{path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ready_path = self.output_dir / "visualization.json"
        # 清除本次就绪标记，不能把同输出目录旧服务的信息误当成本次结果。
        ready_path.unlink(missing_ok=True)
        self.log_file = (self.output_dir / "visualization.log").open("w", encoding="utf-8")
        env = dict(os.environ, NODE_PATH=str(runtime / "lib/node_modules"))
        try:
            self.process = subprocess.Popen(
                [str(required[0]), str(PROJECT_ROOT / "tools/web_observer.cjs"),
                 str(runtime), self.redis_host, str(self.redis_port), str(self.port),
                 str(ready_path)], cwd=runtime, env=env, stdin=subprocess.PIPE,
                stdout=self.log_file, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + 20
            while not ready_path.is_file():
                if self.process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"网页服务未能启动，请查看 {self.output_dir / 'visualization.log'}")
                time.sleep(0.1)
            info = json.loads(ready_path.read_text(encoding="utf-8"))
            print(f"[visualize] 只读实时观测：{info['url']}（实验结束后自动关闭服务）")
            print("[visualize] 地图跟随当前 CLI 仿真；相机仅显示已运行 UE 提供的图像。")
            webbrowser.open(info["url"])
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(b"stop\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=8)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.kill()
                    self.process.wait(timeout=5)
            try:
                self.process.stdin.close()
            except OSError:
                pass
        if self.log_file is not None:
            self.log_file.close()
