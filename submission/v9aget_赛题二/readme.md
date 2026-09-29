# v9aget 赛题二提交版

- 入口：`v9aget.py:V9aget`，继承官方 `CoopAgent`。
- 模型：同目录的 `yolo.pt`，PyTorch 双类别模型，类别 `0=real_vehicle`、`1=model_prop`；SHA-256 为 `35b7edb741eb8b05e0762cec6f61bdc32c302202c421c78f81f0ee04b2395895`。CUDA 上把模型和输入转为 FP16（广义上的低精度量化），CPU 上用 FP32；没有做 INT8/INT4 量化，也没有使用 TensorRT engine。
- 运行环境：Ubuntu 24.04，Python 3.10 及以上；依赖见 `requirements.txt`。CUDA 可用时使用第一块 GPU，否则使用 CPU。PyTorch 安装源应与评测机 CUDA 环境匹配。
- 感知输入：仅用 `obs.self.photo` 的图片字节、本机姿态、队友消息以及 `score_view.sim_time`。图片由官方 SDK 放入观测，Agent 不连接 Redis。
- 视觉诊断：提交版固定为原实验入口的 `000`，原始检测框直接进入控制，不包含 `100`、`101` 等 Runner 真值校正或补框功能。感知异常明确返回空检测，避免官方 Resolver 回退到默认识别器。
- 算法：整合五状态控制、三机搜索和本地运动判定；视觉推理由提交文件内共享的最新帧工作线程完成。
- 使用方式：将本文件夹中的文件一起放入赛题二目录，并指定 `v9aget:V9aget`。无需复制开发仓库、配置 JSON、脚本或输出目录。

## 环境构建与核对

赛方提供官方 `competition` SDK。Ubuntu 24.04 上可在提交文件夹**外**建立 Python 3.12 虚拟环境；进入本文件夹后执行：

```bash
python3.12 -m venv ../v9aget-venv
../v9aget-venv/bin/python -m pip install -r requirements.txt
../v9aget-venv/bin/python -c 'import torch, torchvision, cv2, ultralytics; print(torch.__version__, torchvision.__version__, cv2.__version__, ultralytics.__version__, torch.cuda.is_available())'
```

GPU 评测机应安装与其驱动和 CUDA 运行环境匹配的 PyTorch wheel。`requirements.txt` 固定 PyTorch/torchvision 的公开版本号，不指定 CUDA wheel 来源，也不安装 TensorRT。尚无 Ubuntu 24.04 集群安装或运行日志。

本包使用 PyTorch `.pt` 和运行时 FP16 路径；没有包含 TensorRT engine。

官方原版 CLI 启动示例（执行后会启动仿真）：

```bash
python -m competition run \
  --scenario coop_decoy \
  --agent v9aget:V9aget \
  --mode eval \
  --photo \
  --seed 0 \
  --duration 600 \
  --output output/v9aget
```

v9 需要照片输入，并在 `sensor()` 内自行加载同目录的 `yolo.pt`；`--mode eval` 由官方 Runner 处理。若使用新版 SDK，请将示例中的 `--photo` 改为 `--photo-mode on`。`--seed 0` 表示真目标随机选路；正整数种子控制真目标选路，诱饵仍独立随机。天气由 Runner 从场景 JSON 的 `weather.type` 读取；要更换天气，由评测方通过 `--scenario-json` 指定场景文件，Agent 不读取场景 JSON。

提交到 Linux 集群时由评测方安装依赖并加载同目录中的 `v9aget:V9aget`。

本版只完成静态与接口检查，尚未进行仿真测评；最终成绩和 Linux 集群运行情况待验证。
