# v9aget 赛题二提交版

- 入口：`v9aget.py:V9aget`，继承官方 `CoopAgent`。
- 模型：同目录的 `yolo.pt`，PyTorch 双类别模型，类别 `0=real_vehicle`、`1=model_prop`；SHA-256 为 `a30c839d72a6d996eac62b0ab15a147ee03d3c28c0a3853bb25b16c04445cfc2`。CUDA 上把模型和输入转为 FP16（广义上的低精度量化），CPU 上用 FP32；没有做 INT8/INT4 量化，也没有使用 TensorRT engine。
- 运行环境：Ubuntu 24.04，Python 3.10 及以上；依赖见 `requirements.txt`。CUDA 可用时使用第一块 GPU，否则使用 CPU。PyTorch 安装源应与评测机 CUDA 环境匹配。
- 感知输入：仅用 `obs.self.photo` 的图片字节、本机姿态、队友消息以及 `score_view.sim_time`。图片由官方 SDK 放入观测，Agent 不连接 Redis。
- 视觉诊断：提交版固定为原实验入口的 `000`，原始检测框直接进入控制，不包含 `100`、`101` 等 Runner 真值校正或补框功能。感知异常明确返回空检测，避免官方 Resolver 回退到默认识别器。
- 算法来源：以 `ZqhjGame_Module_Based` 的 `main` 提交 `8d589a5`（V4.5.1）为基础，整合五状态控制、三机搜索和本地运动判定；视觉推理改为提交文件内共享的最新帧工作线程。模型来自 `assets/vehicle_prop_v2/yolo26s_fovmix_blend50.pt`。
- 使用方式：将本文件夹中的文件一起放入赛题二目录，并指定 `v9aget:V9aget`。无需复制开发仓库、配置 JSON、脚本或输出目录。

## 环境构建与核对

赛方提供官方 `competition` SDK。Ubuntu 24.04 上可在提交文件夹**外**建立 Python 3.12 虚拟环境；进入本文件夹后执行：

```bash
python3.12 -m venv ../v9aget-venv
../v9aget-venv/bin/python -m pip install -r requirements.txt
../v9aget-venv/bin/python -c 'import torch, torchvision, cv2, ultralytics; print(torch.__version__, torchvision.__version__, cv2.__version__, ultralytics.__version__, torch.cuda.is_available())'
```

GPU 评测机应安装与其驱动和 CUDA 运行环境匹配的 PyTorch wheel。`requirements.txt` 固定 PyTorch/torchvision 的公开版本号，不指定 CUDA wheel 来源，也不安装 TensorRT。本地已核对的 Windows 环境为 Python 3.12.13、`torch 2.9.1+cu128`、`torchvision 0.24.1+cu128`、`opencv 4.13.0`、`numpy 2.1.3`、`ultralytics 8.4.154`，显卡为 RTX 5060 Laptop GPU。这是本地环境记录；尚无 Ubuntu 24.04 集群安装或运行日志。

V4 默认配置本来使用 PyTorch `.pt`。曾有开发用的 TensorRT FP16 后端，但 engine 加载代码会核对 TensorRT 版本、GPU 型号和计算能力；目前不知道评测集群的对应信息，因此本包保留可加载的 `.pt` 和运行时 FP16 路径。

本地启动命令（从 `hf2026-sim-windows` 官方根目录执行，**执行后会真的启动仿真**）：

```powershell
Set-Location 'D:\Workspace\00_MyRepo\red_m_competiton\hf2026-sim-windows'
$env:PYTHONPATH = (Resolve-Path '..\codex_worktrees\v9aget-submission\submission\v9aget_赛题二').Path
$pythonExe = (Resolve-Path '.\ZqhjGame_Module_Based\.venv-vehicleprop-integration\Scripts\python.exe').Path
& $pythonExe -B -X utf8 -m competition run `
  --scenario coop_decoy `
  --agent 'v9aget:V9aget' `
  --mode eval `
  --photo-mode on `
  --duration 600 `
  --output "..\output\v9aget\run_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
```

`ZqhjGame_Module_Based\run.cmd` 是开发仓库内部任务启动器，不直接加载这个提交入口。提交到 Linux 集群时，由评测方在同目录提供 `yolo.pt`、安装依赖并加载 `v9aget:V9aget`。

本版只完成静态与接口检查，尚未进行仿真测评；最终成绩和 Linux 集群运行情况待验证。
