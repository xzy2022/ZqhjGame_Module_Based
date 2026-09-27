# v9aget 赛题二提交版

- 入口：`v9aget.py:V9aget`，继承官方 `CoopAgent`。
- 模型：同目录的 `yolo.pt`，PyTorch 双类别模型，类别 `0=real_vehicle`、`1=model_prop`；SHA-256 为 `a30c839d72a6d996eac62b0ab15a147ee03d3c28c0a3853bb25b16c04445cfc2`。未进行 INT8 量化，也不依赖 TensorRT；CUDA 上用 FP16，CPU 上用 FP32。
- 运行环境：Ubuntu 24.04，Python 3.10 及以上；依赖见 `requirements.txt`。CUDA 可用时使用第一块 GPU，否则使用 CPU。PyTorch 安装源应与评测机 CUDA 环境匹配。
- 感知输入：仅用 `obs.self.photo` 的图片字节、本机姿态、队友消息以及 `score_view.sim_time`。图片由官方 SDK 放入观测，Agent 不连接 Redis。
- 视觉诊断：提交版固定为原实验入口的 `000`，原始检测框直接进入控制，不包含 `100`、`101` 等 Runner 真值校正或补框功能。感知异常明确返回空检测，避免官方 Resolver 回退到默认识别器。
- 算法来源：以 `ZqhjGame_Module_Based` 的 `main` 提交 `8d589a5`（V4.5.1）为基础，整合五状态控制、三机搜索和本地运动判定；视觉推理改为提交文件内共享的最新帧工作线程。模型来自 `assets/vehicle_prop_v2/yolo26s_fovmix_blend50.pt`。
- 使用方式：将本文件夹中的文件一起放入赛题二目录，并指定 `v9aget:V9aget`。无需复制开发仓库、配置 JSON、脚本或输出目录。

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
