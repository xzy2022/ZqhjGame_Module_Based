# v9aget 赛题二提交版

- 入口：`v9aget.py:V9aget`，继承官方 `CoopAgent`。
- 模型：同目录的 `yolo.pt`，PyTorch 双类别模型，类别 `0=real_vehicle`、`1=model_prop`；SHA-256 为 `a30c839d72a6d996eac62b0ab15a147ee03d3c28c0a3853bb25b16c04445cfc2`。
- 运行环境：Ubuntu 24.04，Python 3.10 及以上；依赖见 `requirements.txt`。CUDA 可用时使用第一块 GPU，否则使用 CPU。PyTorch 安装源应与评测机 CUDA 环境匹配。
- 感知输入：仅用 `obs.self.photo` 的图片字节、本机姿态、队友消息以及 `score_view.sim_time`。图片由官方 SDK 放入观测，Agent 不连接 Redis。
- 算法来源：以 `ZqhjGame_Module_Based` 的 `main` 提交 `8d589a5`（V4.5.1）为基础，整合五状态控制、三机搜索和本地运动判定；视觉推理改为提交文件内共享的最新帧工作线程。模型来自 `assets/vehicle_prop_v2/yolo26s_fovmix_blend50.pt`。
- 使用方式：将本文件夹中的文件一起放入赛题二目录，并指定 `v9aget:V9aget`。无需复制开发仓库、配置 JSON、脚本或输出目录。

本版只完成静态与接口检查，尚未进行仿真测评；最终成绩和 Linux 集群运行情况待验证。
