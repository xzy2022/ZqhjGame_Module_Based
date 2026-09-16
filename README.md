# ZqhjGame_Module_Based

红枫 2026 赛题二的独立算法仓库，包含 PersonalV1、PersonalV2 视觉旁路、个人场景、模型与实验工具。官方 SDK、Python、UE 和引擎由父层官方发行包提供。

日常 VS Code 工作区和以下命令的当前目录均为 **hf2026-sim-windows 官方根目录**，不要先进入本子仓库。`codex/migration-integration`分支中有一些文档，后续不会在被主分支维护，但是需要的话也可以看(也可以把它们拷贝到hf2026-sim-windows 官方根目录层级)。

## 首次安装

在已经准备好 UE 等资源的官方 Windows 发行包根目录执行：

```powershell
git clone https://github.com/xzy2022/ZqhjGame_Module_Based.git ZqhjGame_Module_Based
git -C .\ZqhjGame_Module_Based lfs pull
.\ZqhjGame_Module_Based\setup.ps1 -Vision
```

安装器默认选择父目录 `python/python.exe`，建立子仓库自己的 `.venv`，无需系统 Python 或队友虚拟环境。已经克隆时不要重复 clone；先更新到维护者发布的版本。模型位于 `assets/personal_v2/vision.pt`，使用 Git LFS。

## V1 实验与网页

```powershell
$runId = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$runOutput = Join-Path '..\output\tmp\module_based' ('v1-' + $runId)
.\ZqhjGame_Module_Based\run.cmd control_test_runner `
  --scenario-json '.\ZqhjGame_Module_Based\configs\scenarios\coop_decoy\static-decoys.json' `
  --agent 'personal_hf2026.personal_v1:PersonalV1Agent' `
  --duration 600 --seed 1 --output $runOutput --visualize
```

删除 `--visualize` 就不启动网页。网页复用官方界面，只观察本轮实验；它不会再启动一个仿真，也不会由此启动 UE 相机。已有 Redis 会直接复用；缺失时入口创建本轮 Redis，并只清理自己启动的服务。V1 是理想感知控制实验，网页实时地图不等于 UE 图像感知。

不传输出参数时，各入口的默认输出根为官方目录同级的 `output`。本机即 `D:/Workspace/00_MyRepo/red_m_competiton/output`。显式相对路径相对于调用命令时的目录解释；代码和模型默认路径相对于本子仓库定位。

当前V1版本在下面这些条件下，对于晴天环境测试，可以取得90+的分数：
1. 诱饵被设定为静止，方便直接根据速度判断真目标和诱饵(隔离视觉识别)
2. 相机fov内的车辆直接获得其100%的精准经纬度坐标(隔离从图像获得经纬度坐标)
3. 完全放弃无人机之距离超过200m扣分的分数

在这些假设下，也完成了这些工作：
1. 搜索航线的规划以对场地进行覆盖扫描
2. 无人机发现真目标后，通过通信发起协同跟踪
3. fov内多个目标时，根据经纬度坐标去进行轨迹匹配区分各个实体
4. track中断时，调整无人机的移动方向靠近真目标同时远离最大竞争者(设计原因见track机制)

对于track机制，有非常需要注意的地方。这一机制是仿真系统自带的，而非V1阶段开发时自行设计的：
1. track只会锁定离自己最近的对象。
2. 当最近对象在FOV外，锁定会失败。它根本不会锁定FOV内想要锁定的对象。
3. 当最近对象在FOV内，如果是真目标，则可以锁定。如果是诱饵，有概率锁定为空。

## V2 与采集

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
.\ZqhjGame_Module_Based\run.cmd visual_shadow_study `
  --duration 200 --seed 1 --output "..\output\tmp\module_based\v2-$stamp"

.\ZqhjGame_Module_Based\run.cmd dataset_capture `
  --weather Clear_Skies --duration 20 --seed 1 --fov 30 `
  --output "..\output\tmp\module_based\capture-$stamp"
```

逐条运行并等待上一轮结束；V2 和采集入口自行管理本轮 UE。V2 仍由 V1 控制，视觉仅在旁路观察；采集使用 `oracle_identity` 与 `ideal_positions`，可见性未经人工标注。已有输出不会用作新的采集目录。FOV30 采集会额外记录飞机姿态、协同状态、`camera_calibration.json` 和 `dataset/coop_pairs.jsonl`；相机参数仍是基于未验证假设的推导值。`dataset/samples.jsonl.capture_pose` 将同一 `sim:state` tick 的飞机经纬高、roll/pitch/yaw 和云台 pan/tilt/FOV 作为一个原子快照按照片源时间就近对齐，并保留对齐时间差；它不是已标定的相机外参。

已有 FOV30 批次可直接导出严格的双机 pair 数据集，不需要重新采集。下面命令处理 `runs` 下的全部 run；默认剔除任一侧缺框、缺位姿/姿态/真值、时间差超过 0.1 秒以及触边框的 pair：

```powershell
$pairOutput = 'E:\datasets\fov30-red-m-0916-paired-strict'
.\ZqhjGame_Module_Based\run.cmd coop_pair_dataset `
  --source-root 'E:\datasets\fov30-red-m-0916\runs' `
  --output-root $pairOutput `
  --image-mode hardlink `
  --max-time-delta-s 0.1
```

硬链接要求源图和输出目录位于同一磁盘卷。只处理一个 run 时加 `--run-names seed01-clear-skies-150s-fov30`；需要保留触边框时加 `--keep-edge`。训练/验证划分必须按输出标签中的 `run + session_id` 分组，不能随机拆相邻帧。

同一批原始 run 也可以导出目标/诱饵分类数据集。分类裁剪是无绘制框的原始 RGB PNG；`samples.jsonl` 是不含 run、天气、无人机、目标 ID、框和地理信息的训练索引，`audit.jsonl` 只用于标签溯源和人工核验，不应送入分类模型：

```powershell
$classOutput = 'D:\Workspace\00_MyRepo\red_m_competiton\datasets\fov30-red-m-0916-target-decoy'
.\ZqhjGame_Module_Based\run.cmd target_decoy_dataset `
  --source-root 'E:\datasets\fov30-red-m-0916\runs' `
  --output-root $classOutput `
  --image-mode copy
```

源 run 与输出跨磁盘时必须使用 `--image-mode copy`；若输出也放在 E 盘，可改为 `--image-mode hardlink` 节省源图重复占用。默认沿用旧整理器的行为，触边框裁到图像范围后保留并在审计索引中标记；需要拒绝触边框时加 `--drop-edge`。只处理一个 run 时使用 `--run-names <run名>`，小批实际验证可再加 `--limit-per-run <记录数>`。训练/验证划分应按匿名 `group_id` 分组，避免同一物理车辆的相邻裁剪跨集合泄漏。

非默认目录可在模块名前指定 `--sim-root <SDK目录> --runtime-root <完整发行包目录>`。当前兼容基准为官方 `79b91d2336b61fee901b1342fbdd99a83ee15f40`，保留 OpenSim 2.0.4 官方更新。

