# ZqhjGame_Module_Based

红枫 2026 赛题二的独立算法仓库，包含 PersonalV1、PersonalV2 视觉旁路、个人场景、模型与实验工具。官方 SDK、Python、UE 和引擎由父层官方发行包提供。

日常 VS Code 工作区和以下命令的当前目录均为 **hf2026-sim-windows 官方根目录**，不要先进入本子仓库。详细启动教程统一维护在外层 `docs/human_read/启动方式-模块化项目.md`；本仓库不再维护重复的 `docs`。

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

## V2 与采集

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
.\ZqhjGame_Module_Based\run.cmd visual_shadow_study `
  --duration 200 --seed 1 --output "..\output\tmp\module_based\v2-$stamp"

.\ZqhjGame_Module_Based\run.cmd dataset_capture `
  --weather Clear_Skies --duration 20 --seed 1 `
  --output "..\output\tmp\module_based\capture-$stamp"
```

逐条运行并等待上一轮结束；V2 和采集入口自行管理本轮 UE。V2 仍由 V1 控制，视觉仅在旁路观察；采集使用 `oracle_identity` 与 `ideal_positions`，可见性未经人工标注。已有输出不会用作新的采集目录。

非默认目录可在模块名前指定 `--sim-root <SDK目录> --runtime-root <完整发行包目录>`。当前兼容基准为官方 `79b91d2336b61fee901b1342fbdd99a83ee15f40`，保留 OpenSim 2.0.4 官方更新。历史来源与验收清单保存在 `migration`；其中2026-09-14首轮迁移记录描述的是当时版本。未合入算法主线的车辆发现研究只归档在 `references`。

本轮收尾开发在 `codex/migration-finish-integration` 汇总，尚未由迁移任务推送到 GitHub。云端默认分支 `codex/migration-integration` 的旧版本不包含本轮修复；维护者发布新提交后，队友才能直接取得这些行为。主分支合并和远程发布由维护者执行。
