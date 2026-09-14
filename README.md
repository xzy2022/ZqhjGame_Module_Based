# ZqhjGame_Module_Based

红枫 2026 赛题二多机协同识别的个人算法仓库，包含 PersonalV1、PersonalV2 视觉旁路、实验与数据集采集工具、个人场景及文档。它采用独立 Git 历史，通常放在官方 Windows UE 发行包内，与队友的 `ZqhjGame` 目录并列。

```text
hf2026-sim-windows/                  官方 SDK、UE、引擎和 Redis
├─ competition/
├─ python/
├─ ue-renderer/
├─ opensim-sim.exe
├─ ZqhjGame/                         队友独立仓库，运行本项目无需导入它
└─ ZqhjGame_Module_Based/            本仓库
   ├─ src/personal_hf2026/           算法、实验入口与分析工具
   ├─ configs/scenarios/             迁移后的场景副本
   ├─ assets/personal_v2/vision.pt    视觉权重，使用 Git LFS
   ├─ docs/
   ├─ requirements.txt
   ├─ setup.ps1
   └─ run.cmd
```

先按 [给队友的安装与启动说明](docs/human_read/启动方式-给队友.md) 建立本仓库 `.venv`，再从本仓库根目录运行 `run.cmd`。默认官方根目录为父目录，也支持 `--sim-root`、`--runtime-root` 与相应环境变量。模型使用 Git LFS，首次获取仓库后执行 `git lfs pull`。

PersonalV1 的控制验证采用理想目标坐标；PersonalV2 在同一控制链旁运行视觉观察，视觉尚未接管控制。采集入口使用 `oracle_identity` 控制和 `ideal_positions` 感知；照片中的投影框可见性仍需人工核验。这些实验不能直接解释为纯视觉参赛成绩。

`docs` 从原项目完整复制，旧文档只做路径适配，保留历史事实、实验结果和当时的限制。当前仓库的安装与运行以新队友文档为准；[官方安装说明](docs/安装.md) 和 [比赛手册](docs/codex_read/红枫2026无人集群自主协同智能算法挑战赛参赛手册.md) 中的发行包目录结构仍以官方目录为上下文。

本次迁移保留原官方仓库及其历史。本仓库 `main` 从空提交开始，开发在独立工作树的 `codex/*` 分支进行，成果汇总到集成分支后由用户合入 `main`。GitHub 地址及推送由用户自行配置。

当前集成分支为 `codex/migration-integration`。基准 SDK 隔离方式、V1/V2 和采集实际结果、首轮失败记录及未验证范围见[迁移验收报告](docs/human_read/迁移验收-20260914.md)。逐文件迁移来源见 `migration/source-manifest.json`。
