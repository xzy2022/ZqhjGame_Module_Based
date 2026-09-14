# 车辆发现研究参考归档

归档日期：2026-09-14。本目录保存尚未并入原官方仓库 master 的车辆发现离线研究，供后续移植参考；未接入本项目的 `personal_hf2026` 算法包，也未进行迁移后的实际运行或仿真。

## 来源与范围

- 原仓库分支：`codex/vehicle-discovery-compare`。
- 冻结来源提交：`37ae1d79ad8930090649f871b965e6545407ca11`。
- 原代码包：`competition.user_algorithms.coop_decoy.vehicle_discovery`。
- 原代码目录：`competition/user_algorithms/coop_decoy/vehicle_discovery/`。
- 本目录的 9 个 Python 文件直接复制 Git 提交中的内容，保留原注释、导入、参数和硬编码路径，没有改写算法。
- `docs/车辆发现离线验证.md` 与 `docs/分支开发.md` 复制该提交的原文；其中旧电脑目录、旧解释器、旧命令和历史测试结果仅用于记录当时研究，不代表本次迁移已经验证或可直接运行。

## 依赖与后续移植边界

离线代码依赖 NumPy、OpenCV、PyTorch，以及原 `visual_appearance.py` 间接使用的 Pillow。`static.py` 仍保留 `from ..visual_appearance import ...`，需要后续移植时改为新算法包的对应模块。它还保留原输出目录、manifest 和模型文件路径；不能只把旧命令的模块前缀替换后便视为完成迁移。

新项目 `requirements.txt` 提供个人算法自己的 Python 依赖环境。归档中的队友 `.venv-learning` 路径属于历史记录，本项目无需使用或修改队友仓库。研究使用 OpenCV 写视频，旧文档中的汇总视频和报告没有随代码迁入。

以下数据、权重及产物没有复制：

- `dataset-200s-seed1-20260913-192854` 与 `dataset-fov30-50s-seed1-20260913-195159` 的原图、标注及采集日志。
- 旧 `output/personal_v2/vehicle-discovery/20260913-204344` 下的 manifest、参数冻结文件、A2 训练模型、候选记录、指标、案例及视频。
- 队友仓库、队友模型和任何虚拟环境。

本目录只核对了归档文件与来源提交的字节一致性；未运行既有逻辑测试、离线重放或 UE 仿真。原文的性能结论仍只适用于其固定数据和历史运行。

## 其它研究分支检查

以原 master `384b6f0b7c86ffe370a3d0ab23df3394205269a2` 为比较对象，检查原仓库全部 8 个分支及 detached PersonalV2 工作树提交 `7d211b83e04668789f4b18215630b4fe19f459a8`：

- `codex/dataset-night-batch`、`codex/dataset-night-collect`、`codex/dataset-oracle-weather` 的分支改动虽然提交历史仍分叉，但相关文件内容已与 master 相同，没有另外漏迁的独有代码。
- `codex/vehicle-discovery-base`、`codex/vehicle-static-poc`、`codex/vehicle-motion-poc` 的独有代码都已被本次来源提交完整包含，文件字节相同。
- detached PersonalV2 提交没有相对 master 的独有分支改动。
- 其它研究工作树的代码与文档范围未发现未提交修改；原主工作树的已有场景修改及本次迁移验证输出由主迁移任务单独处理。
