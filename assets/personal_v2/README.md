# PersonalV2 视觉旁路权重

本仓库路径：`assets/personal_v2/vision.pt`。本次迁移复制自原个人算法的 `competition/user_algorithms/coop_decoy/v2_assets/vision.pt`，文件最初来源为队友冻结发布 `ZqhjGame/artifacts/submission/capture-v31/vision.pt`。

SHA256：`819381fa5b383310592238f0cc61843c5ac808228bc4ec9d997295c9c822a0cf`。

配套 `visual_appearance.py` 来自 v31 的 `src/zqhj_patch_vision.py`，原始文件 SHA256 为 `62eb52ddfa824f736415885565dd62fd2020969a8a6260a215d6bb4ed0855106`。只调整模块导入及中文注释；网络、类别顺序、候选提取和图像预处理保持一致。

类别顺序：真目标、诱饵、背景。推理依赖 PyTorch、NumPy、OpenCV、Pillow；实验入口另依赖项目 SDK 和 Redis。使用本仓库的 `.venv/Scripts/python.exe` 和 `requirements.txt` 安装依赖，运行无需导入队友代码。

只迁移外观模型，未迁移 v31 的图像运动验证、图像地理定位、飞行规划或协同状态机。
