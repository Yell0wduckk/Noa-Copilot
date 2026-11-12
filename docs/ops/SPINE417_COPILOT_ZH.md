# NoaCopilot 的 Spine417 推理环境

Windows 发布包解压后从顶层 `MiaCode.exe` 启动。在 NoaCopilot 设置选择 v15 的 `maidata.spine417.mert6+14-muq8-mel.v4.designer21` 谱面 checkpoint。整曲、光标续写和选区共用同一模型。续写和选区用光标或选区前已提交的真实音符构建历史，替换范围按当前难度的 Simai 单元对齐。

默认 Python 在 `app/noa-copilot/env/python.exe`。保持“内置”环境并将计算设备选为 CUDA，可以编辑 WSL 发行版和 Linux Python 的完整路径。WSL 经挂载盘读取同一份源码、音频、checkpoint 和前端权重。Windows 自定义解释器也可在高级设置选择。

### 复现 Windows 发布包

- `MIACODE_SPINE_SOURCE` 指向 v15 worktree，默认使用工作区的 `repos-0wd-v3/worktrees/v15-bms-beat-calibrator`。
- `MIACODE_NOA_PYTHON_ENV` 指向可搬移的完整 Windows Python 安装，默认位于 `third_party/noa-copilot-env`。本次为 CPython 3.13、CPU torch 和 torchaudio 2.11、Transformers 4.47.1、MuQ 0.1.0。解释器、标准库、DLL 和 site-packages 都随包分发。
- 在可加载原始 MERT 和 MuQ 的环境中，将 `PYTHONPATH` 指到 v15 的 `src`，运行 `python scripts/export_spine_frontends.py --mert <MERT原始目录> --muq <MuQ原始目录> --output <MiaCode目录>/third_party/spine417-features`。也可用 `MIACODE_SPINE_MODELS` 指定已有导出目录。发布包仅复制导出的权重和模型配置。
- `MIACODE_WSL_DISTRO` 和 `MIACODE_WSL_PYTHON` 提供该发布包初始的 CUDA 设置。用户在界面中保存的值优先。本机使用 `Ubuntu-22.04` 和 `/home/dev/.venvs/maimai-structure-v1/songformer/bin/python`。
- 使用仓库 Windows Release 打包脚本和 Qt 6.11.1。`app/noa-copilot/runtime.json` 记录打包源码提交。谱面 checkpoint 可放在 `app/noa-copilot/checkpoints` 或从设置选择外部文件。

MERT 权重来自 `m-a-p/MERT-v1-330M`，本次输入快照为 `5240c2708a5acaee1007f43fb9735c7dcd0b78c9`。应用一次运行前 14 层，读取第 6、14 层并使用谱面 checkpoint 学得的混合权重。MuQ 来自 `OpenMuQ/MuQ-large-msd-iter` 的 `0562a57814f6f8bbd9fdea0a25921a2fce1a841a`，应用只运行 Mel 预处理、下采样和前 8 个 Conformer 块。RVQ 目标、分类投影和层末输出归一化均已从发布权重剔除。
