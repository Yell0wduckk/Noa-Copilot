# NoaCopilot 的 Flat320 推理环境

从 Windows 发布包顶层的 `MiaCode.exe` 启动，在 NoaCopilot 设置中选择 Flat320 v18.5 或 v18.4 谱面 checkpoint。内置推理根据权重声明的架构选择解码器。整曲、光标续写和选区替换共用 `maidata.spine.copilot`。续写和选区使用编辑器已有的音符前缀。谱面权重可以放在 `app/noa-copilot/checkpoints`，也可从设置选择外部文件。

Windows CPU 使用包内 `app/noa-copilot/env/python.exe`。选用内置环境和 CUDA 时，编辑器通过 WSL 调用设置中的 Linux Python，读取同一份源码、音频和模型。高级设置也支持自定义 Windows Python。

点底部状态栏的 NoaCopilot，再展开“高级设置”，即可调整 Slide 偏置和随机种子。偏好设置的 NoaCopilot 高级设置也能调整同一组值。Slide 偏置默认 0，可调至 4。每次允许写新滑星时，采样器在温度处理前给三种合法星头的 logit 加上该值。温度为 1 时，偏置约 0.7 会使该步选择滑星相对其他合法动作的赔率约翻倍。整曲生成、光标续写和选区替换共用这些设置。随机种子默认 `-1`，每次请求重新取种。输入非负整数可以在相同环境、权重、采样设置和输入下复现生成。

生成整曲时，已保存的本地工程约每两秒将当前难度的新增谱面同步到预览和 `maidata.txt`，结束时立即写入完整结果。取消生成会保留最近一次已写入的前缀。正在播放时沿用现有音频时钟。光标续写和选区替换在采纳后才修改谱面。

## 重建推理包

1. 在本仓库根目录运行下面的命令。从完整的本地快照导出原精度的声学前端、下采样器和前八个 Conformer 块。第 8 层输出为 25 Hz × 1024，125 Hz 双窗 Mel 由谱面源码另行计算。导出的模型目录包含原许可证和裁剪说明。

   ```powershell
   third_party/noa-copilot-env/python.exe scripts/build/export-mert-v2-l8.py ../MERT-v2-FullSong third_party/flat320-features/mert-v2
   ```

2. 使用 Qt 6.11.1 的 Release 目录构建 `MiaCode` 和 `MiaCodeLauncher`。运行 `scripts/build/package-win.ps1` 打包到新的发布目录。默认从工作区的 `repos-0wd-v3/worktrees/v18-beat-mertv2-calibrator` 导出已提交的 `src/maidata`，将 Git 提交号记录到 `app/noa-copilot/runtime.json`。可用 `MIACODE_FLAT_SOURCE`、`MIACODE_FLAT_MODELS` 和 `MIACODE_NOA_PYTHON_ENV` 分别指定模型源码仓库、已裁剪模型的父目录和可搬移的 Windows Python 安装。

3. 通过 `MIACODE_WSL_DISTRO` 和 `MIACODE_WSL_PYTHON` 指定本包的初始 CUDA 环境。本机使用 `Ubuntu-22.04` 的 `/home/dev/.venvs/sheetsage2-cu126/bin/python`。内置模式升级时会替换上一发布包写入的默认解释器，用户选择的其他解释器继续生效。两种设备均从 `app/noa-copilot/models/mert-v2` 加载本地前端。
