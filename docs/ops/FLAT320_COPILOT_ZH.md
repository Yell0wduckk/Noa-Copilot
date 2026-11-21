# NoaCopilot 的 Flat320 推理环境

从 Windows 发布包顶层的 `MiaCode.exe` 启动，在 NoaCopilot 设置中选择谱面 checkpoint。v19 使用逐拍编辑协议。在 `app/noa-copilot/checkpoints` 中放入权重，或在设置中选择外部文件。启用后，应用启动时加载选定权重。v18.4、v18.5 权重仍可使用随包提供的历史解码器。

## 在编辑器里使用

- 普通编辑可随时停笔，按 Tab 时才发起光标续写。光标在长条、星星路径或多押内部时，从同拍前一个已保留音符组之后的可用位置开始，重新采样当前整个 Moment 的起音和内容。默认补齐本行至四拍边界，遇到已有的下一行则保留该行原来的起拍。可以在高级设置修改每行拍数。
- Ctrl+K 可改写选中的整行。在含骨架的连续行中，Tab 或 Ctrl+K 填充该块，选中部分占位符则只填所选部分。普通拖选后可点击“填充采音”，Alt 拖选松手后直接填充。
- `x` 生成一个非 Slide 音符，`*` 生成单分支 Slide，`x**` 生成双分支 Slide。`1h[4:1]/x` 保留左边的 Hold。草稿占位符在时间轴上显示为 1 号 tap，播放时可听到 key 音。
- 同一推理进程默认预计算 3 条候选，高级设置可改为 1–8 条。↓ 向后翻页时继续预取。当前候选尚在生成时，↓ 会在光标旁提示稍等。当前候选完成后，↓ 可以接看下一条的实时输出，已算完的候选会直接显示。↑ 回看，回到 #0 可听原谱。候选穷尽时光标旁会短暂提示。
- → 采纳一拍并为剩余部分补充候选，Tab 采纳整个当前候选，Esc 保留已经采纳的拍。单击、框选和 Ctrl+单击定位都可以继续查看灰字，随后按 Tab 仍采纳到原来的位置。
- 预览圈和底部时间轴都用普通音符的原色皮肤显示候选。预览圈的候选不透明度为 45%，底部时间轴为 60%，采纳后恢复普通音符的不透明度。候选与正式音符使用相同的音效。流式期间只播放已生成的音符，替换区之外的原谱保留原来的时间。
- 整曲草稿随解码逐拍更新编辑器和预览，并间隔保存当前难度，结束或取消时保存已生成的部分。续写和选区候选在采纳时写入。

## 设备、缓存与采样

谱面模型和 MERT 可分别选择 CPU 或 CUDA。谱面 CPU 默认使用 4 个线程。完整包带有 `app/noa-copilot/env/python.exe`，轻量包可在高级设置中扫描或指定本地 Python。CUDA 支持 Windows 与 WSL Python，两个设备设置共用所选 CUDA 路径。使用两个不同环境时，MERT 只在提取时启动，提取完成后退出。同进程 CUDA 提取完成即释放 MERT 权重和空闲显存，CPU 提取可在切歌时复用权重。设置中的“浏览…”可分别选择解释器。环境扫描分别判断谱面解码与 MERT 能力，随包 MERT 模型需要 torchaudio 的频谱变换。

项目缓存位于音频旁的 `.miacode/.noacopilot/`。MERT 开关控制源音频的 INT8 L8 特征，RoFormer 开关控制 FP16 网格以及随附的 Drafter 和情绪曲线。首次有 BPM 或拍速图、偏移、难度条件发生变化时，编辑器会在约半秒后预热。普通音符编辑复用现有网格，修改定数只更新条件特征。音频结束之后的手写长音仍能延续。同曲替换音频按内容重新绑定缓存。

Slide 偏置默认 0，可调至 4。每次允许新滑星时，采样器在温度处理前给三种合法星头的 logit 加上该值。温度为 1 时，偏置约 0.7 会使该步滑星相对其他合法动作的赔率约翻倍。随机种子默认 `-1`，每次请求重新取种。非负整数可在同一环境、权重及输入下复现采样。所有温度共用同一棵候选树。每一步以温度 1 的原生概率确定分支，最可能的动作达到 0.98 时锁定，其他步骤保留首选和概率至少为 0.03 的次选。温度影响探索顺序，同一锚点下按实际音符去重。

## 重建推理包

1. 在本仓库根目录运行下面的命令。从完整的本地快照导出原精度的声学前端、下采样器和前八个 Conformer 块。第 8 层输出为 25 Hz × 1024，125 Hz 双窗 Mel 由谱面源码另行计算。导出的模型目录包含原许可证和裁剪说明。

   ```powershell
   third_party/noa-copilot-env/python.exe scripts/build/export-mert-v2-l8.py ../MERT-v2-FullSong third_party/flat320-features/mert-v2
   ```

2. 使用 Qt 6.11.1 的 Release 目录构建 `MiaCode` 和 `MiaCodeLauncher`。运行 `scripts/build/package-win.ps1` 打包到新的发布目录。在命令后加 `-NoTorch` 可生成使用外部 Python 的轻量包。默认从工作区的 `repos-0wd-v3/worktrees/v19-flat320-drafter-te` 导出已提交的 `src/maidata`，将 Git 提交号记录到 `app/noa-copilot/runtime.json`。可用 `MIACODE_FLAT_SOURCE`、`MIACODE_FLAT_MODELS` 和 `MIACODE_NOA_PYTHON_ENV` 分别指定模型源码仓库、已裁剪模型的父目录和可搬移的 Windows Python 安装。

3. 通过 `MIACODE_WSL_DISTRO` 和 `MIACODE_WSL_PYTHON` 指定本包的初始 CUDA 环境。本机使用 `Ubuntu-22.04` 的 `/home/dev/.venvs/sheetsage2-cu126/bin/python`。内置模式升级时会替换上一发布包写入的默认解释器，用户选择的其他解释器继续生效。两种设备均从 `app/noa-copilot/models/mert-v2` 加载本地前端。
