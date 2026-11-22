# MiaCode-NoaCopilot

本开发预览版基于 [Team-MiaCode/MiaCode](https://github.com/Team-MiaCode/MiaCode)，新增谱面续写功能。编辑器及本分支对编辑器的修改采用 [GNU GPLv3](LICENSE)。第三方组件和素材的授权见 [第三方声明](THIRD_PARTY_NOTICES.md)。

可在「设置 → NoaCopilot」安装 CPU 或 CUDA 版 PyTorch，包含独立 Python。CUDA 版也支持 CPU 推理。下载源可选清华 TUNA、中国科大、阿里云或官方源。若您的电脑上已有PyTorch环境，可在高级设置中选择。

首次使用续写时，点击同页的「下载 MERT2」，应用会从 [MERT-v2-FullSong 官方仓库](https://huggingface.co/m-a-p/MERT-v2-FullSong) 下载权重，下载完成后会自动裁剪并保留所需的部分。MERT2 权重及其裁剪版采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)，仅限非商业用途。
