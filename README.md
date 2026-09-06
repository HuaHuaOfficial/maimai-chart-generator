# maimai Chart Generator 0.3.0

[English](README.en.md) | 简体中文

面向 maimai DX Simai 谱面的原生 CUDA 生成器。当前版本使用 contextual V4 模型生成候选，并由同一套 CUDA Harness 检查候选与整谱；只支持随仓库发布的最新模型，不兼容旧 checkpoint。

## 功能

- 从音频、BPM、版本和目标定数生成 BASIC～Re:MASTER 谱面。
- 保留 Slide Star 与 Track 的独立表示，并控制星星数量上下限。
- HARD 问题返回局部上下文重生成；相同阻塞点重复失败时扩大因果窗口，不重新生成整首。
- 只有完整整谱许可且 Simai 回读与许可稿一致时才写出 `maidata.txt`。
- Windows GUI 入口为 `启动生成器.pyw`。

## 环境与启动

1. 安装 NVIDIA CUDA 版 PyTorch 和 Python 依赖：`pip install -r requirements.txt`。
2. 自行安装 FFmpeg，并确保 `ffmpeg.exe` 位于系统 PATH。
3. 双击 `启动生成器.pyw`。
4. 选择音频、版本、定数和 BPM；结果默认保存到 `generated`。

MajdataViewX 和 FFmpeg 均不随仓库提供。若自行把 MajdataViewX 放到 `tools/MajdataViewX-v6.2.0`，预览按钮会优先调用它；否则使用内置谱面预览。系统存在 `ffplay` 时，内置预览可以同步播放音频。

## 已知边界

- 需要 NVIDIA CUDA，不提供 CPU 音乐判定回退。
- 音频最多生成前 256 小节。
- 低难度尚无同口径质量校准时会明确标记为不可用，而不是伪造阈值。
- 当前 MultiTouch 指标不是几何合手证明。
- 模型的星星时间预测已经较稳定，星星轨迹与落位仍有改进空间。

## 许可

本项目自研源码采用 [Apache License 2.0](LICENSE)。随仓库提供的 MERT 模型材料不属于 Apache-2.0 授权范围，继续受 [CC BY-NC 4.0](THIRD_PARTY_NOTICES.md) 约束，因此包含该权重的分发仅限非商业用途。

详细说明见 GitHub Wiki。
