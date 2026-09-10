# maimai Chart Studio 1.0.0

本地运行的 maimai 谱面生成器。1.0.0 固定使用一套原生模型链：全曲 planner、五难度 joint WHAT planner、V4 relational renderer，以及共享 CUDA Harness。

## 支持范围

- BASIC、ADVANCED、EXPERT、MASTER、Re:MASTER 五档均由同一 joint WHAT 分层生成，不回退旧的 factorized WHAT 路径。
- 定数精确到 0.1。WHAT 组成先按谱面版本/机制兼容性筛选，再在同 DS、近 BPM 的官谱池内选择联合 profile；缺少精确官谱 profile 的档位会在生成前明确拒绝，不偷换为邻近 DS。
- Star、双押、Hold、Touch、Touch Hold 的界面倍率作用于同一归一化配置分布。
- `stable` 是默认搜索；EXPERT 以上可选 `causal-v1` 小范围因果回溯。两种模式使用同一 planner、renderer 和 Harness。
- Harness 使用完整轨迹接触、动态手数、180 Hz 松手语义和版本能力检查；只有整谱 ACCEPT 并绑定 permit 后才会发布文件。

## 环境

- Windows 10/11
- NVIDIA CUDA GPU
- 与驱动匹配的 CUDA 版 PyTorch
- Python 3.11 或更新版本
- FFmpeg，且 `ffmpeg.exe` 已加入 `PATH`

安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

## 启动

双击 `启动生成器.pyw`。启动器会打开本地网页制作台，音频和素材不会上传。

生成前需要填写或选择：

- 音频、曲名、BPM；
- 自定义歌曲 ID（可留空；DX 前从 3000、DX 及以后从 13000 开始自动分配并持久避重）；
- 可选的封面图片；
- 可选的 BGA MP4；
- 谱面版本、难度与精确定数。

1.0.0 不再提供任意 checkpoint 选择器或后端选择器。发布包只接受清单中的固定模型，并只运行 CUDA 路径。

## 输出目录

一次成功生成写为：

```text
输出目录/
└─ 乐曲名-YYYYMMDD_HHMMSS/
   ├─ 元数据.json
   └─ 歌曲ID/
      ├─ maidata.txt
      ├─ track.mp3
      ├─ bg.png（可选）
      └─ pv.mp4（可选）
```

`track.mp3` 由输入音频转码。选择封面时写出 `bg.png`（非 PNG 输入会转换为 PNG）；选择 BGA 时写出 `pv.mp4`。`元数据.json` 记录最终歌曲 ID、是否自动分配、模型、Harness、各难度摘要、内容 digest 和实际输出路径。

## 模型与实验边界

正式包不包含训练集、实验脚本、历史 checkpoint、失败的 ranker、生成成品、日志、备份或阶段性部署文档。官谱 profile 只控制全曲组成，不给 Touch-on-Slide 等局部配置添加人工奖励；局部编排仍由模型和音乐条件自然产生，Harness 只负责合法性。

## 预览

可选安装 MiaCode 或 MajdataViewX。制作台会优先调用外部工具；不可用时回退到内置预览。标准歌曲素材名为 `track.mp3`、`bg.png` 和 `pv.mp4`。

本项目为非官方工具，不隶属于 SEGA。
