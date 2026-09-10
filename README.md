# maimai Chart Studio 1.1.0

[English](README.en.md) | 简体中文

本地运行的 maimai 谱面生成器。1.1.0 固定使用一套原生模型链：全曲 planner、低难度轻量 V4 联合采样、高难度 joint WHAT planner 与 relational WHERE、以及共享 CUDA Harness。

## 支持范围

- BASIC、ADVANCED 使用轻量 V4 联合采样并保持简单配置；EXPERT、MASTER、Re:MASTER 使用 joint WHAT 再进入 relational WHERE，不回退旧的 factorized WHAT 路径。
- 五档共用 contextual V4 renderer 与 CUDA Harness，但按难度分为轻量低难度路径和重型高难度路径。
- Star、双押、Hold、Touch、Touch Hold 的界面倍率作用于同一归一化配置分布。
- `stable` 是默认搜索；EXPERT 以上可选 `causal-v1` 小范围因果回溯。两种模式使用同一 planner、renderer 和 Harness。
- Harness 使用完整轨迹接触、动态手数、180 Hz 松手语义和版本能力检查；只有整谱 ACCEPT 并绑定 permit 后才会发布文件。

## 环境

- Windows 10/11
- NVIDIA CUDA GPU
- 与驱动匹配的 CUDA 版 PyTorch
- Python 3.11 或更新版本
- FFmpeg 已随发布包内置；无需单独安装或配置 `PATH`

安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

## 启动

双击 `启动生成器.pyw`。启动器会打开本地网页制作台，音频和素材不会上传。

生成前需要填写或选择：

- 音频或带音轨的 MP4、曲名、BPM；
- 自定义歌曲 ID（可留空；DX 前从 3000、DX 及以后从 13000 开始自动分配并持久避重）；
- 可选的封面图片；
- 可选的 BGA MP4；也可以直接把同一个 MP4 同时作为音乐与 BGA；
- 谱面版本、难度与精确定数。
- 更换或上传新音频会清空上一首的封面和 BGA 选择；曲名仍默认使用音频文件名。



## 内置 FFmpeg 与外部预览工具

发布 ZIP 已内置 Windows x64 FFmpeg。MiaCode 和 MajdataViewX 仍是可选外部工具；如需使用，请解压到程序目录的 `tools`。运行时识别以下结构：

```text
maimai Chart Studio 1.1.0/
└─ tools/
   ├─ ffmpeg/
   │  ├─ ffmpeg.exe
   │  ├─ BUILD_INFO.txt
   │  └─ LICENSE-GPL-3.0.txt
   ├─ MiaCode-v1.0.0-win64/
   │  └─ MiaCode.exe
   └─ MajdataViewX-v6.2.0/
      └─ MajdataEdit-Neo.exe
```

- FFmpeg：发布包内置 `tools/ffmpeg/ffmpeg.exe`；运行时优先使用包内版本，系统 `PATH` 仅作为兼容 fallback。
- MiaCode：解压后必须存在 `tools/MiaCode-v1.0.0-win64/MiaCode.exe`。
- MajdataViewX：解压后必须存在 `tools/MajdataViewX-v6.2.0/MajdataEdit-Neo.exe`。

点击 MiaCode 或 MajdataViewX 时会先检查对应入口文件，存在后才选择或打开 `maidata.txt`。

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

`track.mp3` 始终规范化为真实的 320 kbps MP3，并复用 Studio 试听缓存。选择封面时写出 `bg.png`（非 PNG 输入会转换为 PNG）；选择 BGA 时写出只含视频流的 `pv.mp4`，视频流采用 stream copy、不重新编码。带音轨 MP4 可一次导入并自动拆成 `track.mp3 + pv.mp4`。`元数据.json` 记录最终歌曲 ID、是否自动分配、模型、Harness、各难度摘要、内容 digest 和实际输出路径。
- 自动歌曲 ID 注册表使用跨进程锁；同时打开多个 Studio 实例申请 ID 时也不会重复分配。


## 预览

可选安装 MiaCode 或 MajdataViewX。制作台会提供两个并列的外部工具入口；MajdataViewX 不可用时只提示安装位置，不再回退到内置预览。MajdataViewX 可安装到 `tools/MajdataViewX-v6.2.0` 或 `.tools/MajdataViewX-v6.2.0`，也可通过 `MAJDATA_EXE` 指定 `MajdataEdit-Neo.exe`。标准歌曲素材名为 `track.mp3`、`bg.png` 和 `pv.mp4`。
- 封面与 BGA 下方各有固定大小的拖放框；拖入图片或 MP4 会自动导入。BGA 使用包内 FFmpeg 从原视频截取静态缩略图预览，不要求浏览器直接解码 MP4；源文件不会被修改，最终 `pv.mp4` 只做视频流 remux，不重新编码画面。清空按钮只清除当前选择，不删除原始文件。

本项目为非官方工具，不隶属于 SEGA。
