# maimai Chart Studio 1.0.0

[简体中文](README.md) | English

maimai Chart Studio is a local Windows chart generator. Version 1.0.0 ships one fixed native model chain: a whole-song planner, a five-difficulty joint WHAT planner, the V4 relational renderer, and the shared CUDA Harness.

## Supported generation

- BASIC, ADVANCED, EXPERT, MASTER, and Re:MASTER use the same joint WHAT hierarchy. There is no fallback to the old factorized WHAT path.
- Internal levels are controlled to 0.1. Official-chart composition profiles are filtered by mechanics-compatible version first, then exact DS and nearby BPM. A level without an exact official profile is rejected before generation rather than silently mapped to a neighboring DS.
- Star, double-note, Hold, Touch, and Touch Hold controls operate in one normalized configuration distribution.
- `stable` is the default search. `causal-v1` is an optional bounded recovery mode for EXPERT and above; both modes use the same planner, renderer, and Harness.
- A chart is written only after full-chart CUDA evaluation returns ACCEPT and a permit is bound to the exact content digest.

## Requirements

- Windows 10/11
- NVIDIA CUDA GPU
- A CUDA-enabled PyTorch build compatible with the installed driver
- Python 3.11 or newer
- FFmpeg available on `PATH`

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

Double-click `启动生成器.pyw` to open the local Studio. Audio and media assets remain on the computer.

The form requires audio, title, BPM, chart version, difficulty, and exact internal level. Song ID is optional: automatic IDs start at 3000 before DX and 13000 from DX onward, and are persisted without reuse. Cover and BGA MP4 are optional.

## External tools

The release ZIP does not bundle FFmpeg, MiaCode, or MajdataViewX. Download and extract Windows x64 ZIP/7z distributions into the application's `tools` directory; placing the archive file itself there is not sufficient. The runtime recognizes:

```text
maimai Chart Studio 1.0.0/
└─ tools/
   ├─ ffmpeg/
   │  └─ <any extracted nesting>/bin/ffmpeg.exe
   ├─ MiaCode-v1.0.0-win64/
   │  └─ MiaCode.exe
   └─ MajdataViewX-v6.2.0/
      └─ MajdataEdit-Neo.exe
```

- FFmpeg may be nested anywhere below `tools/ffmpeg`; standard Windows x64 builds normally place it at `bin/ffmpeg.exe`. FFmpeg on the system `PATH` is also accepted.
- MiaCode requires `tools/MiaCode-v1.0.0-win64/MiaCode.exe`.
- MajdataViewX requires `tools/MajdataViewX-v6.2.0/MajdataEdit-Neo.exe`.

The Studio checks each editor/viewer executable before asking for a `maidata.txt` file.

## Output layout

```text
output/
└─ Song title-YYYYMMDD_HHMMSS/
   ├─ 元数据.json
   └─ SongId/
      ├─ maidata.txt
      ├─ track.mp3
      ├─ bg.png (optional)
      └─ pv.mp4 (optional)
```



## Preview

MiaCode and MajdataViewX are optional external tools exposed as two separate actions in the Studio. If MajdataViewX is unavailable, the action reports the installation locations instead of opening a built-in preview. Install it at `tools/MajdataViewX-v6.2.0` or `.tools/MajdataViewX-v6.2.0`, or set `MAJDATA_EXE` to the `MajdataEdit-Neo.exe` path.

This is an unofficial project and is not affiliated with SEGA.
