# maimai Chart Studio 1.1.0

[简体中文](README.md) | English

maimai Chart Studio is a local Windows chart generator. Version 1.1.0 ships one fixed native chain: a whole-song planner, lightweight V4 combined sampling for lower difficulties, joint WHAT plus relational WHERE for higher difficulties, and the shared CUDA Harness.

## Supported generation

- BASIC and ADVANCED use lightweight V4 combined sampling and remain oriented toward simple configurations; EXPERT, MASTER, and Re:MASTER use joint WHAT followed by relational WHERE. The old factorized WHAT fallback is not used.
- All five difficulties share the contextual V4 renderer and CUDA Harness, but production intentionally uses a lightweight lower-difficulty path and a heavier higher-difficulty path.
- Star, double-note, Hold, Touch, and Touch Hold controls operate in one normalized configuration distribution.
- `stable` is the default search. `causal-v1` is an optional bounded recovery mode for EXPERT and above; both modes use the same planner, renderer, and Harness.
- A chart is written only after full-chart CUDA evaluation returns ACCEPT and a permit is bound to the exact content digest.

## Requirements

- Windows 10/11
- NVIDIA CUDA GPU
- A CUDA-enabled PyTorch build compatible with the installed driver
- Python 3.11 or newer
- FFmpeg is bundled in both the formal Git repository (via Git LFS) and the release; no separate installation or `PATH` setup is required

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

Double-click `启动生成器.pyw` to open the local Studio. Audio and media assets remain on the computer.

The form requires audio, title, BPM, chart version, difficulty, and exact internal level. Song ID is optional: automatic IDs start at 3000 before DX and 13000 from DX onward, and are persisted without reuse. Cover and BGA MP4 are optional.
- Choosing or uploading a new audio clears the previous cover and BGA selections; the title still defaults to the audio filename.
- Automatic song-ID allocation uses a cross-process lock, so simultaneous Studio instances do not receive the same ID.

## Bundled FFmpeg and optional preview tools

Both the formal Git repository (via Git LFS) and the release ZIP bundle the Windows x64 FFmpeg runtime. A normal LFS-enabled clone materializes `tools/ffmpeg/ffmpeg.exe`; if LFS was skipped during clone, run `git lfs pull`. MiaCode and MajdataViewX remain optional external tools; extract them into the application's `tools` directory if desired. The runtime recognizes:

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

- FFmpeg is tracked in the formal Git repository via LFS and bundled in releases at `tools/ffmpeg/ffmpeg.exe`. The runtime prefers this bundled binary and accepts a system `PATH` installation only as a compatibility fallback.
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

- A video with an audio track can be selected once as both music and BGA. Publication splits it into a normalized 320 kbps `track.mp3` and a video-only `pv.mp4`; the video stream is copied without re-encoding.
MiaCode and MajdataViewX are optional external tools exposed as two separate actions in the Studio. If MajdataViewX is unavailable, the action reports the installation locations instead of opening a built-in preview. Install it at `tools/MajdataViewX-v6.2.0` or `.tools/MajdataViewX-v6.2.0`, or set `MAJDATA_EXE` to the `MajdataEdit-Neo.exe` path.
- Cover and BGA controls use fixed-size drop zones. BGA preview is a static thumbnail extracted by the bundled FFmpeg, so the browser never needs to decode the original MP4; source media is never modified, and the final video-only `pv.mp4` remuxes the original video stream without transcoding. Clear buttons remove only the current selection and never delete the source file.

This is an unofficial project and is not affiliated with SEGA.
