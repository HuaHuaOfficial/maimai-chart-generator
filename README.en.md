# maimai Chart Studio 1.0.0

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

The form requires audio, title, BPM, a custom song ID, a cover image, a BGA MP4, chart version, difficulty, and exact internal level. Version 1.0.0 no longer exposes arbitrary checkpoint or backend selectors; the release accepts only its manifest-listed native model set and CUDA backend.

## Output layout

```text
output/
└─ Song title-YYYYMMDD_HHMMSS/
   ├─ 元数据.json
   └─ CustomSongId/
      ├─ maidata.txt
      ├─ track.mp3
      ├─ bg.mp4
      └─ bg.png
```

The release archive excludes datasets, experiments, obsolete checkpoints and APIs, generated charts, logs, backups, and internal deployment notes. Official-chart profiles control composition only; expressive local structures remain model-emergent and are not manually rewarded.

This is an unofficial project and is not affiliated with SEGA.
