# maimai Chart Generator 0.3.0

English | [简体中文](README.md)

A native CUDA chart generator for maimai DX Simai charts. The current release uses a contextual V4 model for generation and one shared CUDA Harness for candidate and whole-chart checks. Only the bundled latest model set is supported; legacy checkpoints are incompatible.

## Features

- Generates BASIC through Re:MASTER charts from audio, BPM, game version, and target difficulty constants.
- Keeps Slide Stars and Tracks as separate representations and enforces explicit Star count bounds.
- Returns HARD failures to a local context window; repeated failures at the same point expand the causal edit scope without regenerating the full song.
- Writes `maidata.txt` only after a complete whole-chart permit and an identity-preserving Simai round trip.
- The Windows GUI entry point is `启动生成器.pyw`.

## Setup and launch

1. Install a CUDA-enabled PyTorch build and the Python dependencies: `pip install -r requirements.txt`.
2. Install FFmpeg separately and make sure `ffmpeg.exe` is available on PATH.
3. Double-click `启动生成器.pyw`.
4. Select the audio, version, difficulty constants, and BPM. Output is written to `generated` by default.

FFmpeg and MajdataViewX are not bundled. If MajdataViewX is installed at `tools/MajdataViewX-v6.2.0`, the Preview button will use it; otherwise the built-in chart preview is used. Built-in audio playback is available when `ffplay` is on PATH.

## Current boundaries

- NVIDIA CUDA is required; there is no CPU musical-judgement fallback.
- Generation is limited to the first 256 bars.
- Missing low-difficulty calibration is reported explicitly instead of being replaced by a synthetic threshold.
- The current MultiTouch metric is not a proof of geometric hand merging.
- Star timing prediction is currently stable, while route and placement quality remain areas for improvement.

## Licensing

Original project source code is licensed under the [Apache License 2.0](LICENSE). Bundled MERT model material is excluded from that grant and remains subject to [CC BY-NC 4.0](THIRD_PARTY_NOTICES.md), so distributions containing those weights are restricted to non-commercial use.

See the GitHub Wiki for architecture, operation, and troubleshooting details.
