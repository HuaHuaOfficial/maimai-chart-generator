# maimai Chart Studio 1.0.0

## Formal release boundary

1.0.0 is the first formal package built from the current native production runtime. It removes the historical release surface instead of carrying compatibility selectors forward.

- MiaCode and MajdataViewX are now separate peer actions; the unreliable built-in chart preview fallback has been removed. MajdataViewX remains installable through the documented `tools/` / `.tools/` locations or `MAJDATA_EXE`.

## Two-tier difficulty planning

- BASIC and ADVANCED intentionally use lightweight V4 combined sampling; their production path does not invoke the heavy joint WHAT planner and remains oriented toward simple configurations.
- EXPERT, MASTER, and Re:MASTER use the joint WHAT planner followed by relational WHERE and bounded causal recovery.
- The five-slot Joint WHAT checkpoint contains dedicated heads for all slots for model capability and validation, while the production policy selects the lightweight or heavy path by difficulty.
- The old factorized WHAT fallback is not included.

## Fixed native model and backend

- Removed arbitrary model checkpoint selection from the API and UI.
- Removed `auto`/TensorRT backend inputs. CUDA is the sole supported runtime.
- Removed the deprecated `starTargetRatio` alias and unversioned WHAT-profile compatibility path.
- Missing `artist` now resolves to the GUI's explicit empty string, preventing metadata-conditioning drift between GUI and scripted requests.

## Output package

Successful generations use `乐曲名-YYYYMMDD_HHMMSS/歌曲ID/`. A blank ID is allocated persistently from 3000 before DX or 13000 from DX onward without reuse. The pure-ID directory always contains `maidata.txt` and `track.mp3`; optional selections add `bg.png` and `pv.mp4`. `元数据.json` is stored in the outer timestamped directory.
- Automatic song-ID allocation now locks the registry transaction across processes, closing the read-allocate-write race when multiple Studio instances run at once.

## External tools

FFmpeg, MiaCode, and MajdataViewX are optional external Windows x64 packages and are not bundled. Extract them under `tools/` so the following executable paths exist:

```text
tools/ffmpeg/<distribution>/bin/ffmpeg.exe
tools/MiaCode-v1.0.0-win64/MiaCode.exe
tools/MajdataViewX-v6.2.0/MajdataEdit-Neo.exe
```

FFmpeg is discovered recursively below `tools/ffmpeg` or from the system `PATH`. MiaCode and MajdataViewX are checked before the Studio asks the user to choose a `maidata.txt` file. The built-in chart preview has been removed.

## Reliability and operation

- BPM detection and generation can be cancelled from the Studio; the worker process tree is terminated while existing completed files are preserved.
- Native sampling exhaustion is returned to the bounded resume path instead of terminating the whole job as `no allowed factor ids`.
- Native file dialogs use a short-lived topmost owner so they remain visible above the browser-based Studio.

## Current Harness

The release includes shared CUDA rules for complete Slide contacts and queues, version capability, dynamic hand accounting, 180 Hz release semantics, source-head Tap/Hold lifecycle, Track tail windows, and bounded causal recovery. The full-chart Harness and bound permit remain the publication authority.

## Claim boundary

The current version-conditioned joint profile has not been shown to suppress Touch-on-Slide: in the matched Koi comparison, its conditional ratio was not lower than the earlier profile. The confirmed historical discrepancy was request-identity drift caused by omitting `artist`; 1.0.0 does not add a Touch-on-Slide reward.

## Validation

- BASIC/ADVANCED low-head validation loss improved from 2.047 to 1.187.
- The migrated EXPERT/MASTER/Re:MASTER weights and real-bar logits have maximum absolute difference 0.0 from the prior production checkpoint.
- A single Home Street run generated all five difficulties with zero feedback rounds, HARD=0, and QUALITY=0 for every chart.
- The clean release tree independently generated BASIC 5.0 with HARD=0 and QUALITY=0.
- A blank pre-DX song ID produced `3000`; with no cover or BGA selected, the song directory contained only `maidata.txt` and `track.mp3`.
- Python compilation, Web UI syntax, joint/sequence/track tests, CUDA Slide queue, CUDA incremental Muri, motion, compound routes, versioned percentile, and causal recovery checks passed.

## Release contents

Only runtime source, the fixed model/assets, launcher, user documentation, dependency manifest, and legal notices are shipped. Experiments, generated outputs, backups, tools, caches, failed rankers, old interfaces, and stage-by-stage Markdown records are excluded.
