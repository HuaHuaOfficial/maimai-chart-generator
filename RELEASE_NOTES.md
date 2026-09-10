# maimai Chart Studio 1.1.0

## Formal release boundary

1.1.0 is the complete package built from the current native production runtime. It includes the full 0.4.0-to-1.0.0 feature line and the current production fixes; it is not a reduced compatibility patch.

Version 1.0.1 was an unreleased transition node. Its complete working-tree changes are folded directly into 1.1.0; there is no separate 1.0.1 release artifact.

- This complete package includes the two-tier difficulty policy: lightweight V4 combined sampling for BASIC/ADVANCED and joint WHAT plus relational WHERE for EXPERT/MASTER/Re:MASTER.
- It includes cross-process automatic song-ID allocation, cover/BGA preview and clear controls, local token-protected media streaming, and new-audio asset reset behavior.

- MiaCode and MajdataViewX are now separate peer actions; the unreliable built-in chart preview fallback has been removed. MajdataViewX remains installable through the documented `tools/` / `.tools/` locations or `MAJDATA_EXE`.
- The Studio now keeps cover and BGA drop zones fixed-size below the title, BPM, and song ID fields. BGA preview uses a cached FFmpeg-extracted static frame instead of browser video decoding; the original MP4 is preserved unchanged. Clear buttons only clear the selection.

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
- Choosing or uploading a new audio clears the previous cover and BGA selections while keeping the filename-based title default.

## External tools

FFmpeg is now bundled as a Windows x64 runtime. MiaCode and MajdataViewX remain optional external packages. The following paths are used:

```text
tools/ffmpeg/ffmpeg.exe
tools/MiaCode-v1.0.0-win64/MiaCode.exe
tools/MajdataViewX-v6.2.0/MajdataEdit-Neo.exe
```

The bundled FFmpeg at `tools/ffmpeg/ffmpeg.exe` is preferred; recursive discovery below `tools/ffmpeg` and the system `PATH` remain compatibility fallbacks. MiaCode and MajdataViewX are checked before the Studio asks the user to choose a `maidata.txt` file. The built-in chart preview has been removed.

## WHEN edge calibration

- A locally active first bar is no longer left empty solely because the whole-song density planner assigns an anomalously near-zero first-bar expectation; a bounded continuity calibration uses only neighboring opening bars.
- A genuine sustained fade-out is detected from local audio energy and progressively lowers only the ending WHEN budget. Quiet interior sections and non-fade endings are not globally loudness-scaled.
- The regression cases `390982371_nb2-1-30280` and `TheFatRat - Unity` both completed with HARD=0 and QUALITY=0 after this change.

## Reliability and operation

- WHEN edge calibration now repairs anomalously empty first bars when adjacent bars have matching audio activity, and suppresses density only across a detected sustained end fade; interior sections are unchanged.
- Studio audio preview now uses a normalized 320 kbps MP3 cache, so mislabeled or browser-incompatible audio remains playable; the same cached MP3 is reused verbatim as the published `track.mp3`.
- BGA preview now extracts a cached static frame with bundled FFmpeg instead of asking the browser to decode the original MP4; the original BGA is still copied unchanged to `pv.mp4`.
- The localhost media Range path now imports and handles `re` correctly, fixing `<audio>` requests that previously returned HTTP 400.

- BPM detection and generation can be cancelled from the Studio; the worker process tree is terminated while existing completed files are preserved.
- Native sampling exhaustion is returned to the bounded resume path instead of terminating the whole job as `no allowed factor ids`.
- Native file dialogs use a short-lived topmost owner so they remain visible above the browser-based Studio.

## Current Harness

The release includes shared CUDA rules for complete Slide contacts and queues, version capability, dynamic hand accounting, 180 Hz release semantics, source-head Tap/Hold lifecycle, Track tail windows, and bounded causal recovery. The full-chart Harness and bound permit remain the publication authority.

## Claim boundary

The current version-conditioned joint profile has not been shown to suppress Touch-on-Slide: in the matched Koi comparison, its conditional ratio was not lower than the earlier profile. The confirmed historical discrepancy was request-identity drift caused by omitting `artist`; 1.1.0 does not add a Touch-on-Slide reward.

## Validation

- BASIC/ADVANCED low-head validation loss improved from 2.047 to 1.187.
- The migrated EXPERT/MASTER/Re:MASTER weights and real-bar logits have maximum absolute difference 0.0 from the prior production checkpoint.
- A single Home Street run generated all five difficulties with zero feedback rounds, HARD=0, and QUALITY=0 for every chart.
- The clean release tree independently generated BASIC 5.0 with HARD=0 and QUALITY=0.
- A blank pre-DX song ID produced `3000`; with no cover or BGA selected, the song directory contained only `maidata.txt` and `track.mp3`.
- Python compilation, Web UI syntax, joint/sequence/track tests, CUDA Slide queue, CUDA incremental Muri, motion, compound routes, versioned percentile, and causal recovery checks passed.

## Release contents

Only runtime source, the fixed model/assets, bundled FFmpeg runtime, launcher, user documentation, dependency manifest, and legal notices are shipped. Experiments, generated outputs, backups, optional editor/viewer tools, caches, failed rankers, old interfaces, and stage-by-stage Markdown records are excluded.
