# maimai Chart Studio 1.1.0 package boundary

The release contains only:

- `启动生成器.pyw`;
- `src/chart_runtime/`, excluding bytecode and unused historical modules;
- the fixed production model chain and required inference/calibration assets under `models/`;
- the bundled FFmpeg runtime under `tools/ffmpeg/`;
- `README.md`, `README.en.md`, `RELEASE_NOTES.md`, `PACKAGE_CONTENTS.md`;
- `requirements.txt`, `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.

The release excludes:

- `backups/`, `generated/`, `logs/`, `.work/`, tests, experiments, audits, and caches;
- optional editor/viewer binaries under `tools/` or `.tools/`;
- historical deployment receipts and stage-specific Markdown files;
- obsolete checkpoint selectors, backend selectors, unversioned profile adapters, unused timed-window prototypes, and failed ranker candidates;
- `.git/` from the downloadable archive.

FFmpeg is tracked by the formal Git repository via Git LFS and is bundled in releases at `tools/ffmpeg/ffmpeg.exe`; system `PATH` is only a fallback. A normal LFS-enabled clone materializes the same binary. MiaCode and MajdataViewX remain optional external tools.
