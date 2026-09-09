# maimai Chart Studio 1.0.0 package boundary

The release contains only:

- `启动生成器.pyw`;
- `src/chart_runtime/`, excluding bytecode and unused historical modules;
- the fixed production model chain and required inference/calibration assets under `models/`;
- `README.md`, `README.en.md`, `RELEASE_NOTES.md`, `PACKAGE_CONTENTS.md`;
- `requirements.txt`, `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.

The release excludes:

- `backups/`, `generated/`, `logs/`, `.work/`, tests, experiments, audits, and caches;
- optional editor/viewer/FFmpeg binaries under `tools/` or `.tools/`;
- historical deployment receipts and stage-specific Markdown files;
- obsolete checkpoint selectors, backend selectors, unversioned profile adapters, unused timed-window prototypes, and failed ranker candidates;
- `.git/` from the downloadable archive.

FFmpeg is an external prerequisite on `PATH`. MiaCode and MajdataViewX are optional external tools.
