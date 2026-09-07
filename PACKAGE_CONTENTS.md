# 0.4.0 release boundary

The end-user archive contains only:

- `启动生成器.pyw` and `src/chart_runtime/`;
- the single native model set and its required runtime assets under `models/`;
- installation/readme/release notes and license notices.

The archive intentionally excludes:

- `tools/` (including FFmpeg, MajdataViewX, replay and audit scripts);
- `tests/`, `.git/`, `.gitignore`, and `.gitattributes`;
- `logs/`, `generated/`, `star_ratio_validation/`, caches and bytecode.

FFmpeg is an external runtime prerequisite and must be installed on `PATH`.
MajdataViewX is optional and may be installed separately at
`tools/MajdataViewX-v6.2.0` after extraction.
