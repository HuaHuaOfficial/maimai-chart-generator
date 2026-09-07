from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..io.timing import ticks_to_seconds
from ..io.audio import FRAME_SECONDS, extract_log_mel
from ..io.simai import parse_maidata, render_compact_maidata
from ..generator.frontend import NativeFrontend
from ..generator.model_store import available_models,model_spec_from_renderer_path,load_models,ModelSpec
TPB=384


_CACHE_LOCK = threading.Lock()
_AUDIO_FEATURE_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
_MODEL_SESSION_CACHE: OrderedDict[tuple, tuple] = OrderedDict()


def _file_cache_key(path: Path) -> tuple:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved).lower(), int(stat.st_size), int(stat.st_mtime_ns)


def _audio_cache_get(path: Path):
    key = _file_cache_key(path)
    with _CACHE_LOCK:
        value = _AUDIO_FEATURE_CACHE.get(key)
        if value is not None:
            _AUDIO_FEATURE_CACHE.move_to_end(key)
    return key, value


def _audio_cache_put(key: tuple, value: tuple) -> None:
    with _CACHE_LOCK:
        _AUDIO_FEATURE_CACHE[key] = value
        _AUDIO_FEATURE_CACHE.move_to_end(key)
        while len(_AUDIO_FEATURE_CACHE) > 2:
            _AUDIO_FEATURE_CACHE.popitem(last=False)


def _notify(progress: Callable[[str], None] | None, text: str) -> None:
    if progress is not None:
        progress(text)


def _asset_root(root: Path) -> Path:
    return root / "models" / "v2"


def _find_ffmpeg(root: Path) -> Path:
    candidates = (
        root / "tools" / "ffmpeg",
        root / ".tools" / "ffmpeg",
    )
    for directory in candidates:
        if directory.is_dir():
            found = next(directory.glob("**/ffmpeg.exe"), None)
            if found is not None:
                return found
    command = shutil.which("ffmpeg")
    if command:
        return Path(command)
    raise FileNotFoundError("找不到 ffmpeg.exe；请安装 FFmpeg 并加入系统 PATH。")


def _planner_batch(
    mert_bars: np.ndarray,
    bpm_ticks: np.ndarray,
    bpm_values: np.ndarray,
    version_id: int,
    difficulty_slot: int,
    level: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    bars = min(256, len(mert_bars))
    padded_mert = np.zeros((256, mert_bars.shape[1]), np.float32)
    padded_mert[:bars] = mert_bars[:bars]
    mask = np.zeros(256, np.bool_)
    mask[:bars] = True
    bpm = np.zeros(256, np.float32)
    bar_ticks = np.arange(bars, dtype=np.int64) * TPB
    indices = np.clip(
        np.searchsorted(bpm_ticks, bar_ticks, side="right") - 1,
        0,
        len(bpm_values) - 1,
    )
    bpm[:bars] = bpm_values[indices]
    boundary_seconds = ticks_to_seconds(
        np.arange(bars + 1, dtype=np.int64) * TPB,
        bpm_ticks,
        bpm_values,
    )
    bar_duration = np.zeros(256, np.float32)
    bar_duration[:bars] = np.diff(boundary_seconds).astype(np.float32)
    return {
        "mert": torch.from_numpy(padded_mert)[None].to(device),
        "bar_mask": torch.from_numpy(mask)[None].to(device),
        "bpm": torch.from_numpy(bpm)[None].to(device),
        "bar_duration": torch.from_numpy(bar_duration)[None].to(device),
        "version": torch.tensor([version_id], device=device),
        "slot": torch.tensor([difficulty_slot - 2], device=device),
        "level": torch.tensor([round(level * 10)], device=device),
    }, bars


def _write_track_mp3(audio_path: Path, output_path: Path, ffmpeg: Path) -> None:
    if audio_path.suffix.lower() == ".mp3":
        shutil.copy2(audio_path, output_path)
        return
    subprocess.run(
        [
            str(ffmpeg), "-y", "-v", "error", "-i", str(audio_path), "-vn",
            "-c:a", "libmp3lame", "-q:a", "2", str(output_path),
        ],
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _create_generation_dir(root: Path, model_name: str) -> Path:
    """Create a unique model-name plus timestamp directory."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(model_name))
    safe_name = safe_name.strip("._-")[:80] or "model"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for serial in range(1, 1000):
        suffix = "" if serial == 1 else f"_{serial:02d}"
        candidate = root / f"{safe_name}_{stamp}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"无法在 {root} 下创建唯一生成目录")


def prepare_request(
    *,
    root: Path,
    model_key: str | None = None,
    renderer_path: Path | str | None = None,
    audio_path: Path,
    output_dir: Path,
    title: str,
    version_id: int,
    version_name: str,
    levels: dict[int, float],
    bpm: float,
    exploration: float = 0.8,
    extra_metadata: dict | None = None,
    progress: Callable[[str], None] | None = None,
    inference_backend: str = "auto",
) -> dict:
    """Run a packaged or path-selected renderer without any chart dataset."""

    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    if not np.isfinite(bpm) or bpm<=0:raise ValueError('BPM必须是大于0的有限数值')
    if not levels or any(s not in (2,3,4,5,6) or not np.isfinite(ds) or not 1<=ds<=15 or abs(ds*10-round(ds*10))>1e-5 for s,ds in levels.items()):
        raise ValueError('难度需为BASIC至Re:MASTER，定数1.0–15.0且精确到0.1')
    if not np.isfinite(exploration) or exploration<0:raise ValueError('探索强度必须是非负有限数值')
    if extra_metadata is not None and not isinstance(extra_metadata,dict):raise ValueError('附加元数据必须是JSON对象')

    root = Path(root)
    if not torch.cuda.is_available():raise RuntimeError('本原生版本需要 NVIDIA CUDA')
    if inference_backend not in ('auto','cuda'):raise ValueError('本原生版本使用已验证的 CUDA 后端')
    acceleration_info = {'requested':inference_backend,'effective':'cuda','gpu':torch.cuda.get_device_name(0)}
    frontend=NativeFrontend(root)
    if progress is not None:
        progress(
            f"推理后端: 请求={acceleration_info['requested']} / "
            f"实际={acceleration_info['effective']} / "
            f"GPU={acceleration_info.get('gpu') or 'CPU'}"
        )
        if inference_backend == "tensorrt":
            progress(
                "TensorRT 首次启用需要编译；同一 GUI 进程后续生成会复用已编译模块。"
            )
    audio_path = Path(audio_path)
    output_root = Path(output_dir)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if renderer_path is not None:
        spec = model_spec_from_renderer_path(root, renderer_path)
    else:
        model_map = {spec.key: spec for spec in available_models(root)}
        if model_key not in model_map:
            raise ValueError(f"模型不可用：{model_key}")
        spec = model_map[model_key]
    asset_root = _asset_root(root)
    if not asset_root.is_dir():
        raise FileNotFoundError(f"缺少 V2 推理资源：{asset_root}")
    # These two modules intentionally keep their asset root configurable so
    # the same code works in the source tree and the standalone package.
    extra = dict(extra_metadata or {})
    experimental = True
    contextual_repair = True
    if contextual_repair:
        if not torch.cuda.is_available():raise RuntimeError('CUDA Harness 必须可用；禁止回退到CPU逐项检查')
        extra.setdefault('parallelDifficultyWorkers', 2)
    if experimental:
        acceleration_info['requestedByUser']=inference_backend
        acceleration_info['reason']='CUDA Harness is required by this runtime'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffmpeg = _find_ffmpeg(root)
    beat_offset = max(
        0.0, float(extra.get("detectedBeatOffsetSeconds", extra.get("first", 0.0)))
    )
    bpm_ticks, bpm_values = frontend.tempo_arrays(bpm, extra.get("bpmChanges", []))

    _notify(progress, "解码音频并提取 MERT 特征…")
    phase_started = time.perf_counter()
    audio_cache_key, cached_audio = _audio_cache_get(audio_path)
    if cached_audio is None:
        mert, duration, active_start, active_end = frontend.extract_mert(
            audio_path, ffmpeg, device
        )
    else:
        mert, duration, active_start, active_end, _ = cached_audio
    timings["mertSeconds"] = time.perf_counter() - phase_started
    timings["audioFeatureCacheHit"] = cached_audio is not None
    if beat_offset < active_start:
        beat_period = 60.0 / max(float(bpm), 1e-6)
        beat_offset += math.ceil((active_start - beat_offset) / beat_period) * beat_period
    usable_duration = max(0.1, active_end - beat_offset)
    raw_ticks = frontend.duration_to_ticks(usable_duration, bpm_ticks, bpm_values)
    total_ticks = max(TPB, math.ceil(raw_ticks / TPB) * TPB)

    _notify(progress, "提取小节音乐结构…")
    phase_started = time.perf_counter()
    structure, _, _, mert_bars, _ = frontend.structure_and_density(
        mert,
        duration,
        total_ticks,
        bpm_ticks,
        bpm_values,
        beat_offset,
        version_id,
        levels,
        device,
    )
    timings["structureSeconds"] = time.perf_counter() - phase_started
    bars = min(256, len(structure), len(mert_bars))
    structure = structure[:bars].astype(np.float32, copy=False)
    mert_bars = mert_bars[:bars].astype(np.float32, copy=False)
    total_ticks = min(total_ticks, bars * TPB)
    if raw_ticks > total_ticks:
        _notify(progress, "音频超过 256 小节；推理包仅生成前 256 小节。")

    _notify(progress, f"加载 {spec.label}…")
    phase_started = time.perf_counter()
    planner, factor_session, model_cache_hit = load_models(
        spec, device, acceleration_info["effective"]
    )
    timings["modelLoadSeconds"] = time.perf_counter() - phase_started
    timings["modelSessionCacheHit"] = model_cache_hit
    duration_calibration_path = asset_root / "duration_density_calibration.json"
    workload_calibration_path = asset_root / "difficulty_workload_calibration.json"
    duration_scale = 1.0
    duration_calibration = None
    workload_calibration = None
    if duration_calibration_path.is_file():
        duration_calibration = json.loads(duration_calibration_path.read_text(encoding="utf8"))
    if workload_calibration_path.is_file():
        workload_calibration = json.loads(workload_calibration_path.read_text(encoding="utf8"))

    workload_mode = str(extra.get("difficultyWorkloadMode", "auto")).strip().lower()
    def workload_target(level_value: float):
        if workload_calibration is None:
            return None
        levels_doc = workload_calibration.get("levels", {})
        key = str(int(round(float(level_value) * 10)))
        entry = levels_doc.get(key)
        if entry is None and levels_doc:
            nearest = min(levels_doc, key=lambda value: abs(int(value) - int(round(float(level_value) * 10))))
            entry = levels_doc[nearest]
        if entry is None:
            return None
        canonical = float(np.median(bpm_values))
        while canonical < 75.0: canonical *= 2.0
        while canonical >= 150.0: canonical /= 2.0
        tempo = "slow" if canonical < 95 else "mid" if canonical < 120 else "fast" if canonical < 135 else "veryfast"
        chosen = entry.get("tempo", {}).get(tempo, entry)
        requested = workload_mode if float(level_value) >= 13.0 else "auto"
        if requested in {"note", "slide"}: chosen = chosen.get("modes", {}).get(requested, chosen)
        return chosen, canonical, tempo
    phase_started = time.perf_counter()
    density = np.zeros((bars, 5), np.float32)
    plan_info: dict[str, dict] = {}
    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        for slot, level in sorted(levels.items()):
            if not 2 <= slot <= 6:
                continue
            batch, _ = _planner_batch(
                mert_bars,
                bpm_ticks,
                bpm_values,
                version_id,
                slot,
                level,
                device,
            )
            plan = planner(batch)
            raw_total = float(plan["total"][0])
            raw_value = plan["expected"][0, :bars].float().cpu().numpy()
            difficulty_target = workload_target(level)
            target_event_rate = None
            difficulty_blend = 0.0
            if difficulty_target is not None:
                chosen_target, canonical_bpm, tempo_family = difficulty_target
                target_event_rate = float(chosen_target["eventRate"])
                calibrated_total = max(1.0, target_event_rate * usable_duration)
                if duration_calibration is not None:
                    lower = float(duration_calibration["lowerSeconds"]); upper = float(duration_calibration["upperSeconds"])
                    difficulty_blend = 1.0 if (usable_duration < lower or usable_duration > upper) else 0.5
                else:
                    difficulty_blend = 0.5
                target_total = math.exp((1.0 - difficulty_blend) * math.log(max(raw_total, 1e-6)) + difficulty_blend * math.log(calibrated_total))
                duration_scale = target_total / max(raw_total, 1e-6)
            elif duration_calibration is not None:
                lower = float(duration_calibration["lowerSeconds"]); upper = float(duration_calibration["upperSeconds"])
                reference_duration = float(np.clip(usable_duration, lower, upper))
                duration_scale = float(usable_duration / max(reference_duration, 1e-6))
                target_total = float(raw_total * duration_scale)
            else:
                duration_scale = 1.0
                target_total = raw_total
            active_tick_limit = max(1, min(int(math.floor(raw_ticks)), int(total_ticks)))
            bar_starts = np.arange(bars, dtype=np.float64) * TPB
            active_fraction = np.clip((active_tick_limit - bar_starts) / TPB, 0.0, 1.0).astype(np.float32)
            weighted_profile = np.maximum(raw_value, 0.0) * active_fraction
            if float(weighted_profile.sum()) > 1e-8:
                profile = weighted_profile / float(weighted_profile.sum())
            else:
                profile = active_fraction / max(float(active_fraction.sum()), 1e-8)
            profile_blend = 1.0
            if duration_calibration is not None and usable_duration < float(duration_calibration["lowerSeconds"]):
                lower = float(duration_calibration["lowerSeconds"])
                profile_blend = float(np.clip(usable_duration / max(lower, 1e-6), 0.0, 1.0))
                uniform_profile = active_fraction / max(float(active_fraction.sum()), 1e-8)
                profile = profile_blend * profile + (1.0 - profile_blend) * uniform_profile
            value = profile * target_total
            density[:, slot - 2] = value
            plan_info[str(slot)] = {
                "mean": float(value.mean()),
                "cv": float(value.std() / (value.mean() + 1e-8)),
                "total": float(target_total),
                "rawTotal": raw_total,
                "durationScale": float(duration_scale),
                "difficultyRateTarget": target_event_rate,
                "difficultyBlend": float(difficulty_blend),
                "shortProfileBlend": float(profile_blend),
                "workloadMode": workload_mode if float(level) >= 13.0 else "auto",
            }
            if "sectionSwitchProbability" in plan:
                plan_info[str(slot)]["sectionSwitchProbability"] = float(
                    plan["sectionSwitchProbability"][0]
                )
    timings["plannerSeconds"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    if cached_audio is None:
        mel_all = extract_log_mel(audio_path, ffmpeg).numpy().astype(np.float32)
        _audio_cache_put(
            audio_cache_key,
            (mert, duration, active_start, active_end, mel_all),
        )
    else:
        mel_all = cached_audio[4]
    offset_frames = min(len(mel_all) - 1, max(0, int(round(beat_offset / FRAME_SECONDS))))
    mel = mel_all[offset_frames:]
    timings["melSeconds"] = time.perf_counter() - phase_started
    metadata = {
        "title": title,
        "artist": extra.get("artist", "AI Generated"),
        "first": beat_offset,
        "wholebpm": bpm,
        "shortid": extra.get("shortid", 0),
        "genre": extra.get("genre", ""),
        "versionid": version_id,
        "version": version_name,
        "clock_count": 4,
        "difficultyWorkloadMode": workload_mode,
        # This is a direct target ratio against the calibrated official-star
        # reference.  It is deliberately not a "fill the gap from the first
        # draft" ratio: the first draft is model-dependent and made the GUI
        # slider ineffective whenever the model already landed near the
        # reference count.
        "starTargetRatio": float(np.clip(extra.get("starTargetRatio", 0.5), 0.0, 1.0)),
        "chartEndSeconds": min(float(duration - beat_offset), float(ticks_to_seconds(np.asarray([total_ticks]), bpm_ticks, bpm_values)[0])),
        **{f"lv_{slot}": value for slot, value in levels.items()},
    }
    _notify(progress, "预测音乐锚点和全曲风格…")
    phase_started = time.perf_counter()
    anchor_logits = frontend.predict_anchor_logits(
        mel, structure, bpm_ticks, bpm_values, version_id, levels, device
    )
    audio_end_tick = max(1, min(int(math.floor(raw_ticks)), int(total_ticks)))
    # The contextual hybrid keeps the native Sequence timing distribution.
    # The joint planner is used later only to rank bounded intent additions.
    anchors = frontend.select_anchors(
        anchor_logits,
        structure,
        density,
        levels,
        max_tick=audio_end_tick,
    )
    timings["anchorSeconds"] = time.perf_counter() - phase_started
    style_results: dict[str, dict] = {}
    inotes: dict[int, str] = {}
    chart_results: list[dict] = []
    change_map = {int(tick): float(value) for tick, value in zip(bpm_ticks, bpm_values)}
    base_creativity = {2: 0.08, 3: 0.16, 4: 0.30, 5: 0.48, 6: 0.58}
    seed = int(extra.get("generationSeed", 20260901))
    style_seconds = 0.0
    decode_seconds = 0.0
    slot_inputs = []
    for slot, level in sorted(levels.items()):
        creativity = float(np.clip(base_creativity[slot] * max(0.1, exploration), 0, 0.95))
        phase_started = time.perf_counter()
        style_vector, style_info = frontend.predict_style_latent(
            structure,
            version_id,
            slot,
            level,
            bpm,
            metadata,
            creativity,
            seed + slot * 1009,
            device,
        )
        style_seconds += time.perf_counter() - phase_started
        style_results[str(slot)] = style_info
        slot_inputs.append((slot, level, style_vector))

    return dict(root=root, spec=spec, audio_path=audio_path, output_root=output_root,
        title=title, version_id=version_id, version_name=version_name, levels=levels,
        bpm=bpm, bpm_ticks=bpm_ticks, bpm_values=bpm_values, beat_offset=beat_offset,
        duration=duration, total_ticks=total_ticks, audio_end_tick=audio_end_tick,
        ffmpeg=ffmpeg, mel=mel, structure=structure, mert=mert, mert_bars=mert_bars,
        metadata=metadata, extra=extra, device=device, factor_session=factor_session,
        anchors=anchors, anchor_logits=anchor_logits, slot_inputs=slot_inputs,
        plan_info=plan_info, style_results=style_results, timings=timings,
        acceleration_info=acceleration_info, seed=seed, experimental=experimental,
        end_seconds=metadata['chartEndSeconds'], total_started=total_started)
