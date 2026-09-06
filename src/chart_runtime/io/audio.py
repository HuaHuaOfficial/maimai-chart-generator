from __future__ import annotations

import math
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


SAMPLE_RATE = 16_000
N_FFT = 1024
WIN_LENGTH = 640
HOP_LENGTH = 40
N_MELS = 80
FRAME_SECONDS = HOP_LENGTH / SAMPLE_RATE


def _hz_to_mel(value: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)


@lru_cache(maxsize=4)
def mel_filterbank(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> torch.Tensor:
    min_mel = _hz_to_mel(torch.tensor(30.0))
    max_mel = _hz_to_mel(torch.tensor(sample_rate / 2.0))
    points = _mel_to_hz(torch.linspace(min_mel, max_mel, n_mels + 2))
    bins = torch.floor((n_fft + 1) * points / sample_rate).long().clamp(0, n_fft // 2)
    filters = torch.zeros(n_mels, n_fft // 2 + 1)
    for index in range(n_mels):
        left, center, right = bins[index:index + 3].tolist()
        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2)
        if center > left:
            filters[index, left:center] = torch.arange(center - left) / (center - left)
        if right > center:
            filters[index, center:right] = torch.arange(right - center, 0, -1) / (right - center)
    return filters


def decode_mp3(path: Path, ffmpeg: Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    command = [
        str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-3000:])
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if samples.size == 0:
        raise RuntimeError(f"decoded no samples from {path}")
    return torch.from_numpy(samples)


def waveform_to_log_mel(waveform: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(WIN_LENGTH, dtype=waveform.dtype)
    spectrum = torch.stft(
        waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=window, center=True, return_complex=True,
    )
    power = spectrum.abs().square()
    mel = mel_filterbank().to(power) @ power
    mel = torch.log1p(mel).transpose(0, 1)
    mean = mel.mean(dim=0, keepdim=True)
    std = mel.std(dim=0, keepdim=True).clamp_min(1e-4)
    return ((mel - mean) / std).contiguous()


def extract_log_mel(path: Path, ffmpeg: Path) -> torch.Tensor:
    return waveform_to_log_mel(decode_mp3(path, ffmpeg))
