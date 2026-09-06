from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .audio import decode_mp3

def detect_bpm(path:Path,ffmpeg:Path,min_bpm:float=55,max_bpm:float=220)->dict:
    wave=decode_mp3(Path(path),ffmpeg);sr=16000;hop=160;window=torch.hann_window(640)
    # Analyse only the middle 90%; the original audio/generation timeline is untouched.
    total_samples=len(wave);start=total_samples//20;end=total_samples-start
    wave=wave[start:end]
    if len(wave)<2*sr:
        raise ValueError('音频中间90%不足2秒，无法可靠识别BPM，请手动填写')
    if not torch.isfinite(wave).all() or float(wave.abs().max())<1e-7:
        raise ValueError('音频中间90%无有效声音，请手动填写BPM')
    spec=torch.stft(wave,n_fft=1024,hop_length=hop,win_length=640,window=window,return_complex=True).abs().log1p()
    onset=torch.relu(spec[:,1:]-spec[:,:-1]).mean(0);onset=(onset-onset.mean()).clamp_min(0);x=onset.numpy()
    if float(x.std())<1e-8:
        raise ValueError('音频中间90%未检测到有效节奏变化，请手动填写BPM')
    x=x/(x.std()+1e-8)
    n=1<<(2*len(x)-1).bit_length();fft=np.fft.rfft(x,n=n);ac=np.fft.irfft(fft*np.conj(fft),n=n)[:len(x)];ac/=np.maximum(1,np.arange(len(x),0,-1))
    bpms=np.linspace(min_bpm,max_bpm,3301);lags=np.rint(60/(bpms*(hop/sr))).astype(int);valid=(lags>1)&(lags<len(ac));scores=np.full_like(bpms,-np.inf);scores[valid]=ac[lags[valid]]
    # Prefer a musically useful fundamental over extreme half/double choices.
    preference=np.exp(-0.5*((bpms-120)/75)**2);scores=scores*(.85+.15*preference)
    order=np.argsort(scores)[::-1];candidates=[]
    for idx in order:
        value=float(bpms[idx])
        if all(abs(value-old)>2 for old in candidates):candidates.append(value)
        if len(candidates)==5:break
    best=candidates[0]
    if best>190 and best/2>=min_bpm:best/=2
    period=max(1,int(round(60/(best*(hop/sr)))));phase_scores=np.array([x[p::period].sum() for p in range(period)]);phase=int(phase_scores.argmax());offset=(phase+1)*hop/sr
    # Phase is relative to the analysed slice; map it back to the original beat grid.
    offset=(start/sr+offset)%(60/best)
    confidence=float((scores[order[0]]-np.median(scores[np.isfinite(scores)]))/(np.std(scores[np.isfinite(scores)])+1e-8))
    return {'bpm':round(best,3),'beatOffsetSeconds':round(offset,4),'confidence':round(confidence,3),'candidates':[round(v,3) for v in candidates], 'analysisStartSeconds':start/sr,'analysisEndSeconds':end/sr,'analysisFraction':0.9}
