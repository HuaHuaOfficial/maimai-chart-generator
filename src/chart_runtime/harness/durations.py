"""CUDA-vectorized duration support; strings are encoded once as data."""
from functools import lru_cache
import numpy as np
import torch


def _ratio(text):
    left,right=str(text).split(':',1)
    return 240.*float(right)/float(left)


@lru_cache(maxsize=8)
def _encoded(durations):
    # mode: 0 invalid, 1 constant seconds, 2 coefficient/current BPM.
    hm=[];hv=[];wm=[];wv=[];mm=[];mv=[]
    for signature in durations:
        hmode=hvalue=wmode=wvalue=mmode=mvalue=0.
        try:
            text=str(signature)
            if text and not text.startswith('<'):
                if '#' not in text:hmode,hvalue=2.,_ratio(text)
                else:
                    left,right=text.split('#',1)
                    if not left:hmode,hvalue=1.,float(right)
                    elif ':' in right:hmode,hvalue=1.,_ratio(right)/float(left)
                if '##' in text:
                    wait_text,move_text=text.split('##',1);wmode,wvalue=1.,float(wait_text)
                    if '#' in move_text:
                        tempo,ratio=move_text.split('#',1);mmode,mvalue=1.,_ratio(ratio)/float(tempo)
                    elif ':' in move_text:mmode,mvalue=2.,_ratio(move_text)
                    else:mmode,mvalue=1.,float(move_text)
                elif '#' in text:
                    tempo,move_text=text.split('#',1);wmode,wvalue=1.,60./float(tempo)
                    if ':' in move_text:mmode,mvalue=1.,_ratio(move_text)/float(tempo)
                    else:mmode,mvalue=1.,float(move_text)
                else:wmode,wvalue,mmode,mvalue=2.,60.,2.,_ratio(text)
        except (ValueError,ZeroDivisionError,OverflowError):pass
        hm.append(hmode);hv.append(hvalue);wm.append(wmode);wv.append(wvalue);mm.append(mmode);mv.append(mvalue)
    return tuple(np.asarray(x,np.float64) for x in (hm,hv,wm,wv,mm,mv))


def typed_duration_values(durations,bpm):
    if not torch.cuda.is_available():raise RuntimeError('Duration support requires CUDA; CPU fallback is forbidden')
    hm,hv,wm,wv,mm,mv=(torch.as_tensor(x,device='cuda') for x in _encoded(tuple(durations)))
    tempo=torch.as_tensor(float(bpm),device='cuda',dtype=torch.float64)
    hold=torch.where(hm==1,hv,torch.where(hm==2,hv/tempo,torch.zeros_like(hv)))
    wait=torch.where(wm==1,wv,torch.where(wm==2,wv/tempo,torch.full_like(wv,float('nan'))))
    move=torch.where(mm==1,mv,torch.where(mm==2,mv/tempo,torch.zeros_like(mv)))
    return hold,wait,move


def typed_duration_support(durations,bpm):
    hold,wait,move=typed_duration_values(durations,bpm)
    return torch.isfinite(hold)&(hold>0),torch.isfinite(wait)&torch.isfinite(move)&(wait>=0)&(move>0)
