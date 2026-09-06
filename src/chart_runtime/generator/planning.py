"""Native joint intent model used inside generation, never as a judge."""
import threading
from pathlib import Path
import numpy as np
import torch
from .models.joint_plan import JointEventPlanModel
from ..io.timing import sample_positions
from ..io.audio import FRAME_SECONDS

_LOCK=threading.Lock()
_MODELS={}


def star_scores(context):
    path=Path(context.root)/'models/experimental/joint_plan.pt';key=(str(path),path.stat().st_mtime_ns)
    with _LOCK:
        if key not in _MODELS:
            model=JointEventPlanModel();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'],strict=True);model.cuda().eval()
            _MODELS.clear();_MODELS[key]=model
        model=_MODELS[key]
    result=np.zeros((len(context.structure),384),np.float32)
    index=context.slot-4
    if index not in (0,1,2):return result
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for begin in range(0,len(context.structure),32):
            end=min(begin+32,len(context.structure));bars=list(range(begin,end))
            audio=np.stack([sample_positions(context.mel,b*384+np.arange(384),context.bt,context.bv,FRAME_SECONDS) for b in bars])
            levels=torch.zeros((len(bars),3),dtype=torch.int64,device='cuda');levels[:,index]=round(context.level*10)
            _,out=model(torch.from_numpy(audio).cuda(),torch.from_numpy(context.structure[begin:end]).cuda(),torch.full((len(bars),),context.version,dtype=torch.int64,device='cuda'),levels)
            probability=out[index]['stars'].float().softmax(-1)
            expected=probability[...,1]+2*probability[...,2]
            result[begin:end]=expected.cpu().numpy()
    return result
