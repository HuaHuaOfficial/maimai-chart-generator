from __future__ import annotations
import math
import torch
from torch import nn

class SlideTimingPlannerV1(nn.Module):
    def __init__(self,route_count:int,move_classes:int,width:int=24,d:int=128):
        super().__init__(); self.width=width; self.d=d
        self.event=nn.Sequential(nn.Linear(7,d),nn.GELU(),nn.LayerNorm(d),nn.Linear(d,d))
        self.position=nn.Embedding(width,d)
        self.slot=nn.Embedding(7,d); self.version=nn.Embedding(64,d)
        self.ds=nn.Sequential(nn.Linear(1,d),nn.GELU(),nn.LayerNorm(d))
        self.bpm=nn.Sequential(nn.Linear(1,d),nn.GELU(),nn.LayerNorm(d))
        self.phase=nn.Sequential(nn.Linear(2,d),nn.GELU(),nn.LayerNorm(d))
        layer=nn.TransformerEncoderLayer(d,4,384,.1,'gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(d)
        self.cue=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.LayerNorm(d),nn.Linear(d,3))
        self.route=nn.Embedding(route_count,64)
        self.route_proj=nn.Sequential(nn.Linear(64,d),nn.GELU(),nn.LayerNorm(d))
        self.move=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.LayerNorm(d),nn.Linear(d,move_classes))
    def forward(self,b):
        delta=b['delta'].float().clamp(-1536,1536)/384.0
        arity=b['arity'].float()/2.0
        fam=b['families'].float()/2.0
        touch=b['touch'].float()/8.0
        accent=b['accent'].float()
        x=torch.cat((delta[...,None],arity[...,None],fam,touch[...,None],accent[...,None]),-1)
        x=self.event(x)
        pos=torch.arange(self.width,device=x.device)[None]
        phase=b['head_phase'].float()*2*math.pi/384.0
        meta=(self.slot(b['slot'].long().clamp(0,6))+self.version(b['version'].long().clamp(0,63))
              +self.ds((b['ds'].float()/15.0)[:,None])+self.bpm(torch.log1p(b['bpm'].float())[:,None]/6.0)
              +self.phase(torch.stack((phase.sin(),phase.cos()),-1)))
        x=x+self.position(pos)+meta[:,None]
        valid=b['valid'].bool()
        x=self.encoder(x,src_key_padding_mask=~valid)
        head=x[:,8]
        mean=(x*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        state=self.norm(head+0.5*mean)
        cue=self.cue(state)
        move=self.move(state+self.route_proj(self.route(b['route'].long())))
        return {'cue':cue,'move':move,'state':state}
