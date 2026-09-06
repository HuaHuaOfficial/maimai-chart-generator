from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
@dataclass
class V2Config:
    audio_dim:int=256;d_model:int=256;max_bars:int=256;difficulty_count:int=5
    event_families:int=4;positions:int=43;slide_shapes:int=13;duration_classes:int=256
class ChartTransformerV2(nn.Module):
    """Training scaffold: global density planner + factorized event heads."""
    def __init__(self,c:V2Config):
        super().__init__();self.c=c;self.audio=nn.Linear(c.audio_dim,c.d_model);self.bar_pos=nn.Embedding(c.max_bars,c.d_model)
        layer=nn.TransformerEncoderLayer(c.d_model,8,1024,.1,'gelu',batch_first=True,norm_first=True);self.structure=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.difficulty=nn.Embedding(c.difficulty_count,c.d_model);self.density=nn.Linear(c.d_model,c.difficulty_count)
        self.density_version=nn.Embedding(32,c.d_model);self.density_level=nn.Embedding(201,c.d_model);self.density_slot=nn.Embedding(c.difficulty_count,c.d_model)
        self.density_residual=nn.ModuleList(nn.Linear(c.d_model,1) for _ in range(c.difficulty_count))
        for head in self.density_residual:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.anchor=nn.Linear(c.d_model,1);self.retain=nn.Linear(c.d_model,c.difficulty_count-1)
        def heads():return nn.ModuleDict({'family':nn.Linear(c.d_model,c.event_families),'start':nn.Linear(c.d_model,c.positions),'end':nn.Linear(c.d_model,9),'shape':nn.Linear(c.d_model,c.slide_shapes),'duration':nn.Linear(c.d_model,c.duration_classes),'modifiers':nn.Linear(c.d_model,5)})
        self.factor_heads=nn.ModuleDict({name:heads() for name in ('basic','advanced','expert','master_shared','utage')})
    def predict_density(self,x,version=None,levels=None):
        base=self.density(x)
        if version is None or levels is None:return torch.nn.functional.softplus(base)
        version_context=self.density_version(version.clamp(0,31))[:,None]
        outputs=[]
        for slot in range(self.c.difficulty_count):
            context=x+version_context+self.density_level(levels[:,slot].clamp(0,200))[:,None]+self.density_slot.weight[slot][None,None]
            outputs.append(base[:,:,slot]+self.density_residual[slot](torch.nn.functional.gelu(context)).squeeze(-1))
        return torch.nn.functional.softplus(torch.stack(outputs,dim=-1))
    def forward(self,bar_audio,bar_mask=None,version=None,levels=None):
        pos=torch.arange(bar_audio.shape[1],device=bar_audio.device);x=self.structure(self.audio(bar_audio)+self.bar_pos(pos)[None],src_key_padding_mask=bar_mask)
        factors={group:{name:head(x) for name,head in heads.items()} for group,heads in self.factor_heads.items()}
        return {'density':self.predict_density(x,version,levels),'anchor':self.anchor(x).squeeze(-1),'factors':factors,'retain':self.retain(x)}
