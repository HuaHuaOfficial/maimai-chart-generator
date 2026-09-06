import torch
from torch import nn
from chart_runtime.generator.models.anchor import AnchorRelationModel


class JointEventPlanModel(nn.Module):
    def __init__(self,d=256):
        super().__init__();self.anchor=AnchorRelationModel(d);self.timing_feature=nn.Linear(1,d)
        self.intent=nn.ModuleList([nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU()) for _ in range(3)])
        self.arity=nn.ModuleList([nn.Linear(d,3) for _ in range(3)]);self.stars=nn.ModuleList([nn.Linear(d,3) for _ in range(3)])
        self.holds=nn.ModuleList([nn.Linear(d,3) for _ in range(3)]);self.touches=nn.ModuleList([nn.Linear(d,34) for _ in range(3)])
        self.tap_slide=nn.ModuleList([nn.Linear(d,1) for _ in range(3)])
    def forward(self,audio,structure,version,levels):
        pos=torch.arange(384,device=audio.device);base=self.anchor.encoder(self.anchor.audio(audio)+self.anchor.tick(pos)[None]+self.anchor.structure(structure)[:,None]+self.anchor.version(version)[:,None]);timing=[];outputs=[]
        for i,slot_index in enumerate((2,3,4)):
            branch=self.anchor.heads[slot_index](base) if slot_index<3 else self.anchor.master_shared(base)
            hidden=torch.nn.functional.gelu(branch+self.anchor.level(levels[:,i].clamp(0,200))[:,None]);logit=self.anchor.out[slot_index](hidden).squeeze(-1);timing.append(logit)
            z=self.intent[i](base+hidden+self.timing_feature(logit[...,None]))
            outputs.append({'arity':self.arity[i](z),'stars':self.stars[i](z),'holds':self.holds[i](z),'touches':self.touches[i](z),'tap_slide':self.tap_slide[i](z).squeeze(-1)})
        return torch.stack(timing,-1),outputs
