from __future__ import annotations

from dataclasses import asdict,dataclass
import torch
from torch import nn

@dataclass
class DensityCalibratorConfig:
    hidden:int=96
    def to_dict(self):return asdict(self)

class DensityCalibrator(nn.Module):
    def __init__(self,c:DensityCalibratorConfig=DensityCalibratorConfig()):
        super().__init__();self.c=c;self.version=nn.Embedding(32,24);self.slot=nn.Embedding(5,16);self.level=nn.Embedding(201,32);self.net=nn.Sequential(nn.Linear(24+16+32+3,c.hidden),nn.GELU(),nn.Linear(c.hidden,c.hidden),nn.GELU(),nn.Linear(c.hidden,1))
    def forward(self,version,slot,level,bpm):
        continuous=torch.stack((level.float()/150,torch.log1p(bpm)/6,bpm/300),dim=-1);x=torch.cat((self.version(version.clamp(0,31)),self.slot(slot.clamp(0,4)),self.level(level.clamp(0,200)),continuous),dim=-1);return torch.nn.functional.softplus(self.net(x).squeeze(-1))
