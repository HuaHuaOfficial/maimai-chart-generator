from __future__ import annotations
import torch
from torch import nn
class AnchorRelationModel(nn.Module):
    def __init__(self,d=256):
        super().__init__();self.audio=nn.Linear(82,d);self.tick=nn.Embedding(384,d);self.structure=nn.Linear(256,d);self.version=nn.Embedding(32,d);self.level=nn.Embedding(201,d)
        layer=nn.TransformerEncoderLayer(d,8,1024,.1,'gelu',batch_first=True,norm_first=True);self.encoder=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.heads=nn.ModuleList([nn.Linear(d,d) for _ in range(4)]);self.master_shared=nn.Linear(d,d);self.out=nn.ModuleList([nn.Linear(d,1) for _ in range(5)])
    def forward(self,audio,structure,version,levels):
        pos=torch.arange(384,device=audio.device);base=self.encoder(self.audio(audio)+self.tick(pos)[None]+self.structure(structure)[:,None]+self.version(version)[:,None]);outputs=[]
        for i in range(5):
            branch=self.heads[i](base) if i<3 else self.master_shared(base);outputs.append(self.out[i](torch.nn.functional.gelu(branch+self.level(levels[:,i].clamp(0,200))[:,None])).squeeze(-1))
        return torch.stack(outputs,dim=-1)
