"""Resident weights for the one supported native release model set."""
from dataclasses import dataclass
from pathlib import Path
import threading
import torch
from .models.factor import FactorConfig
from .models.relational import V4RelationalRenderer
from .models.planner import PersistentSectionPlanner,PersistentPlannerConfig


@dataclass(frozen=True)
class ModelSpec:
    key:str
    label:str
    planner_checkpoint:Path
    renderer_checkpoint:Path


def available_models(root):
    base=Path(root)/'models'
    return (ModelSpec('native','原生联合生成 · CUDA Harness',base/'planner_v33.pt',base/'renderer_v4_contextual.pt'),)


def model_spec_from_renderer_path(root,path):
    spec=available_models(root)[0];value=Path(path)
    if not value.is_absolute():value=Path(root)/value
    if value.resolve()!=spec.renderer_checkpoint.resolve():
        raise ValueError('此发布包只运行清单中的最新原生模型，不支持旧 checkpoint')
    return spec


_LOCK=threading.Lock()
_CACHE={}


def load_models(spec,device,backend='cuda'):
    paths=(spec.planner_checkpoint,spec.renderer_checkpoint)
    key=(tuple((str(p.resolve()),p.stat().st_mtime_ns,p.stat().st_size) for p in paths),str(device))
    with _LOCK:
        if key in _CACHE:return (*_CACHE[key],True)
        checkpoint=torch.load(paths[0],map_location='cpu',weights_only=False)
        if checkpoint.get('modelClass')!='PersistentSectionPlanner':raise ValueError('Wrong native planning architecture')
        planner=PersistentSectionPlanner(PersistentPlannerConfig(**checkpoint['config']))
        planner.load_state_dict(checkpoint['model'],strict=True);planner.to(device).eval()
        checkpoint=torch.load(paths[1],map_location='cpu',weights_only=False)
        if checkpoint.get('modelClass')!='V4RelationalRenderer' or not checkpoint.get('geometrySequenceState') or not checkpoint.get('contextualPipeline'):
            raise ValueError('Wrong native renderer architecture/capabilities')
        renderer=V4RelationalRenderer(FactorConfig(**checkpoint['config']),geometry_sequence_state=True)
        renderer.load_state_dict(checkpoint['model'],strict=True);renderer.to(device).eval()
        for head in renderer.heads:
            head.touch_hold_enabled=True;head.touch_hold_threshold=float(checkpoint['touchHoldThreshold'])
        _CACHE.clear();_CACHE[key]=(planner,(renderer,checkpoint['vocab']))
        return planner,(renderer,checkpoint['vocab']),False
