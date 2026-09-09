"""GUI and CLI entry for the Model/Generator <-> shared CUDA Harness runtime."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json
import math
import time
import uuid
import torch

from .preparation import prepare_request,_create_generation_dir
from ..domain import Budget,Definition,GenerationRequest
from ..runtime.payloads import Envelope
from ..runtime.session import run_session
from ..runtime.resources import get_workspace
from ..io.codec import Codec
from ..io.publish import publish
from ..generator.renderer import RenderContext
from ..generator.backend import GeneratorBackend
from ..harness.backend import HarnessBackend
from ..harness.calibration import load_calibration
from ..harness.star_policy import target_stars


def _official_stars(root,ds,bpm,event_count):
    document=json.loads((root/'models/v2/difficulty_workload_calibration.json').read_text(encoding='utf8'))['levels']
    key=min(document,key=lambda k:abs(int(k)-round(ds*10)));entry=document[key]
    canonical=bpm
    while canonical<75:canonical*=2
    while canonical>=150:canonical/=2
    tempo='slow' if canonical<95 else 'mid' if canonical<120 else 'fast' if canonical<135 else 'veryfast'
    selected=entry.get('tempo',{}).get(tempo,entry)
    return round(event_count*float(selected['slideRate'])/max(float(selected['eventRate']),1e-6))


def generate(**kwargs):
    start=time.perf_counter();prepared=prepare_request(**kwargs);root=prepared['root'];progress=kwargs.get('progress') or (lambda _:None)
    release_folder=_create_generation_dir(prepared['output_root'],prepared['title'])
    folder=release_folder/prepared['song_id']
    completed={};codecs={};extra=prepared['extra'];workers=int(extra.get('parallelDifficultyWorkers',2)) if extra.get('parallelDifficulties',True) else 1
    workers=max(1,min(2,workers,len(prepared['slot_inputs'])))
    root_src=Path(__file__).resolve().parents[1]
    rule_files=('harness/kernel.py','harness/backend.py','harness/fused.py','harness/source_head.py','harness/slide_queue.py','harness/sampling.py','harness/durations.py','io/codec.py','io/tensor_pack.py','harness/snapshot_fast.py','io/symmetry.py','io/durations.py')
    preference_assets=(root/'models/experimental/one_hand_motion_preference.json',root/'models/experimental/slide_entry_motion_preference.json',root/'models/experimental/muri_policy.json')
    calibration_bytes=(root/'models/experimental/contextual_calibration.npz').read_bytes()+b''.join(asset.read_bytes() for asset in preference_assets if asset.is_file())
    definition=Definition('chart-ir/1',sha256(b''.join((root_src/name).read_bytes() for name in rule_files)).hexdigest(),sha256((root_src/'harness/features.py').read_bytes()).hexdigest(),
                          sha256(calibration_bytes).hexdigest(),sha256((root/'models/v2/playability_tables.json').read_bytes()).hexdigest())
    def run_slot(item):
        slot,ds,style=item;codec=Codec(root);codecs[slot]=codec
        request_id=str(uuid.uuid4());calibration=load_calibration(str(root),prepared['version_id'],slot,round(ds*10),prepared['bpm'])
        star_scale=float(prepared['metadata']['whatStarScale'])
        official_stars=_official_stars(root,ds,prepared['bpm'],len(prepared['anchors'][slot]))
        star_plan_stars=target_stars(official_stars,1.0,len(prepared['anchors'][slot]))
        conditions={'bpm':prepared['bpm'],'end_seconds':prepared['end_seconds'],
                    'bt':prepared['bpm_ticks'],'bv':prepared['bpm_values'],
                    'official_stars':official_stars,'star_scale':star_scale,
                    'star_plan_stars':star_plan_stars,'display_ds':float(ds)}
        request=GenerationRequest(request_id,prepared['version_id'],slot,round(ds*10),prepared['seed']+slot*65537,definition,Envelope(conditions,torch.tensor([ds,prepared['bpm']],device='cuda'),'generation-conditions/1'))
        ctx=RenderContext(root,prepared['anchors'][slot],prepared['mel'],prepared['structure'],prepared['bpm_ticks'],prepared['bpm_values'],prepared['version_id'],slot,ds,
                          prepared['metadata'],torch.device('cuda'),prepared['spec'].renderer_checkpoint,style,.85,request.seed,prepared['factor_session'],progress)
        harness=HarnessBackend(codec,calibration,conditions);generator=GeneratorBackend(codec,ctx,harness,prepared['anchor_logits'][:,:,slot-2],prepared['mert_bars'])
        budget=Budget(int(extra.get('maxFeedbackRounds',12)),4,128*1024*1024)
        progress(f"难度 {slot}: DS={ds:.1f}")
        progress(f'难度 {slot}: 新运行时，生成段 ↔ CUDA Harness')
        try:
            result=run_session(request,generator,harness,budget)
        except Exception as exc:
            details=exc.as_dict() if hasattr(exc,'as_dict') else {'message':str(exc),'type':type(exc).__name__}
            (release_folder/f'failure_{slot}.json').write_text(json.dumps(details,ensure_ascii=False,indent=2,default=str),encoding='utf8')
            raise
        if result.state!='accepted':raise RuntimeError(f'难度{slot}未取得整谱许可：{result.state}；{folder}')
        if generator.relational_plan is not None:
            binding=generator.context.cache.get('relational_what_binding')
            if not isinstance(binding,dict) or result.chart is None or binding.get('chartDigest')!=result.chart.ref.content_digest:
                raise RuntimeError(f'难度{slot}的关系审计未绑定到最终 Harness 谱面摘要；{folder}')
        progress(f'难度 {slot}: 全谱通过，反馈 {len(result.observations)-1} 轮')
        return slot,(result,harness,generator,request)
    phase=time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for slot,value in pool.map(run_slot,prepared['slot_inputs']):completed[slot]=value
    prepared['timings'].update(decodeSeconds=time.perf_counter()-phase,parallelDifficulties=workers>1,parallelDifficultyWorkers=workers)
    prepared['timings']['workspace']=vars(get_workspace('cuda:0').telemetry())
    prepared['timings']['totalSeconds']=time.perf_counter()-start
    result=publish(prepared,completed,codecs,folder,release_folder)
    result['timings']['totalSeconds']=time.perf_counter()-start
    metadata_path=release_folder/'元数据.json'
    metadata_pending=release_folder/'元数据.pending.json'
    metadata_pending.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    metadata_pending.replace(metadata_path)
    return result
