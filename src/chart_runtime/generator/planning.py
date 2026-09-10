"""Native joint intent model used inside generation, never as a judge."""
import threading
import math
import json
from pathlib import Path
import numpy as np
import torch
from .models.joint_plan import JointEventPlanModel
from ..io.timing import sample_positions
from ..io.audio import FRAME_SECONDS
from .intent import EventIntent,IntentChoices
from ..version_semantics import touch_enabled, touch_sensor_capacity

_LOCK=threading.Lock()
_MODELS={}


def predict_heads(context):
    if context.slot not in (2,3,4,5,6):
        raise ValueError('Joint WHAT checkpoint supports BASIC through Re:MASTER')
    path=Path(context.root)/'models/experimental/joint_plan.pt';key=(str(path),path.stat().st_mtime_ns)
    with _LOCK:
        if key not in _MODELS:
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if checkpoint.get('modelClass')!='FiveSlotJointEventPlanModel' or checkpoint.get('slots')!=[2,3,4,5,6]:
                raise ValueError('Wrong 1.1.0 five-slot Joint WHAT checkpoint')
            model=JointEventPlanModel();model.load_state_dict(checkpoint['model'],strict=True);model.cuda().eval()
            _MODELS.clear();_MODELS[key]=model
        model=_MODELS[key]
    results={name:[] for name in ('arity','stars','holds','touches','tap_slide')}
    index=context.slot-2
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for begin in range(0,len(context.structure),32):
            end=min(begin+32,len(context.structure));bars=list(range(begin,end))
            audio=np.stack([sample_positions(context.mel,b*384+np.arange(384),context.bt,context.bv,FRAME_SECONDS) for b in bars])
            levels=torch.zeros((len(bars),5),dtype=torch.int64,device='cuda');levels[:,index]=round(context.level*10)
            _,out=model(torch.from_numpy(audio).cuda(),torch.from_numpy(context.structure[begin:end]).cuda(),torch.full((len(bars),),context.version,dtype=torch.int64,device='cuda'),levels)
            for name in results:results[name].append(out[index][name].float())
    return {name:torch.cat(values) for name,values in results.items()}


COMBINATIONS=tuple((a,s,h) for a in range(3) for s in range(a+1) for h in range(a-s+1))

def shifted_stars(heads,bias):
    result=dict(heads);count=torch.arange(3,device=heads['stars'].device)
    result['stars']=heads['stars']+float(bias)*count
    return result


def _base_score(output,allowed):
    logs={name:output[name].float().log_softmax(-1) for name in ('arity','stars','holds','touches')}
    device=logs['arity'].device;comb=torch.tensor(COMBINATIONS,device=device);a,s,h=comb.unbind(-1)
    score=logs['arity'][...,a]+logs['stars'][...,s]+logs['holds'][...,h]
    mixed=(a==2)&(s==1)&(h==0);aux=output['tap_slide'].float()-math.log(5.)
    score=score+.25*torch.where(mixed,torch.nn.functional.logsigmoid(aux)[...,None],torch.nn.functional.logsigmoid(-aux)[...,None])
    score=score[...,None]+logs['touches'][...,None,:]
    counts=torch.arange(logs['touches'].shape[-1],device=device)
    support=torch.zeros((len(COMBINATIONS),len(counts)),device=device,dtype=torch.bool)
    for arity,stars,holds,touches in allowed:
        if (arity,stars,holds) in COMBINATIONS and 0<=touches<len(counts):
            support[COMBINATIONS.index((arity,stars,holds)),touches]=True
    score=score.masked_fill(~support,-torch.inf)
    if not bool(torch.isfinite(score.flatten(-2)).any(-1).all()):raise RuntimeError('No slot-specific WHAT support')
    return score


def _bias_features(score):
    device=score.device;touches=score.shape[-1];comb=torch.tensor(COMBINATIONS,device=device);a,s,h=comb.unbind(-1);t=torch.arange(touches,device=device)
    return {
        'star':s[:,None].expand(len(COMBINATIONS),touches).reshape(-1).float(),
        'arity2':(a[:,None].expand(len(COMBINATIONS),touches)==2).reshape(-1).float(),
        'hold':(h[:,None].expand(len(COMBINATIONS),touches)>0).reshape(-1).float(),
        'touch':(t[None,:].expand(len(COMBINATIONS),touches)>0).reshape(-1).float(),
        'notesExtra':(a[:,None].expand(len(COMBINATIONS),touches)+t[None,:].expand(len(COMBINATIONS),touches)-1).clamp_min(0).reshape(-1).float(),
    }


def _apply_bias(score,biases):
    flat=score.flatten(-2);features=_bias_features(score)
    for name,value in biases.items():flat=flat+float(value)*features[name]
    return flat


def _apply_scales(flat,features,scales):
    result=flat
    for name,scale in scales.items():
        value=float(scale);feature=features[name]
        if value<=0:
            result=result.masked_fill(feature>0,-torch.inf)
        else:
            result=result+math.log(value)*feature
    return result

def decode_intents(output,allowed,topk=8,biases=None,scales=None):
    score=_base_score(output,allowed)
    flat=_apply_bias(score,biases or {})
    if scales:flat=_apply_scales(flat,_bias_features(score),scales)
    finite=torch.isfinite(flat).sum(-1)
    support=min(int(topk),int(finite.min().item()))
    if support<1:raise RuntimeError('No finite slot-specific WHAT support after scales')
    ids=flat.topk(support,-1).indices.cpu().tolist();touches=score.shape[-1]
    def row(values):
        result=[]
        for value in values:
            arity,stars,holds=COMBINATIONS[value//touches]
            families=(0,)*(arity-stars-holds)+(1,)*holds+(2,)*stars
            result.append(EventIntent(families,value%touches))
        return IntentChoices(tuple(result))
    return [row(values) for values in ids]


def _mechanics_era(version):
    version=int(version)
    if version<3:return 'classic'
    if version<6:return 'green-plus'
    if version<13:return 'multiple-slide'
    if version<19:return 'dx-touch'
    if version<24:return 'festival-chain'
    return 'prism-plus-touch-hold'


def _versioned_profile_pool(context,entry):
    records=entry.get('samples') or []
    if not records or not isinstance(records[0],dict):
        raise ValueError('1.1.0 requires versioned WHAT profile samples')
    requested_version=int(context.version);requested_bpm=float(context.metadata.get('wholebpm',0.) or 0.)
    exact=[row for row in records if int(row['versionId'])==requested_version]
    era=_mechanics_era(requested_version)
    past=[row for row in records if _mechanics_era(row['versionId'])==era and int(row['versionId'])<=requested_version]
    same_era=[row for row in records if _mechanics_era(row['versionId'])==era]
    if len(exact)>=8:pool=exact;tier='exact-version-exact-ds'
    elif len(past)>=8:pool=past;tier='same-mechanics-past-versions-exact-ds'
    elif len(same_era)>=8:pool=same_era;tier='same-mechanics-exact-ds'
    elif past:pool=past;tier='sparse-same-mechanics-past-versions-exact-ds'
    elif same_era:pool=same_era;tier='sparse-same-mechanics-exact-ds'
    else:pool=records;tier='last-resort-exact-ds'
    def proximity(row):
        bpm=float(row.get('bpm',0.) or 0.)
        tempo=abs(math.log2(bpm/requested_bpm)) if bpm>0 and requested_bpm>0 else 0.
        return (tempo+.08*abs(int(row['versionId'])-requested_version),abs(int(row['versionId'])-requested_version),str(row.get('songKey','')))
    # A compatibility fallback deliberately uses the eight nearest charts so
    # distant-tempo charts from older versions cannot dominate a sparse exact
    # version. Exact-version pools can remain wider without semantic drift.
    pool_limit=24 if tier=='exact-version-exact-ds' else 8
    pool=sorted(pool,key=proximity)[:pool_limit]
    values=np.asarray([row['values'] for row in pool],dtype=np.float64)
    return values,pool,{'selectionTier':tier,'mechanicsEra':era,'requestedVersionId':requested_version,
                        'requestedBpm':requested_bpm,'exactVersionCharts':len(exact),'poolCharts':len(pool),
                        'poolVersionIds':sorted({int(row['versionId']) for row in pool}),
                        'poolBpmRange':[min(float(row['bpm']) for row in pool),max(float(row['bpm']) for row in pool)]}


def _profile_entry(context):
    path=Path(context.root)/'models/experimental/what_complexity_profile.json'
    if not path.is_file():raise FileNotFoundError(path)
    doc=json.loads(path.read_text(encoding='utf8'))
    if doc.get('schemaVersion')!=3:raise ValueError('1.1.0 requires WHAT profile schemaVersion 3')
    entries=doc.get('slots',{}).get(str(context.slot),{})
    if not entries:raise ValueError(f'WHAT profile has no support for slot {context.slot}')
    display_levels=context.metadata.get('difficultyDisplayLevels',{}) if isinstance(context.metadata,dict) else {}
    requested=round(float(display_levels.get(str(context.slot),context.level))*10)
    chosen=str(requested)
    if chosen not in entries:raise ValueError(f'No exact official WHAT profile for slot {context.slot} DS {requested/10:.1f}')
    entry=entries[chosen];samples,records,pool_info=_versioned_profile_pool(context,entry)
    return int(chosen),entry,samples,records,pool_info


def _percentile_profile(samples,variation,rng):
    samples=np.asarray(samples,dtype=np.float64);count=len(samples);width=samples.shape[1]
    median=np.median(samples,axis=0);variation=float(np.clip(variation,0.,1.))
    if variation<=1e-12:return median,median,np.full(width,.5),np.full(width,.5)
    percentiles=np.empty_like(samples)
    for row in range(count):
        for column in range(width):
            percentiles[row,column]=((samples[:,column]<samples[row,column]).sum()+.5*(samples[:,column]==samples[row,column]).sum())/count
    center_error=np.abs(percentiles.mean(1)-.5);dispersion=np.abs(percentiles-.5).mean(1);target_dispersion=.5*variation
    central=center_error<=center_error.min()+max(1/count,.02)+1e-12
    distance=np.where(central,np.abs(dispersion-target_dispersion),np.inf);best=distance.min()
    choices=np.flatnonzero(distance<=best+1e-12);selected_index=int(choices[int(rng.integers(0,len(choices)))])
    return samples[selected_index].copy(),samples[selected_index].copy(),percentiles[selected_index].copy(),percentiles[selected_index].copy()


def _target_profile(context,n):
    found=_profile_entry(context)
    if found is None:return None
    source_ds,entry,samples,records,pool_info=found;names=('arity2Rate','holdEventRate','touchEventRate');all_names=names+('notesPerEvent',)
    if not len(samples):return None
    median=np.median(samples[:,:4],axis=0);samples=samples[:,:4]
    variation=float(np.clip(context.metadata.get('whatVariation',.35),0.,1.))
    rng=np.random.default_rng((int(context.seed)*6364136223846793005+int(context.slot)*1442695040888963407)&((1<<63)-1))
    values,sample,percentiles,raw_percentiles=_percentile_profile(samples,variation,rng);rates=values[:3];notes_per_event=float(values[3])
    scales=np.asarray([context.metadata.get('whatArity2Scale',1.),context.metadata.get('whatHoldScale',1.),context.metadata.get('whatTouchScale',1.)],float)
    overrides=(context.metadata.get('whatArity2TargetRate'),context.metadata.get('whatHoldTargetRate'),context.metadata.get('whatTouchTargetRate'))
    for i,value in enumerate(overrides):
        if value is not None:rates[i]=float(value)
    rates=np.clip(rates,[0.,0.,0.],[.8,.5,.5])
    counts={'arity2':round(float(rates[0])*n),'hold':round(float(rates[1])*n),'touch':round(float(rates[2])*n),
            'notesExtra':round(max(0.,notes_per_event-1.)*n)}
    selected_source=None
    if variation>1e-12 and records and isinstance(records[0],dict):
        matches=np.flatnonzero(np.all(np.isclose(samples,sample,rtol=0.,atol=1e-15),axis=1))
        if len(matches):
            row=records[int(matches[0])];selected_source={key:row[key] for key in ('songKey','songId','title','versionId','versionName','bpm')}
    requested_ds=round(float((context.metadata.get('difficultyDisplayLevels',{}) if isinstance(context.metadata,dict) else {}).get(str(context.slot),context.level))*10)
    return counts,{'sourceDs':source_ds,'requestedDs':requested_ds,'exactDisplayedDs':source_ds==requested_ds,
                   'sourceCharts':int(pool_info.get('poolCharts',len(samples))),
                   'unfilteredExactDsCharts':int(entry.get('charts',0)),'variation':variation,'medianRates':dict(zip(names,map(float,median[:3]))),
                   'sampleRates':dict(zip(names,map(float,sample[:3]))),'baseTargetRates':dict(zip(names,map(float,rates))),
                   'medianNotesPerEvent':float(median[3]),'targetNotesPerEvent':notes_per_event,
                   'percentileMetrics':list(all_names),'rawSamplePercentiles':dict(zip(all_names,map(float,raw_percentiles))),
                   'targetPercentiles':dict(zip(all_names,map(float,percentiles))),'meanTargetPercentile':float(percentiles.mean()),
                   'meanAbsolutePercentileDeviation':float(np.abs(percentiles-.5).mean()),
                   'targetMeanAbsolutePercentileDeviation':float(.5*variation),'versionConditioning':pool_info,
                   'selectedOfficialSource':selected_source,
                   'percentileVariationSemantics':'within an exact-DS, version/mechanics-compatible, tempo-nearest official pool, select one real joint profile near the composite median whose mean per-metric percentile deviation best matches variation'}


def _counts(ids,features):
    return {name:int(features[name][ids].sum().item()) for name in features}


def _notes_per_event(ids,score):
    touches=score.shape[-1];comb=torch.tensor(COMBINATIONS,device=ids.device)
    arity=comb[(ids//touches).long(),0].float();touch=(ids%touches).float()
    return float((arity+touch).mean().item())


def _fit_biases(score,targets):
    flat=score.flatten(-2);features=_bias_features(score);biases={name:0. for name in features};grid=np.linspace(-5.,5.,81)
    def ids_for(values):
        adjusted=flat
        for name,value in values.items():adjusted=adjusted+float(value)*features[name]
        return adjusted.argmax(-1)
    current=_counts(ids_for(biases),features)
    for _ in range(5):
        for name in ('notesExtra','arity2','hold','touch','star'):
            best=None
            for x in grid:
                trial=dict(biases);trial[name]=float(x);counts=_counts(ids_for(trial),features)
                key=(abs(counts[name]-targets[name]),sum(abs(counts[k]-targets[k])/max(1,targets[k]) for k in targets),abs(float(x)))
                if best is None or key<best[0]:best=(key,trial,counts)
            biases,current=best[1],best[2]
        if all(abs(current[k]-targets[k])<=1 for k in ('arity2','hold','touch','notesExtra')):break
    return biases,current

def intent_plan(context,target_stars,topk=8):
    topk=int(topk)
    if not 1<=topk<=64:raise ValueError('WHAT candidate support must be 1..64')
    heads=predict_heads(context);ticks=np.asarray(context.ticks,dtype=np.int64)
    document=json.loads((Path(context.root)/'models/experimental/intent_support.json').read_text(encoding='utf8'))
    if document.get('schemaVersion')!=2:raise ValueError('1.1.0 requires five-slot intent support schemaVersion 2')
    touch_capacity=touch_sensor_capacity(context.version)
    allowed={tuple(x) for x in document['slots'][str(context.slot)]['configurations']
             if int(x[3])<=touch_capacity}
    selected={name:value[ticks//384,ticks%384] for name,value in heads.items()};score=_base_score(selected,allowed)
    target=max(0,min(int(target_stars),2*len(ticks)));profile=_target_profile(context,len(ticks))
    targets={'star':target,'arity2':0,'hold':0,'touch':0};profile_info=None
    if profile is not None:
        extra,profile_info=profile;targets.update(extra)
    if not touch_enabled(context.version):
        targets['touch']=0
    base_biases,base_actual=_fit_biases(score,targets)
    scales={'star':float(context.metadata.get('whatStarScale',1.0)),
            'arity2':float(context.metadata.get('whatArity2Scale',1.0)),
            'hold':float(context.metadata.get('whatHoldScale',1.0)),
            'touch':float(context.metadata.get('whatTouchScale',1.0)) if touch_enabled(context.version) else 0.0}
    flat=_apply_bias(score,base_biases);features=_bias_features(score)
    base_ids=flat.argmax(-1)
    final_flat=_apply_scales(flat,features,scales)
    final_ids=final_flat.argmax(-1);final_actual=_counts(final_ids,features)
    scale_biases={name:(-math.inf if value<=0 else math.log(value)) for name,value in scales.items()}
    choices=decode_intents(selected,allowed,topk,base_biases,scales)

    star_bias=base_biases['star']+(scale_biases['star'] if math.isfinite(scale_biases['star']) else -20.)
    adjusted=shifted_stars(heads,star_bias);probability=adjusted['stars'].softmax(-1)
    stars=(probability[...,1]+2*probability[...,2]).cpu().numpy()
    if profile_info is not None:profile_info=dict(profile_info,scales=scales)
    actual_support=min((len(value.candidates) for value in choices),default=0)
    info={'targetStars':target,'baselinePlannedStars':base_actual['star'],'plannedStars':final_actual['star'],'candidateSupport':actual_support,'candidateSupportRequested':topk,
          'starScale':scales['star'],'whatTargets':targets,'whatBaselinePlanned':base_actual,
          'whatPlanned':final_actual,'whatBaseBiases':base_biases,'whatScaleLogWeights':scale_biases,
          'baselineNotesPerEvent':_notes_per_event(base_ids,score),'plannedNotesPerEvent':_notes_per_event(final_ids,score),
          'whatProfile':profile_info,'normalization':'single categorical WHAT distribution; GUI scales are relative odds'}
    return dict(zip(map(int,ticks),choices)),stars,info


def star_scores(context):
    heads=predict_heads(context);p=heads['stars'].softmax(-1)
    return (p[...,1]+2*p[...,2]).cpu().numpy()
