"""Inference-only short-history WHERE planner, with conditional cue calibration.
The neural renderer remains the music scorer; learned sequence likelihood
ratios correct its geometry. Legality is exclusively owned by the Harness.
"""
from pathlib import Path
from functools import lru_cache
from hashlib import sha256
import re,json
import numpy as np
from .sequence_model import Backoff,route_keys,start_keys,cue_keys,gap_bin,SHAPES,K
_RE=re.compile(r'(-|<|>|\^|pp|qq|p|q|v|V[1-8]|s|z|w)([1-8])$')
@lru_cache(maxsize=4)
def _load(path,mtime,size):
 raw=Path(path).read_bytes();d=json.loads(raw)
 if d.get('schema')!='where-sequence-prior/1' or d['shapes']!=SHAPES:raise ValueError('Sequence profile schema mismatch')
 return {k:Backoff.unpack(d[k]) for k in ('route','start','cue')},sha256(raw).hexdigest()
def future_tag(value):
 from .intent import IntentChoices,EventIntent
 if isinstance(value,IntentChoices):value=value.candidates[0]
 value=EventIntent.from_representation(value) if value is not None else EventIntent()
 return min(2,value.button_families.count(0))+3*min(2,value.button_families.count(2))
class Profile:
 def __init__(self,context,plan,vocab):
  p=Path(__file__).with_name('sequence_profile.json');s=p.stat()
  self.models,self.digest=_load(str(p),s.st_mtime_ns,s.st_size)
  self.weight=float(context.metadata.get('whereRouteSequenceWeight',.6))
  self.start_weight=float(context.metadata.get('whereStartSequenceWeight',.3))
  if not 0<=self.weight<=1 or not 0<=self.start_weight<=1:raise ValueError('Sequence weights outside [0,1]')
  self.cue_enabled=bool(context.metadata.get('whereCueCalibration',True))
  self.slot=int(context.slot);self.lev=int(np.clip(round(context.level*2),20,31))
  self.future={int(t):future_tag(plan.get(int(t)+96)) for t in context.ticks}
  from .relational_what import _primary
  from ..io.timing import ticks_to_seconds
  self.chord_ticks=np.array(sorted(int(t) for t,v in plan.items() if _primary(v).button_arity>=2),np.int64)
  self.chord_seconds=ticks_to_seconds(self.chord_ticks,context.bt,context.bv) if len(self.chord_ticks) else np.empty(0)
  self.desc=[]
  for route in vocab['routes']:
   m=_RE.fullmatch(route)
   self.desc.append((int(m[2])-1,SHAPES.index('V' if m[1].startswith('V') else m[1])) if m else (-1,-1))
  self.end=np.array([x[0] for x in self.desc]);self.shape=np.array([x[1] for x in self.desc])
 def movement_fits(self,source,end):
  # A still-active track strictly after its launch cannot share a fresh
  # outer input. Reserve the next declared two-outer event, including
  # the same 1/60-second release already used by the CUDA hand kernel.
  i=int(np.searchsorted(self.chord_ticks,int(source)+96,side='right'))
  return i==len(self.chord_ticks) or end+1/60<=self.chord_seconds[i]+1e-7
 def row(self,tick,start,history):
  old=history[-1] if history and 0<tick-history[-1][0]<=768 else None
  older=history[-2] if old and len(history)>1 and old[0]-history[-2][0]<=768 else None
  return {'lev':self.lev,'start':int(start),'future':self.future.get(int(tick),0),'prev':old,'older':older,'gap':gap_bin(tick-old[0]) if old else 5}
 def route_bias(self,tick,start,history):
  r=self.row(tick,start,history);m=self.models['route']
  q=m.predict(route_keys(r));base=m.predict(route_keys(r),True)
  # Correct transitions, not a second unconditional popularity prior.
  ratio=np.clip(np.log(q/base),-2.,2.)*self.weight
  result=np.zeros(len(self.desc));valid=self.end>=0
  result[valid]=ratio[((self.end[valid]-start)%8)*K+self.shape[valid]]
  return result
 def start_bias(self,tick,history):
  if not history or tick-history[-1][0]>768:return np.zeros(8)
  r=self.row(tick,0,history);m=self.models['start'];keys=start_keys(r)
  q=m.predict(keys);base=m.predict(keys,True)
  ratios=np.clip(np.log(q/base),-2.,2.)*self.start_weight
  return ratios[(np.arange(8)-history[-1][1])%8]
 def cue_probability(self,bpm,taps=1,outer=1):
  row={'slot':self.slot,'lev':self.lev,'tempo':int(np.searchsorted([120,160,200,240],bpm)),'tapCount':taps,'outerCount':outer}
  return float(np.clip(self.models['cue'].predict(cue_keys(row))[1],.02,.98))
def maybe_profile(context,plan,vocab):
 if context.slot<5 or not context.metadata.get('whereSequenceEnabled',True):return None
 p=Path(__file__).with_name('sequence_profile.json')
 if not p.is_file():raise FileNotFoundError('Enabled WHERE sequence profile is missing')
 return Profile(context,plan,vocab)
def append_history(history,tick,rep,profile):
 if profile is None:return
 active=[i for i in range(int(rep['button_arity'])) if int(rep['button_family'][i])==2]
 if len(active)>1:history.clear();return
 if len(active)==1:
  i=active[0];rid=int(rep['button_route'][i]);end,shape=profile.desc[rid]
  if end<0:history.clear();return
  history.append((int(tick),int(rep['button_start'][i]),end,shape))
  del history[:-2]
def _draw(values,ids,temperature,rng,top_p):
 if len(ids)==1:return int(ids[0])
 p=np.exp((values-values.max())/max(.05,float(temperature)));p/=p.sum()
 if top_p<1:
  order=np.argsort(p)[::-1];n=max(1,int(np.searchsorted(np.cumsum(p[order]),max(.05,top_p)))+1)
  ids=np.asarray(ids)[order[:n]];p=p[order[:n]];p/=p.sum()
 return int(rng.choice(ids,p=p))
def cue_calibration_eligible(families):
    # Cue calibration is a pure-Tap style choice. Any event that already
    # contains a Star (including Tap+Star / Star+Tap) must remain natural;
    # phase-chain geometry and existing decorative Taps are not cue targets.
    return tuple(map(int,families))==(0,)

def calibrated_cue(scores,ids,starts,heads,probability,temperature,rng,top_p):
 # Top-p operates WITHIN each cue class, so vocabulary/cardinality cannot
 # inflate the probability of cue. Existing legal support is never enlarged.
 cue=np.array([bool(set(starts[i])&set(heads)) for i in ids]);ids=np.array(ids)
 if cue.all() or not cue.any():chosen=np.ones(len(ids),bool)
 else:chosen=cue if rng.random()<probability else ~cue
 values=scores.float().detach().cpu().numpy()
 return _draw(values[ids[chosen]],ids[chosen],temperature,rng,top_p)
def choose_route(logits,ids,start,snapshot,temperature,rng,top_p):
 p=snapshot.get('_sequence_profile')
 from .sampling import choose_allowed
 if p is None or p.weight==0:return choose_allowed(logits,ids,temperature,rng,top_p)
 ids=np.asarray(ids,np.int64)
 if not len(ids):raise ValueError('No legal routes')
 if len(ids)==1:return int(ids[0])
 values=logits.float().detach().cpu().numpy()+p.route_bias(snapshot['_sequence_tick'],start,snapshot['_sequence_history'])
 # First endpoint, then shape, then the exact legal token (e.g. V pivot).
 scaled=(values[ids]-values[ids].max())/max(.05,float(temperature));prob=np.exp(scaled);prob/=prob.sum()
 if top_p<1:
  order=np.argsort(prob)[::-1];n=max(1,int(np.searchsorted(np.cumsum(prob[order]),max(.05,top_p)))+1)
  ids=ids[order[:n]];prob=prob[order[:n]];prob/=prob.sum()
 end=p.end[ids];shape=p.shape[ids]
 endpoints=np.unique(end);mass=np.array([prob[end==e].sum() for e in endpoints]);mass/=mass.sum()
 e=int(rng.choice(endpoints,p=mass));keep=end==e;ids=ids[keep];shape=shape[keep];prob=prob[keep];prob/=prob.sum()
 shapes=np.unique(shape);mass=np.array([prob[shape==s].sum() for s in shapes]);mass/=mass.sum()
 s=int(rng.choice(shapes,p=mass));keep=shape==s;ids=ids[keep];prob=prob[keep];prob/=prob.sum()
 return int(rng.choice(ids,p=prob))
