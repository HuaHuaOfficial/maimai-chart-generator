"""Train-only, rotation-invariant endpoint/shape and start-transition backoff model."""
from __future__ import annotations
import json, math
from pathlib import Path
from functools import lru_cache
import numpy as np
SHAPES=['-','<','>','^','p','q','pp','qq','v','V','s','z','w']
K=len(SHAPES)
MIRROR={i:SHAPES.index({'<':'>','>':'<','p':'q','q':'p','pp':'qq','qq':'pp','s':'z','z':'s'}.get(s,s)) for i,s in enumerate(SHAPES)}
def gap_bin(delta):return int(np.searchsorted([48,96,192,384,768],delta))
def route_keys(r):
 keys=[('level',r['lev']),('future',r['lev'],r['future'])]
 p=r.get('prev');p2=r.get('older')
 if p is not None:
  delta=(r['start']-p[1])%8;rel=(p[2]-p[1])%8
  keys.append(('previous',r['lev'],r['gap'],p[3],rel,delta))
  if p2 is not None:
   keys.append(('motif',r['lev'],r['gap'],p[3],p2[3],rel,(p2[2]-p2[1])%8,delta,(p[1]-p2[1])%8,r['future']))
 return [str(x) for x in keys]
def start_keys(r):
 p=r.get('prev');p2=r.get('older')
 if p is None:return []
 keys=[('level',r['lev']),('gap',r['lev'],r['gap'],(p[2]-p[1])%8,p[3])]
 if p2 is not None:keys.append(('motion',r['lev'],r['gap'],(p[2]-p[1])%8,p[3],(p[1]-p2[1])%8,r['future']))
 return [str(x) for x in keys]
def cue_keys(r):
 return [str(x) for x in [('slot',r['slot']),('level',r['slot'],r['lev']),('inventory',r['slot'],r['lev'],min(2,r['tapCount']),min(3,r['outerCount'])),('tempo',r['slot'],r['lev'],r['tempo'],min(2,r['tapCount']),min(3,r['outerCount']))]]
class Backoff:
 def __init__(self,size,alpha=64.):self.size=size;self.alpha=alpha;self.total=np.zeros(size);self.tables={}
 def add(self,keys,y):
  self.total[y]+=1
  for key in keys:
   if key not in self.tables:self.tables[key]=np.zeros(self.size)
   self.tables[key][y]+=1
 def predict(self,keys,baseline=False):
  p=(self.total+.5)/(self.total.sum()+self.size*.5)
  for i,key in enumerate(keys):
   if baseline and i:break
   c=self.tables.get(key)
   if c is not None:p=(c+self.alpha*p)/(c.sum()+self.alpha)
  return p
 def pack(self):return {'size':self.size,'alpha':self.alpha,'total':self.total.tolist(),'tables':{k:{str(i):int(c[i]) for i in np.flatnonzero(c)} for k,c in self.tables.items()}}
 @classmethod
 def unpack(cls,d):
  self=cls(d['size'],d['alpha']);self.total=np.array(d['total'],float)
  for k,c in d['tables'].items():
   a=np.zeros(self.size)
   for i,v in c.items():a[int(i)]=v
   self.tables[k]=a
  return self
