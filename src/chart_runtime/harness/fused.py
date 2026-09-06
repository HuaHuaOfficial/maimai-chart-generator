"""Fused execution of the common rule definitions for every scope/batch."""
from __future__ import annotations
import numpy as np
import torch

TIME_EPSILON=1e-7
INPUT_RELEASE_SECONDS=1/60

SOURCE=r'''
typedef long long ll;
typedef unsigned long long ull;
__device__ bool exempt(ll a,ll b,const ll* spans,int S) {
    ll low=a<b?a:b,high=a>b?a:b;
    for(int s=0;s<S;s++)if(low>=spans[2*s]&&high<=spans[2*s+1])return true;
    return false;
}
__device__ void mark(ull* flags,int r,ll a,ll b,const ll* clean,int C,bool exceptions=true) {
    if(exceptions&&exempt(a,b,clean,C))return;
    atomicOr(flags+a,1ULL<<r);atomicOr(flags+b,1ULL<<r);
}
// Rule positions match kernel.HARD_NAMES and are checked by a source manifest.
extern "C" __global__ void inputs(
 const ll* event,const ll* note,const ll* sensor,const bool* outer,const bool* hold,const bool* ex,
 const double* start,const double* end,const ll* batch,
 const ll* te,const ll* tn,const ll* th,const ll* tt,const double* tshoot,const double* tend,const bool* wifi,
 const ll* clean,int C,const double* limit,ull* flags,ll* soft,int I,int T) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=I)return;
 ll e=event[i],b=batch[e];double at=start[i];
 int outer_count=0,held_count=0;int active[64],A=0;
 for(int j=0;j<I;j++) {
   if(batch[event[j]]!=b)continue;
   if(j>i&&sensor[i]==sensor[j]) {
     double bound=fmax(start[i]+2.0/60,end[i]);
     if(start[j]<bound-1e-7)mark(flags,0,e,event[j],clean,C);
     bool taps=(end[i]-start[i]<=1e-7)&&(end[j]-start[j]<=1e-7);
     bool overlap=taps?fabs(start[i]-start[j])<=2.0/60+1e-7:
        start[i]-2.0/60<=end[j]+1e-7&&start[j]-2.0/60<=end[i]+1e-7;
     if(overlap)atomicAdd((ull*)(soft+b*4+2),1ULL);
   }
   bool on=start[j]<=at+1e-7&&end[j]+1.0/60>at+1e-7;
   if(on&&outer[j]){if(A<64)active[A++]=j;outer_count++;}
   if(hold[j]&&start[j]<=at+1e-7&&end[j]>at+1e-7)held_count++;
   if(outer[i]&&outer[j]&&hold[j]&&start[j]<at-1e-7&&end[j]>at+.015&&sensor[i]==sensor[j])mark(flags,2,e,event[j],clean,C);
 }
 if(outer[i]&&outer_count>2)mark(flags,1,e,e,clean,C);
 if(hold[i]&&end[i]>limit[b]+.02)mark(flags,14,e,e,clean,C,false);
 if(hold[i]&&(!isfinite(end[i])||end[i]<=at))mark(flags,16,e,e,clean,C,false);
 for(int t=0;t<T;t++) {
   if(batch[te[t]]!=b)continue;
   bool other=e!=te[t]||note[i]!=tn[t];double delta=at-tshoot[t];
   if(other&&outer[i]&&sensor[i]==tt[t]&&fabs(at-tend[t])<1e-7)mark(flags,4,e,te[t],clean,C);
   if(other&&outer[i]&&sensor[i]==th[t]&&delta>1e-7&&delta<.2-1e-7) {
     atomicAdd((ull*)(soft+b*4+1),1ULL);
     if(!ex[i])mark(flags,5,e,te[t],clean,C);
   }
   if(at<tshoot[t]-1e-7||at>tend[t]+1e-7)continue;
   int count=0;ll lo=te[t],hi=te[t];
   if(outer_count<=64) {
     for(int k=0;k<A;k++){int j=active[k];
       if(hold[j]&&end[j]<=tshoot[t]+1e-7)continue;
       if(!hold[j]&&fabs(start[j]-tshoot[t])<1e-7&&sensor[j]==th[t])continue;
       count++;lo=min(lo,event[j]);hi=max(hi,event[j]);
     }
   } else { // Rare already-invalid dense batch; still evaluate exact support.
     for(int j=0;j<I;j++)if(batch[event[j]]==b&&outer[j]&&start[j]<=at+1e-7&&end[j]+1.0/60>at+1e-7) {
       if(hold[j]&&end[j]<=tshoot[t]+1e-7)continue;
       if(!hold[j]&&fabs(start[j]-tshoot[t])<1e-7&&sensor[j]==th[t])continue;
       count++;lo=min(lo,event[j]);hi=max(hi,event[j]);
     }
   }
   if(count>=2)mark(flags,7,lo,hi,clean,C);
   if(wifi[t]&&count>=1&&at<tend[t]-1e-7)mark(flags,8,lo,hi,clean,C);
   if(held_count>=2&&at<tend[t]-1e-7) {
     ll hlo=te[t],hhi=te[t];
     for(int j=0;j<I;j++)if(batch[event[j]]==b&&hold[j]&&start[j]<=at+1e-7&&end[j]>at+1e-7){hlo=min(hlo,event[j]);hhi=max(hhi,event[j]);}
     mark(flags,3,hlo,hhi,clean,C);
   }
 }
}
extern "C" __global__ void tracks(
 const ll* te,const ll* tn,const ll* th,const ll* route,const ll* path,const ll* contactkey,
 const double* at,const double* shoot,const double* end,const bool* wifi,const ll* batch,
 const ll* ie,const ll* sensor,const bool* outer,const bool* hold,const double* istart,const double* iend,
 const double* limit,const ll* clean,int C,ull* flags,int I,int T) {
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=T)return;
 ll e=te[t],b=batch[e];double time=shoot[t];
 if(!isfinite(end[t])||end[t]<=shoot[t]||shoot[t]<at[t])mark(flags,16,e,e,clean,C,false);
 if(end[t]>limit[b]+.02)mark(flags,15,e,e,clean,C,false);
 int inputs=0,holds=0,downs=0;ll low=e,high=e,hlo=e,hhi=e;
 for(int i=0;i<I;i++) {
   if(batch[ie[i]]!=b)continue;
   if(hold[i]&&istart[i]<=time+1e-7&&iend[i]>time+1e-7){holds++;hlo=min(hlo,ie[i]);hhi=max(hhi,ie[i]);}
   if(!outer[i])continue;
   if(fabs(istart[i]-time)<1.0/60-1e-7)downs++;
   if(istart[i]>time+1e-7||iend[i]+1.0/60<=time+1e-7)continue;
   if(hold[i]&&iend[i]<=time+1e-7)continue;
   if(!hold[i]&&fabs(istart[i]-time)<1e-7&&sensor[i]==th[t])continue;
   inputs++;low=min(low,ie[i]);high=max(high,ie[i]);
 }
 if(inputs>=2)mark(flags,7,low,high,clean,C);
 if(wifi[t]&&inputs>=1)mark(flags,8,low,high,clean,C);
 if(holds>=2)mark(flags,3,hlo,hhi,clean,C);
 int own_nonwifi=0;bool distinct=false;
 for(int u=0;u<T;u++) {
   if(batch[te[u]]!=b)continue;
   bool overlap=shoot[t]<end[u]-1e-7&&shoot[u]<end[t]-1e-7;
   if(u>t) {
     if(overlap&&wifi[t]&&wifi[u])mark(flags,9,e,te[u],clean,C);
     if(path[t]==path[u]&&e!=te[u]&&shoot[t]<end[u]+.2&&shoot[u]<end[t]+.2)mark(flags,12,e,te[u],clean,C);
   }
   if(overlap&&e==te[u]&&tn[t]==tn[u]&&!wifi[u]) {
      own_nonwifi++;
      if(route[t]!=route[u]&&contactkey[t]!=contactkey[u])distinct=true;
   }
 }
 if(wifi[t]&&own_nonwifi>=2)mark(flags,10,e,e,clean,C);
 if(!wifi[t]&&distinct&&downs>=2)mark(flags,11,e,e,clean,C);
}
extern "C" __global__ void contacts(
 const ll* ct,const ll* cs,const double* time,const ll* te,const ll* batch,
 const ll* ie,const ll* sensor,const bool* outer,const bool* ex,const double* start,
 const ll* clean,int C,ull* flags,ll* soft,int Q,int I) {
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=Q)return;
 ll e=te[ct[q]],b=batch[e];
 for(int i=0;i<I;i++)if(batch[ie[i]]==b&&outer[i]&&sensor[i]==cs[q]) {
    double dt=start[i]-time[q];if(dt<=1e-7||dt>=.15-1e-7)continue;
    atomicAdd((ull*)(soft+b*4),1ULL);
    if(!ex[i])mark(flags,6,e,ie[i],clean,C);
 }
}
extern "C" __global__ void versions(
 const ll* ne,const ll* kind,const ll* mods,const ll* batch,const ll* version,
 const ll* ie,const ll* sensor,const bool* outer,const bool* hold,ull* flags,int N,int I,int center) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<N){ll e=ne[i],v=version[batch[e]];ull mask=0;
 if(kind[i]==3&&v<13)mask|=1ULL<<18;
 if((mods[i]&2)&&v<13)mask|=1ULL<<20;
 if((mods[i]&1)&&(kind[i]==1||kind[i]==2)&&v<19)mask|=1ULL<<21;
 atomicOr(flags+e,mask);}
 if(i<I){ll e=ie[i],v=version[batch[e]];if(!outer[i]&&hold[i]&&sensor[i]!=center&&v<24)atomicOr(flags+e,1ULL<<19);}
}
extern "C" __global__ void multitouch(
 const ll* ie,const ll* inode,const ll* sensor,const ll* ipad,const bool* outer,const bool* hold,
 const double* start,const double* iend,const ll* batch,
 const ll* te,const ll* tn,const ll* path,const ll* head,const ll* tail,const double* shoot,const double* tend,const bool* wifi,
 const double* astart,const double* aend,const ll* amask,const ll* atrack,const bool* tadj,
 const ll* clean,int C,ull* flags,int I,int T,int A) {
 int probe=blockIdx.x*blockDim.x+threadIdx.x;if(probe>=I+T)return;
 double at;ll b,owner;
 if(probe<I){at=start[probe];owner=ie[probe];b=batch[owner];}
 else {int p=probe-I;at=shoot[p];owner=te[p];b=batch[owner];}
 int hands=0;ll lo=owner,hi=owner;
 for(int i=0;i<I;i++) {
   if(batch[ie[i]]!=b||start[i]>at+1e-7||iend[i]+1.0/180<=at+1e-7)continue;
   if(outer[i]||hold[i]){hands++;lo=min(lo,ie[i]);hi=max(hi,ie[i]);continue;}
   int sid=(int)sensor[i]-8;ull present=0,reach=1ULL<<sid;
   for(int j=0;j<I;j++)if(ie[j]==ie[i]&&!outer[j]&&!hold[j]&&start[j]<=at+1e-7&&iend[j]+1.0/180>at+1e-7)present|=1ULL<<((int)sensor[j]-8);
   for(int z=0;z<33;z++)for(int a=0;a<33;a++)if((reach>>a)&1ULL)for(int q=0;q<33;q++)if(((present>>q)&1ULL)&&tadj[a*33+q])reach|=1ULL<<q;
   bool first=true;
   for(int j=0;j<i;j++)if(ie[j]==ie[i]&&!outer[j]&&!hold[j]&&((reach>>((int)sensor[j]-8))&1ULL)){first=false;break;}
   if(!first)continue;
   bool all_covered=true;
   for(int j=i;j<I;j++)if(ie[j]==ie[i]&&!outer[j]&&!hold[j]&&((reach>>((int)sensor[j]-8))&1ULL)) {
     bool covered=false;
     for(int t=0;t<T&&!covered;t++)if(batch[te[t]]==b&&shoot[t]<=at+1e-7&&tend[t]+1.0/180>at+1e-7)
       for(int a=0;a<A;a++)if(atrack[a]==t&&astart[a]<=at+1e-7&&aend[a]>at-1e-7&&(((ull)amask[a]&((ull)ipad[j]))!=0)){covered=true;break;}
     if(!covered){all_covered=false;break;}
   }
   if(!all_covered){hands++;lo=min(lo,ie[i]);hi=max(hi,ie[i]);}
 }
 for(int t=0;t<T;t++)if(batch[te[t]]==b&&shoot[t]<=at+1e-7&&tend[t]+1.0/180>at+1e-7) {
   bool duplicate=false;
   for(int u=0;u<t;u++)if(batch[te[u]]==b&&te[u]==te[t]&&tn[u]==tn[t]&&path[u]==path[t]&&head[u]==head[t]&&tail[u]==tail[t]&&fabs(shoot[u]-shoot[t])<1e-7&&fabs(tend[u]-tend[t])<1e-7){duplicate=true;break;}
   if(!duplicate){hands+=wifi[t]?2:1;lo=min(lo,te[t]);hi=max(hi,te[t]);}
 }
 if(hands>2)mark(flags,22,lo,hi,clean,C);
}
'''


class FusedRules:
    def __init__(self,codec):
        import cupy as cp
        self.cp=cp;module=cp.RawModule(code=SOURCE,options=('--std=c++17',),name_expressions=('inputs','tracks','contacts','versions','multitouch'))
        self.functions={name:module.get_function(name) for name in ('inputs','tracks','contacts','versions','multitouch')}
        names=list(codec.vocab['touchPositions']);adj=codec.tables['touchAdjacency']
        self.touch_adjacency=torch.tensor([[a==b or b in adj.get(a,()) for b in names] for a in names],device=codec.device,dtype=torch.bool)

    def run(self,c,clean,versions,limits,center,B):
        from .cuda_views import view
        cp=self.cp;d=c['event_tick'].device
        flags=torch.zeros(len(c['event_tick']),dtype=torch.int64,device=d);soft=torch.zeros((B,4),dtype=torch.int64,device=d)
        cached={}
        def arr(key):
            if key not in cached:cached[key]=view(cp,c[key])
            return cached[key]
        def launch(name,size,args):
            if size:self.functions[name](((size+127)//128,),(128,),args)
        C=np.int32(len(clean));I=np.int32(len(c['input_event']));T=np.int32(len(c['track_event']));Q=np.int32(len(c['contact_track']))
        stream=torch.cuda.current_stream(d)
        with cp.cuda.ExternalStream(stream.cuda_stream):
            cl=view(cp,clean);fl=view(cp,flags);sf=view(cp,soft);lim=view(cp,limits);v=view(cp,versions)
            launch('inputs',int(I),tuple(arr(k) for k in ('input_event','input_note','input_sensor','input_outer','input_hold','input_ex','input_start','input_end','event_batch','track_event','track_note','track_head','track_tail','track_shoot','track_end','track_wifi'))+(cl,C,lim,fl,sf,I,T))
            launch('tracks',int(T),tuple(arr(k) for k in ('track_event','track_note','track_head','track_route','track_path','track_contacts_key','track_start','track_shoot','track_end','track_wifi','event_batch','input_event','input_sensor','input_outer','input_hold','input_start','input_end'))+(lim,cl,C,fl,I,T))
            launch('contacts',int(Q),tuple(arr(k) for k in ('contact_track','contact_sensor','contact_time','track_event','event_batch','input_event','input_sensor','input_outer','input_ex','input_start'))+(cl,C,fl,sf,Q,I))
            N=np.int32(len(c['note_event']))
            launch('versions',max(int(N),int(I)),tuple(arr(k) for k in ('note_event','note_kind','note_modifiers','event_batch'))+(v,)+tuple(arr(k) for k in ('input_event','input_sensor','input_outer','input_hold'))+(fl,N,I,np.int32(center)))
            A=np.int32(len(c['action_event']))
            launch('multitouch',int(I+T),tuple(arr(k) for k in ('input_event','input_note','input_sensor','input_pad','input_outer','input_hold','input_start','input_end','event_batch','track_event','track_note','track_path','track_head','track_tail','track_shoot','track_end','track_wifi','action_start','action_end','action_mask','action_track'))+(view(cp,self.touch_adjacency),cl,C,fl,I,T,A))
        return flags,soft
