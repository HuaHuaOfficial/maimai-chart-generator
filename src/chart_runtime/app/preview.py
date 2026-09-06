from __future__ import annotations
import math, os, re, shutil, subprocess, time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from ..io.timing import ticks_to_seconds
from ..io.simai import parse_inote_ticks, parse_maidata


COLORS={"tap":"#ff5f9e","break":"#ffcf33","ex":"#aefcff","hold":"#7ce577","slide":"#ba8cff","touch":"#56d9ff"}

def split_notes(text:str): return [x for x in re.split(r"[/`]",text) if x]
def point(center,radius,key):
    angle=math.radians(-67.5+(int(key)-1)*45);return center[0]+math.cos(angle)*radius,center[1]+math.sin(angle)*radius

class PreviewWindow(tk.Toplevel):
    def __init__(self,master,maidata_path:Path):
        super().__init__(master);self.title('谱面预览');self.geometry('900x820');self.protocol('WM_DELETE_WINDOW',self.close)
        self.maidata_path=Path(maidata_path);self.song_dir=self.maidata_path.parent;self.player=None;self.playing=False;self.base=0.;self.started=0.;self.events=[];self.duration=1.;self.slot=tk.StringVar();self.lead=tk.DoubleVar(value=2.0);self.position=tk.DoubleVar(value=0.)
        self._build();self._load_fields();self.after(16,self._update)
    def _build(self):
        top=ttk.Frame(self,padding=8);top.pack(fill='x');ttk.Label(top,text='谱面槽').pack(side='left');self.slot_box=ttk.Combobox(top,textvariable=self.slot,state='readonly',width=18);self.slot_box.pack(side='left',padx=6);self.slot_box.bind('<<ComboboxSelected>>',lambda _:self.load_chart())
        self.play_btn=ttk.Button(top,text='播放',command=self.toggle);self.play_btn.pack(side='left',padx=6);ttk.Button(top,text='回到开头',command=lambda:self.seek(0)).pack(side='left')
        ttk.Label(top,text='提前显示(s)').pack(side='left',padx=(18,4));ttk.Spinbox(top,from_=.5,to=5,increment=.1,textvariable=self.lead,width=6).pack(side='left')
        self.time_label=ttk.Label(top,text='0:00.000 / 0:00.000');self.time_label.pack(side='right')
        self.canvas=tk.Canvas(self,bg='#0d1020',highlightthickness=0);self.canvas.pack(fill='both',expand=True,padx=8,pady=4)
        self.scale=ttk.Scale(self,from_=0,to=1,variable=self.position,command=self._slider);self.scale.pack(fill='x',padx=12,pady=(4,10));self._dragging=False;self.scale.bind('<ButtonPress-1>',lambda _:setattr(self,'_dragging',True));self.scale.bind('<ButtonRelease-1>',self._release)
    def _load_fields(self):
        self.fields=parse_maidata(self.maidata_path.read_text(encoding='utf-8-sig'));slots=[k.split('_')[1] for k in self.fields if re.fullmatch(r'inote_[2-7]',k)];self.slot_box['values']=[f'inote_{x}' for x in slots]
        if not slots:raise ValueError('maidata中没有inote_2～inote_7')
        self.slot.set(f'inote_{slots[0]}');self.load_chart()
    def load_chart(self):
        bpm=float(self.fields.get('wholebpm') or 120);parsed=parse_inote_ticks(self.fields[self.slot.get()],bpm);ticks=np.array([x[0] for x in parsed.events],dtype=np.int64);secs=ticks_to_seconds(ticks,np.array(parsed.bpm_ticks),np.array(parsed.bpm_values));first=float(self.fields.get('first') or 0)
        self.events=[(float(s)+first,text,tick) for s,(_,text),tick in zip(secs,parsed.events,ticks)];end=ticks_to_seconds(np.array([parsed.end_tick]),np.array(parsed.bpm_ticks),np.array(parsed.bpm_values))[0]+first;self.duration=max(1.,float(end));self.scale.configure(to=self.duration);self.seek(0)
    def ffplay(self):
        root=Path(os.environ.get('MAIMAI_INFERENCE_ROOT',str(Path(__file__).resolve().parents[3])))
        matches=list((root/'tools'/'ffmpeg').glob('**/ffplay.exe'))
        return matches[0] if matches else shutil.which('ffplay')
    def start_audio(self):
        self.stop_audio();exe=self.ffplay();track=self.song_dir/'track.mp3'
        if exe and track.is_file():
            self.player=subprocess.Popen([str(exe),'-nodisp','-autoexit','-loglevel','quiet','-ss',f'{self.base:.3f}',str(track)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    def stop_audio(self):
        if self.player and self.player.poll() is None:self.player.terminate()
        self.player=None
    def toggle(self):
        if self.playing:self.base=self.current();self.playing=False;self.stop_audio();self.play_btn.configure(text='播放')
        else:self.started=time.perf_counter();self.playing=True;self.start_audio();self.play_btn.configure(text='暂停')
    def current(self): return min(self.duration,self.base+(time.perf_counter()-self.started if self.playing else 0))
    def seek(self,value):
        was=self.playing;self.base=max(0,min(self.duration,float(value)));self.started=time.perf_counter();self.position.set(self.base)
        if was:self.start_audio()
    def _slider(self,_):
        if self._dragging:self.base=float(self.position.get());self.started=time.perf_counter()
    def _release(self,_):self._dragging=False;self.seek(self.position.get())
    def _draw(self,now):
        c=self.canvas;c.delete('all');w=max(400,c.winfo_width());h=max(400,c.winfo_height());cx,cy=w/2,h/2-20;R=min(w,h)*.36
        c.create_oval(cx-R,cy-R,cx+R,cy+R,outline='#6780aa',width=4);c.create_oval(cx-R*.12,cy-R*.12,cx+R*.12,cy+R*.12,outline='#33476e',width=2)
        for k in range(1,9):
            x,y=point((cx,cy),R,k);c.create_oval(x-17,y-17,x+17,y+17,fill='#1d2943',outline='#9bb6e8',width=2);c.create_text(x,y,text=str(k),fill='white',font=('Segoe UI',11,'bold'))
        lead=max(.2,float(self.lead.get()));visible=[]
        for event_time,text,_ in self.events:
            delta=event_time-now
            if delta>lead or delta<-.35:continue
            visible.append((event_time,text));progress=max(0,min(1,1-delta/lead))
            for note in split_notes(text):
                touch=re.match(r'([A-E])([1-8])?',note)
                if touch:
                    area,key=touch.groups();rad={'A':.86,'B':.55,'C':0,'D':.72,'E':.64}[area]*R;x,y=(cx,cy) if not key else point((cx,cy),rad,key);color=COLORS['touch'];size=16 if abs(delta)<.12 else 11;c.create_oval(x-size,y-size,x+size,y+size,fill=color,outline='white')
                    continue
                start=re.match(r'([1-8])',note)
                if not start:continue
                key=start.group(1);x,y=point((cx,cy),R*progress,key);kind='break' if 'b' in note else 'ex' if 'x' in note else 'hold' if 'h' in note else 'slide' if any(s in note for s in '-<>^vpqszVw') else 'tap';color=COLORS[kind];size=14 if abs(delta)<.08 else 10
                c.create_oval(x-size,y-size,x+size,y+size,fill=color,outline='white' if kind in ('break','ex') else color,width=2)
                if kind=='slide':
                    digits=re.findall(r'[1-8]',note.split('[',1)[0]);end=digits[-1] if len(digits)>1 else key;x2,y2=point((cx,cy),R,end);c.create_line(*point((cx,cy),R,key),x2,y2,fill=color,width=3,arrow='last')
                if kind=='hold':c.create_text(x,y-20,text='H',fill=color,font=('Segoe UI',10,'bold'))
        c.create_text(cx,cy,text=f'{now:7.3f}s',fill='white',font=('Consolas',18,'bold'))
        shown='\n'.join(f'{t-now:+.3f}  {s}' for t,s in sorted(visible)[:8]);c.create_text(16,h-150,text=shown,fill='#c9d8ff',anchor='nw',font=('Consolas',10))
    def _update(self):
        now=self.current()
        if self.playing and now>=self.duration:self.playing=False;self.base=self.duration;self.stop_audio();self.play_btn.configure(text='播放')
        if not self._dragging:self.position.set(now)
        def fmt(v):return f'{int(v//60)}:{v%60:06.3f}'
        self.time_label.configure(text=f'{fmt(now)} / {fmt(self.duration)}');self._draw(now);self.after(16,self._update)
    def close(self):self.stop_audio();self.destroy()
