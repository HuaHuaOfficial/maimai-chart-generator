"""DX Studio: offline web frontend with a token-guarded localhost bridge.

The browser UI delegates generation to app.service.generate.
Native file dialogs and optional previews stay on Tk's main thread. Heavy inference
runs in an owned subprocess, so it never blocks the UI and can be cancelled.
"""
from __future__ import annotations
import collections
import json
import math
import mimetypes
import os
from pathlib import Path
import queue
import secrets
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from .. import __version__
from tkinter import filedialog, messagebox
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit
import uuid
import webbrowser

AUDIO_EXTENSIONS = {'.mp3','.wav','.flac','.ogg','.m4a','.aac','.opus','.wma'}
UI_BUILD = 'dx-studio-ui/2'
WEBROOT = Path(__file__).with_name('webui')
VERSIONS = ['maimai','maimai PLUS','GreeN','GreeN PLUS','ORANGE','ORANGE PLUS','PiNK','PiNK PLUS','MURASAKi','MURASAKi PLUS','MiLK','MiLK PLUS','FiNALE','でらっくす','でらっくす PLUS','Splash','Splash PLUS','UNiVERSE','UNiVERSE PLUS','FESTiVAL','FESTiVAL PLUS','BUDDiES','BUDDiES PLUS','PRiSM','PRiSM PLUS','CiRCLE','CiRCLE PLUS']


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(temp, path)


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding='utf8'))
    except (OSError, ValueError):
        return default


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    server_version = 'MaimaiStudio/1'
    def log_message(self, *args):
        pass

    @property
    def app(self):
        return self.server.app

    def _valid_host(self):
        return self.headers.get('Host') == self.app.host

    def _authorized(self):
        if not self._valid_host():
            return False
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.app.host:
            return False
        supplied = self.headers.get('X-Studio-Token', '')
        if not supplied:
            supplied = parse_qs(urlsplit(self.path).query).get('token', [''])[0]
        return secrets.compare_digest(supplied, self.app.token)

    def _headers(self, code, kind, size, extra=None):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, str(v))
        self.end_headers()

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=True, allow_nan=False).encode('utf8')
        self._headers(code, 'application/json; charset=utf-8', len(data))
        self.wfile.write(data)

    def do_GET(self):
        try:
            route = urlsplit(self.path).path
            if not self._valid_host():
                return self._json({'error':'Host rejected'},403)
            if route.startswith('/api/'):
                if not self._authorized():
                    return self._json({'error':'本地会话验证失败，请重新运行启动器。'},403)
                self.app.last_contact = time.monotonic()
                if route == '/api/audio':
                    return self._audio()
                if route == '/api/bootstrap':
                    return self._json(self.app.bootstrap())
                if route == '/api/state':
                    after = int(parse_qs(urlsplit(self.path).query).get('after',['0'])[0])
                    return self._json(self.app.state(after))
                if route == '/api/history':
                    return self._json({'items':self.app.history_items()})
                return self._json({'error':'Unknown endpoint'},404)
            names = {'/':'index.html','/index.html':'index.html','/studio.css':'studio.css','/brand.css':'brand.css','/studio.js':'studio.js','/icon.svg':'icon.svg','/favicon.ico':'icon.svg','/brand-icon.png':'brand-icon.png'}
            if route not in names:
                return self._json({'error':'Not found'},404)
            path = WEBROOT/names[route]
            data = path.read_bytes()
            kind = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.svg':'image/svg+xml','.png':'image/png'}[path.suffix]
            self._headers(200,kind,len(data)); self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self._json({'error':str(exc)},400)

    def do_POST(self):
        try:
            if not self._authorized():
                return self._json({'error':'Local session authentication failed'},403)
            self.app.last_contact = time.monotonic()
            route = urlsplit(self.path).path
            if route == '/api/upload':
                return self._upload()
            size = int(self.headers.get('Content-Length','0'))
            if not 0 <= size <= 2*1024*1024:
                return self._json({'error':'Request too large'},413)
            data = json.loads(self.rfile.read(size) or b'{}', parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Non-finite JSON')))
            if not isinstance(data,dict):
                raise ValueError('请求必须为 JSON 对象。')
            if route == '/api/settings':
                self.app.save_settings(data); result = {'ok':True}
            elif route == '/api/browse':
                result = self.app.browse(str(data.get('kind','audio')))
            elif route in ('/api/generate','/api/bpm'):
                result = self.app.start_job(route.rsplit('/',1)[1],data)
            elif route == '/api/cancel':
                result = self.app.cancel_job()
            elif route == '/api/preview':
                result = self.app.preview(data)
            elif route == '/api/open-output':
                result = self.app.open_output(data)
            else:
                return self._json({'error':'Unknown endpoint'},404)
            self._json(result)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            self._json({'error':str(exc)},400)

    def _audio(self):
        path = self.app.audio_path
        if path is None or not path.is_file():
            return self._json({'error':'No audio selected'},404)
        length=path.stat().st_size; start=0;end=length-1;code=200
        value=self.headers.get('Range','')
        if value:
            import re
            m=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
            if not m or not any(m.groups()):
                self._headers(416,'text/plain',0,{'Content-Range':f'bytes */{length}'});return
            if not m[1]:
                start=max(0,length-int(m[2]))
            else:
                start=int(m[1]); end=min(end,int(m[2])) if m[2] else end
            if start>end or start>=length:
                self._headers(416,'text/plain',0,{'Content-Range':f'bytes */{length}'});return
            code=206
        extra={'Accept-Ranges':'bytes'}
        if code==206:extra['Content-Range']=f'bytes {start}-{end}/{length}'
        kind=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        self._headers(code,kind,max(0,end-start+1),extra)
        with path.open('rb') as f:
            f.seek(start); remaining=end-start+1
            while remaining>0:
                part=f.read(min(262144,remaining))
                if not part:break
                self.wfile.write(part);remaining-=len(part)

    def _upload(self):
        if self.app.busy:
            raise ValueError('任务运行中，暂不能更换音频。')
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<=512*1024*1024:
            return self._json({'error':'音频导入上限为 512 MB。'},413)
        name=Path(unquote(self.headers.get('X-File-Name','audio.mp3')).replace('\\','/')).name
        if Path(name).suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError('不支持此音频格式。')
        folder=self.app.logs/'studio_imports'/uuid.uuid4().hex
        folder.mkdir(parents=True,exist_ok=False)
        path=folder/name
        try:
            with path.open('wb') as f:
                remaining=length
                while remaining:
                    chunk=self.rfile.read(min(1048576,remaining))
                    if not chunk:raise ValueError('音频传输中断。')
                    f.write(chunk);remaining-=len(chunk)
            self._json(self.app.select_audio(path))
        except Exception:
            path.unlink(missing_ok=True)
            raise


class ChartGeneratorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        self.title(f'maimai Chart Studio {__version__}')
        self.runtime_root=Path(os.environ.get('MAIMAI_INFERENCE_ROOT',str(Path(__file__).resolve().parents[3]))).resolve()
        self.logs=self.runtime_root/'logs';self.logs.mkdir(parents=True,exist_ok=True)
        self.native_tasks=queue.Queue();self.lock=threading.RLock();self.events=collections.deque(maxlen=1600);self.seq=0
        self.busy=False;self.kind='';self.job_id='';self.child=None;self.probe=None;self.cancelled=False;self.started=0.;self.duration=0.;self.closed=False
        self.audio_path=None;self.last_result=None;self.last_contact=time.monotonic()
        self.settings=_load_json(self.logs/'studio_settings.json',{})
        if not isinstance(self.settings,dict):self.settings={}
        self.environment={'checking':True}
        previous=self.settings.get('audioPath')
        if previous and Path(previous).is_file() and Path(previous).suffix.lower() in AUDIO_EXTENSIONS:self.audio_path=Path(previous)
        self.token=secrets.token_urlsafe(32)
        self.server=StudioServer(('127.0.0.1',0),Handler);self.server.app=self
        self.host=f'127.0.0.1:{self.server.server_port}'
        self.url='http://'+self.host+'/#token='+self.token
        threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.2},daemon=True).start()
        self.after(50,self._poll_native)
        self.after(15000,self._watchdog)
        if os.environ.get('CHART_RUNTIME_SMOKE_TEST')!='1':
            self.after(150,self._open_browser)
            threading.Thread(target=self._probe_environment,daemon=True).start()
        else:
            self.environment={'checking':False,'cuda':False,'ffmpeg':False,'error':'UI smoke test; environment probe not run'}

    def _python(self):
        p=Path(sys.executable)
        sibling=p.with_name('python.exe')
        return str(sibling if p.name.lower()=='pythonw.exe' and sibling.is_file() else p)

    def _spawn(self,mode,request_path):
        env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8';env['MAIMAI_INFERENCE_ROOT']=str(self.runtime_root)
        env['MAIMAI_MERT_DIR']=str(self.runtime_root/'models/mert');env['HF_HUB_OFFLINE']='1'
        return subprocess.Popen([self._python(),'-X','utf8','-u',str(Path(__file__).with_name('studio_worker.py')),str(self.runtime_root),mode,str(request_path)],cwd=self.runtime_root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))

    def _probe_environment(self):
        try:
            self.probe=self._spawn('probe','-')
            output,_=self.probe.communicate(timeout=90)
            found=False
            for line in output.splitlines():
                if line.startswith('@@MAISTUDIO@@'):
                    record=json.loads(line[13:]);self.environment=record['value'];found=True
            if not found:raise RuntimeError(output[-500:] or '环境检查未返回结果')
        except Exception as exc:
            if self.probe and self.probe.poll() is None:self.probe.kill()
            self.environment={'checking':False,'cuda':False,'ffmpeg':False,'error':str(exc)[-300:]}

    def _open_browser(self):
        if self.closed:return
        candidates=[]
        for key in ('PROGRAMFILES(X86)','PROGRAMFILES','LOCALAPPDATA'):
            root=Path(os.environ.get(key,''))
            candidates.extend([root/'Microsoft/Edge/Application/msedge.exe',root/'Google/Chrome/Application/chrome.exe'])
        executable=next((str(x) for x in candidates if x.is_file()),None)
        try:
            if executable:
                profile=Path(os.environ.get('LOCALAPPDATA',str(self.logs)))/'MaimaiChartStudio/browser'
                subprocess.Popen([executable,'--app='+self.url,'--new-window','--window-size=1440,1030','--no-first-run','--user-data-dir='+str(profile)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            else:webbrowser.open(self.url)
        except Exception:
            webbrowser.open(self.url)

    def _native(self,func):
        done=threading.Event();result={};self.native_tasks.put((func,done,result))
        if not done.wait(300):raise TimeoutError('文件选择超时，请重试。')
        if 'error' in result:raise RuntimeError(result['error'])
        return result.get('value')

    def _dialog(self,chooser):
        # The visible Studio is an Edge/Chrome app window; Tk itself stays
        # withdrawn. A native common dialog owned by that hidden root can be
        # placed behind the maximized browser window. Give each dialog a
        # short-lived, invisible TOPMOST owner instead of making the whole
        # application permanently topmost.
        owner=tk.Toplevel(self)
        owner.withdraw();owner.title('maimai Chart Studio')
        try:owner.attributes('-toolwindow',True)
        except tk.TclError:pass
        owner.overrideredirect(True);owner.geometry('1x1+0+0')
        try:owner.attributes('-alpha',0.01)
        except tk.TclError:pass
        owner.attributes('-topmost',True);owner.deiconify();owner.lift()
        try:owner.focus_force()
        except tk.TclError:pass
        # update(), not only update_idletasks(), guarantees a real HWND exists
        # before Windows creates the owned IFileDialog/common dialog.
        owner.update()
        try:return chooser(owner)
        finally:
            try:owner.attributes('-topmost',False)
            except tk.TclError:pass
            try:owner.destroy()
            except tk.TclError:pass

    def _poll_native(self):
        if self.closed:return
        try:
            while True:
                func,done,result=self.native_tasks.get_nowait()
                try:result['value']=func()
                except Exception as exc:result['error']=str(exc)
                finally:done.set()
        except queue.Empty:pass
        self.after(60,self._poll_native)

    def _watchdog(self):
        if self.closed:return
        if not self.busy and time.monotonic()-self.last_contact>120:
            self.destroy();return
        self.after(15000,self._watchdog)

    def emit(self,kind,value):
        with self.lock:
            self.seq+=1;self.events.append({'seq':self.seq,'kind':kind,'value':value})

    def bootstrap(self):
        profile=_load_json(self.runtime_root/'models/experimental/what_complexity_profile.json',{})
        supported={slot:[int(value)/10 for value in sorted(entries,key=int)] for slot,entries in profile.get('slots',{}).items()}
        return {'build':UI_BUILD,'outputDir':str(self.runtime_root/'generated'),'settings':self.settings,'audio':self.audio_info(),'versions':VERSIONS,'supportedLevels':supported}

    def audio_info(self):
        p=self.audio_path
        return {'path':str(p),'name':p.name,'size':p.stat().st_size} if p and p.is_file() else None

    def select_audio(self,path):
        path=Path(path).resolve()
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:raise ValueError('请选择有效的音频文件。')
        self.audio_path=path;return self.audio_info()

    def browse(self,kind):
        if kind=='audio':
            chosen=self._native(lambda:self._dialog(lambda owner:filedialog.askopenfilename(parent=owner,title='选择乐曲音频',filetypes=[('音频','*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.opus *.wma'),('全部文件','*.*')])))
            return self.select_audio(chosen) if chosen else {'path':''}
        if kind=='output':
            chosen=self._native(lambda:self._dialog(lambda owner:filedialog.askdirectory(parent=owner,title='选择输出目录',initialdir=str(self.runtime_root))))
        elif kind=='cover':
            chosen=self._native(lambda:self._dialog(lambda owner:filedialog.askopenfilename(parent=owner,title='选择封面图片',filetypes=[('封面图片','*.png *.jpg *.jpeg *.webp *.bmp')])))
        elif kind=='bga':
            chosen=self._native(lambda:self._dialog(lambda owner:filedialog.askopenfilename(parent=owner,title='选择BGA MP4',filetypes=[('MP4视频','*.mp4')])))
        else:raise ValueError('不支持此选择器。')
        return {'path':chosen or ''}

    def save_settings(self,data):
        if len(json.dumps(data))>1024*1024:raise ValueError('配置过大。')
        with self.lock:self.settings=dict(data);_write_json(self.logs/'studio_settings.json',data)

    def state(self,after):
        with self.lock:
            elapsed=time.monotonic()-self.started if self.busy else self.duration
            return {'busy':self.busy,'kind':self.kind,'jobId':self.job_id,'elapsed':elapsed,'environment':self.environment,'events':[x for x in self.events if x['seq']>after]}

    def _validate(self,mode,data):
        audio=Path(str(data.get('audioPath',''))).expanduser()
        if not audio.is_file() or audio.suffix.lower() not in AUDIO_EXTENSIONS:raise ValueError('音频路径不存在或格式不受支持。')
        clean=dict(data);clean['audioPath']=str(audio.resolve())
        if mode=='bpm':return clean
        version=int(data.get('versionId',26))
        if not 0<=version<len(VERSIONS):raise ValueError('无效的谱面版本。')
        clean['versionId']=version;clean['versionName']=VERSIONS[version]
        levels={int(k):float(v) for k,v in data.get('levels',{}).items()}
        if not levels or any(s not in (2,3,4,5,6) or not math.isfinite(v) or not 1<=v<=15 or abs(v*10-round(v*10))>1e-5 for s,v in levels.items()):raise ValueError('请选择至少一个难度，定数范围为1.0–15.0，精确到0.1。')
        clean['levels']=levels
        for key,lo,hi in [('bpm',1,2000),('exploration',0,3)]:
            value=float(data.get(key,0))
            if not math.isfinite(value) or not lo<=value<=hi:raise ValueError('无效的 '+key)
            clean[key]=value
        extra=clean.get('extraMetadata',{})
        if not isinstance(extra,dict):raise ValueError('附加元数据必须是对象。')
        extra=dict(extra)
        mode=str(extra.get('causalSearchMode','stable'))
        if mode not in ('stable','causal-v1'):raise ValueError('无效的因果搜索模式。')
        extra['causalSearchMode']=mode
        for key in ('whatStarScale','whatArity2Scale','whatHoldScale','whatTouchScale','whatTouchHoldScale'):
            n=float(extra.get(key,1))
            if not math.isfinite(n) or not 0<=n<=3:raise ValueError('无效的元素倍率：'+key)
            extra[key]=n
        if version<13:extra['whatTouchScale']=extra['whatTouchHoldScale']=0.
        clean['extraMetadata']=extra
        song_id=str(data.get('songId','')).strip()
        if song_id and (song_id in ('.','..') or len(song_id)>80 or song_id.rstrip(' .')!=song_id or re.search(r'[<>:"/\\|?*\x00-\x1f]',song_id)):raise ValueError('请输入有效的自定义歌曲ID。')
        clean['songId']=song_id
        cover_text=str(data.get('coverPath','')).strip()
        if cover_text:
            cover=Path(cover_text).expanduser()
            if not cover.is_file() or cover.suffix.lower() not in ('.png','.jpg','.jpeg','.webp','.bmp'):raise ValueError('请选择有效的封面图片。')
            clean['coverPath']=str(cover.resolve())
        else:clean['coverPath']=''
        bga_text=str(data.get('bgaPath','')).strip()
        if bga_text:
            bga=Path(bga_text).expanduser()
            if not bga.is_file() or bga.suffix.lower()!='.mp4':raise ValueError('请选择有效的BGA MP4。')
            clean['bgaPath']=str(bga.resolve())
        else:clean['bgaPath']=''
        output=str(data.get('outputDir','')).strip()
        clean['outputDir']=str(Path(output).expanduser().resolve() if output else self.runtime_root/'generated')
        return clean

    def start_job(self,mode,data):
        clean=self._validate(mode,data)
        with self.lock:
            if self.busy:raise ValueError('已有任务在运行。')
            self.busy=True;self.kind=mode;self.cancelled=False;self.started=time.monotonic();self.duration=0.;self.job_id=uuid.uuid4().hex
            request_path=self.logs/f'studio_{mode}_request.json'
            _write_json(request_path,clean)
            try:self.child=self._spawn(mode,request_path)
            except Exception:self.busy=False;raise
            self.emit('log','开始 BPM / 首拍检测…' if mode=='bpm' else '已提交真实生成任务，等待引擎初始化…')
            threading.Thread(target=self._read_job,args=(self.child,clean,mode),daemon=True).start()
        return {'ok':True,'jobId':self.job_id}

    def _read_job(self,child,request,mode):
        terminal=False
        with (self.logs/f'studio_{mode}.log').open('w',encoding='utf8') as log:
            for raw in child.stdout:
                log.write(raw);log.flush();line=raw.rstrip()
                if not line:continue
                if line.startswith('@@MAISTUDIO@@'):
                    try:
                        event=json.loads(line[13:]);kind=event['kind'];value=event['value']
                    except (ValueError,KeyError):self.emit('log',line);continue
                    if kind in ('done','bpm','error'):
                        terminal=True
                        if kind=='done':
                            self.last_result=value
                            self._record_history(value)
                    self.emit(kind,value)
                else:self.emit('log',line)
        code=child.wait()
        with self.lock:
            self.duration=time.monotonic()-self.started
            if self.cancelled:self.emit('cancelled',{})
            elif not terminal:self.emit('error',f'后端进程异常结束（退出码 {code}），请查看完整日志。')
            self.busy=False
            if self.child is child:self.child=None

    def cancel_job(self):
        with self.lock:
            if not self.busy or self.child is None:
                return {'ok':True,'cancelled':False,'state':'idle'}
            child=self.child;job_id=self.job_id
            if child.poll() is not None:
                return {'ok':True,'cancelled':False,'state':'already-finished','pid':child.pid,'jobId':job_id}
            # Set this before termination so the reader thread cannot race an
            # expected exit into a false backend-error event.
            self.cancelled=True
        self.emit('log',f'正在中断生成引擎（PID {child.pid}）…')
        try:
            if os.name=='nt':
                killed=subprocess.run(
                    ['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,
                    text=True,encoding='utf8',errors='replace',
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=15,
                )
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
                if child.poll() is None:
                    detail=(killed.stderr or killed.stdout or '').strip()
                    raise RuntimeError('无法中断生成引擎。'+((' '+detail) if detail else ''))
                method='taskkill-tree'
            else:
                child.terminate();child.wait(timeout=10);method='terminate'
        except Exception:
            with self.lock:self.cancelled=False
            raise
        return {'ok':True,'cancelled':True,'state':'terminated','pid':child.pid,'jobId':job_id,'method':method,'exitCode':child.returncode}

    def _record_history(self,result):
        with self.lock:
            entries=_load_json(self.logs/'studio_history.json',[])
            if not isinstance(entries,list):entries=[]
            levels=result.get('levels',{})
            summary=' / '.join(f'{k}: {v}' for k,v in levels.items())
            entry={'id':uuid.uuid4().hex,'title':result.get('title',''),'versionName':result.get('versionName',''),'time':time.strftime('%Y-%m-%d %H:%M'),'outputDir':result['outputDir'],'summary':summary}
            entries.insert(0,entry);_write_json(self.logs/'studio_history.json',entries[:100])

    def history_items(self):
        entries=_load_json(self.logs/'studio_history.json',[])
        return entries if isinstance(entries,list) else []

    def _output_path(self,record_id=None):
        if record_id:
            entry=next((r for r in self.history_items() if r.get('id')==record_id),None)
            if not entry:raise ValueError('找不到这条生成记录。')
            return Path(entry['outputDir'])
        if self.last_result:return Path(self.last_result['outputDir'])
        history=self.history_items()
        if history:return Path(history[0]['outputDir'])
        return None

    def open_output(self,data):
        path=self._output_path(data.get('id'))
        if path is None or not path.is_dir():raise ValueError('输出目录不存在。')
        if os.name=='nt':os.startfile(str(path))
        else:subprocess.Popen(['xdg-open',str(path)])
        return {'ok':True}

    def preview(self,data):
        kind=data.get('kind')
        if kind not in ('preview','miacode'):raise ValueError('无效预览操作。')
        from .external_preview import find_miacode_executable,find_majdata_executable,launch_miacode,launch_majdata_preview
        if kind=='miacode':
            find_miacode_executable(self.runtime_root)
        else:
            find_majdata_executable(self.runtime_root)
        folder=self._output_path(data.get('id'));path=folder/'maidata.txt' if folder else None
        if path is None or not path.is_file():
            value=self._native(lambda:self._dialog(lambda owner:filedialog.askopenfilename(parent=owner,title='选择要预览的谱面',filetypes=[('谱面','maidata.txt'),('文本','*.txt')])))
            if not value:return {'ok':True,'message':'已取消选择。'}
            path=Path(value)
        if kind=='miacode':
            launch_miacode(path,self.runtime_root);return {'ok':True,'message':'已在 MiaCode 中打开谱面。'}
        launch_majdata_preview(path,self.runtime_root);return {'ok':True,'message':'已在 MajdataViewX 中打开谱面。'}

    def destroy(self):
        if getattr(self,'closed',False):return
        self.closed=True
        if getattr(self,'busy',False):
            try:self.cancel_job()
            except Exception:pass
        if self.probe and self.probe.poll() is None:self.probe.terminate()
        if hasattr(self,'server'):
            self.server.shutdown();self.server.server_close()
        super().destroy()


def main():
    ChartGeneratorApp().mainloop()

if __name__=='__main__':main()
