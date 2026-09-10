from __future__ import annotations
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

AUDIO_CACHE_CONTRACT='mp3-320k-v1'
_AUDIO_LOCK=threading.Lock()

def _cache_key(path: Path, contract: str) -> str:
    path=Path(path).resolve();st=path.stat()
    value=f'{contract}|{str(path).casefold()}|{st.st_size}|{st.st_mtime_ns}'
    return hashlib.sha256(value.encode('utf8')).hexdigest()

def find_ffmpeg(root: Path) -> Path:
    direct=Path(root)/'tools/ffmpeg/ffmpeg.exe'
    if direct.is_file():return direct
    tree=Path(root)/'tools/ffmpeg'
    if tree.is_dir():
        found=next((p for p in tree.glob('**/ffmpeg.exe') if p.is_file()),None)
        if found is not None:return found
    command=shutil.which('ffmpeg')
    if command:return Path(command)
    raise FileNotFoundError('找不到 FFmpeg，无法生成媒体缓存。')

def ensure_track_mp3(root: Path, source: Path, ffmpeg: Path | None=None):
    source=Path(source).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    folder=Path(root)/'logs/media_cache/audio';folder.mkdir(parents=True,exist_ok=True)
    target=folder/('audio_'+_cache_key(source,AUDIO_CACHE_CONTRACT)+'.mp3')
    with _AUDIO_LOCK:
        if target.is_file() and target.stat().st_size>0:return target,True
        temp=folder/(target.stem+f'.{os.getpid()}.{time.time_ns()}.tmp.mp3')
        try:
            exe=Path(ffmpeg) if ffmpeg else find_ffmpeg(root)
            cmd=[str(exe),'-y','-v','error','-i',str(source),'-map','0:a:0','-vn','-sn','-dn','-c:a','libmp3lame','-b:a','320k',str(temp)]
            result=subprocess.run(cmd,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode!=0 or not temp.is_file() or temp.stat().st_size==0:
                detail=(result.stderr or '').strip()[-800:]
                raise RuntimeError('FFmpeg 无法生成 320 kbps MP3 缓存。 '+detail)
            os.replace(temp,target)
            cached=sorted(folder.glob('audio_*.mp3'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
            for old in cached[8:]:old.unlink(missing_ok=True)
            return target,False
        finally:
            temp.unlink(missing_ok=True)

def stage_track_mp3(root: Path, source: Path, output: Path, ffmpeg: Path | None=None):
    cached,hit=ensure_track_mp3(root,source,ffmpeg)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(cached,output)
    return cached,hit
