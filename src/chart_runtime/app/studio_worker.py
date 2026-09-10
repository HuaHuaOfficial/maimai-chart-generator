"""Owned UI worker for the fixed 1.0.1 native runtime."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import traceback

PREFIX='@@MAISTUDIO@@'
def emit(kind,value):
    print(PREFIX+json.dumps({'kind':kind,'value':value},ensure_ascii=True,default=str),flush=True)


def main():
    root=Path(sys.argv[1]).resolve(); mode=sys.argv[2]
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(root/'src'))
    os.environ['MAIMAI_INFERENCE_ROOT']=str(root)
    os.environ['MAIMAI_MERT_DIR']=str(root/'models/mert')
    os.environ['HF_HUB_OFFLINE']='1'
    if mode=='probe':
        import torch
        from chart_runtime.app.preparation import _find_ffmpeg
        cuda=bool(torch.cuda.is_available())
        try:ffmpeg=str(_find_ffmpeg(root))
        except Exception:ffmpeg=''
        emit('environment',{'checking':False,'cuda':cuda,'gpu':torch.cuda.get_device_name(0) if cuda else '', 'ffmpeg':ffmpeg})
        return
    data=json.loads(Path(sys.argv[3]).read_text(encoding='utf8'))
    if mode=='bpm':
        from chart_runtime.io.bpm import detect_bpm
        from chart_runtime.app.preparation import _find_ffmpeg
        result=detect_bpm(Path(data['audioPath']),_find_ffmpeg(root))
        emit('bpm',result)
    elif mode=='generate':
        from chart_runtime.app.service import generate
        result=generate(
            root=root,audio_path=Path(data['audioPath']),cover_path=Path(data['coverPath']) if data.get('coverPath') else None,
            bga_path=Path(data['bgaPath']) if data.get('bgaPath') else None,song_id=data.get('songId'),output_dir=Path(data['outputDir']),
            title=data['title'],version_id=int(data['versionId']),version_name=data['versionName'],
            levels={int(k):float(v) for k,v in data['levels'].items()},
            bpm=float(data['bpm']),exploration=float(data['exploration']),
            extra_metadata=data['extraMetadata'],progress=lambda value:emit('log',str(value)),
        )
        emit('done',result)
    else:
        raise ValueError('Unknown UI worker mode: '+mode)

if __name__=='__main__':
    try:main()
    except Exception:
        emit('error',traceback.format_exc())
        raise SystemExit(1)
