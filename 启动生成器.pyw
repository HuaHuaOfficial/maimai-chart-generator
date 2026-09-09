"""Native release entry with persistent startup diagnostics."""
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
os.environ['MAIMAI_INFERENCE_ROOT']=str(ROOT)
os.environ['MAIMAI_MERT_DIR']=str(ROOT/'models/mert')
os.environ['HF_HUB_OFFLINE']='1'
try:
    from chart_runtime.app.gui import ChartGeneratorApp
    from chart_runtime.harness.fused import load_muri_policy
    from chart_runtime import __version__
    app=ChartGeneratorApp()
    folder=ROOT/'logs';folder.mkdir(exist_ok=True)
    receipt={'pid':os.getpid(),'python':sys.executable,'root':str(ROOT),'entry':'启动生成器.pyw','release':__version__,'status':'initialized',
             'muriPolicy':load_muri_policy(ROOT)}
    (folder/'startup_status.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    if os.environ.get('CHART_RUNTIME_SMOKE_TEST')=='1':app.after(1000,app.destroy)
    app.mainloop()
    receipt['status']='closed'
    (folder/'startup_status.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
except Exception:
    folder=ROOT/'logs';folder.mkdir(exist_ok=True)
    error=traceback.format_exc();(folder/'startup_error.log').write_text(error,encoding='utf8')
    if os.name=='nt':
        import ctypes
        ctypes.windll.user32.MessageBoxW(0,error.splitlines()[-1]+'\n\n'+str(folder/'startup_error.log'),'Chart Runtime 启动失败',0x10)
    raise
