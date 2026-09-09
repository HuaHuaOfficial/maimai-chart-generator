from __future__ import annotations
import os, subprocess
from pathlib import Path

ROOT=Path(os.environ.get('MAIMAI_INFERENCE_ROOT', str(Path(__file__).resolve().parents[3])))
TOOLS=(ROOT/'tools'/'MajdataViewX-v6.2.0')
if not TOOLS.is_dir():
    TOOLS=ROOT/'.tools'/'MajdataViewX-v6.2.0'
EDITOR=TOOLS/'MajdataEdit-Neo.exe'


def find_miacode_executable(root: Path | None = None) -> Path:
    """Find the optional MiaCode v1 Windows editor."""
    project_root = Path(
        root or os.environ.get('MAIMAI_INFERENCE_ROOT', str(Path(__file__).resolve().parents[3]))
    ).resolve()
    candidates = []
    configured = os.environ.get('MIACODE_EXE')
    if configured:
        candidates.append(Path(os.path.expandvars(configured)).expanduser())
    candidates.extend(
        (
            project_root / 'tools' / 'MiaCode-v1.0.0-win64' / 'MiaCode.exe',
            project_root / '.tools' / 'MiaCode-v1.0.0-win64' / 'MiaCode.exe',
            project_root.parent / 'tools' / 'MiaCode-v1.0.0-win64' / 'MiaCode.exe',
            project_root.parent / '.tools' / 'MiaCode-v1.0.0-win64' / 'MiaCode.exe',
        )
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        'MiaCode 未找到；可设置 MIACODE_EXE，或安装到 tools/MiaCode-v1.0.0-win64'
        ' 或 .tools/MiaCode-v1.0.0-win64'
    )


def launch_miacode(maidata_path: Path, root: Path | None = None) -> subprocess.Popen:
    maidata_path = Path(maidata_path).resolve()
    if not maidata_path.is_file():
        raise FileNotFoundError(maidata_path)
    executable = find_miacode_executable(root)
    return subprocess.Popen(
        [str(executable), str(maidata_path)],
        cwd=str(executable.parent),
    )

def launch_majdata_preview(maidata_path:Path)->subprocess.Popen:
    maidata_path=Path(maidata_path).resolve()
    if not EDITOR.is_file():raise FileNotFoundError(EDITOR)
    if not maidata_path.is_file():raise FileNotFoundError(maidata_path)
    process=subprocess.Popen([str(EDITOR)],cwd=str(TOOLS))
    env=os.environ.copy();env['MAJDATA_TARGET']=str(maidata_path)
    script=r'''
$ws = New-Object -ComObject WScript.Shell
Start-Sleep -Seconds 3
$null = $ws.AppActivate('MajdataEdit Neo v6.2.0')
Start-Sleep -Milliseconds 500
Set-Clipboard -Value $env:MAJDATA_TARGET
$ws.SendKeys('^o')
Start-Sleep -Seconds 1
$ws.SendKeys('^v')
$ws.SendKeys('{ENTER}')
Start-Sleep -Seconds 5
$null = $ws.AppActivate('MajdataEdit Neo v6.2.0')
$ws.SendKeys('^+z')
'''
    subprocess.Popen(['powershell','-NoProfile','-WindowStyle','Hidden','-Command',script],env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    return process
