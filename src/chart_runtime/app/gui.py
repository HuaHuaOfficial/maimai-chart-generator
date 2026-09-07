from __future__ import annotations

import json
import os
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch

from .preview import PreviewWindow
from .external_preview import launch_majdata_preview
from ..io.bpm import detect_bpm
from .service import available_models, generate
from .preparation import _find_ffmpeg


VERSIONS = [
    (0, "maimai"), (1, "maimai PLUS"), (2, "GreeN"), (3, "GreeN PLUS"),
    (4, "ORANGE"), (5, "ORANGE PLUS"), (6, "PiNK"), (7, "PiNK PLUS"),
    (8, "MURASAKi"), (9, "MURASAKi PLUS"), (10, "MiLK"), (11, "MiLK PLUS"),
    (12, "FiNALE"), (13, "でらっくす"), (14, "でらっくす PLUS"),
    (15, "スプラッシュ"), (16, "スプラッシュ PLUS"),
    (17, "UNiVERSE"), (18, "UNiVERSE PLUS"), (19, "FESTiVAL"),
    (20, "FESTiVAL PLUS"), (21, "BUDDiES"), (22, "BUDDiES PLUS"),
    (23, "PRiSM"), (24, "PRiSM PLUS"), (25, "CiRCLE"), (26, "CiRCLE PLUS"),
]
DIFFICULTIES = [(2, "BASIC"), (3, "ADVANCED"), (4, "EXPERT"), (5, "MASTER"), (6, "Re:MASTER"), (7, "UTAGE")]
FOCUS_MODES = {"平衡（旋律+节奏）":"balanced", "人声优先":"vocal", "旋律优先":"melody", "节奏优先":"rhythm"}
BACKEND_OPTIONS = {"CUDA 原生后端":"cuda"}


class ChartGeneratorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("maimai Chart Runtime 0.4")
        self.geometry("1000x820")
        self.minsize(820, 700)
        self.messages: queue.Queue[tuple[str, str]] = queue.Queue()
        self.runtime_root = Path(os.environ.get("MAIMAI_INFERENCE_ROOT", str(Path(__file__).resolve().parents[3])))
        bundled_models = available_models(self.runtime_root)
        self.model_options = {}
        for spec in bundled_models:
            try:
                path = str(
                    spec.renderer_checkpoint.resolve().relative_to(
                        self.runtime_root.resolve()
                    )
                )
            except ValueError:
                path = str(spec.renderer_checkpoint.resolve())
            self.model_options[spec.label] = path
        self.model_path=tk.StringVar(value="models/renderer_v4_contextual.pt")
        self.model_choice=tk.StringVar(value=bundled_models[0].label)
        self.last_output_dir: Path | None = None
        self._build()
        self.after(100, self._poll)

    def _build(self) -> None:
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        row = 0

        def path_row(label: str, variable: tk.StringVar, command) -> None:
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8)
            ttk.Button(frame, text="浏览…", command=command).grid(row=row, column=2)
            row += 1

        self.mp3 = tk.StringVar()
        self.output = tk.StringVar(value=str(self.runtime_root / "generated"))
        path_row("乐曲音频", self.mp3, self._choose_mp3)
        path_row("输出目录", self.output, self._choose_output)

        self.title_value = tk.StringVar(value="AI Generated Chart")
        self.version = tk.StringVar(value="26 | CiRCLE PLUS")
        self.bpm = tk.StringVar(value="")
        self.threshold = tk.StringVar(value="0.80")
        self.focus = tk.StringVar(value="平衡（旋律+节奏）")
        self.backend = tk.StringVar(value="CUDA 原生后端")
        self.star_target_ratio = tk.DoubleVar(value=0.5)
        fields = [
            ("曲名", ttk.Entry(frame, textvariable=self.title_value)),
            ("版本", ttk.Combobox(frame, textvariable=self.version, state="readonly", values=[f"{i} | {n}" for i, n in VERSIONS])),
            ("BPM（选择音频后自动识别）", ttk.Entry(frame, textvariable=self.bpm)),
            ("全曲风格探索强度", ttk.Entry(frame, textvariable=self.threshold)),
            ("采音侧重", ttk.Combobox(frame, textvariable=self.focus, state="readonly", values=list(FOCUS_MODES))),
            ("推理后端", ttk.Combobox(frame, textvariable=self.backend, state="readonly", values=list(BACKEND_OPTIONS))),
        ]
        for label, widget in fields:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            widget.grid(row=row, column=1, columnspan=2, sticky="ew", padx=8)
            row += 1
        ttk.Label(frame, text="星星目标比例（相对官谱参考）").grid(row=row, column=0, sticky="w", pady=4)
        star_frame=ttk.Frame(frame);star_frame.grid(row=row,column=1,columnspan=2,sticky="ew",padx=8);star_frame.columnconfigure(1,weight=1)
        ttk.Label(star_frame,text="少").grid(row=0,column=0,padx=(0,6))
        ttk.Scale(star_frame,from_=0.0,to=1.0,variable=self.star_target_ratio,orient="horizontal").grid(row=0,column=1,sticky="ew")
        ttk.Label(star_frame,text="多").grid(row=0,column=2,padx=(6,0))
        row += 1
        ttk.Label(frame,text="生成模型").grid(row=row,column=0,sticky="w",pady=4)
        ttk.Label(frame,textvariable=self.model_choice).grid(row=row,column=1,columnspan=2,sticky="w",padx=8)
        row+=1
        device_label = "GPU：" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA Harness 不可用（禁止CPU逐项检查）"
        ttk.Label(frame, text=device_label, foreground="#176b3a" if torch.cuda.is_available() else "#8a5b00").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 4)
        )
        row += 1
        self.contextual_repair = tk.BooleanVar(value=True)
        levels_frame = ttk.LabelFrame(frame, text="各难度精确定数（一次生成）", padding=8)
        levels_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        self.level_vars = {2:tk.StringVar(value="5.0"), 3:tk.StringVar(value="8.0"), 4:tk.StringVar(value="11.0"), 5:tk.StringVar(value="14.0"), 6:tk.StringVar(value="14.5")}
        for column,(slot,name) in enumerate(DIFFICULTIES[:5]):
            ttk.Label(levels_frame,text=name).grid(row=0,column=column,padx=5)
            ttk.Entry(levels_frame,textvariable=self.level_vars[slot],width=8).grid(row=1,column=column,padx=5)
        self.include_remaster=tk.BooleanVar(value=False)
        ttk.Checkbutton(levels_frame,text="生成 Re:MASTER",variable=self.include_remaster).grid(row=2,column=4,pady=(5,0))
        row += 1
        ttk.Button(frame, text="自动识别 BPM / 首拍", command=self._detect_bpm).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(2, 6))
        row += 1

        ttk.Label(frame, text="附加元数据 JSON").grid(row=row, column=0, sticky="nw", pady=4)
        self.metadata = tk.Text(frame, height=10, wrap="word")
        self.metadata.insert("1.0", '{\n  "artist": "",\n  "genre": "",\n  "first": 0,\n  "displayLevel": "14",\n  "bpmChanges": []\n}')
        self.metadata.grid(row=row, column=1, columnspan=2, sticky="nsew", padx=8)
        frame.rowconfigure(row, weight=2)
        row += 1

        self.generate_button = ttk.Button(frame, text="生成谱面", command=self._start)
        self.generate_button.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 6))
        row += 1
        ttk.Button(frame, text="预览输出谱面", command=self._preview).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        row += 1
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=row, column=0, columnspan=3, sticky="ew")
        row += 1
        self.log = tk.Text(frame, height=7, state="disabled", wrap="word")
        self.log.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        frame.rowconfigure(row, weight=1)

    def _resolve_model_path(self, value: str) -> Path:
        path = Path(os.path.expandvars(value.strip())).expanduser()
        if not path.is_absolute():
            path = self.runtime_root / path
        return path.resolve()

    def _choose_model(self) -> None:
        value = filedialog.askopenfilename(
            initialdir=str(self.runtime_root),
            filetypes=[("PyTorch checkpoint", "*.pt *.pth *.bin"), ("All", "*.*")],
        )
        if value:
            self.model_choice.set("自定义 checkpoint")
            selected = Path(value).resolve()
            try:
                self.model_path.set(str(selected.relative_to(self.runtime_root.resolve())))
            except ValueError:
                self.model_path.set(str(selected))

    def _select_bundled_model(self, _event=None) -> None:
        selected = self.model_options.get(self.model_choice.get())
        if selected is not None:
            self.model_path.set(selected)
            if 'renderer_v4_contextual' in selected:self.contextual_repair.set(True)

    def _choose_mp3(self) -> None:
        value = filedialog.askopenfilename(filetypes=[("常见音频", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.opus *.wma"), ("MP3", "*.mp3"), ("All", "*.*")])
        if value:
            self.mp3.set(value)
            # A newly selected audio file is a new chart source: reset the
            # title so an earlier manual title cannot leak into its maidata.
            self.title_value.set(Path(value).stem)
            self._detect_bpm()

    def _detect_bpm(self) -> None:
        path = Path(self.mp3.get())
        if not path.is_file():
            return
        self.bpm.set("识别中…")
        self._bpm_request_id = getattr(self,'_bpm_request_id',0)+1
        request_id=self._bpm_request_id
        self._write_log("正在用音频中间90%识别 BPM 和拍点相位（忽略首尾各5%，不裁剪生成音频）…")
        def worker() -> None:
            try:
                self._save_log('last_bpm_request.json',json.dumps({'audioPath':str(path)},ensure_ascii=False,indent=2))
                result = detect_bpm(path, _find_ffmpeg(self.runtime_root))
                result['_requestId']=request_id
                self.messages.put(("bpm", json.dumps(result, ensure_ascii=False)))
            except Exception:
                details = traceback.format_exc()
                self._save_log('last_bpm_error.log',details)
                self.messages.put(("bpm_error", str(request_id)))
                self.messages.put(("log", "BPM识别失败：" + details.splitlines()[-1]))
        threading.Thread(target=worker, daemon=True).start()

    def _choose_output(self) -> None:
        value = filedialog.askdirectory()
        if value:
            self.output.set(value)

    def _write_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _save_log(self,name,text):
        try:
            folder=self.runtime_root/'logs';folder.mkdir(exist_ok=True)
            (folder/name).write_text(text,encoding='utf8')
        except OSError as exc:
            self.messages.put(('log',f'日志保存失败：{exc}'))

    def _start(self) -> None:
        try:
            extra = json.loads(self.metadata.get("1.0", "end").strip() or "{}")
            if not isinstance(extra,dict):raise ValueError('附加元数据必须是 JSON 对象')
            version_id = int(self.version.get().split("|", 1)[0])
            version_name = self.version.get().split("|", 1)[1].strip()
            levels = {slot:float(self.level_vars[slot].get()) for slot in (2,3,4,5)}
            if self.include_remaster.get(): levels[6]=float(self.level_vars[6].get())
            bpm = float(self.bpm.get())
            threshold = float(self.threshold.get())
            extra["mappingFocus"] = FOCUS_MODES[self.focus.get()]
            extra["difficultyWorkloadMode"] = "auto"
            extra["starTargetRatio"] = float(self.star_target_ratio.get())
            extra['contextualRepair'] = True
            inference_backend = BACKEND_OPTIONS[self.backend.get()]
            model_path = self._resolve_model_path(self.model_path.get())
            if not model_path.is_file():
                raise FileNotFoundError(f"模型 checkpoint 不存在：{model_path}")
            if not (self.runtime_root/"models/v2/playability_tables.json").is_file():
                raise FileNotFoundError("缺少原生模型资源，请完整解压发布包")
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc)); return
        self.generate_button.configure(state="disabled")
        self.progress.start(10)
        self.last_output_dir = None
        self._write_log(f"开始生成：{model_path}；后端={self.backend.get()}…")
        audio_path=Path(self.mp3.get());output_path=Path(self.output.get());chart_title=self.title_value.get()
        request_record = {
            'audioPath':str(audio_path),'outputDir':str(output_path),'title':chart_title,
            'versionId':version_id,'versionName':version_name,'levels':levels,'bpm':bpm,
            'exploration':threshold,'modelPath':str(model_path),'backend':inference_backend,'extraMetadata':extra,
        }

        def worker() -> None:
            try:
                self._save_log('last_generation_request.json',json.dumps(request_record,ensure_ascii=False,indent=2))
                result = generate(
                    root=self.runtime_root, renderer_path=model_path,
                    audio_path=audio_path, output_dir=output_path,
                    title=chart_title, version_id=version_id, version_name=version_name,
                    levels=levels, bpm=bpm, exploration=threshold, extra_metadata=extra,
                    progress=lambda value: self.messages.put(("log", value)),
                    inference_backend=inference_backend,
                )
                self._save_log('last_generation_result.json',json.dumps(result,ensure_ascii=False,indent=2))
                self.messages.put(("done", json.dumps(result, ensure_ascii=False, indent=2)))
            except Exception:
                details = traceback.format_exc()
                self._save_log('last_generation_error.log',details)
                self.messages.put(("error", details))

        threading.Thread(target=worker, daemon=True).start()

    def _preview(self) -> None:
        candidates = sorted(
            Path(self.output.get()).glob("*/maidata.txt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if self.last_output_dir is not None:
            path = self.last_output_dir / "maidata.txt"
        elif candidates:
            path = candidates[0]
        else:
            path = Path(self.output.get()) / "maidata.txt"
        if not path.is_file():
            chosen = filedialog.askopenfilename(filetypes=[("maidata", "maidata.txt"), ("Text", "*.txt")])
            if not chosen:
                return
            path = Path(chosen)
        try:
            launch_majdata_preview(path)
            self._write_log("已交给 MajdataViewX v6.2.0 打开并预览")
        except Exception as exc:
            self._write_log(f"MajdataViewX启动失败，使用内置回退预览：{exc}")
            try:
                PreviewWindow(self, path)
            except Exception as fallback_exc:
                messagebox.showerror("预览失败", str(fallback_exc))

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "log":
                    self._write_log(value)
                elif kind == "bpm":
                    result=json.loads(value)
                    if result.pop('_requestId',getattr(self,'_bpm_request_id',0))!=getattr(self,'_bpm_request_id',0):continue
                    self.bpm.set(str(result["bpm"])); self._write_log(f"BPM识别：{result}")
                    try:
                        extra = json.loads(self.metadata.get("1.0", "end").strip() or "{}")
                        extra["detectedBeatOffsetSeconds"] = result["beatOffsetSeconds"]
                        extra["bpmCandidates"] = result["candidates"]
                        self.metadata.delete("1.0", "end"); self.metadata.insert("1.0", json.dumps(extra, ensure_ascii=False, indent=2))
                    except Exception:
                        pass
                elif kind == "bpm_error":
                    if value and int(value)!=getattr(self,'_bpm_request_id',0):continue
                    self.bpm.set("")
                else:
                    self.progress.stop(); self.generate_button.configure(state="normal")
                    self._write_log(value)
                    if kind == "done":
                        try:
                            result = json.loads(value)
                            output_dir = result.get("outputDir")
                            if output_dir:
                                self.last_output_dir = Path(output_dir)
                                self._write_log(f"本次生成目录：{output_dir}")
                        except Exception:
                            pass
                        messagebox.showinfo("完成", "谱面生成完成")
                    else:
                        messagebox.showerror("生成失败", value[-3000:])
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main() -> None:
    ChartGeneratorApp().mainloop()


if __name__ == "__main__":
    main()
