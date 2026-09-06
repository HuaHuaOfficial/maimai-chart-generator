"""Native audio, music-structure and timing frontend for the pinned model set."""
from pathlib import Path
import os,math,json
import numpy as np
import torch
from transformers import AutoModel,Wav2Vec2FeatureExtractor
from ..io.audio import FRAME_SECONDS,decode_mp3
from ..io.timing import ticks_to_seconds,sample_positions
from .models.structure import ChartTransformerV2,V2Config
from .models.anchor import AnchorRelationModel
from .models.style import StylePrior,StylePriorConfig
from .models.density_calibrator import DensityCalibrator,DensityCalibratorConfig
from .tokens import metadata_tokens,GROUP_BY_SLOT
TPB=384
MERT_SR=24000

def accelerate_module(model,*args,**kwargs):
    # The native release has one tested CUDA backend; no legacy/TRT fallbacks.
    return model

class NativeFrontend:
    def __init__(self,root):
        self.root=Path(root)/'models/v2'

    def tempo_arrays(self,bpm: float, changes: list[dict] | None):
        if not np.isfinite(bpm) or bpm<=0:raise ValueError('BPM must be positive and finite')
        if changes is not None and not isinstance(changes,list):raise ValueError('bpmChanges must be a list')
        points = {0: float(bpm)}
        for item in changes or []:
            if not isinstance(item,dict):raise ValueError('Each BPM change must be an object')
            position=int(item.get('bar',0))*TPB+int(item.get('tick',0));value=float(item['bpm'])
            if position<0 or not np.isfinite(value) or value<=0:raise ValueError('BPM change position and tempo are invalid')
            points[position]=value
        ticks = np.asarray(sorted(points), np.int32)
        # Tempo is part of the immutable chart identity and is serialized to
        # Simai. float32 changes ordinary decimal BPMs (for example 129.05),
        # so the published round trip would no longer match its Harness permit.
        return ticks, np.asarray([points[int(t)] for t in ticks], np.float64)

    def duration_to_ticks(self,seconds: float, bpm_ticks: np.ndarray, bpm_values: np.ndarray) -> int:
        elapsed = 0.0
        cursor = 0
        for index in range(1, len(bpm_ticks)):
            span = int(bpm_ticks[index] - cursor)
            segment_seconds = span * 240.0 / (float(bpm_values[index - 1]) * TPB)
            if elapsed + segment_seconds >= seconds:
                return cursor + int(math.ceil((seconds - elapsed) * float(bpm_values[index - 1]) * TPB / 240.0))
            elapsed += segment_seconds
            cursor = int(bpm_ticks[index])
        return cursor + int(math.ceil(max(0.0, seconds - elapsed) * float(bpm_values[-1]) * TPB / 240.0))

    def extract_mert(self,audio_path: Path, ffmpeg: Path, device: torch.device, batch_size: int = 12, chunk_seconds: int = 20):
        local_source = os.environ.get("MAIMAI_MERT_DIR", "").strip()
        source = local_source or str(self.root.parent/'mert')
        if not Path(source).is_dir():raise FileNotFoundError(f'离线MERT资源缺失：{source}')
        load_options = {"trust_remote_code": True}
        load_options["local_files_only"] = True
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(source, **load_options)
        model = AutoModel.from_pretrained(source, **load_options).to(device).eval()
        model = accelerate_module(model, "mert", allow_tensorrt=False)
        wave = decode_mp3(audio_path, ffmpeg, MERT_SR)
        chunk = chunk_seconds * MERT_SR
        parts, lengths = [], []
        for start in range(0, len(wave), chunk):
            part = wave[start : start + chunk]
            lengths.append(len(part))
            parts.append(torch.nn.functional.pad(part, (0, chunk - len(part))).numpy())
        features = []
        for begin in range(0, len(parts), batch_size):
            current = parts[begin : begin + batch_size]
            values = feature_extractor(current, sampling_rate=MERT_SR, return_tensors="pt", padding=True).input_values.to(device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                encoded = model(values).last_hidden_state
            for local, value in enumerate(encoded):
                actual = lengths[begin + local]
                keep = max(1, int(round(value.shape[0] * actual / chunk)))
                features.append(value[:keep].float().cpu().numpy())
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        duration=len(wave)/MERT_SR
        frame=max(1,int(round(.10*MERT_SR)));hop=max(1,int(round(.05*MERT_SR)));samples=wave.numpy();rms=[]
        for start in range(0,max(1,len(samples)-frame+1),hop):rms.append(float(np.sqrt(np.mean(samples[start:start+frame]**2)+1e-12)))
        rms=np.asarray(rms);threshold=max(float(rms.max())*10**(-45/20),1e-6);active=np.flatnonzero(rms>threshold)
        active_start=float(active[0]*hop/MERT_SR) if len(active) else 0.0;active_end=min(duration,float((active[-1]*hop+frame)/MERT_SR)) if len(active) else duration
        return np.concatenate(features),duration,active_start,active_end

    def structure_and_density(self,
        mert: np.ndarray,
        duration_seconds: float,
        total_ticks: int,
        bpm_ticks: np.ndarray,
        bpm_values: np.ndarray,
        beat_offset: float,
        version_id: int,
        levels: dict[int, float],
        device: torch.device,
    ):
        conditioned = self.root / "checkpoints_density_levelcalibrated" / "best.pt"
        checkpoint = torch.load(conditioned, map_location="cpu", weights_only=False)
        model = ChartTransformerV2(V2Config(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"], strict=False)
        model.to(device).eval()
        bars = min(model.c.max_bars, max(1, math.ceil(total_ticks / TPB)))
        boundaries = np.arange(bars + 1, dtype=np.int64) * TPB
        seconds = beat_offset + ticks_to_seconds(boundaries, bpm_ticks, bpm_values)
        frame_index = np.clip(np.rint(seconds / max(duration_seconds, 1e-6) * len(mert)).astype(int), 0, len(mert))
        bar_audio = np.zeros((bars, mert.shape[1]), np.float32)
        for bar in range(bars):
            left = int(frame_index[bar])
            right = max(left + 1, int(frame_index[bar + 1]))
            bar_audio[bar] = mert[left : min(right, len(mert))].mean(0) if left < len(mert) else 0
        source = torch.from_numpy(bar_audio)[None].to(device)
        pos = torch.arange(bars, device=device)
        level_values = np.zeros(5, np.int64)
        for slot, value in levels.items():
            if 2 <= slot <= 6: level_values[slot - 2] = round(value * 10)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            structure = model.structure(model.audio(source) + model.bar_pos(pos)[None])
            density = model.predict_density(structure, torch.tensor([version_id], device=device), torch.from_numpy(level_values)[None].to(device))
        density_values=density[0].float().cpu().numpy();calibration=[None]*5
        calibrator_path=self.root/'checkpoints_density_calibrator'/'best.pt'
        if calibrator_path.exists():
            calibrator_checkpoint=torch.load(calibrator_path,map_location='cpu',weights_only=False);calibrator=DensityCalibrator(DensityCalibratorConfig(**calibrator_checkpoint['config']));calibrator.load_state_dict(calibrator_checkpoint['model']);calibrator.to(device).eval()
            for slot,value in levels.items():
                if not 2<=slot<=6:continue
                index=slot-2
                with torch.inference_mode():target=float(calibrator(torch.tensor([version_id],device=device),torch.tensor([index],device=device),torch.tensor([round(value*10)],device=device),torch.tensor([float(bpm_values[0])],device=device))[0])
                before=float(density_values[:,index].mean());scale=target/max(before,1e-6);density_values[:,index]*=scale;calibration[index]={'before':before,'target':target,'scale':scale}
            del calibrator
        result = structure[0].float().cpu().numpy(),density_values,calibration,bar_audio,seconds
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def predict_anchor_logits(self,
        mel: np.ndarray,
        structure: np.ndarray,
        bpm_ticks: np.ndarray,
        bpm_values: np.ndarray,
        version_id: int,
        levels: dict[int, float],
        device: torch.device,
    ):
        checkpoint = torch.load(self.root / "checkpoints_anchor" / "best.pt", map_location="cpu", weights_only=False)
        model = AnchorRelationModel()
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        model = accelerate_module(model, "anchor", allow_tensorrt=False)
        level_values = np.zeros(5, np.int64)
        for slot, value in levels.items():
            if 2 <= slot <= 6:
                level_values[slot - 2] = round(value * 10)
        logits = np.zeros((len(structure), 384, 5), np.float32)
        for begin in range(0, len(structure), 24):
            end = min(len(structure), begin + 24)
            audio = np.stack([sample_positions(mel, bar * TPB + np.arange(TPB), bpm_ticks, bpm_values, FRAME_SECONDS) for bar in range(begin, end)])
            batch = end - begin
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                value = model(
                    torch.from_numpy(audio).to(device),
                    torch.from_numpy(structure[begin:end]).to(device),
                    torch.full((batch,), version_id, dtype=torch.long, device=device),
                    torch.from_numpy(np.repeat(level_values[None], batch, 0)).to(device),
                )
            logits[begin:end] = value.float().cpu().numpy()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return logits

    def _integer_bar_targets(self,
        density: np.ndarray,
        bars: int,
        max_tick: int | None = None,
    ) -> np.ndarray:
        """Round a continuous whole-song budget without dropping fractions.
    
        The planner predicts an expected event count, so independently rounding
        every bar turns many legitimate sub-one-event allocations into empty
        bars.  Largest-remainder allocation preserves the model's total budget
        and its relative bar profile; it does not invent a BPM- or position-based
        rule.
        """
    
        values = np.asarray(density[:bars], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=96.0, neginf=0.0)
        values = np.clip(values, 0.0, 96.0)
        if max_tick is not None:
            active_bars = min(bars, max(0, (int(max_tick) + TPB - 1) // TPB))
            values[active_bars:] = 0.0
        base = np.floor(values).astype(np.int64)
        remainder = values - base
        target_total = int(math.floor(float(values.sum()) + 0.5))
        remaining = max(0, target_total - int(base.sum()))
        if remaining:
            order = np.argsort(-remainder, kind="stable")
            for bar in order:
                if remaining <= 0:
                    break
                if base[int(bar)] >= 96:
                    continue
                base[int(bar)] += 1
                remaining -= 1
        return base.astype(np.int64, copy=False)

    def select_anchors(self,
        logits: np.ndarray,
        structure: np.ndarray,
        density: np.ndarray,
        levels: dict[int, float],
        max_tick: int | None = None,
    ):
        priors = json.loads((self.root / "rhythm_priors.json").read_text(encoding="utf8"))
        selected: dict[int, list[int]] = {}
        hierarchy = ((5, None), (4, 5), (3, 4), (2, 3), (6, 5))
        min_gap = {2: 12, 3: 6, 4: 3, 5: 1, 6: 1}
        for slot, parent in hierarchy:
            if slot not in levels:
                continue
            index = slot - 2
            prior = np.asarray(priors["slots"][str(slot)]["positionProbability"], np.float32)
            prior_term = np.log(prior + 1e-9)
            prior_term -= prior_term.mean()
            parent_set = set(selected.get(parent, []))
            song_ticks: list[int] = []
            bar_targets = self._integer_bar_targets(density[:, index], len(structure), max_tick)
            for bar in range(len(structure)):
                target = int(bar_targets[bar])
                if target == 0:
                    continue
                scores = logits[bar, :, index] + 0.18 * prior_term
                if parent_set:
                    parent_local = [tick - bar * TPB for tick in parent_set if bar * TPB <= tick < (bar + 1) * TPB]
                    scores[parent_local] += 0.9
                order = np.argsort(scores)[::-1]
                chosen: list[int] = []
                gap = min_gap[slot]
                for local in order:
                    tick = bar * TPB + int(local)
                    # The final structure bar is rounded up to a complete bar,
                    # but the audio boundary is not.  Do not place a new event in
                    # the rounded, silent tail.
                    if max_tick is not None and tick >= max_tick:
                        continue
                    if all(abs(tick - old) >= gap for old in song_ticks[-192:] + chosen):
                        chosen.append(tick)
                    if len(chosen) >= target:
                        break
                song_ticks.extend(sorted(chosen))
            selected[slot] = sorted(set(song_ticks))
        return selected

    def predict_style_latent(self,structure: np.ndarray, version_id: int, slot: int, level: float, bpm: float, maidata_metadata: dict, creativity: float, seed: int, device: torch.device):
        prior_checkpoint = torch.load(self.root / "checkpoints_style_prior" / "best.pt", map_location="cpu", weights_only=False)
        prior = StylePrior(StylePriorConfig(**prior_checkpoint["config"]))
        prior.load_state_dict(prior_checkpoint["model"])
        prior.to(device).eval()
        bars = min(prior.c.max_bars, len(structure))
        padded = np.zeros((prior.c.max_bars, structure.shape[1]), np.float32)
        padding = np.ones(prior.c.max_bars, np.bool_)
        padded[:bars] = structure[:bars]
        padding[:bars] = False
        metadata = metadata_tokens({"maidataMetadata": maidata_metadata, "notesDesigner": {"id": 0, "name": "ChartTransformer AI"}})
        batch = {
            "structure": torch.from_numpy(padded)[None].to(device), "padding": torch.from_numpy(padding)[None].to(device),
            "version": torch.tensor([version_id], device=device), "difficulty": torch.tensor([slot - 2], device=device),
            "group": torch.tensor([GROUP_BY_SLOT[slot]], device=device), "level": torch.tensor([round(level * 10)], device=device),
            "bpm": torch.tensor([bpm], device=device), "metadata": torch.from_numpy(metadata)[None].to(device),
        }
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = prior(batch)[0].float().cpu().numpy()
        rng = np.random.default_rng(seed)
        codes, entropies = [], []
        for book_logits in logits:
            probability = np.exp((book_logits - book_logits.max()) / (0.55 + creativity * 0.75))
            probability /= probability.sum()
            order = np.argsort(probability)[::-1]
            top_p = 0.68 + 0.28 * creativity
            keep = max(1, int(np.searchsorted(np.cumsum(probability[order]), top_p, side="left")) + 1)
            candidates = order[:keep]
            local = probability[candidates] / probability[candidates].sum()
            codes.append(int(rng.choice(candidates, p=local)))
            entropies.append(float(-(probability * np.log(probability + 1e-12)).sum() / np.log(len(probability))))
        style_checkpoint = torch.load(self.root / "checkpoints_style_vq" / "selected_diverse.pt", map_location="cpu", weights_only=False)
        codebook = style_checkpoint["model"]["quantizer.codebook"].float().numpy()
        latent = np.concatenate([codebook[book, code] for book, code in enumerate(codes)]).astype(np.float32)
        del prior
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return latent, {"codes": codes, "normalizedEntropy": entropies, "creativity": creativity}
