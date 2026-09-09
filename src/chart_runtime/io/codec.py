from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from collections import OrderedDict
import json
import re
import numpy as np
import torch

from .timing import ticks_to_seconds
from .simai import parse_inote_ticks
from .factors import factor_event, parse_slide_tracks
from .durations import hold_seconds, slide_seconds
from .symmetry import semantic_notes

SCHEMA = "chart-ir/1"


class UnsupportedChart(ValueError):
    pass


@dataclass(frozen=True)
class ChartPayload:
    """Immutable source text plus owned CUDA columns and a content fingerprint.

    The backend records Tensor version counters to detect in-place mutation.
    Rows preserve all Notes and all Tracks; no two-note model limit here.
    """
    source: tuple[tuple[int, str], ...]
    columns: object
    digest: str
    definition_id: str
    versions: tuple[tuple[str, int], ...]
    schema_id: str = SCHEMA

    @property
    def device(self):
        return str(self.columns['event_tick'].device)

    @property
    def events(self):
        return dict(self.source)

    def assert_unmodified(self):
        if tuple((key, value._version) for key, value in self.columns.items()) != self.versions:
            raise RuntimeError("Immutable chart buffer was modified after creation")


def _version(tensor):
    # Inference tensors have no version counter, so payload construction always
    # happens in inference_mode(False), including inside a renderer callback.
    return tensor._version


class Codec:
    def __init__(self, root, device='cuda:0'):
        self.root = Path(root)
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise RuntimeError('CUDA is required; CPU musical checking is not available')
        base = self.root/'models'
        self.vocab = json.loads((base/'experimental/vocab.json').read_text(encoding='utf8'))
        self.tables = json.loads((base/'v2/playability_tables.json').read_text(encoding='utf8'))
        self.route_ids = {route: i for i, route in enumerate(self.vocab['routes'])}
        self.sensors = list(self.vocab['touchPositions'])
        self.path_digest = sha256((base/'v2/playability_tables.json').read_bytes()).hexdigest()
        self.cache=OrderedDict()
        keys = {}
        self.paths = {}
        for key, entry in self.tables['slideConflicts'].items():
            path = tuple(sorted((int(x['pad']), round(float(x['startFraction']), 8), round(float(x['endFraction']), 8)) for x in entry.get('contactIntervals', ())))
            reverse = tuple(sorted((pad, round(1-end, 8), round(1-start, 8)) for pad, start, end in path))
            canonical = min(path, reverse)
            if canonical not in keys:
                keys[canonical] = len(keys)
            self.paths[key] = keys[canonical]

    @staticmethod
    @lru_cache(maxsize=32768)
    def parse_event(text):
        normalized = '/'.join(semantic_notes(text))
        notes = factor_event(normalized)['notes']
        for note in notes:
            if note['start'] == 'UNKNOWN':
                raise UnsupportedChart(f"Unsupported Simai token: {note['raw']}")
            if note['family'] == 'slide':
                tracks = parse_slide_tracks(note['raw'])
                # Preserve textual source, but never truncate segmented timing
                # into a first-duration representation that loses semantics.
                if not tracks or note['raw'].count('[') != len(tracks):
                    raise UnsupportedChart(f"Per-segment timing has no supported path asset: {note['raw']}")
                for track in tracks:
                    if track['duration'] is None:
                        raise UnsupportedChart(f"Missing Track timing: {note['raw']}")
        return tuple(notes)

    def encode(self, events, bt, bv):
        key=(tuple((int(t),str(s)) for t,s in sorted(events.items()) if str(s)),tuple(map(int,bt)),tuple(map(float,bv)))
        cached=self.cache.get(key)
        if cached is not None:
            cached.assert_unmodified();self.cache.move_to_end(key)
            return cached
        result=self._encode(events,bt,bv)
        self.cache[key]=result
        while len(self.cache)>32:self.cache.popitem(last=False)
        return result

    def _encode(self, events, bt, bv):
        source = tuple((int(t), str(s)) for t, s in sorted(events.items()) if str(s))
        ticks = np.asarray([t for t, _ in source], np.int64)
        seconds = ticks_to_seconds(ticks, np.asarray(bt), np.asarray(bv))
        data = {k: [] for k in (
            'event_tick', 'event_time', 'event_notes', 'event_lanes', 'event_lane_count', 'event_track_speed',
            'input_event', 'input_note', 'input_sensor', 'input_pad', 'input_outer', 'input_hold', 'input_ex', 'input_start', 'input_end',
            'note_event', 'note_kind', 'note_sensor', 'note_modifiers',
              'track_event', 'track_note', 'track_route', 'track_path', 'track_head', 'track_tail', 'track_start', 'track_shoot', 'track_end', 'track_early', 'track_wifi', 'track_contacts_key',
            'contact_track', 'contact_sensor', 'contact_time',
            'action_start', 'action_end', 'action_mask', 'action_event', 'action_track',
            'queue_track', 'queue_masks', 'queue_skip', 'queue_count',
        )}
        max_notes = max((sum(n['family'] != 'touch' for n in self.parse_event(text)) for _, text in source), default=2)
        max_lanes = max(2, max_notes)
        max_areas = 1
        for ei, ((tick, text), at) in enumerate(zip(source, seconds)):
            at = round(float(at)*1e9)/1e9
            bpm = float(bv[max(0, np.searchsorted(bt, tick, side='right')-1)])
            notes = self.parse_event(text)
            lanes = [int(n['start'])-1 for n in notes if n['family'] != 'touch']
            data['event_tick'].append(tick); data['event_time'].append(at)
            data['event_notes'].append(len(notes)); data['event_lane_count'].append(len(lanes))
            data['event_lanes'].append(lanes + [-1]*(max_lanes-len(lanes)))
            track_speed = 0.
            for ni, note in enumerate(notes):
                family = note['family']; outer = family != 'touch'
                sensor = int(note['start'])-1 if outer else 8+self.sensors.index(note['start'])
                kind = {'tap': 0, 'hold': 1, 'slide': 2, 'touch': 3}[family]
                mods = sum(int(note[k]) << b for b, k in enumerate(('is_break', 'is_ex', 'is_star', 'is_firework', 'is_headless')))
                data['note_event'].append(ei); data['note_kind'].append(kind)
                data['note_sensor'].append(sensor); data['note_modifiers'].append(mods)
                held = family == 'hold' or (not outer and bool(note['duration']))
                length = hold_seconds(note['duration'], bpm) if held else 0.
                pad_name = 'A'+str(sensor+1) if outer else note['start']
                pad = int(self.tables['simplePadMasks'][pad_name])
                if not (family == 'slide' and note['is_headless']):
                    for key, val in zip(('input_event','input_note','input_sensor','input_pad','input_outer','input_hold','input_ex','input_start','input_end'), (ei,ni,sensor,pad,outer,held,note['is_ex'],at,at+length)):
                        data[key].append(val)
                    data['action_start'].append(at); data['action_end'].append(at+length+1/60); data['action_mask'].append(pad); data['action_event'].append(ei); data['action_track'].append(-1)
                if family != 'slide':
                    continue
                for track in parse_slide_tracks(note['raw']):
                    route = track['route']; rid = self.route_ids.get(route, -1)
                    key = f"{note['start']}:{rid}"
                    if key not in self.tables['slideConflicts']:
                        raise UnsupportedChart(f"No complete path table for {note['raw']}")
                    entry = self.tables['slideConflicts'][key]
                    if not entry.get('contactIntervals') or not entry.get('judgeLanes'):
                        raise UnsupportedChart(f"Incomplete contact or queue table for {note['raw']}")
                    wait, move = slide_seconds(track['duration'], bpm)
                    shoot, end = at+wait, at+wait+move
                    wifi = bool(entry.get('isWifi', False)); critical = shoot+float(entry['criticalFraction'])*move
                    delta = min(.6, 14/60+float(entry['lastAreaFraction'])*move/4)
                    early = critical-(delta if wifi else max(delta, 17/60))
                    ti = len(data['track_event'])
                    contacts = entry.get('aEntries', ())
                    start_signature = tuple((int(x['lane']), round(float(x['fraction']),8)) for x in contacts[:2])
                    signature_id = int.from_bytes(sha256(repr(start_signature).encode()).digest()[:8], 'little') & ((1<<63)-1)
                    vals = (ei, ni, rid, self.paths[key], int(note['start'])-1, int(route[-1])-1, at, shoot, end, early, wifi, signature_id)
                    for name, val in zip(('track_event','track_note','track_route','track_path','track_head','track_tail','track_start','track_shoot','track_end','track_early','track_wifi','track_contacts_key'), vals):
                        data[name].append(val)
                    for contact in contacts:
                        data['contact_track'].append(ti); data['contact_sensor'].append(int(contact['lane'])-1); data['contact_time'].append(shoot+float(contact['fraction'])*move)
                    for contact in entry['contactIntervals']:
                        start, finish = shoot+float(contact['startFraction'])*move, shoot+float(contact['endFraction'])*move
                        if not wifi and float(contact['endFraction']) >= 1-1e-7:
                            finish += 1/60
                        data['action_start'].append(start); data['action_end'].append(max(start+1e-9,finish)); data['action_mask'].append(1<<int(contact['pad'])); data['action_event'].append(ei); data['action_track'].append(ti)
                    for lane in entry['judgeLanes']:
                        count = len(lane['areaMasks']); max_areas = max(max_areas,count)
                        data['queue_track'].append(ti); data['queue_masks'].append(list(lane['areaMasks'])); data['queue_skip'].append(list(lane['skipNoPress'])); data['queue_count'].append(count)
                    track_speed = max(track_speed, len(re.findall(r'pp|qq|[-<>^vpqszVw]',route))/max(move,1e-6))
            data['event_track_speed'].append(track_speed)
        floats = {'event_time','event_track_speed','input_start','input_end','track_start','track_shoot','track_end','track_early','contact_time','action_start','action_end'}
        bools = {'input_outer','input_hold','input_ex','track_wifi'}
        with torch.inference_mode(False):
            if getattr(self, 'packed_transfers', True):
                from .tensor_pack import numpy_columns_to_device
                arrays = {}
                for key, values in data.items():
                    if key in ('queue_masks','queue_skip'):
                        values = [v+[0]*(max_areas-len(v)) for v in values]
                    dtype = np.float64 if key in floats else np.bool_ if key in bools else np.int64
                    value = np.asarray(values, dtype=dtype)
                    if key == 'event_lanes': value = value.reshape(-1,max_lanes)
                    if key in ('queue_masks','queue_skip'): value = value.reshape(-1,max_areas)
                    arrays[key] = value
                tensors = numpy_columns_to_device(arrays, self.device)
            else:
                tensors = {}
                for key, values in data.items():
                    if key in ('queue_masks','queue_skip'):
                        values = [v+[0]*(max_areas-len(v)) for v in values]
                    dtype = torch.float64 if key in floats else torch.bool if key in bools else torch.int64
                    tensor = torch.tensor(values, dtype=dtype, device=self.device)
                    if key == 'event_lanes': tensor = tensor.reshape(-1,max_lanes)
                    if key in ('queue_masks','queue_skip'): tensor = tensor.reshape(-1,max_areas)
                    tensors[key] = tensor
            tensors['event_time_ns'] = (tensors['event_time']*1e9).round().to(torch.int64)
        digest = sha256(json.dumps((source, list(map(int,bt)), list(map(float,bv))),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        return ChartPayload(source, MappingProxyType(tensors), digest, self.path_digest,
                            tuple((key,_version(value)) for key,value in tensors.items()))

    def parse(self, inote, default_bpm):
        parsed = parse_inote_ticks(inote, default_bpm)
        if parsed.rounded_positions:
            raise UnsupportedChart('Timing finer than the active renderer tick grid cannot be rounded silently')
        return self.encode(dict(parsed.events), parsed.bpm_ticks, parsed.bpm_values)
