from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import gcd


FIELD_RE = re.compile(r"(?m)^&([A-Za-z0-9_]+)=")
COMMENT_RE = re.compile(r"(?m)\|\|[^\r\n]*(?:\r?\n|$)")
DIRECTIVE_RE = re.compile(r"\(([-+0-9.]+)\)|\{([^{}]+)\}")


@dataclass
class ParsedChart:
    events: list[tuple[int, str]]
    end_frame: int
    collisions: int


@dataclass
class ParsedTickChart:
    events: list[tuple[int, str]]
    end_tick: int
    bpm_ticks: list[int]
    bpm_values: list[float]
    rounded_positions: int


def parse_maidata(text: str) -> dict[str, str]:
    matches = list(FIELD_RE.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[match.group(1)] = text[match.end():end].strip()
    return fields


def _clean_inote(inote: str) -> str:
    inote = COMMENT_RE.sub("", inote)
    return re.sub(r"\s+", "", inote)


def parse_inote(inote: str, default_bpm: float, frame_seconds: float = 0.02) -> ParsedChart:
    """Convert simai timing into fixed-time events while preserving note spelling."""
    source = _clean_inote(inote)
    bpm = float(default_bpm) if default_bpm > 0 else 120.0
    divisor = 4.0
    absolute_step: float | None = None
    current_seconds = 0.0
    by_frame: dict[int, list[str]] = {}
    collisions = 0

    parts = source.split(",")
    for part in parts:
        stop = False
        for directive in DIRECTIVE_RE.finditer(part):
            if directive.group(1) is not None:
                try:
                    candidate = float(directive.group(1))
                    if candidate > 0:
                        bpm = candidate
                except ValueError:
                    pass
            else:
                value = directive.group(2)
                if value.startswith("#"):
                    try:
                        candidate = float(value[1:])
                        if candidate > 0:
                            absolute_step = candidate
                    except ValueError:
                        pass
                else:
                    try:
                        candidate = float(value)
                        if candidate > 0:
                            divisor = candidate
                            absolute_step = None
                    except ValueError:
                        pass

        note = DIRECTIVE_RE.sub("", part)
        if note == "E":
            note = ""
            stop = True
        note = note.strip(",")
        if note and not note.startswith("###"):
            frame = max(0, int(round(current_seconds / frame_seconds)))
            if frame in by_frame:
                collisions += 1
            by_frame.setdefault(frame, []).append(note)
        if stop:
            break
        step = absolute_step if absolute_step is not None else 240.0 / max(1e-6, bpm * divisor)
        current_seconds += step

    events: list[tuple[int, str]] = []
    for frame, notes in sorted(by_frame.items()):
        merged = "/".join(notes)
        events.append((frame, merged))
    end_frame = max(int(round(current_seconds / frame_seconds)), events[-1][0] + 1 if events else 1)
    return ParsedChart(events=events, end_frame=end_frame, collisions=collisions)


def parse_inote_ticks(inote: str, default_bpm: float, ticks_per_bar: int = 384) -> ParsedTickChart:
    """Parse simai onto MA2's exact 1/384-bar tick grid."""
    source = _clean_inote(inote)
    bpm = float(default_bpm) if default_bpm > 0 else 120.0
    divisor = Fraction(4, 1)
    absolute_step: Fraction | None = None
    position = Fraction(0, 1)
    by_tick: dict[int, list[str]] = {}
    tempo: dict[int, float] = {0: bpm}
    rounded_positions = 0

    for part in source.split(","):
        stop = False
        for directive in DIRECTIVE_RE.finditer(part):
            if directive.group(1) is not None:
                try:
                    candidate = float(directive.group(1))
                    if candidate > 0:
                        bpm = candidate
                        tempo[int(round(float(position)))] = bpm
                except ValueError:
                    pass
            else:
                value = directive.group(2)
                if value.startswith("#"):
                    try:
                        seconds = Fraction(value[1:])
                        absolute_step = seconds * Fraction(str(bpm)) * ticks_per_bar / 240
                    except (ValueError, ZeroDivisionError):
                        pass
                else:
                    try:
                        candidate = Fraction(value)
                        if candidate > 0:
                            divisor = candidate
                            absolute_step = None
                    except (ValueError, ZeroDivisionError):
                        pass

        note = DIRECTIVE_RE.sub("", part)
        if note == "E":
            note = ""
            stop = True
        note = note.strip(",")
        rounded = int(round(float(position)))
        if position.denominator != 1:
            rounded_positions += 1
        if note and not note.startswith("###"):
            by_tick.setdefault(max(0, rounded), []).append(note)
        if stop:
            break
        position += absolute_step if absolute_step is not None else Fraction(ticks_per_bar, 1) / divisor

    events = [(tick, "/".join(notes)) for tick, notes in sorted(by_tick.items())]
    end_tick = max(int(round(float(position))), events[-1][0] + 1 if events else 1)
    bpm_ticks = sorted(tempo)
    return ParsedTickChart(
        events=events,
        end_tick=end_tick,
        bpm_ticks=bpm_ticks,
        bpm_values=[tempo[tick] for tick in bpm_ticks],
        rounded_positions=rounded_positions,
    )


def collect_event_characters(events: list[tuple[int, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, event in events:
        counts.update(event)
    return counts


def is_dx_only_event(event: str) -> bool:
    if re.search(r"(?:^|[/`])[A-E](?:[1-8])?", event):
        return True
    if "x" in event or "]b" in event:
        return True
    return False


def render_tick_grid_maidata(
    *, title: str, source_name: str, version_name: str, version_id: int,
    difficulty_slot: int, internal_level: float, bpm: float,
    events: dict[int, str], total_ticks: int, ticks_per_bar: int = 384,
    bpm_changes: dict[int, float] | None = None,
) -> str:
    lines = [
        f"&title={title}",
        "&artist=AI Generated",
        "&first=0",
        f"&wholebpm={bpm:g}",
        f"&versionid={version_id}",
        f"&version={version_name}",
        "&clock_count=4",
        "&chartgenerator=ChartTransformer",
        f"&source_track={source_name}",
        "",
        f"&lv_{difficulty_slot}={internal_level:.1f}",
        f"&des_{difficulty_slot}=ChartTransformer AI",
        f"&inote_{difficulty_slot}=({bpm:g}){{{ticks_per_bar}}}",
    ]
    row: list[str] = []
    bpm_changes = bpm_changes or {}
    for tick in range(total_ticks):
        prefix = f"({bpm_changes[tick]:g})" if tick in bpm_changes and tick != 0 else ""
        row.append(prefix + events.get(tick, "") + ",")
        if len(row) == 48:
            lines.append("".join(row))
            row.clear()
    if row:
        lines.append("".join(row))
    lines.append("E")
    return "\n".join(lines) + "\n"

def render_compact_maidata(*, title:str, source_name:str, version_name:str, version_id:int,
    difficulty_slot:int, internal_level:float, bpm:float, events:dict[int,str], total_ticks:int,
    ticks_per_bar:int=384, bpm_changes:dict[int,float]|None=None,first:float=0.0)->str:
    lines=[f"&title={title}","&artist=AI Generated",f"&first={first:g}",f"&wholebpm={bpm:g}",f"&versionid={version_id}",f"&version={version_name}","&clock_count=4","&chartgenerator=ChartTransformer",f"&source_track={source_name}","",f"&lv_{difficulty_slot}={internal_level:.1f}",f"&des_{difficulty_slot}=ChartTransformer AI",f"&inote_{difficulty_slot}=({bpm:g})"]
    changes=bpm_changes or {};bars=(total_ticks+ticks_per_bar-1)//ticks_per_bar
    for bar in range(bars):
        start=bar*ticks_per_bar;locals_=[tick-start for tick in events if start<=tick<start+ticks_per_bar]
        locals_ += [tick-start for tick in changes if start<=tick<start+ticks_per_bar and tick!=0]
        unit=ticks_per_bar
        for value in locals_:unit=gcd(unit,int(value))
        divisor=max(1,ticks_per_bar//max(1,unit));slots=[]
        for index in range(divisor):
            tick=start+index*unit;prefix=f"({changes[tick]:g})" if tick in changes and tick!=0 else "";slots.append(prefix+events.get(tick,"")+',')
        lines.append(f"{{{divisor}}}"+''.join(slots))
    lines.append('E');return '\n'.join(lines)+'\n'
