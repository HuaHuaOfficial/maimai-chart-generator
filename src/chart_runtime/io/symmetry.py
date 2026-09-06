"""Exact D8 canonicalization for one Simai event group.

This is a legality/preference key only.  Difficulty models must retain absolute
position.  Timing expressions inside ``[...]`` are deliberately untouched.
"""

from __future__ import annotations

import re

def transform_pad_value(value,symmetry):
    if value==32:return 32
    group,index=value>>3,value&7
    if symmetry>=8:index=((2 if group<2 else 3)-index)&7
    return (group<<3)|((index+(symmetry&7))&7)


_REFLECT_SYMBOLS = str.maketrans({"<": ">", ">": "<", "p": "q", "q": "p", "s": "z", "z": "s"})


def _pad_name(group: str, lane: str, symmetry: int) -> str:
    if not lane:
        return group
    group_name = "A" if not group else group
    # MaiMuriDX encodes lane 8 with low bits zero.
    group_offset = {"A": 0, "B": 8, "D": 16, "E": 24}[group_name]
    value = group_offset | (int(lane) & 7)
    transformed = int(transform_pad_value(value, symmetry))
    output_group = ("A", "B", "D", "E")[transformed >> 3]
    output_lane = transformed & 7
    return f"{output_group}{8 if output_lane == 0 else output_lane}"


def transform_note(note: str, symmetry: int) -> str:
    """Apply one of 8 rotations or 8 reflections to a Simai note string."""

    symmetry = int(symmetry)
    if not 0 <= symmetry < 16:
        raise ValueError(f"invalid D8 symmetry: {symmetry}")
    reflected = symmetry >= 8
    result: list[str] = []
    depth = 0
    index = 0
    while index < len(note):
        char = note[index]
        if char == "[":
            depth += 1
            result.append(char)
            index += 1
            continue
        if char == "]":
            depth = max(0, depth - 1)
            result.append(char)
            index += 1
            continue
        if depth:
            result.append(char)
            index += 1
            continue
        if char in "ABDE" and index + 1 < len(note) and note[index + 1] in "12345678":
            result.append(_pad_name(char, note[index + 1], symmetry))
            index += 2
            continue
        if char in "12345678":
            result.append(_pad_name("", char, symmetry)[1:])
            index += 1
            continue
        result.append(char.translate(_REFLECT_SYMBOLS) if reflected else char)
        index += 1
    return "".join(result)


def semantic_notes(text: str) -> list[str]:
    notes = [value for value in re.split(r"[/`]", str(text)) if value]
    output = []
    for note in notes:
        if len(note) > 1 and all(char in "12345678" for char in note):
            output.extend(note)
        else:
            output.append(note)
    return output


def transform_event_group(text: str, symmetry: int) -> str:
    return "/".join(sorted(transform_note(note, symmetry) for note in semantic_notes(text)))


def canonical_event_group(text: str) -> str:
    """Return the lexicographically smallest exact D8 event-group spelling."""

    return min(transform_event_group(text, symmetry) for symmetry in range(16))


def canonical_event_window(events: list[tuple[int, str]]) -> tuple[tuple[int, str], ...]:
    """Canonicalize a timed event window using one shared D8 transform."""

    if not events:
        return ()
    ordered = sorted((int(tick), str(text)) for tick, text in events)
    origin = ordered[0][0]
    return min(
        tuple(
            (tick - origin, transform_event_group(text, symmetry))
            for tick, text in ordered
        )
        for symmetry in range(16)
    )
