"""Confirmed hard note/slide availability by maimai game version."""
from __future__ import annotations

import re

DX_VERSION = 13
GREEN_PLUS_VERSION = 3
PINK_VERSION = 6
FESTIVAL_VERSION = 19
PRISM_PLUS_VERSION = 24

_SHAPE_RE = re.compile(r"pp|qq|[-<>^vpqszVw]")
BASE_SLIDE_SHAPES = frozenset(("-", "<", ">", "^"))
GREEN_PLUS_SLIDE_SHAPES = frozenset(("v", "p", "q", "s", "z"))
PINK_SLIDE_SHAPES = frozenset(("pp", "qq", "V", "w"))


def _outer_arc_steps(start: int, route: str) -> int | None:
    match = re.fullmatch(r"([<>^])([1-8])", str(route))
    if match is None:
        return None
    shape, end_text = match.groups()
    start_lane = int(start) + 1
    end_lane = int(end_text)
    if shape == "^":
        distance = (end_lane - start_lane) % 8
        return min(distance, (-distance) % 8)
    if shape == ">":
        return (end_lane - start_lane) % 8
    return (start_lane - end_lane) % 8


def slide_route_allowed(version: int, start: int, route: str) -> bool:
    """Hard candidate gate for one singular V4 route string."""
    v = int(version); shapes = _SHAPE_RE.findall(str(route))
    if not shapes:
        return False
    if v < FESTIVAL_VERSION and len(shapes) != 1:
        return False
    allowed = set(BASE_SLIDE_SHAPES)
    if v >= GREEN_PLUS_VERSION:
        allowed.update(GREEN_PLUS_SLIDE_SHAPES)
    if v >= PINK_VERSION:
        allowed.update(PINK_SLIDE_SHAPES)
    if any(shape not in allowed for shape in shapes):
        return False
    if v < GREEN_PLUS_VERSION and shapes[0] in {"<", ">", "^"}:
        steps = _outer_arc_steps(start, str(route))
        return steps is not None and 1 <= steps <= 3
    return True


def slide_break_head_enabled(version: int) -> bool:
    return int(version) >= PINK_VERSION


def slide_ex_head_enabled(version: int) -> bool:
    return int(version) >= DX_VERSION


def slide_ex_break_head_enabled(version: int) -> bool:
    return int(version) >= FESTIVAL_VERSION


def multiple_slide_enabled(version: int) -> bool:
    return int(version) >= PINK_VERSION


def chaining_slide_enabled(version: int) -> bool:
    return int(version) >= FESTIVAL_VERSION


def break_slide_track_enabled(version: int) -> bool:
    return int(version) >= FESTIVAL_VERSION


def sanitize_slide_head_modifiers(version: int, bits: int) -> int:
    value = int(bits) & 0b00011
    if not slide_break_head_enabled(version):
        value &= ~0b00001
    if not slide_ex_head_enabled(version):
        value &= ~0b00010
    if (value & 0b00011) == 0b00011 and not slide_ex_break_head_enabled(version):
        value &= ~0b00010
    return value


def touch_enabled(version: int) -> bool:
    return int(version) >= DX_VERSION


def touch_hold_enabled(version: int) -> bool:
    return int(version) >= DX_VERSION


def touch_sensor_capacity(version: int) -> int:
    if not touch_enabled(version):
        return 0
    return 17 if int(version) < FESTIVAL_VERSION else 33


def touch_sensor_allowed(version: int, sensor: str) -> bool:
    if not touch_enabled(version):
        return False
    name = str(sensor)
    if int(version) < FESTIVAL_VERSION:
        return name == "C" or name.startswith(("B", "E"))
    return name == "C" or name.startswith(("A", "B", "D", "E"))


def touch_hold_sensor_allowed(version: int, sensor: str) -> bool:
    if not touch_hold_enabled(version):
        return False
    return str(sensor) == "C" or int(version) >= PRISM_PLUS_VERSION
