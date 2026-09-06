from __future__ import annotations
from dataclasses import asdict, dataclass
import re

SHAPE_RE=re.compile(r'pp|qq|[-<>^vpqszVw]')
NOTE_START_RE=re.compile(r'^(?:[A-E](?:[1-8])?|[1-8])')
MODIFIER_RE=re.compile(r'[bxf$?!@]')
@dataclass
class NoteFactors:
    raw:str;family:str;start:str;end:str|None;shapes:list[str];duration:str|None
    is_break:bool;is_ex:bool;is_star:bool;is_firework:bool;is_headless:bool
def split_event(text:str)->list[str]:return [x for x in re.split(r'[/`]',text) if x]
def factor_note(note:str)->NoteFactors:
    touch=re.match(r'([A-E](?:[1-8])?)',note);button=re.match(r'([1-8])',note)
    start=touch.group(1) if touch else button.group(1) if button else 'UNKNOWN'
    duration_m=re.search(r'\[([^]]+)\]',note);duration=duration_m.group(1) if duration_m else None
    body=note.split('[',1)[0];shapes=SHAPE_RE.findall(body);digits=re.findall(r'[1-8]',body)
    family='touch' if touch else 'slide' if shapes else 'hold' if 'h' in note else 'tap'
    end=digits[-1] if family=='slide' and len(digits)>1 else None
    return NoteFactors(note,family,start,end,shapes,duration,'b' in note,'x' in note,'$' in note,'f' in note,'?' in note or '!' in note)
def note_route(note:str)->str|None:
    """Return the complete slide route after the start sensor.

    Keeping the route separate from start/duration/modifiers preserves connected
    slides while avoiding the v1 mistake of treating the complete event string
    as one opaque class.
    """
    body=note.split('[',1)[0]
    body=NOTE_START_RE.sub('',body,count=1)
    body=MODIFIER_RE.sub('',body)
    return body if SHAPE_RE.search(body) else None
def factor_event(text:str)->dict:
    notes=[factor_note(x) for x in split_event(text)]
    return {'raw':text,'arity':len(notes),'notes':[asdict(x) for x in notes]}

def parse_slide_tracks(note: str):
    tracks=[]
    for i,segment in enumerate(note.split('*')):
        match=re.search(r'\[([^]]+)\]',segment)
        body=segment.split('[',1)[0]
        if i==0:body=NOTE_START_RE.sub('',body,count=1)
        route=MODIFIER_RE.sub('',body)
        if not SHAPE_RE.search(route):return []
        if '*' in note and match is None:return []
        modifier_source=segment if match is None else segment[:match.start()]+segment[match.end():]
        bits=sum(int(char in modifier_source)<<bit for bit,char in enumerate(('b','x','$','f','?')))
        tracks.append({'route':route,'duration':match.group(1) if match else None,'modifiers':bits})
    return tracks

def is_compound_slide(note):return '*' in note and len(parse_slide_tracks(note))>1
