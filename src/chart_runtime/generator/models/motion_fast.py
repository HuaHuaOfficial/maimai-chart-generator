"""Exact incremental adapter for the existing motion-state feature."""
from __future__ import annotations
from collections import deque
from .motion import _features

class MotionStateTracker:
    def __init__(self):
        self.outer=deque(maxlen=9)
    def state(self):
        return _features(list(self.outer))
    def append(self,representation):
        arity=min(2,int(representation['button_arity']))
        if arity<=0:return
        lane=int(representation['button_start'][0])
        families=tuple(int(x) for x in representation['button_family'][:arity])
        self.outer.append((lane,families))
    def seed(self,representations):
        self.outer.clear()
        for rep in representations:self.append(rep)
        return self
