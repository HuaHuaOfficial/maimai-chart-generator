"""Position-free WHAT contract shared by planning and realization."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EventIntent:
    button_families: tuple[int, ...] = ()
    touch_count: int = 0

    def __post_init__(self):
        if len(self.button_families)>2 or any(f not in (0,1,2) for f in self.button_families):
            raise ValueError('Intent supports at most two Tap/Hold/Slide buttons')
        if not 0<=self.touch_count<=33:raise ValueError('Invalid Touch count')

    @property
    def button_arity(self):return len(self.button_families)

    def __getitem__(self,key):
        if key=='button_arity':return self.button_arity
        if key=='button_family':return self.button_families
        raise KeyError(key)

    def as_dict(self):
        return {'button_families':list(self.button_families),'touch_count':self.touch_count}

    @classmethod
    def from_representation(cls,rep):
        if isinstance(rep,cls):return rep
        return cls(tuple(int(f) for f in rep['button_family'][:int(rep['button_arity'])]),
                   sum(bool(x) for x in rep['touch_presence']))


@dataclass(frozen=True)
class IntentChoices:
    """Ranked, internally consistent proposals from the WHAT model."""
    candidates: tuple[EventIntent, ...]

    def __post_init__(self):
        if not self.candidates:raise ValueError('Empty WHAT proposal set')

    def rotated(self,offset):
        n=len(self.candidates);i=int(offset)%n
        return IntentChoices(self.candidates[i:]+self.candidates[:i])
