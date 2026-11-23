"""Rollback-owned tries for sliding two-beat, note-prefix repetition penalties."""
from dataclasses import dataclass

# Analysis/control inputs are not chart notes and never affect matching progress.
IGNORED = frozenset((1,2,3,13,16,20,21,22))


def key(token):
    if token.type in IGNORED or token.type >= 25: return None
    # CONT tokens have the same sampled family/value as their numeric head.
    return ({14:8,15:11,24:23}.get(token.type,token.type),token.value)


class Node:
    def __init__(self):
        self.children = {}
        self.progress = {}


def reference_trie(beats, power):
    root = Node()
    for left,right in zip(beats,beats[1:]):
        sequence = left+right
        total = sum(note for _,note in sequence)
        if not total: continue
        node = root; matched = 0
        for symbol,note in sequence:
            # Shared/duplicate references do not stack penalties.
            node.progress[symbol] = max(node.progress.get(symbol,0.),(matched/total)**power)
            node = node.children.setdefault(symbol,Node())
            matched += note
    return root


@dataclass(frozen=True)
class TrajectoryPenalty:
    history_beats: int
    power: float
    moment: bool
    nonmoment: bool
    moment_strength: float
    nonmoment_strength: float
    completed: tuple = ()
    current: tuple = ()
    # Each window starts at an integer beat and survives at most two EOBs.
    active: tuple = ()

    @classmethod
    def from_options(cls, options):
        return cls(options.trajectory_history_beats,options.trajectory_power,
                   options.trajectory_moment,options.trajectory_nonmoment,
                   options.trajectory_moment_penalty,options.trajectory_nonmoment_penalty)

    def costs(self):
        result = {}
        for _,node in self.active:
            for symbol,progress in node.progress.items():
                is_moment = symbol[0]==0
                enabled = self.moment if is_moment else self.nonmoment
                strength = self.moment_strength if is_moment else self.nonmoment_strength
                if enabled and strength>0:
                    result[symbol] = max(result.get(symbol,0.),strength*progress)
        return result

    def advance(self, token, grammar):
        if self.history_beats<2 or not (self.moment and self.moment_strength>0 or self.nonmoment and self.nonmoment_strength>0):
            return self
        symbol = key(token)
        if symbol is None: return self
        # PAD at a route endpoint or Touch's separate pad is not another note.
        touch_start=19 if getattr(grammar,'no_placeholders',False) else 5
        note = token.type==touch_start or token.type==4 and grammar.phase not in ('route_pad','touch_pad')
        current = (*self.current,(symbol,int(note)))
        active = tuple((age,node.children[symbol]) for age,node in self.active if symbol in node.children)
        completed = self.completed
        if symbol==(0,96):
            completed = (*completed,current)[-max(0,self.history_beats):] if self.history_beats>0 else ()
            current = ()
            active = tuple((age+1,node) for age,node in active if age<1)
            if self.history_beats>=2:
                root = reference_trie(completed,self.power)
                if root.children: active = (*active,(0,root))
        return TrajectoryPenalty(self.history_beats,self.power,self.moment,self.nonmoment,
                                 self.moment_strength,self.nonmoment_strength,completed,current,active)


def adjust(logits, costs, family=None, supports=None):
    """Subtract before temperature/top filters; disabled heads are bit-identical.

    A type decision receives a penalty only when its supported field values all
    continue a matched path. This avoids penalizing unrelated values sharing a
    family; multi-value decisions receive their exact per-value penalties below.
    """
    if family is None:
        deductions = {}
        for f,values in supports.items():
            penalties = [costs.get((f,v),0.) for v in values]
            if penalties and min(penalties)>0: deductions[f]=min(penalties)
    else:
        # Account for any amount already applied at the family decision.
        base = min((costs.get((family,v),0.) for v in supports[family]),default=0.)
        deductions = {v:max(0.,cost-base) for (f,v),cost in costs.items() if f==family}
    if not any(deductions.values()): return logits
    result = logits.clone()
    for index,value in deductions.items():
        if value: result[index] -= value
    return result
