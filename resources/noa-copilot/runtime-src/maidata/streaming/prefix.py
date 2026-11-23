"""Resident immutable prefix snapshots for the continuous v8 decoder."""
from .codec import input_row
from .sequence import BOUNDARY, EventToken
from .trajectory_penalty import TrajectoryPenalty


def remember(states, tokens, oracle, cache, row, trajectory, options):
    if states is None:
        return
    key = tuple((k, v) for k, v in vars(options).items() if k.startswith('trajectory_'))
    states.append(dict(tokens=tuple(tokens), oracle=oracle.fork(), cache=cache,
                       row=row, trajectory=trajectory, key=key))
    del states[:-4]


def prime(decoder, memory, oracle, tokens, options, device, states, cancelled=None):
    from .generation import pack_row, GenerationCancelled
    key = tuple((k, v) for k, v in vars(options).items() if k.startswith('trajectory_'))
    matching = [s for s in (states or ()) if s['key'] == key
                and len(s['tokens']) <= len(tokens) and tuple(tokens[:len(s['tokens'])]) == s['tokens']]
    previous = max(matching, key=lambda s: len(s['tokens'])) if matching else None
    reused = len(previous['tokens']) if previous else 0
    if previous:
        oracle, cache, row, trajectory = (previous['oracle'].fork(), previous['cache'],
                                           previous['row'], previous['trajectory'])
    else:
        cache = None
        row = input_row(EventToken(BOUNDARY), oracle.state.grammar, oracle.timing)
        trajectory = TrajectoryPenalty.from_options(options)
    for token in tokens[reused:]:
        if cancelled is not None and cancelled():
            raise GenerationCancelled()
        output = decoder(pack_row(row, device), memory, cache=cache, need_cache=True,
                         positions=([row['causal_beats']], [row['slide_beats']]))
        grammar = oracle.state.grammar
        context = grammar.flag_context
        trajectory = trajectory.advance(token, grammar)
        oracle.accept(token)
        row = input_row(token, oracle.state.grammar, oracle.timing, context=context)
        cache = output.cache
        if token.type == 0 and token.value == 96:
            cache = cache.evict(oracle.state.grammar.beat, options.rollback_tokens)
        oracle.state.groups.clear()
    remember(states, tokens, oracle, cache, row, trajectory, options)
    return oracle, cache, row, trajectory, reused, len(tokens) - reused
