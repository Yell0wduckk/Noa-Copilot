"""One continuous decode, persistent Physical state and rollback-safe previews."""
from copy import copy
import torch
from maidata.decoder.windows import Onset
from maidata.inference.options import GenerationOptions
from maidata.inference.sampling import Sampler
from maidata.inference.progress import emit
from maidata.inference.generation import GenerationError
from .sequence import EventToken, CONT_FAMILY, BOUNDARY, WIDTHS, HOLD_CONT, SLIDE_CONT
from maidata.chart.routes import shape_route_table, slide_motion_groups
from .codec import input_row
from .network import StreamTokens
from .physical import Oracle

def chart_from_groups(groups, timing, metadata):
    from maidata.chart.types import EncodedChart, Moment
    start=min(0,groups[0].tick) if groups else 0
    cursor,tokens=start,[Moment(0)]
    for group in groups:
        if group.tick!=cursor:
            tokens.append(Moment(timing.delta_ms(cursor,group.tick)))
        tokens.extend(group.content)
        cursor=group.tick
    return EncodedChart(tuple(tokens),metadata,start)



class GenerationCancelled(Exception):
    pass


def pack_row(row,device):
    return StreamTokens(**{k:torch.tensor([[v]],device=device,dtype=torch.long if k in ('types','vocab_ids') else torch.float32)
                           for k,v in row.items()})


def candidates(decoder,output,oracle,sampler,forced,duration,options,expansion,history,trajectory=None,relaxed_tick=None):
    widths = getattr(decoder,'widths',WIDTHS)
    g = oracle.state.grammar
    relaxed_sampler=copy(sampler) if relaxed_tick is not None else sampler
    if relaxed_tick is not None: relaxed_sampler.physical_hard_mask=False
    def permitted(family,value,evaluation):
        tick=g.beat*96+value if family==0 and value!=96 else g.tick
        return (relaxed_sampler if tick==relaxed_tick else sampler).allowed(evaluation.raw.decision())
    from .trajectory_penalty import adjust
    from .version import values as version_values
    costs = {} if trajectory is None else trajectory.costs()

    def choices_for(family,available):
        choices = {v:e for v,e in oracle.support(family).items() if v in available}
        if family == 0:
            excluded = getattr(sampler,'excluded_moments',())
            lower = getattr(sampler, 'onset_start', float('-inf'))
            upper = getattr(sampler, 'onset_end', float('inf'))
            choices = {v:e for v,e in choices.items() if v==96 or
                       lower <= g.beat*96+v < upper and g.beat*96+v not in excluded}
        choices = {v:e for v,e in choices.items() if permitted(family,v,e)}
        if family == 4 and g.phase == 'route_pad':
            from maidata.chart.routes import chain_paths
            kept = {}
            for v,e in choices.items():
                routes = g.routes+(shape_route_table()[g.shape][g.pad][v],)
                paths = chain_paths('A'+str(g.head_pad+1),routes)
                count = sum(len(oracle.physical.geometry.paths[p].entries) for p in paths)
                if count<=options.max_slide_judge_areas: kept[v] = e
            choices = kept
        return choices

    supports, deferred = {}, {}
    families=g.allowed_types()
    for family in families:
        if expansion >= options.max_tokens_per_moment and (
                family in (4,5,19) and g.phase in ('after_pad','after_flags','after_note','after_slide') or
                family in (9,17) and g.phase in ('route_chain','after_slide') or family==18): continue
        available=set(version_values(g,family,oracle.profile))
        if not available: continue
        lazy=(getattr(g,'no_placeholders',False) and len(families)>1
              and family in (8,9,23) and not any(f==family and cost for (f,v),cost in costs.items()))
        if lazy:
            if not permitted(family,0,oracle.branch_value(family)): continue
            if not any(permitted(family,v,e) for v,e in oracle.basic_support(family).items() if v in available): continue
            deferred[family]=available
            supports[family]=dict.fromkeys(available)
        else:
            choices=choices_for(family,available)
            if choices: supports[family]=choices
    ordered = tuple(supports) if getattr(decoder,'markers',False) and len(supports)==1 else sampler.order(
        adjust(output.type_logits[0,-1],costs,supports=supports),[f in supports for f in range(output.type_logits.shape[-1])])
    for family in ordered:
        if family in deferred:
            supports[family]=choices_for(family,deferred[family])
            if not supports[family]: continue
        if widths[family]==1:
            yield EventToken(family,0),supports[family][0],output.cache
            continue
        logits = decoder.field_logits(output.hidden[0,-1],family,g.flag_context)
        allowed = [v in supports[family] for v in range(len(logits))]
        if getattr(decoder,'markers',False) and len(supports)==1:
            adjusted=logits.clone() if any(f==family and c for (f,v),c in costs.items()) else logits
            for (f,v),cost in costs.items():
                if f==family and cost: adjusted[v]-=cost
        else: adjusted=adjust(logits,costs,family=family,supports=supports)
        values = (sampler.grouped_order(adjusted, slide_motion_groups(), allowed) if family == 9 else
                  sampler.order(adjusted,allowed))
        for value in values:
            if getattr(g,'physical_v6',False) and family==23 and value==96:
                yield EventToken(24,value),supports[family][value],output.cache
                continue
            t = {8:HOLD_CONT,11:SLIDE_CONT}.get(family,family) if value==96 else family
            yield EventToken(t,value),supports[family][value],output.cache


def prepare_memory(decoder, conditions, timing, config, device, *, level=None, difficulty=None, designer=None):
    chart_condition = None
    if getattr(decoder,'pure_e_main',False):
        from .meta_conditioning import metadata as chart_metadata, designer_id, DESIGNER_VOCAB
        level = conditions.level if level is None else level
        difficulty = conditions.difficulty if difficulty is None else difficulty
        designer = getattr(conditions,'designer',None) if designer is None else designer
        chart_condition = decoder.prepare_chart_meta(chart_metadata([(level,difficulty,
            designer_id(designer,config.get('decoder_designer_vocab',DESIGNER_VOCAB)))],device))
    audio_rope = {}
    audio = decoder.prepare_audio(conditions.audio,prepared=audio_rope)
    return decoder.memory(audio,conditions.audio.beats,conditions.audio.valid,
        chart_condition=chart_condition,
        global_audio=decoder.prepare_global_audio(conditions.z,prepared=audio_rope) if getattr(decoder,'global_cross',False) and getattr(conditions,'z',None) is not None and config.get('inference_global_audio_enabled',True) else None,
        timing=timing)


@torch.inference_mode()
def generate(decoder,conditions,timing,geometry,config,metadata,duration,*,device,options=None,
             progress=None,on_groups=None,return_chart=True,stream=None,level=None,difficulty=None,designer=None,profile=None,
             prefix=(), onset_start=None, onset_end=None, token_budget=None, cancelled=None,
             prefix_states=None, execution_report=None, prepared_memory=None, **unused):
    options = options or GenerationOptions()
    start = int(conditions.coverage.first_beat)
    end = start+int(conditions.coverage.draft_beats)
    if profile is None and getattr(decoder,'markers',False):
        from maidata.version_mask import profile_for_chart, VERSION_ORDER
        if metadata.get('song_id') is not None:
            profile=profile_for_chart(metadata['song_id'],metadata.get('difficulty'),metadata.get('version'))
        elif metadata.get('cabinet') in ('SD','DX'):
            profile=profile_for_chart(0 if metadata['cabinet']=='SD' else 10000,6,VERSION_ORDER[-1])
    if getattr(decoder,'physical_v6',False):
        from .v6_oracle import PhysicalOracle as oracle_class, timing_v6
        timing=timing_v6(timing); metadata=dict(metadata,duration_contract='global-tick96.v1')
    else: oracle_class=Oracle
    oracle = oracle_class(timing,geometry,duration,start,end,notes_only=getattr(decoder,'notes_only',False),markers=getattr(decoder,'markers',False),profile=profile,physical_enabled=options.physical_enabled,**(dict(state_tokens=False,no_placeholders=getattr(decoder,"no_placeholders",True)) if getattr(decoder,"physical_v6",False) else {}))
    sampler = Sampler(temperature=options.temperature,seed=options.seed,physical_hard_mask=options.physical_hard_mask,physical_enabled=options.physical_enabled,
        top_k=options.top_k,top_p=options.top_p,max_slide_judge_areas=options.max_slide_judge_areas)
    decoder.eval()
    memory = (prepare_memory(decoder, conditions, timing, config, device,
                             level=level, difficulty=difficulty, designer=designer)
              if prepared_memory is None else prepared_memory)
    row = input_row(EventToken(BOUNDARY),oracle.state.grammar,timing)
    cache = None
    from .trajectory_penalty import TrajectoryPenalty
    trajectory = TrajectoryPenalty.from_options(options)
    if prefix or prefix_states is not None:
        from .prefix import prime
        oracle, cache, row, trajectory, reused, computed = prime(
            decoder, memory, oracle, prefix, options, device, prefix_states, cancelled)
        if execution_report is not None:
            execution_report.update(reused_rows=reused, computed_rows=computed)
    sampler.onset_start = float('-inf') if onset_start is None else onset_start
    sampler.onset_end = float('inf') if onset_end is None else onset_end
    frames,emitted,tokens,beat_ends,reports,groups,bypass = [],[],[],[],[],[],[]
    backtracks = expansion = 0
    search_backtracks = {}
    relaxed_ticks = set()
    sampler.excluded_moments = set()
    moment_origin = None
    escape_beat = False
    skips = []
    published_skips = 0
    published_beats = published_content = published_events = published_tokens = published_bypass = 0

    def publish(force=False):
        nonlocal published_beats,published_content,published_events,published_tokens,published_bypass,published_skips
        safe = len(tokens) if force or not frames else frames[0][1]
        while published_beats<len(beat_ends) and beat_ends[published_beats]['tokens']<=safe:
            point = beat_ends[published_beats]
            onsets = []
            for tick,content in emitted[published_content:point['content']]:
                if onsets and onsets[-1].tick == tick:
                    old = onsets[-1]; onsets[-1] = Onset(tick,old.delta_ms,old.content+(content,))
                else:
                    previous = onsets[-1].tick if onsets else groups[-1].tick if groups else min(0,tick)
                    onsets.append(Onset(tick,timing.delta_ms(previous,tick),(content,)))
            new_tokens = tokens[published_tokens:point['tokens']]
            report = dict(start_beat=point['beat']-1,end_beat=point['beat'],terminal_tick=point['beat']*96,
                mode='free',generated_tokens=len(new_tokens),history_tokens=0,
                backtracks=point['backtracks']-(reports[-1]['total_backtracks'] if reports else 0),
                total_backtracks=point['backtracks'],physical_bypass=bypass[published_bypass:point['bypass']],
                generation_skips=skips[published_skips:point['skips']])
            if options.trace: report['trace'] = [vars(t) for t in new_tokens]
            groups.extend(onsets); reports.append(report)
            if on_groups is not None: on_groups(tuple(onsets))
            emit(stream,'window_end',window=len(reports),groups=tuple(onsets),event_report=report)
            emit(progress,'window_done',window=len(reports),windows=end-start,tokens=len(new_tokens),
                 backtracks=report['backtracks'],cursor_beat=point['beat'],onsets=len(onsets))
            published_beats += 1
            published_content,published_events,published_tokens,published_bypass = point['content'],point['events'],point['tokens'],point['bypass']
            published_skips = point['skips']

    stopped_by_budget = False
    while (oracle.state.grammar.phase != 'finished' and
           (onset_end is None or oracle.state.grammar.beat*96 < onset_end)):
        if cancelled is not None and cancelled():
            raise GenerationCancelled()
        output = decoder(pack_row(row,device),memory,cache=cache,need_cache=True,
                         positions=([row['causal_beats']],[row['slide_beats']]))
        history = ()
        saved=oracle.fork(preserve_support=True)
        grammar = saved.state.grammar
        if escape_beat:
            token = EventToken(1,0) if grammar.phase=='subdiv' else EventToken(0,96)
            choices = iter(((token,saved.support(token.type)[token.value],output.cache),))
        else:
            choices = candidates(decoder,output,saved,sampler,None,duration,options,expansion,history,trajectory,
                                 grammar.tick if grammar.tick in relaxed_ticks else None)
        frames.append((saved,len(tokens),len(emitted),len(beat_ends),len(bypass),row,expansion,
                       cache,choices,None,trajectory,moment_origin,len(skips)))
        if len(frames)>options.rollback_tokens+1: del frames[0]
        retry = False
        failed_tick = failed_origin = failed_phase = None
        boundary_nt = None
        while frames:
            saved,nt,ne,nb,np,last_row,saved_expansion,previous_cache,choices,beat_length,previous_trajectory,moment_origin,ns = frames[-1]
            row = last_row; expansion = saved_expansion
            del tokens[nt:]; del emitted[ne:]; del beat_ends[nb:]; del bypass[np:]; del skips[ns:]
            try:
                token,evaluation,next_cache = next(choices)
            except StopIteration:
                exhausted=frames.pop(); backtracks += 1
                if failed_tick is None:
                    failed_tick = saved.state.grammar.tick
                    failed_phase = saved.state.grammar.phase
                    failed_origin = moment_origin if failed_phase not in ('moment','subdiv') else None
                    boundary_nt = nt if failed_origin is None else failed_origin[1]
                tick = failed_tick
                search_backtracks[tick]=search_backtracks.get(tick,0)+1
                if frames and frames[-1][1]>boundary_nt and search_backtracks[tick]<options.max_backtracks_per_moment: continue
                if not sampler.physical_hard_mask or tick in relaxed_ticks:
                    phase = failed_phase
                    origin = failed_origin
                    if origin is None:
                        origin = (saved,nt,ne,nb,np,last_row,saved_expansion,previous_cache,
                                  beat_length,previous_trajectory,ns)
                        escape_beat = True
                    saved,nt,ne,nb,np,row,expansion,cache,beat_length,trajectory,ns = origin
                    oracle = saved.fork()
                    del tokens[nt:]; del emitted[ne:]; del beat_ends[nb:]; del bypass[np:]; del skips[ns:]
                    sampler.excluded_moments.add(tick)
                    skipped = dict(tick=tick,phase=phase,reason='candidate_search_exhausted',
                                   backtracks=search_backtracks[tick],physical_enabled=options.physical_enabled,
                                   action='skip_beat_remainder' if escape_beat else 'skip_moment')
                    skips.append(skipped)
                    emit(progress,'generation_skip',**skipped)
                    frames.clear(); moment_origin = None; retry = True
                    break
                anchor=next((f for f in frames if f[0].state.grammar.tick==tick),exhausted)
                saved,nt,ne,nb,np,row,expansion,cache,_,beat_length,trajectory,moment_origin,ns = anchor
                oracle = saved.fork()
                del tokens[nt:]; del emitted[ne:]; del beat_ends[nb:]; del bypass[np:]; del skips[ns:]
                relaxed_ticks.add(tick)
                bypass.append(dict(tick=oracle.state.grammar.tick,field='search',
                    reasons=['physical_search_exhausted'],mandatory_physical=['multitouch'],
                    scope='moment',backtracks=search_backtracks[tick]))
                frames.clear(); retry = True
                break
            if token.type==0 and token.value!=96:
                moment_origin = (saved,nt,ne,nb,np,last_row,saved_expansion,previous_cache,
                                 beat_length,previous_trajectory,ns)
            oracle = saved.fork(preserve_support=True)
            context = oracle.state.grammar.flag_context
            trajectory = previous_trajectory.advance(token,oracle.state.grammar)
            oracle.accept(token,evaluation)
            for tick,content in oracle.state.groups: emitted.extend((tick,t) for t in content)
            oracle.state.groups.clear()
            row = input_row(token,oracle.state.grammar,timing,context=context)
            tokens.append(token); cache = next_cache
            if evaluation.raw.conflicts:
                bypass.append(dict(tick=oracle.state.grammar.tick,field=token.type,reasons=sorted(evaluation.raw.decision().physical)))
            if token.type in ((4,9,19) if getattr(oracle.state.grammar,'no_placeholders',False) else (4,5,9)): expansion += 1
            if token.type == 12: expansion = 0
            if token.type == 0 and token.value == 96:
                escape_beat = False
                cache = cache.evict(oracle.state.grammar.beat,options.rollback_tokens)
                beat_ends.append(dict(beat=oracle.state.grammar.beat,tokens=len(tokens),content=len(emitted),
                                      events=len(oracle.state.grammar.events),bypass=len(bypass),backtracks=backtracks,skips=len(skips)))
            break
        else:
            raise GenerationError(f'Simai stream has no legal completion at beat {oracle.state.grammar.tick/96:g}')
        if retry: continue
        publish()
        if (token_budget is not None and len(tokens) >= token_budget and
                oracle.state.grammar.phase in ('moment', 'finished')):
            stopped_by_budget = True
            break
    if tokens and (not beat_ends or beat_ends[-1]['tokens'] < len(tokens)):
        beat_ends.append(dict(beat=oracle.state.grammar.beat+1,tokens=len(tokens),content=len(emitted),
                              events=len(oracle.state.grammar.events),bypass=len(bypass),
                              backtracks=backtracks,skips=len(skips)))
    publish(force=True)
    if prefix_states is not None:
        from .prefix import remember
        remember(prefix_states, (*prefix, *tokens), oracle, cache, row, trajectory, options)
    if execution_report is not None:
        execution_report.update(tokens=len(tokens), token_limit=stopped_by_budget,
                                physical_bypass=sum(len(r['physical_bypass']) for r in reports))
    return (chart_from_groups(groups,timing,metadata) if return_chart else None),reports
