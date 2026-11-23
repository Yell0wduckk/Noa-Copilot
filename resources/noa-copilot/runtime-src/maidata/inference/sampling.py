"""Hierarchical field sampling against current, prefix-only rule queries."""
import torch

from .options import resolve_seed, validate_slide_limit


class Sampler:
    def __init__(self, *, temperature=.4, seed=20261010, physical_hard_mask=True,
                 physical_enabled=True, top_k=0, top_p=1., floor_tick=None,
                 max_slide_judge_areas=40, **_unused):
        self.floor_tick = floor_tick
        validate_slide_limit(max_slide_judge_areas)
        self.max_slide_judge_areas = max_slide_judge_areas
        if not 0 <= temperature < float('inf'):
            raise ValueError('temperature must be finite and nonnegative')
        self.temperature = temperature
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')
        if not 0 < top_p <= 1:
            raise ValueError('top_p must be in (0, 1]')
        self.top_k, self.top_p = top_k, top_p
        self.seed = resolve_seed(seed)
        self.generator = torch.Generator(device='cpu').manual_seed(self.seed)
        self.physical_enabled = physical_enabled
        self.physical_hard_mask = physical_hard_mask and physical_enabled

    def filtered_scores(self, scores, allowed=None, *, truncate=True, **_unused):
        values = scores.detach().float().cpu().flatten().clone()
        if torch.isnan(values).any() or torch.isposinf(values).any():
            raise ValueError('nonfinite model sampling scores')
        if allowed is not None:
            values.masked_fill_(~torch.as_tensor(allowed, dtype=torch.bool, device='cpu'), -torch.inf)
        if truncate and (self.top_k or self.top_p < 1) and torch.isfinite(values).any():
            indices = values.argsort(descending=True, stable=True)
            indices = indices[torch.isfinite(values[indices])]
            if self.top_k:
                indices = indices[:self.top_k]
            if self.top_p < 1:
                if self.temperature == 0:
                    indices = indices[:1]
                else:
                    logits = values[indices]
                    probabilities = ((logits - logits.max()) / self.temperature).softmax(0)
                    indices = indices[(probabilities.cumsum(0) - probabilities) < self.top_p]
            keep = torch.zeros_like(values, dtype=torch.bool)
            keep[indices] = True
            values.masked_fill_(~keep, -torch.inf)
        return values

    def log_probabilities(self, values):
        if not torch.isfinite(values).any():
            return torch.full_like(values, -torch.inf, dtype=torch.float64)
        return (values.double() / (self.temperature or 1.)).log_softmax(0)

    def order(self, scores, allowed=None, **kwargs):
        yield from self.order_filtered(self.filtered_scores(scores, allowed, **kwargs))

    def grouped_order(self, scores, groups, allowed=None):
        values = self.filtered_scores(scores, allowed, truncate=False)
        group_scores = torch.stack([torch.logsumexp(values[list(group)], 0) for group in groups])
        for group_index in self.order(group_scores):
            members = groups[group_index]
            member_scores = self.filtered_scores(values[list(members)])
            retained = torch.isfinite(member_scores).nonzero().flatten()
            if len(retained) == 1:
                yield members[int(retained[0])]
            else:
                for index in self.order_filtered(member_scores):
                    yield members[index]

    def order_filtered(self, values):
        values = values.clone()
        while torch.isfinite(values).any():
            index = (int(values.argmax()) if self.temperature == 0 else
                     int(torch.multinomial(((values - values.max()) / self.temperature).softmax(0),
                                           1, generator=self.generator)))
            values[index] = -torch.inf
            yield index

    def allowed(self, evaluation):
        d = evaluation.raw.decision() if hasattr(evaluation, 'raw') else evaluation
        return not d.basic_errors and (not self.physical_enabled or (
            'multitouch' not in d.physical and (not self.physical_hard_mask or not d.physical)))

    @staticmethod
    def slide_judge_area_count(oracle):
        from maidata.chart.routes import chain_paths
        state = oracle.physical.grammar.state
        count, previous = 0, None
        for key in chain_paths(state.note.pad, state.routes):
            entries = oracle.physical.geometry.paths[key].entries
            count += len(entries) - int(bool(entries) and previous == entries[0][1])
            if entries:
                previous = entries[-1][1]
        return count
