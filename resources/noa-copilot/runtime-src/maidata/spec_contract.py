"""Dispatch CompactSpec and YMT3+MuQ data paths."""
from . import compactspec_contract as compact
from . import ymt3_muq_contract as ymt3_muq


def enabled(config):
    return compact.enabled(config) or ymt3_muq.enabled(config)


def extractor(config):
    if ymt3_muq.enabled(config):
        from .preparation import ymt3_muq as ymt3_extractor
        return ymt3_extractor
    if not compact.enabled(config):
        raise ValueError('unsupported spectral extractor configuration')
    from .preparation import compactspec
    return compactspec
