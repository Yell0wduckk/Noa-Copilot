"""Explicit AdamW beta validation shared by the joint recipe."""


def adamw_betas(config, prefix):
    values = []
    for name in ('beta1', 'beta2'):
        key = prefix + '_' + name
        value = config[key]
        if type(value) not in (int, float) or not 0 <= value < 1:
            raise ValueError(f'{key} must be a finite number in [0, 1)')
        values.append(float(value))
    return tuple(values)
