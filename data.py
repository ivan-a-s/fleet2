import json
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _expand_arrays(d, n):
    if isinstance(d, dict):
        if 'array' in d:
            kind = d['array']
            if kind == 'logistic':
                age = np.arange(n)
                return (d['max_val'] - d['min_val']) / (1 + np.exp(d['k'] * (age - d['x0']))) + d['min_val']
            elif kind == 'linspace':
                return np.linspace(d['start'], d['end'], n + 1)[:-1]
            elif kind == 'step':
                return [d['below'] if y < d['threshold'] else d['above'] for y in range(n)]
            elif kind == 'constant':
                return [d['value']] * n
            else:
                raise ValueError(f"Unknown array type: {kind}")
        return {k: _expand_arrays(v, n) for k, v in d.items()}
    if isinstance(d, list):
        return [_expand_arrays(v, n) for v in d]
    return d


with open(os.path.join(_HERE, 'data.json')) as _f:
    _raw = json.load(_f)

_MAX_AGE = _raw['settings']['max_age']
PARAMS = _expand_arrays(_raw, _MAX_AGE)

MAX_AGE      = PARAMS['settings']['max_age']
AIR_DENSITY  = PARAMS['settings']['air_density']
GRAVITY      = PARAMS['settings']['gravity']
START_YEAR   = PARAMS['settings']['start_year']
END_YEAR     = PARAMS['settings']['end_year']
DISCOUNT_RATE = PARAMS['settings']['discount_rate']
GROWTH_RATE  = PARAMS['settings']['growth_rate']
