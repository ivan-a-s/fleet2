"""
Vehicle-level plots for sanity-checking against the Paper 1 model.

Two groups of plots, mirroring the old Vehicle.Plots and Fleet.Plots:

  age_profiles(fleet, k, y)
      One figure per attribute, all powertrains overlaid, vs vehicle age.
      Mirrors old Vehicle.Plots: survival_rate, average_speed, fuel_consumption,
      range, annual_distance, emissions, annual_costs.

  cross_powertrain(fleet, k)
      Stacked-bar per powertrain × sample year.
      Mirrors old Fleet.Plots: vehicle_mass, capital_cost, tco, lca_emissions.

Run this file directly to show all plots for all vehicle types.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
from model import Fleet, get_uncertainty_distributions, START_YEAR, END_YEAR, MAX_AGE, DISCOUNT_RATE
from data import PARAMS

SAMPLE_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]

_CYCLE   = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11})

PT_COLOR = {p: _CYCLE[i % len(_CYCLE)]
            for i, p in enumerate(['dice', 'he', 'phe', 'be', 'fc', 'hice', 'dhice'])}

KEY_LABELS = {
    'frame':                   'Frame',
    'trailer':                 'Trailer',
    'payload':                 'Payload',
    'ice':                     'ICE',
    'motor':                   'Motor',
    'battery':                 'Battery',
    'fc':                      'Fuel Cell',
    'diesel_tank':             'Diesel Tank',
    'h2_700bar':               'H₂ Tank (700 bar)',
    'h2_350bar':               'H₂ Tank (350 bar)',
    'electronic_controller':   'Electronic Controller',
    'combustion_transmission': 'Combustion Transmission',
    'electric_transmission':   'Electric Transmission',
    'after_treatment':         'Aftertreatment',
    'engine':                  'Engine',
    'h2_tank':                 'H₂ Tank',
    'tank':                    'Diesel Tank',
    'charger':                 'Charger',
    'capital':                 'Capital',
    'operational':             'Operational',
    'fuel':                    'Fuel',
    'driver':                  'Driver',
    'fc_replacements':         'FC Replacements',
    'revenue':                 'Revenue',
    'embodied':                'Embodied',
    'supply':                  'Supply',
    'use':                     'Use',
}

PT_LABELS = {
    'dice':  'DICE',
    'he':    'HE',
    'phe':   'PHE',
    'be':    'BE',
    'fc':    'FC',
    'hice':  'HICE',
    'dhice': 'DHICE',
}

K_LABELS = {
    'sleeper':  'Sleeper',
    'day_cab':  'Day Cab',
    'straight': 'Straight',
}

def _unique_keys(fleet, k, attr):
    """All unique non-zero component keys across every vehicle of type k, in first-appearance order."""
    seen = []
    for p in fleet.P[k]:
        for y in SAMPLE_YEARS:
            if (k, p, y) not in fleet.vehicles:
                continue
            for key, val in getattr(fleet.vehicles[k, p, y], attr).items():
                if key not in seen and np.any(np.asarray(val) != 0):
                    seen.append(key)
    return seen

def _colours(keys):
    """Standard cycle color for each key, assigned by position."""
    return {key: _CYCLE[i % len(_CYCLE)] for i, key in enumerate(keys)}

def _bar_layout(n, year_gap=5, fill=0.80, internal_gap=0.10):
    width   = (year_gap * fill - (n - 1) * internal_gap) / max(n, 1)
    offsets = [i * (width + internal_gap) - (year_gap * fill) / 2 + width / 2 for i in range(n)]
    return width, offsets

def _stacked_bar(ax, x, comps, width, col):
    """Stacked bar using pre-built name→color dict col."""
    bottom = 0.0
    for key, val in comps.items():
        val = float(np.asarray(val).flat[0])
        if val == 0:
            continue
        ax.bar(x, val, bottom=bottom, width=width, color=col[key], label=key)
        bottom += val
    return bottom

def _legend(ax, keys, col, **kw):
    """Legend in key order, one entry per key, using pre-built colors."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=col[k]) for k in keys]
    labels  = [KEY_LABELS.get(k, k) for k in keys]
    ax.legend(handles, labels, **kw)


# ---------------------------------------------------------------------------
# Age profiles — all powertrains on one figure, vs vehicle age
# ---------------------------------------------------------------------------

def age_profiles(fleet, k='sleeper', y=START_YEAR):
    """Plots mirroring old Vehicle.Plots, shown for all powertrains of type k at model year y."""
    P   = fleet.P[k]
    age = np.arange(MAX_AGE)

    # Survival rate
    fig, ax = plt.subplots()
    ax.set_title(f'Survival rate — {K_LABELS.get(k, k)} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('Survival rate')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.params['survival_rate'], color=PT_COLOR.get(p, 'gray'), label=PT_LABELS.get(p, p))
    ax.set_ylim(0, None)
    ax.legend()

    # Average speed
    fig, ax = plt.subplots()
    ax.set_title(f'Average speed — {K_LABELS.get(k, k)} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('km/h')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.average_speed, color=PT_COLOR.get(p, 'gray'), label=PT_LABELS.get(p, p))
    ax.set_ylim(0, None)
    ax.legend()

    # Fuel consumption — one line per fuel per powertrain
    fig, ax = plt.subplots()
    ax.set_title(f'Fuel consumption — {K_LABELS.get(k, k)} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('L or kg or kWh per km')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        for f, fc in v.fuel_consumption.items():
            ax.plot(age, fc, color=PT_COLOR.get(p, 'gray'), label=f'{PT_LABELS.get(p, p)} ({f})',
                    linestyle='--' if f != 'diesel' else '-')
    ax.set_ylim(0, None)
    ax.legend(fontsize=7)

    # Range
    fig, ax = plt.subplots()
    ax.set_title(f'Range — {K_LABELS.get(k, k)} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('km')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.range, color=PT_COLOR.get(p, 'gray'), label=PT_LABELS.get(p, p))
    ax.set_ylim(0, None)
    ax.legend()

    # Annual distance (actual vs target)
    fig, ax = plt.subplots()
    ax.set_title(f'Annual distance — {K_LABELS.get(k, k)} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('km / year')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.annual_distance,             color=PT_COLOR.get(p, 'gray'), label=PT_LABELS.get(p, p))
        ax.plot(age, v.params['target_distance'],   color=PT_COLOR.get(p, 'gray'), linestyle='--', alpha=0.4)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, title='dashed = target')

    # Emissions vs age
    fig, ax = plt.subplots()
    ax.set_title(f'Annual emissions — {K_LABELS.get(k, k)} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('tCO2e / year')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.embodied         / 1000, color=PT_COLOR.get(p, 'gray'), linestyle=':',  label=f'{PT_LABELS.get(p, p)} embodied')
        ax.plot(age, v.emissions_supply / 1000, color=PT_COLOR.get(p, 'gray'), linestyle='--', label=f'{PT_LABELS.get(p, p)} supply')
        ax.plot(age, v.emissions_use    / 1000, color=PT_COLOR.get(p, 'gray'), linestyle='-',  label=f'{PT_LABELS.get(p, p)} use')
    ax.set_ylim(0, None)
    ax.legend(fontsize=7)

    # Annual costs vs age — total only
    fig, ax = plt.subplots()
    ax.set_title(f'Annual costs — {K_LABELS.get(k, k)} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('$ thousands / year')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        total = sum(v.annual_cost.values())
        ax.plot(age, total / 1000, color=PT_COLOR.get(p, 'gray'), linewidth=2, label=PT_LABELS.get(p, p))
    ax.set_ylim(0, None)
    ax.legend()


# ---------------------------------------------------------------------------
# Cross-powertrain stacked bars — all powertrains × sample years
# ---------------------------------------------------------------------------

def _iter_pys(fleet, k):
    P = fleet.P[k]
    width, offsets = _bar_layout(len(P))
    for i, p in enumerate(P):
        for y in SAMPLE_YEARS:
            if (k, p, y) in fleet.vehicles:
                yield p, y, offsets[i], width

def vehicle_mass(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'Vehicle mass — {K_LABELS.get(k, k)}')
    ax.set_ylabel('kg')
    keys = _unique_keys(fleet, k, 'mass')
    col  = _colours(keys)
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        total = _stacked_bar(ax, y + offset, v.mass, width, col)
        ax.text(y + offset, total * 1.01, PT_LABELS.get(p, p), ha='center', va='bottom', fontsize=9, rotation=90)
    ax.set_xticks(SAMPLE_YEARS)
    _legend(ax, keys, col, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()

def capital_cost(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'Capital cost — {K_LABELS.get(k, k)}')
    ax.set_ylabel('$ thousands')
    keys = _unique_keys(fleet, k, 'capital')
    col  = _colours(keys)
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        comps = {kk: vv / 1000 for kk, vv in v.capital.items()}
        total = _stacked_bar(ax, y + offset, comps, width, col)
        ax.text(y + offset, total * 1.01, PT_LABELS.get(p, p), ha='center', va='bottom', fontsize=9, rotation=90)
    ax.set_xticks(SAMPLE_YEARS)
    _legend(ax, keys, col, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()

def tco(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'TCO breakdown — {K_LABELS.get(k, k)}')
    ax.set_ylabel('$ thousands (NPV-discounted)')
    keys = _unique_keys(fleet, k, 'annual_cost')
    col  = _colours(keys)
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        comps = {key: v._discount(arr) / 1000 for key, arr in v.annual_cost.items()}
        total = _stacked_bar(ax, y + offset, comps, width, col)
        ax.text(y + offset, total * 1.01, PT_LABELS.get(p, p), ha='center', va='bottom', fontsize=9, rotation=90)
    ax.set_xticks(SAMPLE_YEARS)
    _legend(ax, keys, col, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()

def lca_emissions(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'Lifetime LCA emissions — {K_LABELS.get(k, k)}')
    ax.set_ylabel('tCO2e (survival-weighted)')
    lca_keys = ['embodied', 'supply', 'use']
    col      = _colours(lca_keys)
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        surv = np.array(v.params['survival_rate'])
        comps = {
            'embodied': float(v.embodied[0]) / 1000,
            'supply':   float(np.sum(v.emissions_supply * surv)) / 1000,
            'use':      float(np.sum(v.emissions_use    * surv)) / 1000,
        }
        total = _stacked_bar(ax, y + offset, comps, width, col)
        ax.text(y + offset, total * 1.01, PT_LABELS.get(p, p), ha='center', va='bottom', fontsize=9, rotation=90)
    ax.set_xticks(SAMPLE_YEARS)
    _legend(ax, lca_keys, col, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()


def npv(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'NPV breakdown — {K_LABELS.get(k, k)}')
    ax.set_ylabel('$ thousands (NPV-discounted)')
    npv_keys = ['revenue'] + _unique_keys(fleet, k, 'annual_cost')
    col      = _colours(npv_keys)
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        x = y + offset
        vals = {'revenue': v._discount(v.annual_revenue) / 1000}
        vals.update({key: -v._discount(arr) / 1000 for key, arr in v.annual_cost.items()})
        pos_bottom = neg_bottom = 0.0
        for key in npv_keys:
            val = vals.get(key, 0.0)
            if val == 0:
                continue
            bottom = pos_bottom if val > 0 else neg_bottom
            ax.bar(x, val, bottom=bottom, width=width, color=col[key], label=key)
            if val > 0:
                pos_bottom += val
            else:
                neg_bottom += val
        ax.text(x, pos_bottom * 1.01, PT_LABELS.get(p, p), ha='center', va='bottom', fontsize=9, rotation=90)
        ax.scatter(x, v.npv / 1000, marker='x', color='black', zorder=5, s=60, linewidths=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(SAMPLE_YEARS)
    _legend(ax, npv_keys, col, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    inputs_distributions = dict(get_uncertainty_distributions(PARAMS))
    param_cps = dict(zip(inputs_distributions.keys(), np.random.rand(len(inputs_distributions)).astype('float32')))
    fleet = Fleet(PARAMS, param_cps, exclude_powertrains=('phe',))

    for k in ['sleeper']:
        age_profiles(fleet, k=k, y=START_YEAR)
        vehicle_mass(fleet, k=k)
        capital_cost(fleet, k=k)
        # tco(fleet, k=k)
        lca_emissions(fleet, k=k)
        npv(fleet, k=k)

    plt.show()
