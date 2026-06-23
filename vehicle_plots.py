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
import numpy as np
import matplotlib.pyplot as plt
from model import Fleet, get_uncertainty_distributions, START_YEAR, END_YEAR, MAX_AGE, DISCOUNT_RATE
from data import PARAMS

SAMPLE_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]

_CYCLE   = plt.rcParams['axes.prop_cycle'].by_key()['color']
PT_COLOR = {p: _CYCLE[i % len(_CYCLE)]
            for i, p in enumerate(['dice', 'he', 'phe', 'be', 'fc', 'hice', 'dhice'])}

def _comp_colors(keys):
    cmap = plt.cm.tab20
    return {k: cmap(i / max(len(keys), 1)) for i, k in enumerate(sorted(keys))}

def _bar_layout(n, year_gap=5, fill=0.80, internal_gap=0.10):
    width   = (year_gap * fill - (n - 1) * internal_gap) / max(n, 1)
    offsets = [i * (width + internal_gap) - (year_gap * fill) / 2 + width / 2 for i in range(n)]
    return width, offsets

def _stacked_bar(ax, x, comps, width, colors):
    bottom = 0.0
    for label in sorted(comps):
        v = float(np.asarray(comps[label]).flat[0])   # age-0 scalar
        if v == 0:
            continue
        ax.bar(x, v, bottom=bottom, width=width, color=colors.get(label, 'gray'), label=label)
        bottom += v
    return bottom

def _dedup_legend(ax, **kw):
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), **kw)


# ---------------------------------------------------------------------------
# Age profiles — all powertrains on one figure, vs vehicle age
# ---------------------------------------------------------------------------

def age_profiles(fleet, k='sleeper', y=START_YEAR):
    """Plots mirroring old Vehicle.Plots, shown for all powertrains of type k at model year y."""
    P   = fleet.P[k]
    age = np.arange(MAX_AGE)

    # Survival rate
    fig, ax = plt.subplots()
    ax.set_title(f'Survival rate — {k} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('Survival rate')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.params['survival_rate'], color=PT_COLOR.get(p, 'gray'), label=p)
    ax.set_ylim(0, None)
    ax.legend()

    # Average speed
    fig, ax = plt.subplots()
    ax.set_title(f'Average speed — {k} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('km/h')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.average_speed, color=PT_COLOR.get(p, 'gray'), label=p)
    ax.set_ylim(0, None)
    ax.legend()

    # Fuel consumption — one line per fuel per powertrain
    fig, ax = plt.subplots()
    ax.set_title(f'Fuel consumption — {k} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('L or kg or kWh per km')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        for f, fc in v.fuel_consumption.items():
            ax.plot(age, fc, color=PT_COLOR.get(p, 'gray'), label=f'{p} ({f})',
                    linestyle='--' if f != 'diesel' else '-')
    ax.set_ylim(0, None)
    ax.legend(fontsize=7)

    # Range
    fig, ax = plt.subplots()
    ax.set_title(f'Range — {k} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('km')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.range, color=PT_COLOR.get(p, 'gray'), label=p)
    ax.set_ylim(0, None)
    ax.legend()

    # Annual distance (actual vs target)
    fig, ax = plt.subplots()
    ax.set_title(f'Annual distance — {k} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('km / year')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.annual_distance,             color=PT_COLOR.get(p, 'gray'), label=p)
        ax.plot(age, v.params['target_distance'],   color=PT_COLOR.get(p, 'gray'), linestyle='--', alpha=0.4)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, title='dashed = target')

    # Emissions vs age
    fig, ax = plt.subplots()
    ax.set_title(f'Annual emissions — {k} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('tCO2e / year')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.embodied         / 1000, color=PT_COLOR.get(p, 'gray'), linestyle=':',  label=f'{p} embodied')
        ax.plot(age, v.emissions_supply / 1000, color=PT_COLOR.get(p, 'gray'), linestyle='--', label=f'{p} supply')
        ax.plot(age, v.emissions_use    / 1000, color=PT_COLOR.get(p, 'gray'), linestyle='-',  label=f'{p} use')
    ax.set_ylim(0, None)
    ax.legend(fontsize=7)

    # Annual costs vs age — total only
    fig, ax = plt.subplots()
    ax.set_title(f'Annual costs — {k} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('$ thousands / year')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        total = sum(v.annual_cost.values())
        ax.plot(age, total / 1000, color=PT_COLOR.get(p, 'gray'), linewidth=2, label=p)
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
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(f'Vehicle mass — {k}')
    ax.set_ylabel('kg')
    all_keys = set(kk for p in fleet.P[k] for y in SAMPLE_YEARS
                   if (k, p, y) in fleet.vehicles for kk in fleet.vehicles[k, p, y].mass)
    colors = _comp_colors(all_keys)
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        total = _stacked_bar(ax, y + offset, v.mass, width, colors)
        ax.text(y + offset, total * 1.01, p, ha='center', va='bottom', fontsize=6, rotation=90)
    ax.set_xticks(SAMPLE_YEARS)
    _dedup_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()

def capital_cost(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(f'Capital cost — {k}')
    ax.set_ylabel('$ thousands')
    all_keys = set(kk for p in fleet.P[k] for y in SAMPLE_YEARS
                   if (k, p, y) in fleet.vehicles for kk in fleet.vehicles[k, p, y].capital)
    colors = _comp_colors(all_keys)
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        comps = {kk: vv / 1000 for kk, vv in v.capital.items()}
        total = _stacked_bar(ax, y + offset, comps, width, colors)
        ax.text(y + offset, total * 1.01, p, ha='center', va='bottom', fontsize=6, rotation=90)
    ax.set_xticks(SAMPLE_YEARS)
    _dedup_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()

def tco(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(f'TCO breakdown — {k}')
    ax.set_ylabel('$ thousands (NPV-discounted)')
    colors = _comp_colors({'capital', 'operational', 'fuel', 'driver', 'fc_replacements'})
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        comps = {key: v._discount(arr) / 1000 for key, arr in v.annual_cost.items()}
        total = _stacked_bar(ax, y + offset, comps, width, colors)
        ax.text(y + offset, total * 1.01, p, ha='center', va='bottom', fontsize=6, rotation=90)
    ax.set_xticks(SAMPLE_YEARS)
    _dedup_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()

def lca_emissions(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(f'Lifetime LCA emissions — {k}')
    ax.set_ylabel('tCO2e (survival-weighted)')
    colors = _comp_colors({'embodied', 'supply', 'use'})
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        surv = np.array(v.params['survival_rate'])
        comps = {
            'embodied': float(v.embodied[0]) / 1000,
            'supply':   float(np.sum(v.emissions_supply * surv)) / 1000,
            'use':      float(np.sum(v.emissions_use    * surv)) / 1000,
        }
        total = _stacked_bar(ax, y + offset, comps, width, colors)
        ax.text(y + offset, total * 1.01, p, ha='center', va='bottom', fontsize=6, rotation=90)
    ax.set_xticks(SAMPLE_YEARS)
    _dedup_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()


def npv(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(f'NPV breakdown — {k}')
    ax.set_ylabel('$ thousands (NPV-discounted)')
    cost_keys = {'capital', 'operational', 'fuel', 'driver', 'fc_replacements'}
    colors = _comp_colors(cost_keys | {'revenue'})

    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        x = y + offset

        rev = v._discount(v.annual_revenue) / 1000
        ax.bar(x, rev, width=width, color=colors.get('revenue', 'green'), label='revenue')
        ax.text(x, rev * 1.01, p, ha='center', va='bottom', fontsize=6, rotation=90)

        bottom = 0.0
        for key, arr in v.annual_cost.items():
            val = -v._discount(arr) / 1000
            ax.bar(x, val, bottom=bottom, width=width, color=colors.get(key, 'gray'), label=key)
            bottom += val

        ax.scatter(x, v.npv / 1000, marker='x', color='black', zorder=5, s=60, linewidths=1.5)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(SAMPLE_YEARS)
    _dedup_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    inputs_distributions = dict(get_uncertainty_distributions(PARAMS))
    param_cps = dict(zip(inputs_distributions.keys(), np.random.rand(len(inputs_distributions)).astype('float32')))
    fleet = Fleet(PARAMS, param_cps, exclude_powertrains=('phe',))

    for k in fleet.K:
        age_profiles(fleet, k=k, y=START_YEAR)
        vehicle_mass(fleet, k=k)
        capital_cost(fleet, k=k)
        tco(fleet, k=k)
        lca_emissions(fleet, k=k)
        npv(fleet, k=k)

    plt.show()
