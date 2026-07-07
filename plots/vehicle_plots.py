"""
Vehicle-level plots for sanity-checking against the Paper 1 model.

Two groups of plots, mirroring the old Vehicle.Plots and Fleet.Plots:

  age_profiles(fleet, k, y)
      One figure per attribute, all powertrains overlaid, vs vehicle age.
      Mirrors old Vehicle.Plots: survival_rate, average_speed, fuel_consumption,
      range, annual_distance, emissions, annual_costs.

  cross_powertrain(fleet, k)
      Stacked-bar per powertrain x sample year.
      Mirrors old Fleet.Plots: vehicle_mass, capital_cost, tco, lca_emissions.

Run this file directly to show all plots for all vehicle types.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from model import Fleet, get_uncertainty_distributions, START_YEAR, MAX_AGE
from data import PARAMS
from plot_utils import (SAMPLE_YEARS, PT_COLOR, PT_LABELS, K_LABELS,
                        _unique_keys, _colours, _bar_layout, _stacked_bar, _legend)


# ---------------------------------------------------------------------------
# Age profiles -- all powertrains on one figure, vs vehicle age
# ---------------------------------------------------------------------------

def age_profiles(fleet, k='sleeper', y=START_YEAR):
    """Plots mirroring old Vehicle.Plots, shown for all powertrains of type k at model year y."""
    P   = fleet.P[k]
    age = np.arange(MAX_AGE)

    # Survival rate
    fig, ax = plt.subplots()
    ax.set_title(f'Survival rate -- {K_LABELS.get(k, k)} {y}')
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
    ax.set_title(f'Average speed -- {K_LABELS.get(k, k)} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('km/h')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.average_speed, color=PT_COLOR.get(p, 'gray'), label=PT_LABELS.get(p, p))
    ax.set_ylim(0, None)
    ax.legend()

    # Fuel consumption -- one line per fuel per powertrain
    fig, ax = plt.subplots()
    ax.set_title(f'Fuel consumption -- {K_LABELS.get(k, k)} {y}')
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
    ax.set_title(f'Range -- {K_LABELS.get(k, k)} {y}')
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
    ax.set_title(f'Annual distance -- {K_LABELS.get(k, k)} {y}')
    ax.set_xlabel('Vehicle age (years)')
    ax.set_ylabel('km / year')
    for p in P:
        if (k, p, y) not in fleet.vehicles:
            continue
        v = fleet.vehicles[k, p, y]
        ax.plot(age, v.annual_distance,           color=PT_COLOR.get(p, 'gray'), label=PT_LABELS.get(p, p))
        ax.plot(age, v.params['target_distance'], color=PT_COLOR.get(p, 'gray'), linestyle='--', alpha=0.4)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, title='dashed = target')

    # Emissions vs age
    fig, ax = plt.subplots()
    ax.set_title(f'Annual emissions -- {K_LABELS.get(k, k)} {y}')
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

    # Annual costs vs age -- total only
    fig, ax = plt.subplots()
    ax.set_title(f'Annual costs -- {K_LABELS.get(k, k)} {y}')
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
# Cross-powertrain stacked bars -- all powertrains x sample years
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
    ax.set_title(f'Vehicle mass -- {K_LABELS.get(k, k)}')
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
    ax.set_title(f'Capital cost -- {K_LABELS.get(k, k)}')
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
    ax.set_title(f'TCO breakdown -- {K_LABELS.get(k, k)}')
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

def _embodied_components(v):
    """
    Per-component lifetime embodied emissions (kgCO2e) for one Vehicles cohort.
    Mirrors model.py Vehicles._calculate_emissions -- keep in sync if that logic changes.
    frame/trailer/tire/trailer_tire are lump sums at age 0 (per the model's own treatment);
    the fuel-cell stack's replacement emissions are the only age>0 contribution, so they're
    survival-weighted here to match how supply/use emissions are aggregated below.
    """
    p = v.params
    frame_emb = float(p['components']['frame']['embodied_emissions'])
    comps = {'frame': float(p['components']['frame']['mass']) * frame_emb}

    trailer = p['components'].get('trailer')
    if trailer is not None:
        t_emb = float(trailer.get('embodied_emissions', frame_emb))
        comps['trailer'] = float(trailer['mass']) * float(p.get('trailers_per_truck', 0)) * t_emb

    tire = p['components'].get('tire')
    if tire is not None:
        comps['tire'] = float(tire['mass']) * float(tire.get('embodied_emissions', frame_emb))

    trailer_tire = p['components'].get('trailer_tire')
    if trailer_tire is not None:
        tt_emb = float(trailer_tire.get('embodied_emissions', frame_emb))
        comps['trailer_tire'] = float(trailer_tire['mass']) * tt_emb

    for name, comp in p['components'].items():
        if name in ('frame', 'trailer', 'tire', 'trailer_tire'):
            continue
        if comp['type'] == 'ess' and 'embodied_emissions' in comp:
            comps[name] = float(comp['capacity']) * float(comp['embodied_emissions'])
        elif comp['type'] in ('converter', 'transmission') and 'mass' in comp:
            comps[name] = float(comp['mass']) * float(comp.get('embodied_emissions', frame_emb))

    fc_comp = p['components'].get('fc')
    if fc_comp is not None and np.any(v.fc_replacements > 0):
        surv     = np.asarray(p['survival_rate'])
        comp_emb = float(fc_comp.get('embodied_emissions', frame_emb))
        comps['fc_replacements'] = float(np.sum(v.fc_replacements * surv)) * float(fc_comp['mass']) * comp_emb

    return comps

def _unique_embodied_keys(fleet, k):
    """All unique non-zero embodied-component keys across every vehicle of type k."""
    seen = []
    for p in fleet.P[k]:
        for y in SAMPLE_YEARS:
            if (k, p, y) not in fleet.vehicles:
                continue
            for key, val in _embodied_components(fleet.vehicles[k, p, y]).items():
                if key not in seen and val != 0:
                    seen.append(key)
    return seen

def lca_emissions(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'Lifetime LCA emissions -- {K_LABELS.get(k, k)}')
    ax.set_ylabel('tCO2e (survival-weighted)')
    lca_keys = ['embodied', 'supply', 'use']
    col      = _colours(lca_keys)
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        surv = np.array(v.params['survival_rate'])
        comps = {
            'embodied': sum(_embodied_components(v).values()) / 1000,
            'supply':   float(np.sum(v.emissions_supply * surv)) / 1000,
            'use':      float(np.sum(v.emissions_use    * surv)) / 1000,
        }
        total = _stacked_bar(ax, y + offset, comps, width, col)
        ax.text(y + offset, total * 1.01, PT_LABELS.get(p, p), ha='center', va='bottom', fontsize=9, rotation=90)
    ax.set_xticks(SAMPLE_YEARS)
    _legend(ax, lca_keys, col, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()

def embodied_emissions(fleet, k='sleeper'):
    """Embodied emissions broken down by physical component (frame, trailer, tire,
    ICE, motor, battery, H2 tank, etc.), one stacked bar per powertrain x sample year --
    same layout as capital_cost/tco/npv. Complements lca_emissions, which only shows the
    embodied/supply/use split without opening up what's inside 'embodied'."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'Embodied emissions breakdown -- {K_LABELS.get(k, k)}')
    ax.set_ylabel('tCO2e (lifetime)')
    keys = _unique_embodied_keys(fleet, k)
    col  = _colours(keys)
    for p, y, offset, width in _iter_pys(fleet, k):
        v = fleet.vehicles[k, p, y]
        comps = {kk: vv / 1000 for kk, vv in _embodied_components(v).items()}
        total = _stacked_bar(ax, y + offset, comps, width, col)
        ax.text(y + offset, total * 1.01, PT_LABELS.get(p, p), ha='center', va='bottom', fontsize=9, rotation=90)
    ax.set_xticks(SAMPLE_YEARS)
    _legend(ax, keys, col, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.tight_layout()


def npv(fleet, k='sleeper'):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title(f'NPV breakdown -- {K_LABELS.get(k, k)}')
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
    param_cps = {path: np.float32(0.5) for path, _ in get_uncertainty_distributions(PARAMS)}
    fleet = Fleet(PARAMS, param_cps)

    for k in ['straight']:
        age_profiles(fleet, k=k, y=START_YEAR)
        vehicle_mass(fleet, k=k)
        capital_cost(fleet, k=k)
        # tco(fleet, k=k)
        lca_emissions(fleet, k=k)
        embodied_emissions(fleet, k=k)
        npv(fleet, k=k)

    plt.show()
