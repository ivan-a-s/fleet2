"""
ZEV mandate policy plots for the fleet2 HDT adoption model.

MC-ready: loads results/baseline.npz and results/zev_mandate.npz when available;
falls back to a single deterministic run at median parameters otherwise.

  load_data()                                         -- load MC or build single-run fleets
  npv_comparison(base, mandate, is_mc, k, years)      -- NPV stacked bars + box/whisker
  sales_comparison(base, mandate, is_mc, k)           -- new sales by powertrain
  emissions_comparison(base, mandate, is_mc, k)       -- supply + use emissions
  mandate_tracking(mandate, is_mc, policy)             -- ZEV penetration vs target + penalty
  government_revenue(base, mandate, is_mc)             -- net government revenue

Run directly to show all comparison plots.
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '../..'))
sys.path.insert(0, os.path.join(_HERE, '..'))

import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model import Fleet, get_uncertainty_distributions, START_YEAR, END_YEAR, ZEV_POWERTRAINS
from data import PARAMS
from scenarios import SCENARIOS
from plot_utils import (PT_COLOR, PT_LABELS, K_LABELS, KEY_LABELS,
                        _colours, _bar_layout, _legend, add_2007_axis, year0_value)

_YEARS = np.arange(START_YEAR, END_YEAR + 1)
_ALL_K = list(PARAMS['vehicles']['types'].keys())
_ALL_P = {k: list(PARAMS['vehicles']['types'][k]['powertrains'].keys()) for k in _ALL_K}

_MANDATE_SCENARIO = 'zev_mandate_100k_rn'   # default scenario shown in plots


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(results_dir=None):
    """
    Load MC NPZ files; fall back to single deterministic runs if missing.

    Returns (base, mandate, is_mc).
      is_mc=True  -> base, mandate are np.lib.npyio.NpzFile with (n_runs, 26) arrays
      is_mc=False -> base, mandate are Fleet objects at median params
    """
    root = Path(_HERE).parent.parent
    rdir = Path(results_dir) if results_dir else root / 'results'
    base_path = rdir / 'baseline.npz'
    if base_path.exists():
        for name in (_MANDATE_SCENARIO, 'zev_mandate'):
            mandate_path = rdir / f'{name}.npz'
            if mandate_path.exists():
                if name != _MANDATE_SCENARIO:
                    warnings.warn(f'{_MANDATE_SCENARIO}.npz not found; using {name}.npz')
                return np.load(base_path), np.load(mandate_path), True
    warnings.warn(f'MC results not found at {rdir}; falling back to single deterministic run.')
    inputs    = dict(get_uncertainty_distributions(PARAMS))
    param_cps = {k: 0.5 for k in inputs}
    scen      = SCENARIOS.get(_MANDATE_SCENARIO, SCENARIOS['zev_mandate'])
    base      = Fleet(PARAMS, param_cps)
    mandate   = Fleet(PARAMS, param_cps, policies=scen)
    return base, mandate, False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _band(ax, t, arr, col, alpha=0.15, label=None, lw=1.5, ls='-'):
    """
    Median line + p5/p95 fill for a 2-D MC array (n_runs, T).
    Falls back to a plain line for 1-D single-run data.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        p5, med, p95 = np.percentile(arr, [5, 50, 95], axis=0)
        ax.fill_between(t, p5, p95, alpha=alpha, color=col)
        line, = ax.plot(t, med, color=col, linewidth=lw, linestyle=ls, label=label)
    else:
        line, = ax.plot(t, arr, color=col, linewidth=lw, linestyle=ls, label=label)
    return line


def _box_overlay(ax, x, vals, bar_width):
    """p10/p25/p50/p75/p90 box-and-whisker centred on x."""
    p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
    bw = bar_width * 0.38
    ax.plot([x, x], [p10, p90], color='black', linewidth=0.9, zorder=6)
    for y_tick in (p10, p90):
        ax.plot([x - bw/2, x + bw/2], [y_tick, y_tick], color='black', linewidth=0.9, zorder=6)
    ax.add_patch(plt.Rectangle(
        (x - bw/2, p25), bw, p75 - p25,
        linewidth=0.9, edgecolor='black', facecolor='white', zorder=7,
    ))
    ax.plot([x - bw/2, x + bw/2], [p50, p50], color='black', linewidth=1.5, zorder=8)


def _nruns(data):
    """Return number of MC runs from any (n_runs, T) array in an NpzFile."""
    for key in data.files:
        arr = data[key]
        if arr.ndim == 2:
            return arr.shape[0]
    return 1


def _median_fleets():
    """Build mandate Fleet object at median (CP=0.5) parameters."""
    inputs    = dict(get_uncertainty_distributions(PARAMS))
    param_cps = {k: 0.5 for k in inputs}
    return Fleet(PARAMS, param_cps,
                 policies=SCENARIOS.get(_MANDATE_SCENARIO, SCENARIOS['zev_mandate']))


# ---------------------------------------------------------------------------
# NPV comparison
# ---------------------------------------------------------------------------

def npv_comparison(mandate, is_mc, k='sleeper', years=(2030, 2040, 2050)):
    """
    NPV stacked bars for the mandate scenario, vehicle type k, at key cohort years.

    Cost breakdown comes from median-param Fleet objects (built on the fly if MC mode).
    If MC npv_* keys are present, overlays a p10/p25/p50/p75/p90 box-and-whisker on
    each bar's total NPV to show MC uncertainty.
    """
    fleet_mandate = _median_fleets() if is_mc else mandate

    npv_keys = ['revenue']
    seen = set()
    for p in fleet_mandate.P[k]:
        for y in years:
            if (k, p, y) not in fleet_mandate.vehicles:
                continue
            for key in fleet_mandate.vehicles[k, p, y].annual_cost:
                if key not in seen:
                    npv_keys.append(key)
                    seen.add(key)
    col = _colours(npv_keys)

    P = fleet_mandate.P[k]
    years_sorted = sorted(years)
    gap = min(y2 - y1 for y1, y2 in zip(years_sorted[:-1], years_sorted[1:])) if len(years_sorted) > 1 else 10
    width, offsets = _bar_layout(len(P), year_gap=gap)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=110, constrained_layout=True)
    ax.set_title(f'NPV  --  {K_LABELS.get(k, k)}  --  mandate scenario')
    ax.set_ylabel('$ thousands (NPV-discounted)')

    for i, p in enumerate(P):
        for y in years_sorted:
            if (k, p, y) not in fleet_mandate.vehicles:
                continue
            v = fleet_mandate.vehicles[k, p, y]
            x = y + offsets[i]
            vals = {'revenue': v._discount(v.annual_revenue) / 1000}
            vals.update({key: -v._discount(arr) / 1000
                         for key, arr in v.annual_cost.items()})
            pos_bottom = neg_bottom = 0.0
            for key in npv_keys:
                val = vals.get(key, 0.0)
                if val == 0:
                    continue
                bottom = pos_bottom if val > 0 else neg_bottom
                ax.bar(x, val, bottom=bottom, width=width, color=col[key])
                if val > 0:
                    pos_bottom += val
                else:
                    neg_bottom += val
            mc_key = f'npv_{k}_{p}_{y}'
            if is_mc and mc_key in mandate.files:
                _box_overlay(ax, x, mandate[mc_key] / 1000, width)
            ax.text(x, pos_bottom * 1.01,
                    PT_LABELS.get(p, p),
                    ha='center', va='bottom', fontsize=7, rotation=90)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(years_sorted)
    _legend(ax, npv_keys, col, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    return fig, ax


# ---------------------------------------------------------------------------
# Sales comparison
# ---------------------------------------------------------------------------

def sales_comparison(base, mandate, is_mc, k='sleeper'):
    """
    New vehicle sales by powertrain: base (solid) vs mandate (dashed).
    MC mode: median line + p5/p95 fill bands.
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110, constrained_layout=True)
    ax.set_title(f'Annual sales  --  {K_LABELS.get(k, k)}'
                 + ('  (shaded: p5-p95)' if is_mc else ''))
    ax.set_xlabel('Year')
    ax.set_ylabel('New vehicles (thousands)')

    handles, labels = [], []
    global_max = 0.0

    for p in _ALL_P[k]:
        col = PT_COLOR.get(p, 'gray')
        lbl = PT_LABELS.get(p, p)
        added = False
        for data, ls in [(base, '-'), (mandate, '--')]:
            if is_mc:
                key = f'sales_{k}_{p}'
                if key not in data.files:
                    continue
                arr = data[key] / 1e3   # (n_runs, T)
                med_val = float(np.percentile(arr, 50))
                global_max = max(global_max, float(np.percentile(arr, 95).max()))
                line = _band(ax, _YEARS, arr, col, alpha=0.10, lw=1.2, ls=ls)
            else:
                arr = np.array([float(data.sales.get((k, p, t), 0.0))
                                for t in _YEARS]) / 1e3
                global_max = max(global_max, float(arr.max()) if arr.size else 0.0)
                line, = ax.plot(_YEARS, arr, color=col, linestyle=ls, linewidth=1.2)
            if not added:
                handles.append(plt.Line2D([0], [0], color=col, linewidth=1.5))
                labels.append(lbl)
                added = True

    ax.set_ylim(0, global_max * 1.15 if global_max > 0 else 1)
    ax.legend(handles=handles, labels=labels,
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    return fig, ax


# ---------------------------------------------------------------------------
# Emissions comparison
# ---------------------------------------------------------------------------

def emissions_comparison(base, mandate, is_mc, k='sleeper'):
    """
    Fleet supply + use emissions over time for baseline vs mandate (vehicle type k).
    MC mode: median line + p5/p95 fill bands.
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110, constrained_layout=True)
    ax.set_title(f'Fleet emissions  --  {K_LABELS.get(k, k)}'
                 + ('  (shaded: p5-p95)' if is_mc else ''))
    ax.set_xlabel('Year')
    ax.set_ylabel('Mt CO2e / yr')

    lines_added = {}

    for data, scenario_label, ls in [(base, 'Baseline', '-'), (mandate, 'Mandate', '--')]:
        for etype in ('supply', 'use'):
            lbl  = f'{scenario_label} {etype}'
            col  = {'supply': '#e07b39', 'use': '#4472c4'}[etype]
            if is_mc:
                key = f'emissions_{k}_{etype}'
                if key not in data.files:
                    continue
                arr = data[key] / 1e9   # kg -> Mt
                _band(ax, _YEARS, arr, col, alpha=0.12, label=lbl, lw=1.5, ls=ls)
            else:
                arr = np.asarray(data.emissions[k][etype]) / 1e9
                ax.plot(_YEARS, arr, color=col, linestyle=ls, linewidth=1.5, label=lbl)

    ax.set_ylim(0, None)
    if is_mc:
        base_total = base[f'emissions_{k}_supply'] / 1e9 + base[f'emissions_{k}_use'] / 1e9
    else:
        base_total = (np.asarray(base.emissions[k]['supply'])
                       + np.asarray(base.emissions[k]['use'])) / 1e9
    add_2007_axis(ax, year0_value(base_total))
    ax.legend(fontsize=8)
    return fig, ax


# ---------------------------------------------------------------------------
# Mandate tracking (ZEV penetration + penalty fraction)
# ---------------------------------------------------------------------------

def mandate_tracking(mandate, is_mc, policy):
    """
    Single-panel plot: ZEV sales share, mandate target, and credit price fraction,
    all on the same 0-100% axis.

    policy: the ZEVMandate object (for the deterministic target line).
    MC mode: median line + p5/p95 fill bands.
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110, constrained_layout=True)
    ax.set_title('ZEV Mandate: penetration and credit price')
    ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Target trajectory
    target_arr = np.array([policy.target_at(t) for t in _YEARS])
    ax.plot(_YEARS, target_arr, color='black', linewidth=1.5, linestyle=':', label='Target')

    # ZEV sales share
    if is_mc:
        n = _nruns(mandate)
        zev = np.zeros((n, len(_YEARS)))
        tot = np.zeros_like(zev)
        for k_ in _ALL_K:
            for p_ in _ALL_P[k_]:
                key = f'sales_{k_}_{p_}'
                if key not in mandate.files:
                    continue
                arr = mandate[key]
                tot += arr
                if p_ in ZEV_POWERTRAINS:
                    zev += arr
        share = zev / np.maximum(tot, 1.0)
        _band(ax, _YEARS, share, col='#2e86ab', alpha=0.15, label='ZEV share', lw=1.5)
    else:
        zev_s = np.array([
            sum(mandate.sales.get((k_, p_, t), 0.0)
                for k_ in _ALL_K for p_ in mandate.P[k_] if p_ in ZEV_POWERTRAINS)
            for t in _YEARS
        ])
        tot_s = np.array([
            sum(mandate.sales.get((k_, p_, t), 0.0)
                for k_ in _ALL_K for p_ in mandate.P[k_])
            for t in _YEARS
        ])
        ax.plot(_YEARS, zev_s / np.maximum(tot_s, 1.0),
                color='#2e86ab', linewidth=1.5, label='ZEV share')

    # Credit price fraction
    if is_mc and 'mandate_penalty_frac' in mandate.files:
        arr = mandate['mandate_penalty_frac']
        _band(ax, _YEARS, arr, col='#c0392b', alpha=0.15, label='Credit price', lw=1.5)
    elif not is_mc and mandate.penalty_history:
        pen_arr = np.array([mandate.penalty_history.get(int(t), 0.0) for t in _YEARS])
        ax.plot(_YEARS, pen_arr, color='#c0392b', linewidth=1.5, label='Credit price')

    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # RHS axis: actual penalty in $k (shares the same 0-1.05 fraction scale)
    ax_r = ax.twinx()
    pmax_k = policy.penalty_max / 1000
    ax_r.set_ylim(0, 1.05 * pmax_k)
    ax_r.set_ylabel(f'Credit price ($k, max={pmax_k:.0f}k)')
    ax_r.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.0f}k'))

    return fig, ax


# ---------------------------------------------------------------------------
# Government revenue
# ---------------------------------------------------------------------------

def government_revenue(base, mandate, is_mc):
    """
    Net government revenue from the ZEV mandate per year.

    Revenue = penalties collected from non-ZEV purchasers
            - rebates paid to ZEV purchasers
    = sum over vehicle types k of system_costs_k_zev_mandate
    (positive = net revenue; approaches zero when rebates balance penalties)

    MC mode: median line + p5/p95 fill band.
    """
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110, constrained_layout=True)
    ax.set_title('Net government revenue from ZEV mandate'
                 + ('  (shaded: p5-p95)' if is_mc else ''))
    ax.set_xlabel('Year')
    ax.set_ylabel('$M / yr')

    for data, label, col, ls in [
        (base,    'Baseline',     '#888888', '--'),
        (mandate, 'ZEV mandate',  '#c0392b', '-'),
    ]:
        if is_mc:
            rev = np.zeros((_nruns(data), len(_YEARS)))
            for k_ in _ALL_K:
                key = f'system_costs_{k_}_zev_mandate'
                if key in data.files:
                    rev += data[key]
            rev_m = rev / 1e6   # $ -> $M
            _band(ax, _YEARS, rev_m, col=col, alpha=0.15, label=label, lw=1.5, ls=ls)
        else:
            rev = sum(
                data.system_costs[k_].get('zev_mandate', np.zeros(len(_YEARS)))
                for k_ in data.K
            )
            ax.plot(_YEARS, rev / 1e6, color=col, linestyle=ls, linewidth=1.5, label=label)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend(fontsize=8)
    return fig, ax


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.chdir(os.path.join(_HERE, '../..'))
    warnings.filterwarnings('always')

    base, mandate, is_mc = load_data()
    policy = SCENARIOS.get(_MANDATE_SCENARIO, SCENARIOS['zev_mandate']).zev_mandate

    npv_comparison(mandate, is_mc, k='sleeper', years=(2030, 2040, 2050))
    sales_comparison(base, mandate, is_mc, k='sleeper')
    emissions_comparison(base, mandate, is_mc, k='sleeper')
    mandate_tracking(mandate, is_mc, policy)
    government_revenue(base, mandate, is_mc)
    plt.show()
