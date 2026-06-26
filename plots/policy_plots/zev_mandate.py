"""
ZEV mandate policy plots for the fleet2 HDT adoption model.

Comparison plots showing base scenario vs ZEV mandate (30% by 2030, 100% by 2050,
fleet-wide scope, $200k/vehicle max penalty):

  npv_comparison(fleet_base, fleet_mandate)   -- NPV stacked bars, all powertrains
  sales_comparison(fleet_base, fleet_mandate) -- new sales by powertrain over time

Run directly to show all comparison plots.
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '../..'))   # root
sys.path.insert(0, os.path.join(_HERE, '..'))      # plots/

import warnings
import numpy as np
import matplotlib.pyplot as plt
from model import Fleet, get_uncertainty_distributions, START_YEAR, ZEV_POWERTRAINS
from data import PARAMS
from policies import Policies, ZEVMandate
from plot_utils import (SAMPLE_YEARS, PT_COLOR, PT_LABELS, K_LABELS,
                        _colours, _bar_layout, _legend)


# ---------------------------------------------------------------------------
# NPV comparison -- all powertrains, base (hatched) vs mandate (solid)
# ---------------------------------------------------------------------------

def npv_comparison(fleet_base, fleet_mandate):
    """NPV stacked bars for all powertrains, one figure per vehicle type."""
    for k in fleet_base.K:
        npv_keys = ['revenue']
        seen = set()
        for fleet in (fleet_base, fleet_mandate):
            for p in fleet.P[k]:
                for y in SAMPLE_YEARS:
                    if (k, p, y) not in fleet.vehicles:
                        continue
                    for key in fleet.vehicles[k, p, y].annual_cost:
                        if key not in seen:
                            npv_keys.append(key)
                            seen.add(key)
        col = _colours(npv_keys)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_title(f'NPV -- {K_LABELS.get(k, k)}  (hatched = base, solid = mandate)')
        ax.set_ylabel('$ thousands (NPV-discounted)')

        P = fleet_base.P[k]
        width, offsets = _bar_layout(len(P) * 2)

        for i, p in enumerate(P):
            for y in SAMPLE_YEARS:
                if (k, p, y) not in fleet_base.vehicles:
                    continue
                for j, (fleet, hatch) in enumerate([(fleet_base, '//'), (fleet_mandate, '')]):
                    if (k, p, y) not in fleet.vehicles:
                        continue
                    v   = fleet.vehicles[k, p, y]
                    x   = y + offsets[i * 2 + j]
                    vals = {'revenue': v._discount(v.annual_revenue) / 1000}
                    vals.update({key: -v._discount(arr) / 1000
                                 for key, arr in v.annual_cost.items()})
                    pos_bottom = neg_bottom = 0.0
                    for key in npv_keys:
                        val = vals.get(key, 0.0)
                        if val == 0:
                            continue
                        bottom = pos_bottom if val > 0 else neg_bottom
                        ax.bar(x, val, bottom=bottom, width=width, color=col[key], hatch=hatch)
                        if val > 0:
                            pos_bottom += val
                        else:
                            neg_bottom += val
                    lbl = f'{PT_LABELS.get(p, p)}\n{"Base" if j == 0 else "Mandate"}'
                    ax.text(x, pos_bottom * 1.01, lbl, ha='center', va='bottom',
                            fontsize=7, rotation=90)
                    ax.scatter(x, v.npv / 1000, marker='x', color='black',
                               zorder=5, s=60, linewidths=1.5)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(SAMPLE_YEARS)
        _legend(ax, npv_keys, col, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        plt.tight_layout()


# ---------------------------------------------------------------------------
# Sales comparison -- all powertrains, base vs mandate
# ---------------------------------------------------------------------------

def sales_comparison(fleet_base, fleet_mandate):
    """New vehicle sales by powertrain: base (solid) vs mandate (dashed). One subplot per k."""
    T    = fleet_base.years
    K    = fleet_base.K
    fig, axes = plt.subplots(1, len(K), figsize=(4 * len(K), 3.5), sharex=True, sharey=True,
                              constrained_layout=True, dpi=150)
    axes = np.atleast_1d(axes)
    fig.suptitle('Annual sales by powertrain  (solid = base, dashed = mandate)')

    all_handles, all_labels = [], []
    global_max = 0
    for ax, k in zip(axes, K):
        ax.set_title(K_LABELS.get(k, k))
        ax.set_xlabel('Year')
        ax.set_ylabel('New vehicles (thousands)')
        for p in fleet_base.P[k]:
            col = PT_COLOR.get(p, 'gray')
            lbl = PT_LABELS.get(p, p)
            for fleet, ls in [(fleet_base, '-'), (fleet_mandate, '--')]:
                arr  = np.array([float(fleet.sales.get((k, p, t), 0.0)) for t in T]) / 1e3
                line, = ax.plot(T, arr, color=col, linestyle=ls)
                global_max = max(global_max, float(arr.max()) if arr.size else 0)
                if ls == '-' and lbl not in all_labels:
                    all_handles.append(line); all_labels.append(lbl)
        ax.set_ylim(0, None)

    for ax in axes:
        ax.set_ylim(0, global_max * 1.1)
    axes[-1].legend(handles=all_handles, labels=all_labels,
                    bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    return fig, axes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    warnings.filterwarnings('always')

    inputs    = dict(get_uncertainty_distributions(PARAMS))
    param_cps = {k: 0.5 for k in inputs}

    fleet_base    = Fleet(PARAMS, param_cps)
    fleet_mandate = Fleet(PARAMS, param_cps,
                          policies=Policies(zev_mandate=ZEVMandate(
                              targets={'2030': 0.30, '2050': 1.00},
                              penalty=200_000,
                              scope='fleet',
                          )))

    npv_comparison(fleet_base, fleet_mandate)
    sales_comparison(fleet_base, fleet_mandate)
    plt.show()
