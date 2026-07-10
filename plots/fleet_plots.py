"""
Fleet-level plots for the fleet2 HDT adoption model.

extract_outputs(fleet) -- pulls fleet results into a plain dict of arrays.
    For a single run each leaf is a 1-D array over years.
    After Monte Carlo, stack n_runs results with merge_outputs() and each
    leaf becomes a 2-D array (n_runs x n_years); Plotting handles both shapes.

Plotting class -- mirrors the style of Paper 1's parallel_test.py:
    plot_by_both(result, ...)  one subplot per vehicle type k, one line per category
    plot_by_inner(result, ...) single plot, summed across k, one line per category
    plot_lines(...)            mean line + p5-p95 fill (fill omitted for single runs)

merge_outputs(list_of_dicts) -- stacks a list of single-run dicts into MC arrays.

Run directly to show all plots for a single deterministic run.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from model import Fleet, get_uncertainty_distributions
from data import PARAMS
from plot_utils import (Plotting, PT_COLOR, EMIS_COLOR, COST_COLOR, FUEL_COLOR,
                        FUEL_TO_MJ)


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------

def extract_outputs(fleet):
    """
    Pull fleet results into a nested dict of 1-D numpy arrays (one value per year).
    Structure mirrors Paper 1 parallel_test.py so merge_outputs() can stack MC runs.
    """
    T = fleet.years
    all_fuels = sorted({key[1] for key in fleet.fuel_usage})
    return {
        'Emissions': {
            k: {
                'Use':      fleet.emissions[k]['use']      / 1e9,
                'Supply':   fleet.emissions[k]['supply']   / 1e9,
                'Embodied': fleet.emissions[k]['embodied'] / 1e9,
            } for k in fleet.K
        },
        'Cost': {
            k: {
                'Capital':     fleet.system_costs[k]['capital']     / 1e9,
                'Operational': fleet.system_costs[k]['operational'] / 1e9,
                'Fuel':        fleet.system_costs[k]['fuel']        / 1e9,
                'Driver':      fleet.system_costs[k]['driver']      / 1e9,
                'Carbon Tax':  fleet.system_costs[k]['carbon_tax']  / 1e9,
            } for k in fleet.K
        },
        'Stock': {
            k: {
                p: np.array([float(fleet.total_stock.get((k, p, t), 0.0)) for t in T]) / 1e3
                for p in fleet.P[k]
            } for k in fleet.K
        },
        'Sales': {
            k: {
                p: np.array([float(fleet.sales.get((k, p, t), 0.0)) for t in T]) / 1e3
                for p in fleet.P[k]
            } for k in fleet.K
        },
        'Fuel Usage': {
            k: {
                f: np.array([float(fleet.fuel_usage.get((k, f, t), 0.0)) for t in T])
                   * FUEL_TO_MJ.get(f, 1.0) / 1e6   # -> TJ useful energy
                for f in all_fuels
            } for k in fleet.K
        },
    }


def merge_outputs(items):
    """Stack a list of single-run extract_outputs() dicts into 2-D MC arrays."""
    if isinstance(items[0], dict):
        return {k: merge_outputs([item[k] for item in items]) for k in items[0]}
    return np.stack(items)          # (n_runs, n_years)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    param_cps = {path: np.float32(0.5) for path, _ in get_uncertainty_distributions(PARAMS)}
    fleet   = Fleet(PARAMS, param_cps)
    outputs = extract_outputs(fleet)

    p = Plotting()

    p.plot_by_inner(outputs['Emissions'],
                    title='Fleet LCA emissions',
                    x_label='Year', y_label='MtCO2e / year',
                    add_total=True, color_map=EMIS_COLOR, emissions_2007=True)

    p.plot_by_inner(outputs['Cost'],
                    title='System costs',
                    x_label='Year', y_label='$ billions / year',
                    add_total=True, color_map=COST_COLOR)

    p.plot_by_both(outputs['Stock'],
                   title='Total stock by powertrain',
                   x_label='Year', y_label='Vehicles (thousands)',
                   add_total=True, color_map=PT_COLOR)

    p.plot_by_both(outputs['Sales'],
                   title='Annual sales by powertrain',
                   x_label='Year', y_label='New vehicles (thousands)',
                   color_map=PT_COLOR)

    p.plot_by_both(outputs['Fuel Usage'],
                   title='Fleet useful energy demand',
                   x_label='Year', y_label='Useful energy (TJ / year)',
                   color_map=FUEL_COLOR)

    plt.show()
