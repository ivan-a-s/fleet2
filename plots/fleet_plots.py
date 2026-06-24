"""
Fleet-level plots for the fleet2 HDT adoption model.

extract_outputs(fleet) — pulls fleet results into a plain dict of arrays.
    For a single run each leaf is a 1-D array over years.
    After Monte Carlo, stack n_runs results with merge_outputs() and each
    leaf becomes a 2-D array (n_runs × n_years); Plotting handles both shapes.

Plotting class — mirrors the style of Paper 1's parallel_test.py:
    plot_by_both(result, ...)  one subplot per vehicle type k, one line per category
    plot_by_inner(result, ...) single plot, summed across k, one line per category
    plot_lines(...)            mean line + p5–p95 fill (fill omitted for single runs)

merge_outputs(list_of_dicts) — stacks a list of single-run dicts into MC arrays.

Run directly to show all plots for a single deterministic run.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
from model import Fleet, get_uncertainty_distributions, START_YEAR, END_YEAR
from data import PARAMS

_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11})

PT_COLOR = {p: _CYCLE[i % len(_CYCLE)]
            for i, p in enumerate(['dice', 'he', 'be', 'fc', 'hice', 'dhice'])}

DISPLAY_LABELS = {
    'dice':     'DICE',
    'he':       'HE',
    'phe':      'PHE',
    'be':       'BE',
    'fc':       'FC',
    'hice':     'HICE',
    'dhice':    'DHICE',
    'sleeper':  'Sleeper',
    'day_cab':  'Day Cab',
    'straight': 'Straight',
}

# MJ of useful (traction) energy per unit of fuel consumed.
# Diesel/H2: LHV × powertrain efficiency. Electric: 3.6 MJ/kWh × drivetrain efficiency.
# h2 efficiency uses FC (~60%); hice/dhice are a minor share so the error is small.
FUEL_TO_MJ = {
    'diesel':      35.8 * 0.43,   # L   → 15.4 MJ
    'h2':         120.0 * 0.60,   # kg  → 72.0 MJ
    'h2_p':       120.0 * 0.60,
    'h2_pe':      120.0 * 0.60,
    'slow_charge':  3.6 * 0.90,   # kWh →  3.2 MJ
    'fast_charge':  3.6 * 0.90,
}

FUEL_COLOR = {
    'diesel':      PT_COLOR['dice'],
    'slow_charge': PT_COLOR['be'],
    'fast_charge': _CYCLE[6 % len(_CYCLE)],
    'h2':          PT_COLOR['fc'],
    'h2_p':        PT_COLOR['hice'],
    'h2_pe':       PT_COLOR['dhice'],
}
EMIS_COLOR = {e: _CYCLE[i % len(_CYCLE)]
              for i, e in enumerate(['Use', 'Supply', 'Embodied'])}
COST_COLOR = {c: _CYCLE[i % len(_CYCLE)]
              for i, c in enumerate(['Capital', 'Operational', 'Fuel', 'Driver'])}


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
                   * FUEL_TO_MJ.get(f, 1.0) / 1e6   # → TJ useful energy
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
# Plotting class
# ---------------------------------------------------------------------------

class Plotting:
    def __init__(self, sample_years=None):
        self.T = np.arange(START_YEAR, END_YEAR + 1)
        self.sample_years = sample_years or np.array([2025, 2030, 2035, 2040, 2045, 2050])

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _sum_by_outer(result):
        """Sum across outer keys (vehicle types k), keep inner keys (categories)."""
        summed = {}
        for subdict in result.values():
            for cat, arr in subdict.items():
                arr = np.asarray(arr, dtype=float)
                summed[cat] = summed[cat] + arr if cat in summed else arr.copy()
        return summed

    @staticmethod
    def plot_lines(t, result, ax, x_label=None, y_label=None,
                   add_total=False, color_map=None):
        """
        Plot each category in result as a line.
        If v is 1-D (single run): plain line.
        If v is 2-D (MC, axis-0 = runs): mean line + p5–p95 fill band.
        Returns the local y-axis maximum for shared-axis sizing.
        """
        local_max = 0
        for k, v in result.items():
            v   = np.asarray(v, dtype=float)
            col = (color_map or {}).get(k)
            if v.ndim == 1:
                ax.plot(t, v, label=DISPLAY_LABELS.get(k, k), color=col)
                local_max = max(local_max, v.max() if v.size else 0)
            else:
                p5, p95 = np.percentile(v, [5, 95], axis=0)
                mean    = np.mean(v, axis=0)
                ax.fill_between(t, p5, p95, alpha=0.2, color=col)
                ax.plot(t, mean, label=DISPLAY_LABELS.get(k, k), color=col)
                local_max = max(local_max, float(p95.max()))

        if add_total:
            arrays = [np.asarray(v, dtype=float) for v in result.values()]
            total  = sum(arrays)
            if total.ndim == 1:
                ax.plot(t, total, label='Total', linewidth=2, color='black')
                local_max = max(local_max, total.max() if total.size else 0)
            else:
                p5, p95 = np.percentile(total, [5, 95], axis=0)
                ax.fill_between(t, p5, p95, alpha=0.1, color='black')
                ax.plot(t, np.mean(total, axis=0), label='Total', linewidth=2, color='black')
                local_max = max(local_max, float(p95.max()))

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(0, None)
        return local_max

    # -- layout methods ------------------------------------------------------

    def plot_by_both(self, result, title=None, x_label=None, y_label=None,
                     add_total=False, color_map=None):
        """One subplot per vehicle type k; one line per category (powertrain/component)."""
        n    = len(result)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharex=True, sharey=True,
                                 constrained_layout=True, dpi=150)
        axes = np.atleast_1d(axes)

        all_handles, all_labels = [], []
        global_max = 0
        for ax, (k, subdict) in zip(axes, result.items()):
            local_max = self.plot_lines(self.T, subdict, ax, x_label, y_label,
                                        add_total, color_map)
            global_max = max(global_max, local_max)
            ax.set_title(DISPLAY_LABELS.get(k, k))
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in all_labels:
                    all_handles.append(h); all_labels.append(l)

        for ax in axes:
            ax.set_ylim(0, global_max * 1.1)
        axes[-1].legend(handles=all_handles, labels=all_labels,
                        bbox_to_anchor=(1.05, 1), loc='upper left',
                        borderaxespad=0., fontsize=8)
        if title:
            fig.suptitle(title)
        return fig, axes

    def plot_by_inner(self, result, title=None, x_label=None, y_label=None,
                      add_total=False, color_map=None):
        """Single plot; sum across vehicle types k, one line per category."""
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
        summed  = self._sum_by_outer(result)
        self.plot_lines(self.T, summed, ax, x_label, y_label, add_total, color_map)
        ax.legend(fontsize=8)
        if title:
            fig.suptitle(title)
        return fig, ax


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    inputs_distributions = dict(get_uncertainty_distributions(PARAMS))
    param_cps = dict(zip(inputs_distributions.keys(),
                         np.random.rand(len(inputs_distributions)).astype('float32')))
    fleet   = Fleet(PARAMS, param_cps)
    outputs = extract_outputs(fleet)

    p = Plotting()

    p.plot_by_inner(outputs['Emissions'],
                    title='Fleet LCA emissions',
                    x_label='Year', y_label='MtCO₂e / year',
                    add_total=True, color_map=EMIS_COLOR)

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
