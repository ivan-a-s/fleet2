"""
Global sensitivity diagnostic (SRRC) for fleet2 Monte Carlo output.

Reads already-saved run.py MC results (results/<scenario>.npz + _meta.json) --
runs no new simulations.  Rank-transforms the per-run cp input samples and a
chosen scalar output metric, standardizes both, and fits an OLS regression;
each input's squared coefficient (SRRC^2) approximates its share of the
output's variance.  The regression's own rank-R^2 says how much of that
variance the (monotonic, additive) SRRC model actually explains -- a low R^2
signals nonlinearity/interaction (e.g. the ZEV-mandate bisection loop,
production-cap kinks) that SRRC cannot see.

This is an approximate, cheap diagnostic for day-to-day development.  A full
Sobol'/Saltelli variance decomposition needs its own sampling design (A/B/AB
matrices, not the plain random cp draws run.py generates) and a much larger
run budget -- planned separately for pre-publication use, not implemented here.

Run from the fleet2 root:
    python verification/global_sensitivity.py
    python verification/global_sensitivity.py --scenario zev_mandate --metric zev_share --year 2040
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '../plots'))

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from data import START_YEAR
from plot_utils import _CYCLE

METRICS = ('emissions_total', 'zev_share', 'system_cost_total')

POS_COLOR = _CYCLE[0]   # SRRC > 0 -- input increase raises the metric
NEG_COLOR = _CYCLE[3]   # SRRC < 0 -- input increase lowers the metric


# ---------------------------------------------------------------------------
# Load saved MC output
# ---------------------------------------------------------------------------

def _load(scenario, results_dir):
    npz_path  = Path(results_dir) / f'{scenario}.npz'
    meta_path = Path(results_dir) / f'{scenario}_meta.json'
    if not npz_path.exists() or not meta_path.exists():
        sys.exit(f"No saved results for scenario {scenario!r} in {results_dir}/ -- "
                 f"run `python run.py --scenarios {scenario}` first.")
    data = np.load(npz_path)
    meta = json.loads(meta_path.read_text())
    if '_mc_cp_samples' not in data.files or 'col_labels' not in meta:
        sys.exit(f"results/{scenario}.npz predates input-sample saving -- "
                 f"re-run `python run.py --scenarios {scenario}` to regenerate it.")
    return data, meta


# ---------------------------------------------------------------------------
# Metric extraction (by key prefix -- stays correct if k/p/category sets change)
# ---------------------------------------------------------------------------

def _keys_with(data, prefix, suffixes=None):
    return [key for key in data.files
            if key.startswith(prefix) and (suffixes is None or key.endswith(suffixes))]

def _sum_keys(data, keys, year_idx):
    if not keys:
        raise ValueError(f"No matching keys found for year_idx={year_idx}")
    total = np.zeros(data[keys[0]].shape[0], dtype=np.float64)
    for key in keys:
        total += data[key][:, year_idx].astype(np.float64)
    return total

def _compute_metric(data, metric, year):
    year_idx = year - START_YEAR
    if metric == 'emissions_total':
        keys = _keys_with(data, 'emissions_', suffixes=('_supply', '_use'))
        return _sum_keys(data, keys, year_idx)
    elif metric == 'zev_share':
        zev   = _sum_keys(data, _keys_with(data, 'zev_stock_'), year_idx)
        stock = _sum_keys(data, _keys_with(data, 'total_stock_'), year_idx)
        return np.divide(zev, stock, out=np.zeros_like(zev), where=stock > 0)
    elif metric == 'system_cost_total':
        keys = _keys_with(data, 'system_costs_')
        return _sum_keys(data, keys, year_idx)
    else:
        raise ValueError(f"Unknown metric {metric!r}; choices are {METRICS}")


# ---------------------------------------------------------------------------
# SRRC
# ---------------------------------------------------------------------------

def _srrc(X, y):
    """Standardized Rank Regression Coefficients + rank-R^2.  X: (n, p), y: (n,)."""
    Xr = np.apply_along_axis(rankdata, 0, X)
    yr = rankdata(y)
    Xs = (Xr - Xr.mean(axis=0)) / Xr.std(axis=0)
    ys = (yr - yr.mean()) / yr.std()
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    ss_res = np.sum((ys - Xs @ beta) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return beta, r2


# ---------------------------------------------------------------------------
# Report + plot
# ---------------------------------------------------------------------------

def _print_table(labels, beta, r2, excluded_labels, scenario, metric, year, n_runs):
    order = np.argsort(-np.abs(beta))
    print(f"\nSRRC sensitivity -- scenario={scenario}  metric={metric}  year={year}  "
          f"n_runs={n_runs}\n")
    print(f"{'Parameter':<55} {'SRRC':>8} {'SRRC^2 (%)':>11}")
    print("-" * 76)
    for i in order:
        print(f"{labels[i]:<55} {beta[i]:>8.3f} {100 * beta[i]**2:>11.2f}")
    print("-" * 76)
    note = ("decomposition looks reliable" if r2 > 0.7 else
            "treat with caution -- likely nonlinearity/interaction" if r2 < 0.5 else
            "moderate confidence")
    print(f"rank-R^2 = {r2:.3f}  ({note})")
    if excluded_labels:
        print(f"\nExcluded (zero-variance leaves, unaffected by cp): {len(excluded_labels)}")
        for label in excluded_labels:
            print(f"  {label}")


def _plot(labels, beta, r2, scenario, metric, year, n_runs):
    order = np.argsort(-beta**2)
    labels_sorted = [labels[i] for i in order]
    sq_sorted     = beta[order] ** 2
    colors        = [POS_COLOR if beta[i] >= 0 else NEG_COLOR for i in order]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(labels)), 5), dpi=150)
    ax.bar(range(len(labels_sorted)), sq_sorted, color=colors)
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Fraction of output variance (SRRC$^2$)')
    ax.set_title(f'SRRC sensitivity -- {scenario} / {metric} / {year}\n'
                f'n_runs={n_runs}, rank-R$^2$={r2:.2f}')

    handles = [plt.Rectangle((0, 0), 1, 1, color=POS_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=NEG_COLOR)]
    ax.legend(handles, ['Increases metric', 'Decreases metric'], fontsize=9)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    data, meta = _load(args.scenario, args.results_dir)

    X          = data['_mc_cp_samples']
    col_labels = meta['col_labels']
    n_cols     = meta['n_cols']
    zero_var   = set(meta['zero_variance_cols'])
    keep       = [i for i in range(n_cols) if i not in zero_var]

    y = _compute_metric(data, args.metric, args.year)

    beta, r2 = _srrc(X[:, keep], y)
    labels   = [col_labels[i] for i in keep]
    excluded = [col_labels[i] for i in sorted(zero_var)]

    _print_table(labels, beta, r2, excluded, args.scenario, args.metric, args.year, meta['n_runs'])
    _plot(labels, beta, r2, args.scenario, args.metric, args.year, meta['n_runs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SRRC global sensitivity diagnostic from saved run.py MC output',
    )
    parser.add_argument('--scenario', default='baseline',
                        help='Scenario name (default: baseline)')
    parser.add_argument('--metric', default='emissions_total', choices=METRICS,
                        help='Output metric to explain (default: emissions_total)')
    parser.add_argument('--year', type=int, default=2050,
                        help='Calendar year to evaluate the metric at (default: 2050)')
    parser.add_argument('--results-dir', default='results',
                        help='Directory containing <scenario>.npz/_meta.json (default: results)')
    args, _ = parser.parse_known_args()
    main(args)
