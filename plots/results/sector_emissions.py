"""
Sector emissions box plots for 2030, 2040, and 2050 across all policy scenarios.

Each panel shows the distribution of total sector emissions (use + supply, summed
across all vehicle types) for each scenario from the Monte Carlo output.

Whiskers: 5th-95th percentile.  Box: IQR.  Red dot: mean.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '../..'))   # repo root

import numpy as np
import matplotlib.pyplot as plt

from data import START_YEAR

RESULTS_DIR = os.path.join(_HERE, '..', '..', 'results')

SCENARIOS = [
    ('baseline',    'Baseline'),
    ('carbon_tax',  'Carbon Tax'),
    ('lcfs',        'LCFS'),
    ('zev_mandate', 'ZEV Mandate'),
    ('gvwl',        'GVWL'),
    ('full_policy', 'Full Policy'),
]

SCENARIO_COLORS = {
    'baseline':    '#aec6cf',
    'carbon_tax':  '#f4a261',
    'lcfs':        '#2a9d8f',
    'zev_mandate': '#e76f51',
    'gvwl':        '#8ecae6',
    'full_policy': '#6a0572',
}

VEHICLE_TYPES = ['sleeper', 'day_cab', 'straight']
PLOT_YEARS    = [2030, 2040, 2050]


def _box_plot(data, ax, x, edgecolor='black', facecolor='#cce6ff', width=0.6):
    """Draw a 5-95 percentile box plot at position x with a red mean dot."""
    p05, p25, p50, p75, p95 = np.percentile(data, [5, 25, 50, 75, 95])
    bxp_stats = [{
        'med':    p50,
        'q1':     p25,
        'q3':     p75,
        'whislo': p05,
        'whishi': p95,
        'fliers': [],
    }]
    ax.bxp(
        bxp_stats,
        positions=[x],
        widths=width,
        manage_ticks=False,
        patch_artist=True,
        boxprops=dict(facecolor=facecolor, edgecolor=edgecolor, alpha=1),
        whiskerprops=dict(color=edgecolor),
        capprops=dict(color=edgecolor),
        medianprops=dict(color=edgecolor),
    )
    ax.scatter([x], [np.mean(data)], color='red', zorder=3, s=6)


def _load_sector_emissions(scenario):
    """
    Return total sector emissions array, shape (n_runs, 26), in MtCO2e/yr.
    Sums use + supply across all vehicle types.
    """
    path = os.path.join(RESULTS_DIR, f'{scenario}.npz')
    d    = np.load(path)
    total = np.zeros(d[f'emissions_sleeper_use'].shape, dtype=np.float64)
    for k in VEHICLE_TYPES:
        for cat in ('use', 'supply'):
            total += d[f'emissions_{k}_{cat}']
    return total / 1e9   # kg -> MtCO2e


def main():
    year_indices = [y - START_YEAR for y in PLOT_YEARS]

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True,
                             constrained_layout=True, dpi=150)

    for ax, year, yi in zip(axes, PLOT_YEARS, year_indices):
        positions, labels = [], []

        for x, (scenario, label) in enumerate(SCENARIOS, start=1):
            emissions = _load_sector_emissions(scenario)   # (n_runs, 26) MtCO2e
            data      = emissions[:, yi]
            _box_plot(data, ax, x,
                      facecolor=SCENARIO_COLORS.get(scenario, '#cce6ff'))
            positions.append(x)
            labels.append(label)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=40, ha='right')
        ax.set_title(str(year))
        ax.set_ylim(0, None)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r'Sector Emissions (MtCO$_2$e yr$^{-1}$)')
    fig.suptitle('BC HDT Sector Emissions by Policy Scenario', fontsize=13)

    plt.show()


if __name__ == '__main__':
    main()
