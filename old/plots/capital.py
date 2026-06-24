""" Plot capital costs. """
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

from parallel_test import Plotting

def box_plot(data, ax, x, width):
    p05, p25, p50, p75, p95 = np.percentile(data, q=[5, 25, 50, 75, 95])
    bxp_data = [{
        'med': p50,
        'q1': p25,
        'q3': p75,
        'whislo': p05,
        'whishi': p95,
        'fliers': []
    }]
    ax.bxp(bxp_data, positions=[x], widths=width,
           manage_ticks=False, medianprops=dict(color='black'))

fname = '../Results_19_3_2026/Foresight.pkl'

with open(fname, 'rb') as f:
    outputs = pickle.load(f)
outputs = outputs['Capital']

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True, dpi=300)
width = 0.08
gap = 0.02

years = [2025, 2030, 2050]
for iK, (k, v) in enumerate(outputs.items()):
    for iP, (p, vv) in enumerate(v.items()):
        for iY, year in enumerate(years):
            ax[iK].set_prop_cycle(None)
            x = iY + (iP - len(v.keys())/2) * (width + gap)
            bottom=0
            for key, value in vv.items():
                y = value[:, year-2025]
                ax[iK].bar(x, np.mean(y), bottom=bottom, width=width)
                bottom += np.mean(y)
            total = sum(value[:, year-2025] for key, value in vv.items())
            box_plot(total, ax[iK], x, width*0.6)
            p95 = np.percentile(total, q=95)
            ax[iK].text(x, p95 + 10, p, ha='center', va='bottom', fontsize=8, rotation=90)
        ax[iK].set_xticks(range(len(years)))
        ax[iK].set_xticklabels(years, rotation=45, ha='right')

ymin, ymax = ax[0].get_ylim()
ax[0].set_ylim(ymin, ymax * 1.05)
ax[0].set_ylabel('Capital cost ($CAD thousand, 2024)')
ax[0].set_title('Sleeper')
ax[1].set_title('Day Cab')
ax[2].set_title('Straight')

colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
handles = [
    plt.Rectangle((0, 0), 1, 1, color=colours[i % len(colours)])
    for i, _ in enumerate(vv.keys())
]
ax[-1].legend(handles, vv.keys(), loc='upper right', fontsize=8)
