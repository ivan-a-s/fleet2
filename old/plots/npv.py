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

fname = '../Results_19_3_2026/Policy Package.pkl'
# fname = '../Results_19_3_2026/Base.pkl'

with open(fname, 'rb') as f:
    outputs = pickle.load(f)
outputs = outputs['TCO']

fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True, dpi=300)
width = 0.08
gap = 0.02

years = [2030, 2040, 2050]
legend_items = []
colours = []
for iK, (k, v) in enumerate(outputs.items()):
    for iP, (p, vv) in enumerate(v.items()):
        for iY, year in enumerate(years):
            ax[iK].set_prop_cycle(None)
            x = iY + (iP - len(v.keys())/2) * (width + gap)
            pos_bottom = 0
            neg_bottom = 0
            for iColour, (key, value) in enumerate(vv.items()):
                colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][iColour]
                y = value[:, year-2025]
                if np.mean(y) != 0:
                    if key not in legend_items:
                        legend_items.append(key)
                        colours.append(colour)
                    if np.mean(y) > 0:
                        ax[iK].bar(x, np.mean(y), bottom=pos_bottom, width=width, color=colour)
                        pos_bottom += np.mean(y)
                    else:
                        ax[iK].bar(x, np.mean(y), bottom=neg_bottom, width=width, color=colour)
                        neg_bottom += np.mean(y)
            total = sum(value[:, year-2025] for key, value in vv.items())
            box_plot(total, ax[iK], x, width)
            ax[iK].text(x, pos_bottom + 0.05, p, ha='center', va='bottom', fontsize=7, rotation=90)
        ax[iK].set_xticks(range(len(years)))
        ax[iK].set_xticklabels(years, rotation=45, ha='right')

ymin, ymax = ax[0].get_ylim()
ax[0].set_ylim(ymin, ymax * 1.2)
ax[0].set_ylabel('NPV components ($CAD million, 2024)')
ax[0].set_title('Sleeper')
ax[1].set_title('Day Cab')
ax[2].set_title('Straight')

handles = [
    plt.Rectangle((0, 0), 1, 1, color=colours[i])
    for i, _ in enumerate(legend_items)
]
# ax[-1].legend(handles, legend_items, loc='lower right')
ax[-1].legend(handles, legend_items,
              loc='upper left',
              bbox_to_anchor=(1.05, 1))




fname = '../Results_19_3_2026/Foresight (P).pkl'

with open(fname, 'rb') as f:
    outputs = pickle.load(f)
outputs = outputs['TCO']

fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True, dpi=300)
width = 0.08
gap = 0.02

years = [2025, 2030, 2050]
for iK, (k, v) in enumerate(outputs.items()):
    for iP, (p, vv) in enumerate(v.items()):
        for iY, year in enumerate(years):
            ax[iK].set_prop_cycle(None)
            x = iY + (iP - len(v.keys())/2) * (width + gap)
            pos_bottom = 0
            neg_bottom = 0
            for key, value in vv.items():
                y = value[:, year-2025]
                if np.mean(y) > 0:
                    ax[iK].bar(x, np.mean(y), bottom=pos_bottom, width=width)
                    pos_bottom += np.mean(y)
                else:
                    ax[iK].bar(x, np.mean(y), bottom=neg_bottom, width=width)
                    neg_bottom += np.mean(y)
            total = sum(value[:, year-2025] for key, value in vv.items())
            box_plot(total, ax[iK], x, width*0.6)
            ax[iK].text(x, pos_bottom + 0.05, p, ha='center', va='bottom', fontsize=6, rotation=90)
        ax[iK].set_xticks(range(len(years)))
        ax[iK].set_xticklabels(years, rotation=45, ha='right')

ymin, ymax = ax[0].get_ylim()
ax[0].set_ylim(ymin*1.1, ymax * 1.2)
ax[0].set_ylabel('NPV components ($CAD million, 2024)')
ax[0].set_title('Sleeper')
ax[1].set_title('Day Cab')
ax[2].set_title('Straight')

colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
handles = [
    plt.Rectangle((0, 0), 1, 1, color=colours[i % len(colours)])
    for i, _ in enumerate(vv.keys())
]
ax[-1].legend(handles, legend_items, loc='upper right', fontsize=8)

