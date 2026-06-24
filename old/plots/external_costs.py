""" Plot the social costs. """
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy


def box_plot(data, ax, x):
    p05, p25, p50, p75, p95 = np.percentile(data, q=[5, 25, 50, 75, 95])
    bxp_data = [{
        'med': p50,
        'q1': p25,
        'q3': p75,
        'whislo': p05,
        'whishi': p95,
        'fliers': []
    }]
    ax.bxp(bxp_data, positions=[x], widths=0.3,
           manage_ticks=False, medianprops=dict(color='black'))



fname = '../Base Case.pkl'
with open(fname, 'rb') as f:
    outputs = pickle.load(f)

external_cost = outputs['External cost']

files = {
    'Base Case': '../Base Case.pkl',
    'Policy Package': '../Policy Package.pkl',
}

fig, ax = plt.subplots(1, 2, sharey=False, figsize=(10, 4)) 
ax = ax.flatten()
positions = []
labels = []

for iFile, (label, fname) in enumerate(files.items()):
    with open(fname, 'rb') as f:
        outputs = pickle.load(f)

    external_cost = outputs['External cost']

    for x, (key, value) in enumerate(external_cost.items()):
        box_plot(external_cost[key][:, -1], ax[iFile], x)
        positions.append(x)
        labels.append(key)
    ax[iFile].set_xticks(positions)
    ax[iFile].set_xticklabels(labels, rotation=45, ha='right')
    ax[iFile].set_title(key)
