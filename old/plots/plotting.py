""" Plot emissions in the different scenarios. """
import numpy as np
import matplotlib.pyplot as plt
import pickle

def sum_by_inner(result):
    summed = {}
    for k, subdict in result.items():
        total_arr = None
        for cat, values in subdict.items():
            arr = np.asarray(values)
            if total_arr is None:
                total_arr = arr.copy()
            else:
                total_arr += arr
        summed[k] = total_arr
    return summed

def sum_by_outer(result):
    summed = {}
    for k, subdict in result.items():
        for cat, values in subdict.items():
            arr = np.asarray(values)  # turns list into array
            if cat not in summed:
                summed[cat] = arr.copy()
            else:
                summed[cat] += arr
    return summed


def box_plot(data, ax, x, edgecolor='black', facecolor='#cce6ff', width=0.4, annotate=False):
    p05, p25, p50, p75, p95 = np.percentile(data, q=[5, 25, 50, 75, 95])
    bxp_data = [{
        'med': p50,
        'q1': p25,
        'q3': p75,
        'whislo': p05,
        'whishi': p95,
        'fliers': []
    }]
    mean = np.mean(data)
    ax.bxp(
        bxp_data,
        positions=[x],
        widths=width,
        manage_ticks=False,
        patch_artist=True,
        boxprops=dict(facecolor=facecolor, edgecolor=edgecolor, alpha=1),
        whiskerprops=dict(color=edgecolor),
        capprops=dict(color=edgecolor),
        medianprops=dict(color=edgecolor)
    )
    ax.scatter([x], [mean], color='red', zorder=3, s=3)
    if annotate:
        values = [p05, p25, p50, p75, p95]
        labels = ['5', '25', '50', '75', '95']
        for val, lab in zip(values, labels):
            ax.text(
                x + width/2 + 0.02,
                val,
                lab,
                va='center',
                ha='left',
                fontsize=5,
                color='black'
            )

