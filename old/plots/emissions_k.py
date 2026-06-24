""" Plot costs under each policy scenario. """
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

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


files = {
    'Base': '../Results_19_3_2026/Base.pkl',
    'Break Mandate': '../Results_19_3_2026/Break Mandate.pkl',
    'ZEV Mandate': '../Results_19_3_2026/ZEV Mandate.pkl',
    'ZEV GVWL Increase': '../Results_19_3_2026/ZEV GVWL Increase.pkl',
    'Carbon Tax': '../Results_19_3_2026/Carbon Tax.pkl',
    'LCFS': '../Results_19_3_2026/LCFS.pkl',
    'Policy Package': '../Results_19_3_2026/Policy Package.pkl',
}

# Initial emissions
with open('../Results_19_3_2026/Base.pkl', 'rb') as f:
    outputs = pickle.load(f)
emissions = outputs['Emissions']
base_total = {}
for k in emissions.keys():
    base_total[k] = emissions[k]['Fuel Combustion'] + emissions[k]['Fuel Supply']
base_initial = [np.mean(base_total[k][:, -1]) for k in base_total.keys()]


# ==================
# COST (inc. policy), across all time periods
# ==================
positions = []
labels = []
x = 1
fig, ax = plt.subplots(1,3, figsize=(10, 4), dpi=300)
for label, fname in files.items():

    with open(fname, 'rb') as f:
        outputs = pickle.load(f)

    emissions = outputs['Emissions']
    total = {}
    for k in emissions.keys():
        total[k] = emissions[k]['Fuel Combustion'] + emissions[k]['Fuel Supply']

    for i, (key, value) in enumerate(total.items()):
        box_plot(value[:, -1], ax[i], x, width=0.4)
        print(label, key, np.percentile(value[:, -1]/base_total[key][:, -1], [5, 95]))
    positions.append(x)
    labels.append(label)
    x += 1


# --- Axis labels / ticks ---
ax[0].set_ylabel("Emissions (MtCO$_2$e)")
for a in ax:
    a.set_xticks(positions)
    a.set_xticklabels(labels, rotation=45, ha='right')

for i, a in enumerate(ax):
    baseline = base_initial[i]
    a.set_ylim(0, a.get_ylim()[1])
    ymin, ymax = a.get_ylim()
    axp = a.twinx()
    ymax_pct = ymax / baseline * 100
    ymin_pct = ymin / baseline * 100
    axp.set_ylim(ymin_pct, ymax_pct)
    axp.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)
axp.set_ylabel(r"% 2007 levels")

ax[0].set_title('Sleeper')
ax[1].set_title('Day Cab')
ax[2].set_title('Straight Truck')
plt.tight_layout()