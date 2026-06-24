""" Plot emissions in the different scenarios. """
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

with open('../Results_19_3_2026/Base.pkl', 'rb') as f:
    outputs = pickle.load(f)

emissions = outputs['Emissions']
inner = sum_by_outer(emissions)
combustion = inner['Fuel Combustion']
supply = inner['Fuel Supply']
base_total_fuel = combustion + supply
emissions_2007 = np.mean(base_total_fuel[:, 0]) / 1.344827

files = {
    'Base': '../Results_19_3_2026/Base.pkl',
    'Break Mandate': '../Results_19_3_2026/Break Mandate.pkl',
    'ZEV Mandate': '../Results_19_3_2026/ZEV Mandate.pkl',
    'ZEV GVWL Increase': '../Results_19_3_2026/ZEV GVWL Increase.pkl',
    'Carbon Tax': '../Results_19_3_2026/Carbon Tax.pkl',
    # 'ZEV Rebate': '../Results_19_3_2026/ZEV Rebate.pkl',
    'LCFS': '../Results_19_3_2026/LCFS.pkl',
    'Policy Package': '../Results_19_3_2026/Policy Package.pkl',
    # 'Accelerated Retirement (PP)': '../Results_19_3_2026/Accelerated Retirement (PP).pkl',
}

# # ===============
# #    EMISSIONS
# # ===============
# EMISSIONS_2025 = 5.7
# 3.9 Fuel 2022, 2,9 Fuel 2007 (74 % less) => 15 %
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(10, 4), dpi=300)   # share y + less squish
ax = ax.flatten()

positions = []
labels = []
x = 1

annotate=False
for label, fname in files.items():

    with open(fname, 'rb') as f:
        outputs = pickle.load(f)

    emissions = outputs['Emissions']
    inner = sum_by_outer(emissions)
    combustion = inner['Fuel Combustion']
    supply = inner['Fuel Supply']
    total_fuel = combustion + supply

    box_plot(total_fuel[:, 5], ax[0], x)
    box_plot(total_fuel[:, 15], ax[1], x)
    box_plot(total_fuel[:, -1], ax[2], x, annotate=annotate)
    annotate=False

    # print(label, 100-100*np.percentile(total_fuel[:, -1]/base_total_fuel[:, -1], [5, 95]))

    positions.append(x)
    labels.append(label)
    x += 1

# --- Axis labels / ticks ---
for a in ax:
    a.set_xticks(positions)
    a.set_xticklabels(labels, rotation=45, ha='right')

labels = ['2030', '2040', '2050']
targets = [40, 60, 80]
for i, a in enumerate(ax):
    a.set_title(labels[i])
    a.set_ylim(0, a.get_ylim()[1])
    axp = a.twinx()
    axp.set_ylim(0, a.get_ylim()[1] / emissions_2007 * 100)
    axp.axhline(100 - targets[i], color='red', linestyle='--', linewidth=1.5)
    if i < len(ax) - 1:
        axp.set_yticklabels([])
    axp.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)

ax[0].set_ylabel("Emissions (MtCO$_2$e)")
axp.set_ylabel(r"% 2007 levels")

plt.tight_layout()

# np.percentile(total_fuel[:, -1]/emissions_2007, np.arange(100))



