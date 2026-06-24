""" Plot Policy Package without policies. """
import numpy as np
import matplotlib.pyplot as plt
import pickle

from parallel_test import Plotting

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

files = {
    'Policy Package (PP)': '../Results_19_3_2026/Policy Package.pkl',
    'PP (no ZEV Mandate)': '../Results_19_3_2026/Policy Package (no ZEVM).pkl',
    'PP (no ZEV GVWL Increase)': '../Results_19_3_2026/Policy Package (no ZEV GVWL Increase).pkl',
    'PP (no LCFS)': '../Results_19_3_2026/Policy Package (no LCFS).pkl',
    'Accelerated Retirement (PP)': '../Results_19_3_2026/Accelerated Retirement (PP).pkl',
}


with open('../Results_19_3_2026/Policy Package.pkl', 'rb') as f:
    outputs = pickle.load(f)
base_emissions = outputs['Emissions']
inner = sum_by_outer(base_emissions)
base_combustion = inner['Fuel Combustion']
base_supply = inner['Fuel Supply']
base_total_fuel = base_combustion + base_supply

# # ===============
# #    EMISSIONS
# # ===============
# EMISSIONS_2025 = 5.7
# 3.9 Fuel 2022, 2,9 Fuel 2007 (74 % less) => 15 %
fig, ax = plt.subplots(1, 3, sharey=False, figsize=(10, 4))   # share y + less squish
ax = ax.flatten()

positions = []
labels = []
x = 1

for label, fname in files.items():

    with open(fname, 'rb') as f:
        outputs = pickle.load(f)

    emissions = outputs['Emissions']
    inner = sum_by_outer(emissions)
    combustion = inner['Fuel Combustion']
    supply = inner['Fuel Supply']
    total_fuel = combustion + supply

    box_plot(combustion[:, -1], ax[0], x)
    box_plot(supply[:, -1], ax[1], x)
    box_plot(total_fuel[:, -1], ax[2], x)

    positions.append(x)
    labels.append(label)
    x += 1
    percent = 100 - 100 * base_total_fuel[:, -1]/(total_fuel[:, -1])
    print(np.percentile(percent, q=[5, 25, 50, 75, 95]))
    percent = 100 - 100 * total_fuel[:, -1]/(base_total_fuel[:, -1])
    print(np.percentile(percent, q=[5, 25, 50, 75, 95]))

# --- Axis labels / ticks ---
for a in ax:
    a.set_xticks(positions)
    a.set_xticklabels(labels, rotation=45, ha='right')


labels = ['Fuel Use', 'Fuel Supply', 'Total Fuel']
initial_emissions=[np.mean(base_combustion[:, 0]), np.mean(base_supply[:, 0]), np.mean(base_total_fuel[:, 0])]

for i, a in enumerate(ax):
    a.set_title(labels[i])
    a.set_ylim(0, a.get_ylim()[1])
    axp = a.twinx()
    axp.set_ylim(0, a.get_ylim()[1]/(initial_emissions[i]/1.3448) * 100)  # right y-axis as percentage
    a.set_ylabel("Emissions (MtCO2)")
    axp.set_ylabel(r"% of 2007 emissions")
    # axp.axhline(20, color='red', linestyle='--', linewidth=1.5)

plt.tight_layout()

