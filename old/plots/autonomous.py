""" Generate plots for the autonomous scenario. """
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


files = {
    'Base': '../Results_19_3_2026/Base.pkl',
    'Policy Package': '../Results_19_3_2026/Policy Package.pkl',
    'Base (AP)': '../Results_19_3_2026/Autonomous Permits (Base).pkl',
    'Policy package (AP)': '../Results_19_3_2026/Autonomous Permits (PP).pkl',
}


with open('../Results_19_3_2026/Base.pkl', 'rb') as f:
    outputs = pickle.load(f)
emissions = outputs['Emissions']
inner = sum_by_outer(emissions)
base_combustion = inner['Fuel Combustion']
base_supply = inner['Fuel Supply']
base_total_fuel = base_combustion + base_supply


with open('../Results_19_3_2026/Policy Package.pkl', 'rb') as f:
    outputs = pickle.load(f)
emissions = outputs['Emissions']
inner = sum_by_outer(emissions)
pp_combustion = inner['Fuel Combustion']
pp_supply = inner['Fuel Supply']
pp_total_fuel = pp_combustion + pp_supply

# EMISSIONS
positions = []
labels = []
x = 1
fig, ax = plt.subplots(1,2, figsize=(10, 4), tight_layout=True, dpi=300)
for label, fname in files.items():

    with open(fname, 'rb') as f:
        outputs = pickle.load(f)

    emissions = outputs['Emissions']
    inner = sum_by_outer(emissions)
    combustion = inner['Fuel Combustion']
    supply = inner['Fuel Supply']
    total_fuel = combustion + supply

    box_plot(total_fuel[:, -1], ax[0], x)

    print(label, np.percentile(total_fuel[:, -1]/(np.mean(base_total_fuel[:, 0]) / 1.344827), [5, 95]))
    print(label, np.percentile(total_fuel[:, -1]/(base_total_fuel[:, -1]), [5, 95]))
    print(label, np.percentile(total_fuel[:, -1]/(pp_total_fuel[:, -1]), [5, 95]))

    positions.append(x)
    labels.append(label)
    x += 1

# --- Axis labels / ticks ---
initial_emissions = np.mean(base_total_fuel[:, 0] / 1.344827)
a = ax[0]
a.set_xticks(positions)
a.set_xticklabels(labels, rotation=45, ha='right')
a.set_title('Total Fuel Emissions 2050')
axp = a.twinx()
ymin, ymax = a.get_ylim()
ymax_pct = ymax / initial_emissions * 100
ymin_pct = ymin / initial_emissions * 100
a.set_ylim(0, ymax)
axp.set_ylim(0, ymax_pct)
a.set_ylabel(r"Emissions (MtCO$_2$e)")
axp.set_ylabel(r"% 2007 levels")
axp.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)

# COSTS
positions = []
labels = []
x = 1
for label, fname in files.items():

    with open(fname, 'rb') as f:
        outputs = pickle.load(f)

    costs = copy.deepcopy(outputs['Cost'])
    for k in costs.keys():
        costs[k].update(outputs['Policy cost'][k])

    outer = sum_by_inner(costs)
    total_cost = sum(outer[k] for k in outer.keys())
    total_activity = sum(outputs['Activity'][k] for k in outputs['Activity'].keys())
    activity_cost = total_cost / total_activity
    # activity_cost = activity_cost[:, -1]
    box_plot(activity_cost[:, -1]*100, ax[1], x)
    positions.append(x)
    labels.append(label)
    x += 1

# --- Axis labels / ticks ---
initial_costs = np.mean(activity_cost[:, 0])
a = ax[1]
a.set_xticks(positions)
a.set_xticklabels(labels, rotation=45, ha='right')
a.set_title('Activity cost 2050')
axp = a.twinx()
ymin, ymax = a.get_ylim()
ymax_pct = ymax / initial_costs
ymin_pct = ymin / initial_costs
axp.set_ylim(ymin_pct, ymax_pct)
a.set_ylabel("Activity cost (¢/t-km)")
axp.set_ylabel(r"% 2025 cost")
axp.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)

