""" Plot the impact of pyrolysis on cost, electricity usage, and water usage, and emissions. """
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

from parallel_test import Plotting
plotting = Plotting()
sum_by_outer = plotting.sum_by_outer
sum_by_inner = plotting.sum_by_inner

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
    'Base (WE)': '../Results_19_3_2026/Base.pkl',
    'Base (P)': '../Results_19_3_2026/Base (P).pkl',
    'Base (EP)': '../Results_19_3_2026/Base (EP).pkl',
    'Policy Package (WE)': '../Results_19_3_2026/Policy Package.pkl',
    'Policy Pakcage (P)': '../Results_19_3_2026/Policy Package (P).pkl',
    'Policy Pakcage (EP)': '../Results_19_3_2026/Policy Package (EP).pkl',
}

# Water consumption
fig, ax = plt.subplots(1, 2, figsize=(10,4))
iFuel=0
fuel_labels = []
for iFile, (label, fname) in enumerate(files.items()):
    with open(fname, 'rb') as f:
        outputs = pickle.load(f)
    water_usage = outputs['Water Usage']

    # Total water
    total_water = copy.deepcopy(water_usage)
    for k, v in total_water.items():
        total_water[k] = sum(v[kk] for kk in v.keys())
    total_water = sum(total_water[k] for k in total_water.keys()) / 1000 / 1e6
    box_plot(total_water[:, -1], ax[0], iFile)

    # Per fuel
    if label == 'Base Case (WE)':
        fuel_energy = sum_by_outer(outputs['Fuel Energy'])
        fuel_energy['Electricity'] = fuel_energy['Slow Charge'] + fuel_energy['Fast Charge']
        fuel_energy.pop('Slow Charge')
        fuel_energy.pop('Fast Charge')
        fuel_energy.pop('Hydrogen (pyrolysis)')
        fuel_energy.pop('Hydrogen (pyrolysis + elec.)')
        water_intensity = sum_by_outer(water_usage)
        water_intensity['Electricity'] = water_intensity['Slow Charge'] + water_intensity['Fast Charge']
        water_intensity.pop('Slow Charge')
        water_intensity.pop('Fast Charge')
        water_intensity.pop('Hydrogen (pyrolysis)')
        water_intensity.pop('Hydrogen (pyrolysis + elec.)')
        for k in ['Diesel', 'Electricity', 'Hydrogen']:
            water_intensity[k] *= 3600e3/(fuel_energy[k])
            box_plot(water_intensity[k][:, -1], ax[1], iFuel)
            iFuel += 1
            if k == 'Hydrogen':
                fuel_labels.append('Hydrogen (WE)')
            else:
                fuel_labels.append(k)
    if label == 'Policy Pakcage (P)':
        fuel_energy = sum_by_outer(outputs['Fuel Energy'])
        water_intensity = sum_by_outer(water_usage)
        for _, (k, v) in enumerate(water_intensity.items()):
            if k == 'Hydrogen':
                water_intensity[k] *= 3600e3/(fuel_energy[k])
                box_plot(water_intensity[k][:, -1], ax[1], iFuel)
                iFuel += 1
                fuel_labels.append('Hydrogen (P)')
    if label == 'Policy Pakcage (EP)':
        fuel_energy = sum_by_outer(outputs['Fuel Energy'])
        water_intensity = sum_by_outer(water_usage)
        for _, (k, v) in enumerate(water_intensity.items()):
            if k == 'Hydrogen':
                water_intensity[k] *= 3600e3/(fuel_energy[k])
                box_plot(water_intensity[k][:, -1], ax[1], iFuel)
                iFuel += 1
                fuel_labels.append('Hydrogen (EP)')

ax[0].set_xticks(np.arange(len(files)))
ax[0].set_xticklabels(list(files.keys()), rotation=45, ha='right')
ax[0].set_ylabel('Water consumption (million tonnes/yr)')
ax[0].set_ylim([0, ax[0].get_ylim()[1]])

ax[1].set_xticks(np.arange(len(fuel_labels)))
ax[1].set_xticklabels(fuel_labels, rotation=45, ha='right')
ax[1].set_ylabel('Water intensity (L per kWh useful energy)')
ax[1].set_ylim([0, ax[1].get_ylim()[1]])


# Electricity consumption
fig, ax = plt.subplots(1, 2, figsize=(10,4))
iFuel=0
fuel_labels = []
for iFile, (label, fname) in enumerate(files.items()):
    with open(fname, 'rb') as f:
        outputs = pickle.load(f)
    electricity_usage = outputs['Electricity Usage']

    # Total water
    total_electricty = copy.deepcopy(electricity_usage)
    for k, v in total_electricty.items():
        total_electricty[k] = sum(v[kk] for kk in v.keys())
    total_electricty = sum(total_electricty[k] for k in total_electricty.keys()) / 1e9
    box_plot(total_electricty[:, -1], ax[0], iFile)

    # Per fuel
    if label == 'Base Case (WE)':
        fuel_energy = sum_by_outer(outputs['Fuel Energy'])
        fuel_energy['Electricity'] = fuel_energy['Slow Charge'] + fuel_energy['Fast Charge']
        electricity_intensity = sum_by_outer(electricity_usage)
        electricity_intensity['Electricity'] = electricity_intensity['Slow Charge'] + electricity_intensity['Fast Charge']
        for k in ['Diesel', 'Electricity', 'Hydrogen']:
            electricity_intensity[k] *= 3600e3/(fuel_energy[k])
            box_plot(electricity_intensity[k][:, -1], ax[1], iFuel)
            iFuel += 1
            if k == 'Hydrogen':
                fuel_labels.append('Hydrogen (WE)')
            else:
                fuel_labels.append(k)
    if label == 'Policy Pakcage (P)':
        fuel_energy = sum_by_outer(outputs['Fuel Energy'])
        electricity_intensity = sum_by_outer(electricity_usage)
        for _, (k, v) in enumerate(electricity_intensity.items()):
            if k == 'Hydrogen':
                electricity_intensity[k] *= 3600e3/(fuel_energy[k])
                box_plot(electricity_intensity[k][:, -1], ax[1], iFuel)
                iFuel += 1
                fuel_labels.append('Hydrogen (P)')
    if label == 'Policy Pakcage (EP)':
        fuel_energy = sum_by_outer(outputs['Fuel Energy'])
        electricity_intensity = sum_by_outer(electricity_usage)
        for _, (k, v) in enumerate(electricity_intensity.items()):
            if k == 'Hydrogen':
                electricity_intensity[k] *= 3600e3/(fuel_energy[k])
                box_plot(electricity_intensity[k][:, -1], ax[1], iFuel)
                iFuel += 1
                fuel_labels.append('Hydrogen (EP)')

ax[0].set_xticks(np.arange(len(files)))
ax[0].set_xticklabels(list(files.keys()), rotation=45, ha='right')
ax[0].set_ylabel('Electricity consumption (TWh/yr)')

ax[1].set_xticks(np.arange(len(fuel_labels)))
ax[1].set_xticklabels(fuel_labels, rotation=45, ha='right')
ax[1].set_ylabel('Elec. intensity (kWh per kWh useful energy)')



# ===============
#  Water and electricity
# ===============
fig, ax = plt.subplots(1, 2, figsize=(10,4), dpi=300)
iFuel=0
fuel_labels = []
for iFile, (label, fname) in enumerate(files.items()):
    with open(fname, 'rb') as f:
        outputs = pickle.load(f)
    electricity_usage = outputs['Electricity Usage']
    water_usage = outputs['Water Usage']

    # Total water
    total_water = copy.deepcopy(water_usage)
    for k, v in total_water.items():
        total_water[k] = sum(v.values())
    total_water = sum(total_water.values()) / 1000 / 1e6
    box_plot(total_water[:, -1], ax[0], iFile)
    print('water:', np.percentile(total_water[:, -1], [5, 95]))

    # Total
    total_electricty = copy.deepcopy(electricity_usage)
    for k, v in total_electricty.items():
        total_electricty[k] = sum(v[kk] for kk in v.keys())
    total_electricty = sum(total_electricty[k] for k in total_electricty.keys()) / 1e9
    box_plot(total_electricty[:, -1], ax[1], iFile)
    print('elec:', np.percentile(total_electricty[:, -1], [5, 95]))


ax[0].set_xticks(np.arange(len(files)))
ax[0].set_xticklabels(list(files.keys()), rotation=45, ha='right')
ax[0].set_ylabel('Water consumption (Mt/yr)')
ax[0].yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)
ax[0].set_title('Water Consumption')

ax[1].set_xticks(np.arange(len(files)))
ax[1].set_xticklabels(list(files.keys()), rotation=45, ha='right')
ax[1].set_ylabel('Electricity consumption (TWh/yr)')
ax[1].yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)
ax[1].set_title('Electricity Consumption')





# # ===============
# #    EMISSIONS
# # ===============
EMISSIONS_2025 = 5.7
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

# --- Axis labels / ticks ---
for a in ax:
    a.set_xticks(positions)
    a.set_xticklabels(labels, rotation=45, ha='right')

labels = ['Fuel Combustion', 'Fuel Supply', 'Total Fuel']
initial_emissions=[np.mean(combustion[:, 0]), np.mean(supply[:, 0]), np.mean(total_fuel[:, 0])]
for i, a in enumerate(ax):
    a.set_title(labels[i])
    a.set_ylim(0, a.get_ylim()[1])
    axp = a.twinx()
    axp.set_ylim(0, a.get_ylim()[1]/initial_emissions[i] * 100)  # right y-axis as percentage
    a.set_ylabel("Emissions (MtCO2)")
    axp.set_ylabel("Percentage of 2025 emissions (%)")
    axp.axhline(15, color='red', linestyle='--', linewidth=1.5)

plt.tight_layout()



# # ===============
# #    EMISSIONS
# # ===============
with open('../Results_19_3_2026/Base.pkl', 'rb') as f:
    outputs = pickle.load(f)
emissions = outputs['Emissions']
inner = sum_by_outer(emissions)
base_combustion = inner['Fuel Combustion']
base_supply = inner['Fuel Supply']
base_total_fuel = base_combustion + base_supply

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

    box_plot(total_fuel[:, 5], ax[0], x)
    box_plot(total_fuel[:, 15], ax[1], x)
    box_plot(total_fuel[:, -1], ax[2], x)

    positions.append(x)
    labels.append(label)
    x += 1

# --- Axis labels / ticks ---
for a in ax:
    a.set_xticks(positions)
    a.set_xticklabels(labels, rotation=45, ha='right')

labels = ['2030', '2040', '2050']
targets = [40, 60, 80]
initial_emissions=[np.mean(total_fuel[:, 0]), np.mean(total_fuel[:, 0]), np.mean(total_fuel[:, 0])]
for i, a in enumerate(ax):
    a.set_title(labels[i])
    a.set_ylim(0, a.get_ylim()[1])
    axp = a.twinx()
    axp.set_ylim(0, a.get_ylim()[1]/initial_emissions[i] * 100)  # right y-axis as percentage
    a.set_ylabel("Emissions (MtCO2)")
    axp.set_ylabel("Percentage of 2025 emissions (%)")
    axp.axhline((100-targets[i])*0.74, color='red', linestyle='--', linewidth=1.5)

plt.tight_layout()


# # ===============
# #    EMISSIONS
# # ===============
with open('../Results_19_3_2026/Base.pkl', 'rb') as f:
    outputs = pickle.load(f)
emissions = outputs['Emissions']
inner = sum_by_outer(emissions)
base_combustion = inner['Fuel Combustion']
base_supply = inner['Fuel Supply']
base_total_fuel = base_combustion + base_supply

# 3.9 Fuel 2022, 2,9 Fuel 2007 (74 % less) => 15 %
fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4, 4), dpi=300)   # share y + less squish
# ax = ax.flatten()
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

    box_plot(total_fuel[:, -1], ax, x)

    positions.append(x)
    labels.append(label)
    x += 1

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha='right')

targets = [40, 60, 80]
ax.set_ylim(0, ax.get_ylim()[1])
axp = ax.twinx()
axp.set_ylim(0, ax.get_ylim()[1]/(np.mean(base_total_fuel[:, 0]/1.34)) * 100)  # right y-axis as percentage
ax.set_ylabel(r"Emissions (MtCO$_2$e)")
axp.set_ylabel(r"% 2007 levels")
axp.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)

plt.tight_layout()


# ==================
# COST (inc. policy), across all time periods
# ==================
positions = []
labels = []
x = 1
fig, ax = plt.subplots(1,3, figsize=(10, 4))
for label, fname in files.items():

    with open(fname, 'rb') as f:
        outputs = pickle.load(f)

    costs = copy.deepcopy(outputs['Cost'])
    for k in costs.keys():
        costs[k].update(outputs['Policy cost'][k])

    outer = sum_by_inner(costs)

    for k in outer.keys():
        # outer[k] = outer[k][:, 10]/outputs['Activity'][k][:, 10]
        outer[k] = np.sum(outer[k], axis=-1)/np.sum(outputs['Activity'][k], axis=-1)
    for i, (key, value) in enumerate(outer.items()):
        box_plot(value, ax[i], x)
    positions.append(x)
    labels.append(label)
    x += 1

# --- Axis labels / ticks ---
ax[0].set_ylabel("Activity cost ($)")
for a in ax:
    a.set_xticks(positions)
    a.set_xticklabels(labels, rotation=45, ha='right')

# Initial costs
outer_cost = sum_by_inner(costs)
initial_costs = [np.mean(outer_cost[k][:, 0]/outputs['Activity'][k][:, 0]) for k in outer.keys()]
for i, a in enumerate(ax):
    baseline = initial_costs[i]
    ymin, ymax = a.get_ylim()
    ymax = max(ymax, baseline)
    axp = a.twinx()
    ymax_pct = ymax / baseline * 100
    ymin_pct = ymin / baseline * 100
    axp.set_ylim(ymin_pct, ymax_pct)
axp.set_ylabel("Percentage of initial cost (%)")

ax[0].set_title('Sleeper')
ax[1].set_title('Day-cab')
ax[2].set_title('Straight Truck')
fig.suptitle('Activity cost (including policies)')
plt.tight_layout()





