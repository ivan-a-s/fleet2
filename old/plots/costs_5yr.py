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


def box_plot(data, ax, x, edgecolor='black', facecolor='#cce6ff', width=0.4):
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
    'Break Mandate': '../Results_19_3_2026/Break Mandate.pkl',
    'ZEV Mandate': '../Results_19_3_2026/ZEV Mandate.pkl',
    'ZEV GVWL Increase': '../Results_19_3_2026/ZEV GVWL Increase.pkl',
    'Carbon Tax': '../Results_19_3_2026/Carbon Tax.pkl',
    'LCFS': '../Results_19_3_2026/LCFS.pkl',
    'Policy Package': '../Results_19_3_2026/Policy Package.pkl',
    # 'ZEV Rebate': '../ZEV Rebate.pkl',
}

with open('../Results_19_3_2026/Base.pkl', 'rb') as f:
    outputs = pickle.load(f)
cost = copy.deepcopy(outputs['Cost'])
base_capital = sum_by_outer(cost)['Capital']
outer = sum_by_inner(cost)
base_total = sum(outer.values())
base_activity = sum(outputs['Activity'].values())
base_activity_cost = base_total / base_activity




# ==================
# COST (both)
# ==================
positions = []
labels = []
x = 1
YEARS = [2030, 2040, 2050]

fig, ax = plt.subplots(1, 3, figsize=(10, 4), sharey=True, dpi=300)
ax = ax.flatten()

for label, fname in files.items():
    with open(fname, 'rb') as f:
        outputs = pickle.load(f)
    
    # --- 1. Calculate Policy Cost (Your original logic) ---
    costs_p = copy.deepcopy(outputs['Cost'])
    for k in costs_p.keys():
        costs_p[k].update(outputs['Policy cost'][k])
    
    total_p = sum(sum_by_inner(costs_p).values())
    activity = sum(outputs['Activity'].values())
    activity_cost_p = total_p / activity

    # --- 2. Calculate No Policy Cost (Skipping the update) ---
    costs_np = copy.deepcopy(outputs['Cost']) # No policy update here
    total_np = sum(sum_by_inner(costs_np).values())
    activity_cost_np = total_np / activity

    for i, year in enumerate(YEARS):
        idx = year - 2025
        prop_cycle = plt.rcParams['axes.prop_cycle']
        width = 0.4
        if label in ['ZEV Mandate', 'Carbon Tax', 'LCFS', 'Policy Package', 'ZEV Rebate']:
            width=0.5
            box_plot(activity_cost_p[:, idx]*100, ax[i], x - width/4, edgecolor='black', width=width*0.4)
            box_plot(activity_cost_np[:, idx]*100, ax[i], x + width/4, edgecolor='lightsteelblue', facecolor='white', width=width*0.4)
        else:
            box_plot(activity_cost_p[:, idx]*100, ax[i], x, edgecolor='black', width=width)
        ax[i].set_title(f'{year}')

    positions.append(x)
    labels.append(label)
    x += 1

# --- Axis labels / ticks ---
ax[0].set_ylabel("Activity cost (¢/tkm)")
for a in ax:
    a.set_xticks(positions)
    a.set_xticklabels(labels, rotation=45, ha='right')

# Initial costs logic (kept as is)
initial_costs = np.mean(base_activity_cost[:, 0])*100
for i, a in enumerate(ax):
    axp = a.twinx()
    axp.set_ylim(a.get_ylim()[0] / initial_costs * 100, a.get_ylim()[1] / initial_costs * 100)
    if i < len(ax) - 1:
        axp.set_yticklabels([])
    a.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)

axp.set_ylabel(r"% 2025 cost")

# for a in ax:
#     a.yaxis.set_ticks_position('left')
#     a.yaxis.set_label_position('left')

# fig.suptitle('Including Policies (Black); Not Including Policies (Light Blue)')
plt.tight_layout()



with open('../Results_19_3_2026/Accelerated Retirement.pkl', 'rb') as f:
    outputs = pickle.load(f)
cost = copy.deepcopy(outputs['Cost'])
capital = sum_by_outer(cost)['Capital']
outer = sum_by_inner(cost)
total = sum(outer.values())
activity = sum(outputs['Activity'].values())
activity_cost = total / activity

ratio = activity_cost / base_activity_cost
print(np.percentile(ratio[:, 0], q=[5, 95]))
print(np.percentile(capital[:, 0]/base_capital[:, 0], q=[5, 95]))


