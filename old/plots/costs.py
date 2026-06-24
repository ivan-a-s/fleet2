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
}
# Initial costs
with open('../Base.pkl', 'rb') as f:
    outputs = pickle.load(f)
base_cost = copy.deepcopy(outputs['Cost'])
base_cost = sum_by_inner(base_cost)
base_activity = outputs['Activity']
base_activity_cost = {}
for k in base_cost.keys():
    base_activity_cost[k] = base_cost[k] / base_activity[k]
initial_costs = {
    k: np.average(base_activity_cost[k]) for k in base_activity_cost.keys()
}
initial_costs = [initial_costs[k]*100 for k in initial_costs.keys()]


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

    activity = outputs['Activity']
    cost = copy.deepcopy(outputs['Cost'])
    cost = sum_by_inner(cost)

    cost_p = copy.deepcopy(outputs['Cost'])
    for k in cost_p.keys():
        cost_p[k].update(outputs['Policy cost'][k])
    cost_p = sum_by_inner(cost_p)

    total_p = sum(cost_p.values())
    total_activity = sum(activity.values())
    activity_cost = total_p/total_activity

    for i, (key, value) in enumerate(cost.items()):
        if label in ['ZEV Mandate', 'Carbon Tax', 'LCFS', 'Policy Package']:
            box_plot(np.average(cost_p[key]/activity[key], axis=-1)*100, ax[i], x - 0.125, edgecolor='black', width=0.2)
            box_plot(np.average(cost[key]/activity[key], axis=-1)*100, ax[i], x + 0.125, edgecolor='lightsteelblue', width=0.2)
        else:
            box_plot(np.average(cost_p[key]/activity[key], axis=-1)*100, ax[i], x, edgecolor='black', width=0.4)
        print(label, key, 100*np.percentile(np.mean(cost_p[key], axis=0)/np.mean(base_cost[key], axis=0), [5, 95]))
    print('total', key, 100*np.percentile(np.mean(total_p, axis=0)/np.mean(sum(base_cost.values()), axis=0), [5, 95]))
    positions.append(x)
    labels.append(label)
    x += 1

# --- Axis labels / ticks ---
ax[0].set_ylabel("Activity cost (¢/tkm)")
for a in ax:
    a.set_xticks(positions)
    a.set_xticklabels(labels, rotation=45, ha='right')

for i, a in enumerate(ax):
    baseline = initial_costs[i]
    ymin, ymax = a.get_ylim()
    axp = a.twinx()
    ymax_pct = ymax / baseline * 100
    ymin_pct = ymin / baseline * 100
    axp.set_ylim(ymin_pct, ymax_pct)
    a.yaxis.grid(True, linestyle='--', color='gray', alpha=0.3)
axp.set_ylabel(r"% 2025 cost")

ax[0].set_title('Sleeper')
ax[1].set_title('Day Cab')
ax[2].set_title('Straight Truck')
# fig.suptitle('Including Policies (Black); Not Including Policies (Light Blue)')
plt.tight_layout()