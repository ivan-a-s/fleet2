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
    'Base': '../Base.pkl',
    'Break Mandate': '../Break Mandate.pkl',
    'Accelerated Retirement': '../Accelerated Retirement.pkl',
    'ZEV Mandate': '../ZEV Mandate.pkl',
    'ZEV GVWL Increase': '../ZEV GVWL Increase.pkl',
    'Carbon Tax': '../Carbon Tax.pkl',
    'LCFS': '../LCFS.pkl',
    'Policy Package': '../Policy Package.pkl',
}

# ==================
#       COST
# ==================
positions = []
for label, fname in files.items():
    with open(fname, 'rb') as f:
        outputs = pickle.load(f)

    cost = copy.deepcopy(outputs['Cost'])
    cost_by_type = sum_by_inner(cost)
    total_cost = sum(cost_by_type.values())

    policy_cost = copy.deepcopy(outputs['Policy cost'])
    policy_cost_by_type = sum_by_inner(policy_cost)
    total_policy_cost = sum(policy_cost_by_type.values())
    
    activity_by_type = copy.deepcopy(outputs['Activity'])
    total_activity = sum(activity_by_type.values())
    
    x = 1
    fig, ax = plt.subplots(1,2, figsize=(10, 4), sharey=True)
    for iYear, year in enumerate(np.arange(2025, 2051)):
        box_plot((total_cost[:, iYear]+total_policy_cost[:, iYear])/total_activity[:, iYear]*100, ax[0], year)
        box_plot(total_cost[:, iYear]/total_activity[:, iYear]*100, ax[1], year)
        positions.append(x)
        x += 1

    # Initial costs
    initial_costs = [
        np.mean(total_cost[:, 0]/total_activity[:, 0])*100,
        np.mean((total_cost[:, 0] + total_policy_cost[:, 0])/total_activity[:, 0])*100
    ]
    axp_list = []

    for i, a in enumerate(ax):
        baseline = initial_costs[i]
        a.set_ylim([8.5, 12.5])
        ymin, ymax = a.get_ylim()
        ymax = max(ymax, baseline)

        axp = a.twinx()
        axp_list.append(axp)

        ymax_pct = ymax / baseline * 100
        ymin_pct = ymin / baseline * 100
        axp.set_ylim(ymin_pct, ymax_pct)

        # Kill RHS ticks & labels by default
        axp.tick_params(right=False, labelright=False)
        axp.set_ylabel("")

    axp_list[-1].tick_params(right=True, labelright=True)
    axp_list[-1].set_ylabel("Percentage of initial cost (%)")
    ax[1].set_ylabel("")  # suppress duplicate left label
    ax[0].set_ylabel("Activity cost (¢/t-km)")

    ax[0].set_title('Including policies')
    ax[1].set_title('Not including policies')
    fig.suptitle(label)
    plt.tight_layout()

