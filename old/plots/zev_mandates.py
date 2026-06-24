""" Plot stock trajectories. """
import numpy as np
import matplotlib.pyplot as plt
import pickle

from parallel_test import Plotting
from model import ZEVMandate
from data import PARAMS as params

fname = '../Results_19_3_2026/ZEV Mandate.pkl'
with open(fname, 'rb') as f:
    outputs = pickle.load(f)

p_zev = {
    k: sum(outputs['Sales'][k][p] for p in ['BE', 'FC', 'HICE']) / sum(outputs['Sales'][k][p] for p in outputs['Sales'][k].keys()) * 100
    for k in outputs['Sales'].keys()
}

p_zev = (sum(outputs['Sales'][k][p] for k in outputs['Sales'].keys() for p in ['BE', 'FC', 'HICE'])/
    sum(outputs['Sales'][k][p] for k in outputs['Sales'].keys() for p in outputs['Sales']['Sleeper'].keys()))

mean = np.mean(p_zev, axis=0)
p5, p95 = np.percentile(p_zev, [5, 95], axis=0)

zev_mandate = ZEVMandate(
    params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    penalty=30_000,
    rebates=False,
)
cost_per_non_zev = 30_000 * np.maximum(0, (zev_mandate.targets - p_zev)/(1.0-p_zev))
rebate = np.minimum(30_000, 30_000 * (1 - p_zev) / p_zev)
cost_per_hdt = cost_per_non_zev * (1-p_zev) - rebate * p_zev

t = np.arange(2025, 2051)
plt.figure()
plt.plot(t, mean)
plt.fill_between(t,p5,p95,alpha=0.1)

p_low = 5
p_high = 95
plt.figure(figsize=(4,3), dpi=300)
mean = np.mean(cost_per_non_zev, axis=0)
p5, p95 = np.percentile(cost_per_non_zev, [p_low, p_high], axis=0)
plt.plot(t, mean, label='non-ZEV penalty')
plt.fill_between(t,p5,p95,alpha=0.1)

mean = np.mean(rebate, axis=0)
p5, p95 = np.percentile(rebate, [p_low, p_high], axis=0)
plt.plot(t, mean, label='ZEV rebate')
plt.fill_between(t,p5,p95,alpha=0.1)

mean = np.mean(cost_per_hdt, axis=0)
p5, p95 = np.percentile(cost_per_hdt, [p_low, p_high], axis=0)
plt.plot(t, mean, label='Average per HDT')
plt.fill_between(t,p5,p95,alpha=0.1)

plt.xlabel('Years')
plt.ylabel('Cost ($)')
# plt.legend(
#     loc="upper left",
#     bbox_to_anchor=(1, 1)
# )
plt.legend(loc='upper left', fontsize=8, framealpha=0.5)

