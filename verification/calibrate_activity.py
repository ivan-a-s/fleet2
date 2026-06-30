"""
Compute recalibrated activity parameters to account for the sleeper payload change
from fixed 16 t to age-varying (16 t for ages 0-9, 10 t for ages 10-24).

Run with:
    C:\\Users\\ivana\\anaconda3\\python.exe verification/calibrate_activity.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from model import Fleet, PARAMS, START_YEAR, MAX_AGE, get_uncertainty_distributions

param_cps = {path: np.float32(0.5) for path, _ in get_uncertainty_distributions(PARAMS)}
fleet = Fleet(PARAMS, param_cps)

k = 'sleeper'
v_ref = fleet.vehicles[k, 'dice', START_YEAR - MAX_AGE]
dist        = v_ref.annual_distance
surv        = v_ref.params['survival_rate']
payload_new = np.array(v_ref.params['payload'])
GROWTH      = float(PARAMS['fleet']['activity_growth'])

weights  = np.array([surv[a] * (1 + GROWTH) ** (-a) for a in range(MAX_AGE)])
denom_old = float(np.sum(dist * 16_000.0 / 1000.0 * weights))
denom_new = float(np.sum(dist * payload_new  / 1000.0 * weights))
ratio     = denom_new / denom_old

print(f"Old denom: {denom_old:.4f}  New denom: {denom_new:.4f}  Ratio: {ratio:.8f}")

init_act    = float(PARAMS['fleet']['initial_activity'])
old_s_prop  = float(PARAMS['vehicles']['types']['sleeper']['shared']['activity_proportion'])
old_dc_prop = float(PARAMS['vehicles']['types']['day_cab']['shared']['activity_proportion'])
old_st_prop = float(PARAMS['vehicles']['types']['straight']['shared']['activity_proportion'])

old_s_act  = init_act * old_s_prop
old_dc_act = init_act * old_dc_prop
old_st_act = init_act * old_st_prop

new_s_act  = old_s_act * ratio
new_dc_act = old_dc_act
new_st_act = old_st_act
new_total  = new_s_act + new_dc_act + new_st_act

new_s_prop  = new_s_act  / new_total
new_dc_prop = new_dc_act / new_total
new_st_prop = new_st_act / new_total

print(f"\nOld initial_activity:    {init_act:.1f}")
print(f"New initial_activity:    {new_total:.1f}")
print(f"Change:                  {new_total - init_act:.1f}  ({100*(new_total/init_act - 1):.3f}%)")
print(f"\nOld proportions:  sleeper={old_s_prop:.6f}  day_cab={old_dc_prop:.6f}  straight={old_st_prop:.6f}")
print(f"New proportions:  sleeper={new_s_prop:.6f}  day_cab={new_dc_prop:.6f}  straight={new_st_prop:.6f}")
print(f"\n--- Paste into data.json ---")
print(f'"initial_activity": {new_total:.1f},')
print(f'sleeper  "activity_proportion": {new_s_prop:.6f},')
print(f'day_cab  "activity_proportion": {new_dc_prop:.6f},')
print(f'straight "activity_proportion": {new_st_prop:.6f},')
