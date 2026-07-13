"""
Find the battery capacity (kWh) that maximises NPV for BE and PHE cohorts, at CP=0.5
(median parameters, no policies), in 2025, 2030, 2035, 2040, 2045, 2050.

Battery capacity is currently hand-set per (vehicle type, powertrain) in data.json
(params.py). This script checks whether those values are NPV-optimal, or whether a
different size would do better, holding every other parameter at its normal
select_vehicle_params(k, p, t) value.

Partial-equilibrium approximation: revenue_per_tkm is pinned to the value the
baseline (vibes-sized) fleet actually realised for that year, via
fleet.revenue_per_tkm_history[k, t]. Fleet._run() overwrites
self.params[...]['revenue_per_tkm'] in place every year, so a post-hoc call to
select_vehicle_params() for an arbitrary year would otherwise silently pick up
whatever year the mutation was last left at (2050) rather than that year's own
value. This also means the reported optimum is a single cohort's best response to
the *existing* freight market price, not a re-solved fleet equilibrium -- one
cohort's battery choice does not measurably move the fleet-wide average cost per
t-km that sets revenue_per_tkm, so treating it as fixed is a reasonable first pass.

Run with:
    C:\\Users\\ivana\\anaconda3\\python.exe verification/optimize_battery_capacity.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import minimize_scalar

from model import Fleet, Vehicles, PARAMS, get_uncertainty_distributions

YEARS           = [2025, 2030, 2035, 2040, 2045, 2050]
POWERTRAINS     = ['be', 'phe']
VEHICLE_TYPES   = ['sleeper', 'day_cab', 'straight']
CAPACITY_BOUNDS = (10.0, 3000.0)   # kWh search range
GRID_POINTS     = 121


def build_baseline():
    """CP=0.5 for every uncertain parameter, no policies -- matches model.py's __main__."""
    param_cps = {path: np.float32(0.5) for path, _ in get_uncertainty_distributions(PARAMS)}
    return Fleet(PARAMS, param_cps)


def npv_at_capacity(fleet, k, p, t, capacity):
    params = fleet.select_vehicle_params(k, p, t)
    params['revenue_per_tkm'] = fleet.revenue_per_tkm_history[k, t]
    params['components']['battery']['capacity'] = capacity
    v = Vehicles(params, fleet.params['fuels'], fleet.params['vehicles']['costs'],
                 p=p, k=k, foresight=fleet.foresight)
    return v.npv


def optimize_one(fleet, k, p, t):
    """Coarse grid scan to bracket the optimum, then a bounded refine within one grid step."""
    lo, hi = CAPACITY_BOUNDS
    grid = np.linspace(lo, hi, GRID_POINTS)
    npvs = np.array([npv_at_capacity(fleet, k, p, t, x) for x in grid])
    i0   = int(np.argmax(npvs))
    step = grid[1] - grid[0]
    bracket = (max(lo, grid[i0] - step), min(hi, grid[i0] + step))
    res = minimize_scalar(lambda x: -npv_at_capacity(fleet, k, p, t, x),
                           bounds=bracket, method='bounded')
    x_opt    = float(res.x)
    npv_opt  = -float(res.fun)
    at_bound = np.isclose(x_opt, lo, atol=1.0) or np.isclose(x_opt, hi, atol=1.0)
    return x_opt, npv_opt, at_bound


def main():
    fleet = build_baseline()
    header = (f"{'k':<9}{'p':<5}{'year':<6}{'current kWh':>12}{'optimal kWh':>13}"
              f"{'NPV @ current':>16}{'NPV @ optimal':>16}{'delta $':>13}{'delta %':>9}")
    print(header)
    print('-' * len(header))
    for k in VEHICLE_TYPES:
        for p in POWERTRAINS:
            for t in YEARS:
                v_base  = fleet.vehicles[k, p, t]
                cur_cap = float(v_base.params['components']['battery']['capacity'])
                cur_npv = v_base.npv

                x_opt, npv_opt, at_bound = optimize_one(fleet, k, p, t)

                delta     = npv_opt - cur_npv
                delta_pct = 100.0 * delta / abs(cur_npv) if cur_npv else float('nan')
                flag      = '  <- hit search bound' if at_bound else ''
                print(f"{k:<9}{p:<5}{t:<6}{cur_cap:>12.0f}{x_opt:>13.0f}"
                      f"{cur_npv:>16,.0f}{npv_opt:>16,.0f}{delta:>13,.0f}{delta_pct:>8.1f}%{flag}")


if __name__ == '__main__':
    main()
