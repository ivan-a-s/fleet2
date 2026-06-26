"""
Carbon tax policy plots for the fleet2 HDT adoption model.

Shows how a carbon tax affects vehicle economics and fleet composition:

  npv_by_powertrain(fleet)   -- stacked bar NPV breakdown per powertrain x sample year
  system_costs(fleet)        -- fleet system costs line chart (same style as fleet_plots)
  sales_by_powertrain(fleet) -- new sales by powertrain over time
  stock_by_powertrain(fleet) -- total stock by powertrain over time

Run directly with an illustrative BC-style carbon tax trajectory.
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '../..'))   # root
sys.path.insert(0, os.path.join(_HERE, '..'))      # plots/

import numpy as np
import matplotlib.pyplot as plt
from model import Fleet, get_uncertainty_distributions
from data import PARAMS
from policies import Policies, CarbonTax
from plot_utils import Plotting, PT_COLOR, COST_COLOR


# ---------------------------------------------------------------------------
# NPV by powertrain -- stacked bar (same style as vehicle_plots.npv)
# ---------------------------------------------------------------------------

def npv_by_powertrain(fleet):
    """Stacked bar NPV breakdown per powertrain x sample year, one figure per vehicle type."""
    from vehicle_plots import npv as _npv
    for k in fleet.K:
        _npv(fleet, k=k)


# ---------------------------------------------------------------------------
# System costs -- line chart (same style as fleet_plots)
# ---------------------------------------------------------------------------

def system_costs(fleet):
    """Fleet system costs line chart, summed across vehicle types."""
    from fleet_plots import extract_outputs
    outputs = extract_outputs(fleet)
    p = Plotting()
    fig, ax = p.plot_by_inner(outputs['Cost'],
                               title='System costs (carbon tax scenario)',
                               x_label='Year', y_label='$ billions / year',
                               add_total=True, color_map=COST_COLOR)
    return fig, ax


# ---------------------------------------------------------------------------
# Sales and stock by powertrain
# ---------------------------------------------------------------------------

def sales_by_powertrain(fleet):
    """New vehicle sales by powertrain over time, one subplot per vehicle type."""
    T = fleet.years
    result = {
        k: {p: np.array([float(fleet.sales.get((k, p, t), 0.0)) for t in T]) / 1e3
            for p in fleet.P[k]}
        for k in fleet.K
    }
    p = Plotting()
    fig, axes = p.plot_by_both(result,
                               title='Annual sales by powertrain (carbon tax scenario)',
                               x_label='Year', y_label='New vehicles (thousands)',
                               color_map=PT_COLOR)
    return fig, axes


def stock_by_powertrain(fleet):
    """Total stock by powertrain over time, one subplot per vehicle type."""
    T = fleet.years
    result = {
        k: {p: np.array([float(fleet.total_stock.get((k, p, t), 0.0)) for t in T]) / 1e3
            for p in fleet.P[k]}
        for k in fleet.K
    }
    p = Plotting()
    fig, axes = p.plot_by_both(result,
                               title='Total stock by powertrain (carbon tax scenario)',
                               x_label='Year', y_label='Vehicles (thousands)',
                               add_total=True, color_map=PT_COLOR)
    return fig, axes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    inputs    = dict(get_uncertainty_distributions(PARAMS))
    param_cps = {k: 0.5 for k in inputs}

    # Illustrative BC carbon tax: ~$95/t in 2025, rising to $170/t by 2030
    policy = Policies(carbon_tax=CarbonTax({'2025': 95, '2030': 170, '2050': 170}))
    fleet  = Fleet(PARAMS, param_cps, policies=policy)

    npv_by_powertrain(fleet)
    system_costs(fleet)
    sales_by_powertrain(fleet)
    stock_by_powertrain(fleet)
    plt.show()
