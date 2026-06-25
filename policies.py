"""
Policy classes for the fleet2 HDT adoption model.

Two-phase interface:
  pre_apply(params, k, p, t)  — modifies the params dict before Vehicles() is constructed.
                                  Use for physics policies (GVWL exemption, survival rate
                                  modifications) that propagate through mass → FC → cost.
  apply(v)                    — writes cost terms into v.annual_cost after construction,
                                  then calls v._calculate_tco_npv() to update tco and npv.
                                  Use for cost policies (carbon tax, LCFS, ZEV rebates).

Each policy class implements only the method(s) it needs.
Policies is a container that calls both hooks from Fleet._make_vehicle().

Usage:
    from policies import Policies, CarbonTax
    policies = Policies(carbon_tax=CarbonTax({'2025': 80, '2030': 170, '2050': 170}))
    fleet = Fleet(PARAMS, param_cps, policies=policies)
"""
import numpy as np
from data import START_YEAR, END_YEAR, MAX_AGE

_YEAR0 = START_YEAR - MAX_AGE


class CarbonTax:
    def __init__(self, price: dict):
        """
        price: {year_str: $/tCO2e}, e.g. {'2025': 80, '2030': 170, '2050': 170}
        Linearly interpolates between anchor years; pre-START_YEAR values are zero.
        """
        years        = np.arange(_YEAR0, END_YEAR + 1)
        anchor_years = sorted(int(y) for y in price)
        anchor_vals  = [float(price[str(y)]) for y in anchor_years]
        arr          = np.interp(years, anchor_years, anchor_vals,
                                 left=anchor_vals[0], right=anchor_vals[-1])
        arr[years < START_YEAR] = 0.0
        self._price_arr = arr.astype(np.float32)

    def apply(self, v):
        idx  = np.clip(v.operation_years - _YEAR0, 0, len(self._price_arr) - 1)
        cost = (v.emissions_supply + v.emissions_use) / 1000.0 * self._price_arr[idx]
        v.annual_cost['carbon_tax'] = cost.astype(np.float32)
        v._calculate_tco_npv()


class Policies:
    def __init__(self, carbon_tax=None):
        self.carbon_tax = carbon_tax

    def pre_apply(self, params, k, p, t):
        """Modify params dict before Vehicles() is constructed (physics policies).
        Future GVWL exemption: set params['gvwl_exemption_kg'] for ZEV powertrains.
        Note: 'gvwl_increase' is already in data.json as the physical limit — use 'gvwl_exemption_kg'
        as the active hook key to avoid collision.
        """
        pass

    def apply(self, v):
        """Write cost terms into v.annual_cost after construction (cost policies)."""
        if self.carbon_tax:
            self.carbon_tax.apply(v)
