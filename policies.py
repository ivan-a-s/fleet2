"""
Policy classes for the fleet2 HDT adoption model.

Two-phase interface:
  pre_apply(params, k, p, t)  -- modifies the params dict before Vehicles() is constructed.
                                  Use for physics policies (GVWL exemption, survival rate
                                  modifications) that propagate through mass -> FC -> cost.
  apply(v)                    -- writes cost terms into v.annual_cost after construction,
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

_YEAR0          = START_YEAR - MAX_AGE
_ZEV_POWERTRAINS = frozenset({'be', 'fc', 'hice'})


class GVWLExemption:
    """
    Grants ZEV powertrains additional GVWL headroom (kg), increasing payload capacity.
    Amounts reflect BC regulation; override via the `increases` constructor argument.
    """
    _DEFAULT_INCREASES = {'sleeper': 5000, 'day_cab': 3000, 'straight': 2000}

    def __init__(self, increases: dict = None):
        self._increases = increases if increases is not None else self._DEFAULT_INCREASES

    def pre_apply(self, params, k, p, t):
        if p in _ZEV_POWERTRAINS:
            params['gvwl_exemption_kg'] = float(self._increases.get(k, 0.0))


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


class LCFS:
    """
    BC Low Carbon Fuel Standard.

    Per-fuel cost formula (summed over all fuels f consumed by the vehicle):
        cost = annual_fuel[f] x (CI_f - TCI_J x EER_f x LHV_f) x credit_price / 1000

    where:
        CI_f   = supply + use emissions intensity of fuel f (kgCO2e per unit fuel)
        TCI_J  = _DIESEL_CI / diesel_lhv x (1 - target[year])   (kgCO2e/J; age array)
        EER_f  = eer[v.p] for non-diesel fuels; 1.0 for diesel always
        LHV_f  = lower heating value of fuel f (J per unit fuel)

    Positive cost = deficit (fuel CI above target); negative cost = credit.

    eer: {powertrain: float} -- EER for the primary non-diesel fuel of that powertrain.
         Powertrains not listed default to 1.0 (diesel-equivalent treatment for all fuels).
         Diesel fuel always uses EER=1.0 regardless of powertrain.

    credit_price: $/tCO2e  (default 300)
    start_target: CI reduction fraction required in START_YEAR (default 0.183 = 18.3%)
    end_target:   CI reduction fraction required in END_YEAR   (default 0.76  = 76%)
    """
    # Diesel CI: supply 0.88 + use 2.52 kgCO2e/L (from params.py fuels.diesel)
    _DIESEL_CI = 0.88 + 2.52

    def __init__(self, credit_price: float = 300.0,
                 start_target: float = 0.183, end_target: float = 0.76,
                 eer: dict = None):
        self._credit_price = float(credit_price)
        self._eer          = dict(eer) if eer else {}
        years = np.arange(_YEAR0, END_YEAR + 1)
        arr   = np.zeros(len(years))
        mask  = years >= START_YEAR
        arr[mask] = np.linspace(start_target, end_target, int(mask.sum()))
        self._target_arr = arr.astype(np.float32)

    def set_baseline_fc(self, k, v):
        pass   # kept for API compatibility with model.py; no longer used

    def apply(self, v):
        idx        = np.clip(v.operation_years - _YEAR0, 0, len(self._target_arr) - 1)
        target     = self._target_arr[idx]
        diesel_lhv = float(v._all_fuels['diesel']['lhv'])
        tci_per_j  = self._DIESEL_CI / diesel_lhv * (1.0 - target)   # kgCO2e/J, age array
        eer        = self._eer.get(v.p, 1.0)
        lcfs_cost  = np.zeros(len(v.age))
        for f, annual_vol in v.annual_fuel.items():
            ei    = v.fuels[f]['emissions_intensity']
            ci_f  = float(ei['supply']) + float(ei['use'])   # kgCO2e per unit fuel
            lhv_f = float(v.fuels[f]['lhv'])
            eer_f = 1.0 if f == 'diesel' else eer
            lcfs_cost += annual_vol * (ci_f - tci_per_j * eer_f * lhv_f)
        v.annual_cost['lcfs'] = (lcfs_cost * self._credit_price / 1000.0).astype(np.float32)


class ZEVMandate:
    """
    Endogenous ZEV sales mandate enforced via a per-year convergence loop in Fleet._run().

    When the ZEV share of new sales falls below target[t], non-ZEV vehicles incur a penalty
    and ZEV vehicles receive a rebate.  Fleet._run() iterates until the ZEV share meets the
    target (or the iteration limit is reached).

    targets : {year_str: fraction}                      for scope='fleet'
              {k: {year_str: fraction}}                 for scope='per_k'
    penalty : maximum $/vehicle annual penalty/rebate
    scope   : 'fleet' (aggregate ZEV share across all k) or 'per_k' (independent per type)
    """

    def __init__(self, targets: dict, penalty: float, scope: str = 'fleet',
                 revenue_neutral: bool = False):
        self.scope           = scope
        self.penalty_max     = float(penalty)
        self.revenue_neutral = revenue_neutral
        years            = np.arange(_YEAR0, END_YEAR + 1)

        if scope == 'per_k':
            self._target_arr = {}
            for k, kdict in targets.items():
                anchor_years = sorted(int(y) for y in kdict)
                anchor_vals  = [float(kdict[str(y)]) for y in anchor_years]
                arr          = np.interp(years, anchor_years, anchor_vals,
                                         left=0.0, right=anchor_vals[-1])
                arr[years < START_YEAR] = 0.0
                self._target_arr[k] = arr.astype(np.float32)
        else:
            anchor_years = sorted(int(y) for y in targets)
            anchor_vals  = [float(targets[str(y)]) for y in anchor_years]
            arr          = np.interp(years, anchor_years, anchor_vals,
                                     left=0.0, right=anchor_vals[-1])
            arr[years < START_YEAR] = 0.0
            self._target_arr = arr.astype(np.float32)

        self._year0 = int(years[0])

    def target_at(self, t, k=None) -> float:
        """Return mandate ZEV fraction for year t (and vehicle type k if per_k scope)."""
        idx = int(t) - self._year0
        if self.scope == 'per_k':
            arr = self._target_arr.get(k)
            return float(arr[idx]) if arr is not None and 0 <= idx < len(arr) else 0.0
        return float(self._target_arr[idx]) if 0 <= idx < len(self._target_arr) else 0.0


class Policies:
    def __init__(self, carbon_tax=None, gvwl_exemption=None, lcfs=None, zev_mandate=None):
        self.carbon_tax     = carbon_tax
        self.gvwl_exemption = gvwl_exemption
        self.lcfs           = lcfs
        self.zev_mandate    = zev_mandate

    def pre_apply(self, params, k, p, t):
        """Modify params dict before Vehicles() is constructed (physics policies)."""
        if self.gvwl_exemption:
            self.gvwl_exemption.pre_apply(params, k, p, t)

    def apply(self, v):
        """Write cost terms into v.annual_cost after construction, then recompute TCO/NPV."""
        if self.carbon_tax:
            self.carbon_tax.apply(v)
        if self.lcfs:
            self.lcfs.apply(v)
        v._calculate_tco_npv()
