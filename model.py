"""
Multinomial logit fleet adoption model for heavy-duty trucks (HDT) in BC, 2025-2050.

Architecture
------------
Each Monte Carlo run draws one cumulative-probability vector (param_cps) and passes it to
Fleet(), which realises all uncertain parameters at once so shared components (e.g. ICE
efficiency) receive a single consistent draw.

The simulation proceeds in three layers:
  1. Vehicles  -- one cohort object per (vehicle type k, powertrain p, model year y).
                 Computes mass, fuel consumption (FASTSim polynomial surrogate), range,
                 annual distance, emissions, capital cost, annual cost, TCO, NPV.
  2. Fleet     -- year-by-year roll-over of surviving cohorts, new vehicle creation, and
                 market-share allocation via multinomial logit with production caps.
  3. Aggregate -- totals over the stock for fuel use, emissions, and system costs.

To do:
 Needs changing later:
 - Proper verification of key parameters (fuel consumption).

 Nice to have (in order of priority):
 - Alternate to hard production cap.
 - Add a resource-haul vehicle type.
 - Convervative behavioiur (favour incumbent).
 - Scale factors to relate cost to scale somehow.
 - Activity price elasticity.
 - Check diesel LHV
 - Improve the battery accessory load estimation.
 - Variance in use (logit change like NREL).
 - Add autonomous vehicles again.
 - Nested logit (all diesel compete together).
 - Simplify params (one motor and then just change the mass with the powertrain etc.)
 - HDRD proportion may change over time. Make this possible (could be through CI distribution).
 - Sobol analysis.
 - Hotel load for sleepers and fridge units?
 - Make scrappage/usage decisions for vehicles?

 Checked up to:
  - _calculate_fuel_consumption

"""
import os
import math
import warnings
import numpy as np
import copy
import scipy.stats as stats
import json
from pprint import pprint

from data import PARAMS, MAX_AGE, START_YEAR, END_YEAR, DISCOUNT_RATE, GROWTH_RATE


def load_model_params(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

_SURROGATES = load_model_params(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'vehicle_modelling', 'surrogates.json')
)

def estimate_fuel_consumption(input_data, model_params):
    """
    Evaluate the degree-2 interaction polynomial surrogate trained by fuel_consumption.py.
    Features are: mass, drag_coef, accessory_load, inv_eff (= 1/peak_eff) and their pairwise
    products. Returns scalar fuel consumption in L/km, kg/km, or kWh/km depending on the
    surrogate (the caller in _split_surrogate_output interprets the units).
    """
    base  = {
        'mass':           input_data['mass'],
        'drag_coef':      input_data['drag_coef'],
        'accessory_load': input_data['accessory_load'],
        'inv_eff':        1 / input_data['peak_eff'],
    }
    total = model_params['intercept']
    for name, coef in model_params['features'].items():
        if ' ' in name:
            v1, v2 = name.split(' ')
            total += base[v1] * base[v2] * coef
        else:
            total += base[name] * coef
    return total

# Surrogate model mappings
# hice reuses he surrogate (H2 ICE hybrid mirrors HEV mechanics); dhice reuses dice surrogate (no regen)
SURROGATE_NAME = {
    'hice': 'he',
    'dhice': 'dice',
}
# Which component efficiency to pass as peak_eff to the surrogate
EFF_COMPONENT = {
    'dice': 'ice', 'he': 'ice',
    'be':   'motor', 'fc': 'fc',
    'hice': 'ice', 'dhice': 'ice',
    'phe':  'motor',
}
ZEV_POWERTRAINS     = {'be', 'fc', 'hice'}
HICE_POWERTRAINS    = {'hice', 'dhice'}
CHARGER_POWERTRAINS = {'be'}
ALL_POWERTRAINS     = frozenset({'dice', 'he', 'phe', 'be', 'fc', 'hice', 'dhice'})

# Nested-logit tree for market-share allocation (_shadow_price_shares). Each non-leaf node is
# (name, lambda_key, children); leaves are bare powertrain strings. lambda_key indexes into
# Fleet.nest_lambdas (params.py: fleet.nest_lambdas); the root's lambda_key is None, meaning its
# scale is fixed at 1.0 -- nothing sits above the root to rescale against, so it is not a
# params.py value. dhice sits in Hydrogen (not Liquid) despite being a 75%-diesel/25%-H2
# dual-fuel ICE -- grouped by ZEV-adjacent substitution pattern with fc/hice, not by fuel share.
NEST_TREE = ('root', None, (
    ('Liquid', 'liquid', (
        ('Conventional', 'conventional', ('dice', 'he')),
        'phe',
    )),
    ('Hydrogen', 'hydrogen', ('fc', 'hice', 'dhice')),
    ('Electric', 'electric', ('be',)),
))


def _walk_leaves(node):
    if isinstance(node, str):
        yield node
    else:
        for c in node[2]:
            yield from _walk_leaves(c)


_NEST_TREE_LEAVES = frozenset(_walk_leaves(NEST_TREE))
assert _NEST_TREE_LEAVES == ALL_POWERTRAINS, (
    f"NEST_TREE leaves {_NEST_TREE_LEAVES} != ALL_POWERTRAINS {ALL_POWERTRAINS}"
)
COST_CATEGORIES     = {
    'system': ('capital', 'operational', 'fuel', 'driver', 'fc_replacements'),
    'policy': ('carbon_tax', 'lcfs', 'zev_mandate'),
}
_YEAR0 = START_YEAR - MAX_AGE   # first year in all realised arrays (e.g. 2000)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_uncertainty_distributions(d, current_path=()):
    """Walk the nested params dict; return (path, group) for each leaf with a 'dist' key.
    group is the value of the optional 'group' field in the dist spec, or None.
    Distributions in the same group share one cp in a Monte Carlo run (see __main__)."""
    paths = []
    if isinstance(d, dict):
        if 'dist' in d:
            paths.append((current_path, d.get('group', None)))
        else:
            for k, v in d.items():
                paths.extend(get_uncertainty_distributions(v, current_path + (k,)))
    return paths

def set_param_(param, cp):
    if isinstance(param, (int, float, np.floating)):
        return param
    elif param['dist'] == 'const':
        return param['val']
    elif param['dist'] == 'triangle':
        c = (param['mode'] - param['min']) / (param['max'] - param['min'])
        return stats.triang.ppf(cp, c, loc=param['min'], scale=param['max'] - param['min'])
    elif param['dist'] == 'uniform':
        return stats.uniform.ppf(cp, loc=param['min'], scale=param['max'] - param['min'])
    else:
        raise ValueError(f"Unknown distribution: {param['dist']}")

def set_param(param, cp=0.5, Y=np.arange(START_YEAR - MAX_AGE, END_YEAR + 1)):
    """
    Convert a cumulative probability cp in [0,1] to a realised parameter value.
    'linear'  -- linearly interpolated array over Y; start and end are themselves distributions.
    'interp'  -- piecewise-linear over specified anchor years, realised at each cp.
    Scalar distributions (const/triangle/uniform) are handled by set_param_().
    The same cp is applied to all distribution specs in a single MC run, so correlated
    parameters (e.g. 2025 and 2050 end-points of a cost curve) move together.
    """
    if param['dist'] == 'linear':
        start = set_param_(param['start'], cp)
        end   = set_param_(param['end'],   cp)
        return np.concatenate([np.ones(MAX_AGE) * start,
                                np.linspace(start, end, len(Y) - MAX_AGE)])
    elif param['dist'] == 'interp':
        year_keys = sorted([int(k) for k in param.keys() if k.isdigit()])
        values    = [set_param_(param[str(y)], cp) for y in year_keys]
        return np.interp(Y, year_keys, values, left=values[0], right=values[-1])
    else:
        return set_param_(param, cp)

def convert_to_float32(d):
    for k, v in d.items():
        if isinstance(v, dict):
            convert_to_float32(v)
        elif isinstance(v, (int, float)):
            d[k] = np.float32(v)
        elif isinstance(v, np.ndarray) and v.dtype != np.float32:
            d[k] = v.astype(np.float32)
    return d

def set_year(input_dict, year=START_YEAR, years=np.arange(START_YEAR - MAX_AGE, END_YEAR + 1)):
    """
    Slice all time-varying arrays in a nested dict to the scalar value at `year`.
    Called by select_vehicle_params() so that a Vehicles object sees scalar costs,
    efficiencies, etc. appropriate to its model year rather than full time series.
    Arrays covering age (target_distance, drive_cycle, survival_rate) are excluded
    from this slicing -- they vary by vehicle age, not calendar year.

    Non-mutating: always returns a new object; the input is never modified.
    This allows select_vehicle_params() to pass in shallow-copied dicts without
    needing deepcopy to protect self.params from mutation.
    """
    if isinstance(input_dict, np.ndarray):
        idx = year - _YEAR0
        return input_dict[max(0, min(idx, len(input_dict) - 1))]
    elif isinstance(input_dict, dict):
        return {key: set_year(value, year, years) for key, value in input_dict.items()}
    return input_dict


# ---------------------------------------------------------------------------
# Vehicles class
# ---------------------------------------------------------------------------

class Vehicles:
    """
    Represents a single vehicle cohort (vehicle type k, powertrain p, model year y).

    params -- time-sliced vehicle params from Fleet.select_vehicle_params()
    fuels  -- full (unsliced) fuel params from PARAMS['fuels'], covering all Y years
    costs  -- full (unsliced) vehicle cost arrays from PARAMS['vehicles']['costs']
    p            -- powertrain key string, e.g. 'dice', 'be', 'fc'
    k            -- vehicle type key string, e.g. 'sleeper', 'day_cab', 'straight'
    foresight    -- 0..1, how much of the future fuel-price / FC-replacement-cost trajectory
                    this cohort "knows" at construction (see _calculate_annual_cost).
    """

    def __init__(self, params, fuels, costs, p, k, foresight=0.0):
        self.params       = params
        self._all_fuels   = fuels                                    # full dict -- needed for en-route fast_charge pricing
        self.fuels        = {f: fuels[f] for f in params['fuels']}
        self.costs        = costs
        self.p            = p
        self.k            = k
        self.foresight    = float(foresight)
        self.age          = np.arange(params['max_age'], dtype=int)
        self.model_year   = int(params['model_year'])
        self.operation_years = self.model_year + self.age
        self._Y_start     = START_YEAR - MAX_AGE   # first year in all realized arrays (e.g. 2000)

        self._discount_factor = (
            np.asarray(params['survival_rate']) / (1.0 + DISCOUNT_RATE) ** self.age
        )

        self._calculate_mass()
        self._calculate_fuel_consumption()
        self._calculate_range()
        self._calculate_annual_distance()
        self._track_fc_replacements()
        self._calculate_emissions()
        self._calculate_capital_cost()
        self._calculate_annual_cost()
        self._calculate_tco_npv()

    # -- Mass ------------------------------------------------------------------

    def _calculate_mass(self):
        gvwl_increase = float(self.params.get('gvwl_exemption_kg', 0.0))
        self.mass = {'frame': float(self.params['components']['frame']['mass'])}
        if 'trailer' in self.params['components']:
            self.mass['trailer'] = float(self.params['components']['trailer']['mass'])
        for comp_name, comp in self.params['components'].items():
            if comp['type'] == 'converter':
                self.mass[comp_name] = float(comp['mass'])
            elif comp['type'] == 'ess':
                # ESS mass = specific_mass (kg per unit capacity) x capacity
                self.mass[comp_name] = float(comp['specific_mass']) * float(comp['capacity'])
            elif comp['type'] == 'transmission':
                self.mass[comp_name] = float(comp['mass'])
        self.unloaded_mass = sum(self.mass.values())

        # Payload penalty: heavier drivetrains displace payload.
        # payload_frac = 1 when unloaded_mass == default_unloaded_mass (no penalty).
        # payload_frac < 1 when unloaded_mass > default, scaled by p_weighed_out
        # (the fraction of loads that are weight-limited rather than volume-limited).
        # The ratio (available_headroom / reference_headroom) gives the fractional
        # payload capacity remaining after the heavier drivetrain takes up mass budget.
        # payload_frac is a scalar -- it depends only on unloaded_mass vs the reference mass budget.
        # The per-age payload (from drive_cycles) is then scaled by this fraction.
        payload_frac = max(0.0, 1.0 - self.params['p_weighed_out'] * (
            1.0 - (self.params['gvwl'] + gvwl_increase - self.unloaded_mass)
                / (self.params['gvwl'] - self.params['default_unloaded_mass'])
        ))
        self.mass['payload'] = np.asarray(self.params['payload']) * payload_frac
        self.total_mass      = self.unloaded_mass + self.mass['payload']  # age-array

    # -- Fuel consumption ------------------------------------------------------

    def _calculate_fuel_consumption(self):
        if self.p == 'phe':
            battery_fuel = next((f for f in self.fuels if 'charge' in f), None)
            charge_eff   = float(self.fuels[battery_fuel].get('refuel_efficiency', 1.0)) if battery_fuel else 1.0
            motor_eff    = float(self.params['components']['motor']['efficiency'])
            ice_eff      = float(self.params['components']['ice']['efficiency'])
            self.average_speed    = self.params['average_speed']
            self.fuel_consumption = {f: np.zeros(len(self.age)) for f in self.fuels}
            for dc in np.unique(self.params['drive_cycle']):
                a_rep = next(a for a in self.age if self.params['drive_cycle'][a] == dc)
                feat  = {
                    'mass':           float(self.total_mass[a_rep]),
                    'drag_coef':      self.params['drag_coef'],
                    'accessory_load': self.params['accessory_load'],
                }
                be_raw = estimate_fuel_consumption({**feat, 'peak_eff': motor_eff}, _SURROGATES['be'][dc])
                he_raw = estimate_fuel_consumption({**feat, 'peak_eff': ice_eff},   _SURROGATES['he'][dc])
                mask   = np.array([self.params['drive_cycle'][a] == dc for a in self.age])
                self.fuel_consumption[battery_fuel][mask] = be_raw / charge_eff
                self.fuel_consumption['diesel'][mask]     = he_raw
            self._phe_cd_fc = self.fuel_consumption[battery_fuel].copy()
            self._phe_cs_fc = self.fuel_consumption['diesel'].copy()
            return

        surrogate = SURROGATE_NAME.get(self.p, self.p)
        peak_eff  = float(self.params['components'][EFF_COMPONENT[self.p]]['efficiency'])

        self.average_speed    = self.params['average_speed']
        self.fuel_consumption = {f: np.zeros(len(self.age)) for f in self.fuels}

        for dc in np.unique(self.params['drive_cycle']):
            # Use the total mass at a representative age for this drive cycle
            a_rep        = next(a for a in self.age if self.params['drive_cycle'][a] == dc)
            model_params = _SURROGATES[surrogate][dc]
            raw          = estimate_fuel_consumption({
                'mass':           float(self.total_mass[a_rep]),
                'drag_coef':      self.params['drag_coef'],
                'accessory_load': self.params['accessory_load'],
                'peak_eff':       peak_eff,
            }, model_params)
            per_fuel = self._split_surrogate_output(raw)
            mask     = np.array([self.params['drive_cycle'][a] == dc for a in self.age])
            for f, val in per_fuel.items():
                if f in self.fuel_consumption:
                    self.fuel_consumption[f][mask] = val

        # Convert from ESS (tank/battery) units/km to source (grid/pump) units/km.
        # The surrogate is trained on vehicle energy demand, but fuel costs and emissions
        # are quoted per unit of delivered energy at the pump/meter.  Dividing by
        # refuel_efficiency (e.g. 0.95 for slow AC charging, 0.86 for DC fast charging)
        # grosses up from battery kWh to grid kWh, or from tank kg to pump kg for H2.
        for f in self.fuel_consumption:
            eff = float(self.fuels[f].get('refuel_efficiency', 1.0))
            if eff < 1.0:
                self.fuel_consumption[f] = self.fuel_consumption[f] / eff

    def _split_surrogate_output(self, raw_val):
        """
        Map the scalar surrogate output to a per-fuel consumption dict (L/km, kg/km, or kWh/km).

        The surrogate always returns a single number representing the primary energy carrier
        (diesel L/km for ICE/HEV, kWh/km battery for BEV, kg/km H2 for FCEV).  For
        multi-fuel powertrains (PHE, DHICE) the total energy is split into components using
        the fuel proportions declared in data.json, via LHV-based energy accounting.
        These are battery-to-wheel values; the caller divides by refuel_efficiency to get
        grid/pump values.
        """
        DIESEL_LHV = float(self._all_fuels['diesel']['lhv'])  # J/L
        H2_LHV     = float(self._all_fuels['h2']['lhv'])      # J/kg
        fp = self.params['fuels']   # {fuel: {proportion: x}}

        if self.p in ('dice', 'he'):
            # Surrogate gives net diesel L/km; HEV regen already reflected in the trained model
            return {'diesel': raw_val}

        elif self.p == 'be':
            e_fuel = next(f for f in self.fuels if 'charge' in f)
            return {e_fuel: raw_val}

        elif self.p == 'fc':
            # Surrogate is trained on a fuel-cell HEV model; output is H2 kg/km
            return {'h2': raw_val}

        elif self.p == 'hice':
            # Reuses the diesel ICE surrogate; output is diesel L/km which is converted to
            # kg/km H2 via the LHV ratio (same mechanical energy, different fuel energy density)
            return {'h2': raw_val * DIESEL_LHV / H2_LHV}

        elif self.p == 'dhice':
            # Reuses the DICE surrogate (diesel L/km total); split total energy into diesel
            # and H2 fractions using the proportions declared in data.json, then convert
            # each fraction back to its own units via LHV
            d_prop = fp.get('diesel', {}).get('proportion', 0.75)
            h_prop = fp.get('h2',     {}).get('proportion', 0.25)
            total  = d_prop + h_prop
            total_energy = raw_val * DIESEL_LHV
            return {
                'diesel': total_energy * (d_prop / total) / DIESEL_LHV,
                'h2':     total_energy * (h_prop / total) / H2_LHV,
            }

        return {}

    # -- Range -----------------------------------------------------------------
    
    def _calculate_range(self):
        """
        Compute per-age range (km) as the binding energy-storage constraint across all ESS
        components, and set self.refuel_rate for use in the daily-distance loop.

        Range per ESS = capacity x usable_fraction / fc_tank  [km]
        where fc_tank = fuel_consumption (source units/km) x refuel_efficiency converts back
        to tank/battery units (kWh, kg, L) so that capacity and FC are in commensurable units.
        refuel_efficiency is only relevant to charging losses, not to how far the stored
        energy propels the vehicle.  The binding range is the minimum across all ESS.

        self.refuel_rate and self._binding_fuel belong to the ESS that set the minimum range
        (the constraining tank/battery), so they are always paired with the correct FC in the
        time-budget formula in _calculate_annual_distance.
        For H2 tanks: pump flow rate (kg/hr).
        For batteries: fast-charger wall power x fast_eff = kW delivered to battery.
        """
        battery_fuel = next((f for f in self.fuels if 'charge' in f), None)
        ESS_FUEL = {
            'diesel_tank': 'diesel',
            'h2_700bar':   'h2',
            'h2_350bar':   'h2',
            'battery':     battery_fuel,
        }
        self.range         = np.full(len(self.age), 1e6)
        self.refuel_rate   = 0.0
        self._binding_fuel = next(iter(self.fuel_consumption), None)

        for comp_name, comp in self.params['components'].items():
            if comp['type'] != 'ess':
                continue
            f = ESS_FUEL.get(comp_name)
            if f is None or f not in self.fuel_consumption:
                continue
            fc = self.fuel_consumption[f].copy()
            if np.all(fc == 0):
                continue
            fc[fc == 0] = 1e-9
            refuel_eff = float(self.fuels[f].get('refuel_efficiency', 1.0))
            r = float(comp['capacity']) * float(comp['usable_capacity']) / (fc * refuel_eff)
            prev = self.range.copy()
            self.range = np.minimum(self.range, r)
            if comp_name == 'battery':
                rate_raw = float(comp['refuel_rate'])
                if rate_raw > 0 and battery_fuel:
                    fast_eff = float(self._all_fuels['fast_charge']['refuel_efficiency'])
                    rate = rate_raw * fast_eff   # kW delivered to battery
                else:
                    rate = 0.0
            else:
                rate = float(comp['refuel_rate'])
            if np.any(self.range < prev):
                self.refuel_rate   = rate
                self._binding_fuel = f

    # -- Annual distance -------------------------------------------------------

    def _calculate_annual_distance(self):
        """
        Age-by-age loop: applies battery degradation and range-limited daily distance.

        target_distance (km/year) is converted to a daily working-day target:
          daily_target = annual_km / 365 x (7/5)
        i.e., trucks are assumed to work 5 days out of 7, so each working day covers
        more than 1/365 of the annual distance.

        When daily_target <= range, the vehicle drives the full target.  Otherwise it
        can extend its range with a refuelling/recharging stop using a time-budget formula:
          achievable = (time_left - 0.25h) x speed x R / (fc x speed + R)
        where time_left = shortfall / speed (hours that would have been spent driving),
        0.25h is the fixed stop overhead, and R = self.refuel_rate.  This is derived by
        solving simultaneously for stop time and extra distance given a fixed time budget.

        Battery degradation follows a linear capacity-fade model:
          effective_range[a] = range[a] x max(0, 1 - deg_per_yearxa - deg_per_cyclexcycles)
        Cycle count accumulates from annual_distance x fuel_consumption / battery_capacity.

        self._enroute_distance tracks km driven via en-route fast charging (non-zero only
        for slow-charge BETs when range < target), used in _calculate_annual_cost to apply
        fast-charge pricing to that portion of electricity consumption.

        Annual distance is converted back from working-day to calendar-year basis:
          annual_km = daily_km x (5/7) x 365
        """
        daily_target = self.params['target_distance'] / 365.0 * 7.0 / 5.0

        battery_comp       = self.params['components'].get('battery')
        battery_fuel       = next((f for f in self.fuels if 'charge' in f), None) if battery_comp else None
        battery_cap        = float(battery_comp['capacity'])      if battery_comp else 0.0
        deg_per_year       = float(battery_comp['deg_per_year'])  if battery_comp else 0.0
        deg_per_cycle      = float(battery_comp['deg_per_cycle']) if battery_comp else 0.0
        battery_refuel_eff = float(self.fuels[battery_fuel]['refuel_efficiency']) if battery_fuel else 1.0

        binding_refuel_eff = float(self.fuels[self._binding_fuel].get('refuel_efficiency', 1.0)) \
                             if self._binding_fuel else 1.0

        range_ = self.range.copy()
        self.annual_distance   = np.zeros(len(self.age))
        self._enroute_distance = np.zeros(len(self.age))
        cycles = 0.0

        if self.p == 'phe':
            electric_range   = self.range.copy()
            self.annual_fuel = {battery_fuel: np.zeros(len(self.age)),
                                'diesel':     np.zeros(len(self.age))}
            self._enroute_distance = np.zeros(len(self.age))
            cycles = 0.0
            for a in self.age:
                if battery_cap > 0 and self._phe_cd_fc[a] > 0:
                    elec_r = electric_range[a] * max(0.0, 1.0 - deg_per_year * a - deg_per_cycle * cycles)
                else:
                    elec_r = 0.0
                cd_daily = min(daily_target[a], elec_r)
                cs_daily = max(0.0, daily_target[a] - cd_daily)
                self.annual_distance[a]        = daily_target[a] * 5.0 / 7.0 * 365.0
                annual_cd                      = cd_daily * 5.0 / 7.0 * 365.0
                annual_cs                      = cs_daily * 5.0 / 7.0 * 365.0
                self.annual_fuel[battery_fuel][a] = annual_cd * self._phe_cd_fc[a]
                self.annual_fuel['diesel'][a]     = annual_cs * self._phe_cs_fc[a]
                if battery_cap > 0 and self._phe_cd_fc[a] > 0:
                    cycles += annual_cd * self._phe_cd_fc[a] * battery_refuel_eff / battery_cap
            self.range = electric_range
            return

        for a in self.age:
            # Battery range degradation (1% per year of age + 0.002% per charge cycle)
            if (battery_fuel and battery_fuel in self.fuel_consumption
                    and self.fuel_consumption[battery_fuel][a] > 0):
                range_[a] = self.range[a] * max(0.0, 1.0 - deg_per_year * a - deg_per_cycle * cycles)

            if daily_target[a] <= range_[a]:
                daily      = daily_target[a]
                achievable = 0.0
            else:
                # Estimate extra distance achievable during a refuelling/recharging stop
                shortfall = daily_target[a] - range_[a]
                time_left = shortfall / max(self.average_speed[a], 1.0)
                fc_a = self.fuel_consumption[self._binding_fuel][a] * binding_refuel_eff
                if self.refuel_rate > 0 and fc_a > 0:
                    # Time budget formula: (available_time - 0.25h overhead) x achievable rate
                    achievable = max(0.0,
                        (time_left - 0.25) * self.average_speed[a]
                        * self.refuel_rate / (fc_a * self.average_speed[a] + self.refuel_rate)
                    )
                else:
                    achievable = 0.0
                daily = range_[a] + achievable

            self.annual_distance[a]   = daily      * 5.0 / 7.0 * 365.0
            self._enroute_distance[a] = achievable * 5.0 / 7.0 * 365.0

            if (battery_fuel and battery_cap > 0
                    and self.fuel_consumption[battery_fuel][a] > 0):
                cycles += (self.annual_distance[a]
                           * self.fuel_consumption[battery_fuel][a] * battery_refuel_eff / battery_cap)

        self.range = range_  # save degraded-by-age range back to attribute
        self.annual_fuel = {
            f: self.annual_distance * self.fuel_consumption[f]
            for f in self.fuel_consumption
        }

        # Split slow_charge into depot (slow) + en-route (fast) portions so that
        # fuel_usage and emissions reflect which charger type delivered the energy.
        # Depot portion = (annual_distance - enroute_distance) x fc_slow [grid kWh].
        # En-route portion = enroute_distance x fc_fast [grid kWh], where
        #   fc_fast = fc_slow x slow_eff / fast_eff (same battery kWh, more wall losses).
        if battery_fuel == 'slow_charge' and np.any(self._enroute_distance > 0):
            slow_eff    = float(self.fuels[battery_fuel].get('refuel_efficiency', 1.0))
            fast_data   = self._all_fuels.get('fast_charge', {})
            fast_eff    = float(fast_data.get('refuel_efficiency', slow_eff))
            fc_per_km   = self.fuel_consumption[battery_fuel]
            enroute_kwh = self._enroute_distance * fc_per_km * slow_eff / fast_eff
            self.annual_fuel[battery_fuel] = self.annual_fuel[battery_fuel] - self._enroute_distance * fc_per_km
            self.annual_fuel['fast_charge'] = enroute_kwh
            if 'fast_charge' not in self.fuels and fast_data:
                self.fuels['fast_charge'] = fast_data

    # -- FC replacements -------------------------------------------------------

    def _track_fc_replacements(self):
        """
        Track cumulative operating hours and flag ages at which the fuel-cell stack
        must be replaced (fc_replacements[a] = 1.0).  The stack lifetime is in hours;
        the counter resets to zero after each replacement event.
        """
        self.fc_replacements = np.zeros(len(self.age))
        fc_comp = self.params['components'].get('fc')
        if fc_comp is None:
            return
        fc_lifetime = float(fc_comp['lifetime'])
        fc_hours = 0.0
        for a in self.age:
            fc_hours += self.annual_distance[a] / max(self.average_speed[a], 1.0)
            if fc_hours > fc_lifetime:
                self.fc_replacements[a] = 1.0
                fc_hours = 0.0

    # -- Emissions -------------------------------------------------------------

    def _calculate_emissions(self):
        """
        Three emission streams:
          embodied      -- manufacturing emissions (kgCO2e): initial manufacture at age 0;
                          FC stack replacements at their actual replacement ages, if any.
                          Frame uses the 'frame' component's embodied_emissions (kgCO2e/kg);
                          trailer (sleeper/day_cab only) uses its own 'trailer' component,
                          multiplied by trailers_per_truck for lifetime replacements. Both are
                          shared across vehicle types like every other component; only the
                          mass (frame_mass/trailer_mass) is vehicle-type-specific.
                          tire/trailer_tire use the 'tire' component's embodied_emissions,
                          lump-summed at age 0 like frame/trailer (no age-distribution): their
                          'mass' is each vehicle type's *lifetime* tire-replacement mass, not an
                          instantaneous physical mass, so unlike frame/trailer they are NOT added
                          to Vehicles.mass/unloaded_mass in _calculate_mass. Unlike the trailer's
                          own structural mass, trailer_tire is NOT multiplied by trailers_per_truck:
                          tire wear is driven by cumulative distance travelled, not by how many
                          physical trailer units carried that distance, so GREET's replacement
                          schedule (calibrated over one reference truck lifetime) already covers
                          the full truck life regardless of how many trailers cycle through it.
                          Converter/transmission components use their own embodied_emissions
                          field (same distribution, falls back to shared factor if absent).
                          ESS components use a dedicated kgCO2e/unit-capacity field.
                          All embodied_emissions distributions share one MC cp ("embodied"
                          group in data.json) -- high/low manufacturing decarbonisation moves
                          all factors together.
                          Survival is handled by _aggregate (n = surviving vehicle count),
                          so replacement emissions at late ages scale down automatically.
          emissions_supply -- upstream (well-to-tank) emissions per year (kgCO2e/yr).
          emissions_use    -- tailpipe (tank-to-wheel) emissions per year (kgCO2e/yr).
                             Zero for ZEVs (no combustion at point of use).
        Supply and use emissions scale with annual_fuel, already in source units/km so
        the emissions intensities (kgCO2e per source unit) apply directly.
        """
        p = self.params
        emb = float(p['components']['frame']['embodied_emissions'])
        embodied_total = float(p['components']['frame']['mass']) * emb
        trailer_comp = p['components'].get('trailer')
        if trailer_comp is not None:
            trailer_emb = float(trailer_comp.get('embodied_emissions', emb))
            embodied_total += (
                float(trailer_comp['mass']) * float(p.get('trailers_per_truck', 0)) * trailer_emb
            )
        tire_comp = p['components'].get('tire')
        if tire_comp is not None:
            embodied_total += float(tire_comp['mass']) * float(tire_comp.get('embodied_emissions', emb))
        trailer_tire_comp = p['components'].get('trailer_tire')
        if trailer_tire_comp is not None:
            tt_emb = float(trailer_tire_comp.get('embodied_emissions', emb))
            embodied_total += float(trailer_tire_comp['mass']) * tt_emb
        for _, comp in p['components'].items():
            if comp['type'] == 'ess' and 'embodied_emissions' in comp:
                embodied_total += float(comp['capacity']) * float(comp['embodied_emissions'])
            elif comp['type'] in ('converter', 'transmission') and 'mass' in comp:
                comp_emb = float(comp.get('embodied_emissions', emb))
                embodied_total += float(comp['mass']) * comp_emb
        self.embodied = np.concatenate([[embodied_total], np.zeros(len(self.age) - 1)])
        # FC stack replacement embodied emissions: distributed to actual replacement ages.
        # Survival is handled by _aggregate (n = surviving vehicle count), same as fc_replacement costs.
        fc_comp = self.params['components'].get('fc')
        if fc_comp is not None and np.any(self.fc_replacements > 0):
            comp_emb = float(fc_comp.get('embodied_emissions', emb))
            self.embodied = self.embodied + self.fc_replacements * float(fc_comp['mass']) * comp_emb

        self.emissions_supply = np.zeros(len(self.age))
        self.emissions_use    = np.zeros(len(self.age))
        for f, annual in self.annual_fuel.items():
            ei = self.fuels[f]['emissions_intensity']
            self.emissions_supply += annual * float(ei['supply'])
            self.emissions_use    += annual * float(ei['use'])

    # -- Capital cost ----------------------------------------------------------

    def _cap_cost(self, key):
        """Get cost value at this vehicle's model year (scalars returned as-is)."""
        val = self.costs.get(key)
        if val is None:
            return 0.0
        if isinstance(val, np.ndarray):
            idx = np.clip(self.model_year - self._Y_start, 0, len(val) - 1)
            return float(val[idx])
        return float(val)

    def _op_cost_array(self, key):
        """
        Get cost array indexed to operation years (one value per age), blended toward the
        construction-year ('present') value by self.foresight: 0 = frozen at model_year for
        the whole life (no foresight), 1 = actual future trajectory (full foresight).
        """
        val = self.costs.get(key)
        if val is None:
            return np.zeros(len(self.age))
        if isinstance(val, np.ndarray):
            idx_future  = np.clip(self.operation_years - self._Y_start, 0, len(val) - 1)
            idx_present = np.clip(self.model_year      - self._Y_start, 0, len(val) - 1)
            blended = (1.0 - self.foresight) * val[idx_present] + self.foresight * val[idx_future]
            return blended.astype(float)
        return np.full(len(self.age), float(val))

    def _calculate_capital_cost(self):
        """
        One-time purchase cost broken into labelled components (for plotting).

        Components are looked up from vehicles.costs in data.json:
          engine     -- $/unit: hice/dhice use the hydrogen-engine price, all others diesel engine
          motor      -- $/kW x motor capacity
          battery    -- $/kWh x battery capacity (also includes ESS for phe/he)
          h2_tank    -- $/kg x tank capacity
          fc         -- $/kW x fuel-cell capacity
          tank       -- $/L x diesel tank capacity
          charger    -- depots only (not sleeper -- sleeper trucks use en-route charging)
          after_treatment -- NOx catalyst; present in data.json for dice, he, phe, dhice

        self.capital_total is the scalar sum used in annual_cost['capital'].
        """
        c = {'base': float(self.params['base_cost'])}
        for comp_name, comp in self.params['components'].items():
            cap = float(comp.get('capacity', 0))
            if comp_name == 'ice':
                key = 'hice_engine' if self.p in HICE_POWERTRAINS else 'diesel_engine'
                c['engine'] = self._cap_cost(key)
            elif comp_name == 'motor':
                c['motor'] = self._cap_cost('motor') * cap
            elif comp_name == 'battery':
                c['battery'] = self._cap_cost('battery') * cap
            elif comp_name in ('h2_700bar', 'h2_350bar'):
                c['h2_tank'] = self._cap_cost(comp_name) * cap
            elif comp_name == 'fc':
                c['fc'] = self._cap_cost('fc') * cap
            elif comp_name == 'diesel_tank':
                c['tank'] = self._cap_cost('tank') * cap
            elif comp_name == 'combustion_transmission':
                c['combustion_transmission'] = self._cap_cost('combustion_transmission')
            elif comp_name == 'electric_transmission':
                c['electric_transmission'] = self._cap_cost('electric_transmission')
            elif comp_name == 'after_treatment':
                c['after_treatment'] = self._cap_cost('after_treatment')
        if self.p in CHARGER_POWERTRAINS and self.k != 'sleeper':
            c['charger'] = self._cap_cost('charger_50kw')
        if self.p == 'phe' and self.k != 'sleeper':
            c['charger'] = 0.25 * self._cap_cost('charger_50kw')
        self.capital       = c
        self.capital_total = sum(c.values())

    # -- Annual cost & TCO -----------------------------------------------------

    def _calculate_annual_cost(self):
        """
        Five cost components, all in $/year as age arrays:

          capital        -- full purchase price at age 0, zero thereafter.  Placed here
                           (rather than amortised) so _discount() gives the correct NPV.
          operational    -- maintenance, tyres, etc. proportional to km driven (actual distance).
          fuel           -- fuel cost in source units x time-varying price, blended between
                           the construction-year price and the actual future price by
                           self.foresight (0 = frozen at construction year, 1 = actual future
                           trajectory -- see class docstring).
                           For slow-charge BETs with en-route fast charging, km driven via
                           en-route stops are re-billed at fast-charge rates with a different
                           efficiency: grid_kWh = km x slow_fc x slow_eff / fast_eff.
          driver         -- wage proportional to target_distance (not actual), because the
                           driver is paid whether or not the vehicle is range-constrained.
          fc_replacements -- time-varying $/kW x stack capacity at each replacement age,
                           also subject to self.foresight (see _op_cost_array).

        Revenue is also computed here: annual_distance x payload_tonnes x revenue_per_tkm.
        """
        fuel_cost = np.zeros(len(self.age))
        for f, annual in self.annual_fuel.items():
            cost_arr    = np.asarray(self.fuels[f]['cost'])
            idx_future  = np.clip(self.operation_years - self._Y_start, 0, len(cost_arr) - 1)
            idx_present = np.clip(self.model_year      - self._Y_start, 0, len(cost_arr) - 1)
            price = (1.0 - self.foresight) * cost_arr[idx_present] + self.foresight * cost_arr[idx_future]
            fuel_cost += annual * price

        # FC replacement cost (time-varying $/kW x capacity x replacement events)
        fc_comp = self.params['components'].get('fc')
        fc_replacement_cost = np.zeros(len(self.age))
        if fc_comp is not None and np.any(self.fc_replacements > 0):
            fc_cap = float(fc_comp['capacity'])
            fc_replacement_cost = self.fc_replacements * fc_cap * self._op_cost_array('fc')

        self.annual_cost = {
            'capital':         np.array([self.capital_total if a == 0 else 0.0 for a in self.age]),
            'operational':     self.annual_distance * float(self.params['running_cost']),
            'fuel':            fuel_cost,
            'driver':          self.params['target_distance'] * float(self.params['driver_cost']),
            'fc_replacements': fc_replacement_cost,
            'carbon_tax':      np.zeros(len(self.age), dtype=np.float32),
            'lcfs':            np.zeros(len(self.age), dtype=np.float32),
            'zev_mandate':     np.zeros(len(self.age), dtype=np.float32),
        }
        self.annual_revenue = (
            self.annual_distance
            * self.mass['payload'] / 1000.0
            * float(self.params['revenue_per_tkm'])
        )

    def _discount(self, annual):
        """Survival-weighted NPV: sum_a  annual[a] x survival_rate[a] / (1+r)^a."""
        return float(np.sum(np.asarray(annual) * self._discount_factor))

    def _calculate_tco_npv(self):
        """
        TCO = NPV sum of all cost components (capital + operating + fuel + driver + FC stack).
        NPV = NPV(revenue) - TCO.

        A higher NPV signals a more profitable vehicle from the operator's perspective and is
        the utility term that drives multinomial logit market-share allocation in Fleet.
        Note: TCO here is the total discounted cost of ownership, not cost per km.
        """
        self.tco = sum(self._discount(v) for v in self.annual_cost.values())
        self.npv = self._discount(self.annual_revenue) - self.tco


# ---------------------------------------------------------------------------
# Fleet class
# ---------------------------------------------------------------------------

def _market_share_limit(prev_share, init, cagr_nacent, cagr_mature, threshold=0.15):
    """Maximum market share achievable this year given last year's share (supply constraint)."""
    prev_share = max(prev_share, init / (1 + cagr_nacent))
    if prev_share < threshold:
        return prev_share * (1 + cagr_nacent) / (1 + GROWTH_RATE)
    return prev_share * (1 + cagr_mature) / (1 + GROWTH_RATE)


def _prune_tree(node, active_leaves):
    """
    Return a copy of `node` (a NEST_TREE node) keeping only leaves in active_leaves, dropping
    any subtree that becomes empty as a result. Returns None if nothing survives.
    """
    if isinstance(node, str):
        return node if node in active_leaves else None
    name, lambda_key, children = node
    kept = [p for p in (_prune_tree(c, active_leaves) for c in children) if p is not None]
    return (name, lambda_key, tuple(kept)) if kept else None


def _bottom_up_utility(node, V, nest_lambdas, out):
    """
    Post-order pass: out[leaf name or id(subtree node)] = utility this node contributes to its
    parent (a leaf's own utility, or a nest's lambda-scaled log-sum-exp of its children).
    """
    if isinstance(node, str):
        out[node] = V[node]
        return out[node]
    name, lambda_key, children = node
    lam = 1.0 if lambda_key is None else nest_lambdas[lambda_key]
    # Plain-Python log-sum-exp (not np.array/.max()/np.sum/np.exp): every NEST_TREE node has
    # only 1-7 children, and numpy's per-call dispatch overhead on arrays that small dwarfs the
    # actual arithmetic -- this recurses ~3.6M times during a ZEV-mandate MC run, where that
    # overhead was measured (cProfile, 2026-07) to be ~80% of total _shadow_price_shares runtime.
    # Same math, bit-identical up to normal floating-point ULP noise (well inside the snapshot's
    # 0.01% RTOL).
    scaled = [_bottom_up_utility(c, V, nest_lambdas, out) / lam for c in children]
    m = max(scaled)
    u_node = lam * (m + math.log(sum(math.exp(s - m) for s in scaled)))
    out[node if isinstance(node, str) else id(node)] = u_node
    return u_node


def _top_down_shares(node, p_node, nest_lambdas, node_utils, shares):
    """Pre-order pass: distributes p_node down to leaves via each node's own conditional softmax."""
    if isinstance(node, str):
        shares[node] = p_node
        return
    name, lambda_key, children = node
    lam = 1.0 if lambda_key is None else nest_lambdas[lambda_key]
    # Plain-Python softmax -- see _bottom_up_utility's comment on why (same tiny-array numpy
    # overhead, same fix).
    scaled = [node_utils[c if isinstance(c, str) else id(c)] / lam for c in children]
    m      = max(scaled)
    exp_s  = [math.exp(s - m) for s in scaled]
    total  = sum(exp_s)
    for c, e in zip(children, exp_s):
        _top_down_shares(c, p_node * (e / total), nest_lambdas, node_utils, shares)


def _leaf_ancestor_path(node, target):
    """
    Return a tuple of ancestor nodes (root first) from `node` down to (not including) the leaf
    named `target`, or None if `target` is not in this subtree. Lets a single leaf's shadow-cost
    bisection recompute only its own ancestor chain (_recompute_path_utilities/_path_share
    below) instead of the whole tree on every probe.
    """
    if isinstance(node, str):
        return () if node == target else None
    name, lambda_key, children = node
    for c in children:
        sub = _leaf_ancestor_path(c, target)
        if sub is not None:
            return (node,) + sub
    return None


def _recompute_path_utilities(path, leaf, V_leaf, nest_lambdas, node_utils_cache):
    """
    Recompute bottom-up utility along `path` (ancestors of `leaf`, root first, from
    _leaf_ancestor_path) given a trial utility V_leaf for `leaf`, reusing node_utils_cache for
    every sibling not on the path -- valid since a sibling subtree's utility doesn't depend on
    leaf's own mu. Returns a small dict {leaf: V_leaf, id(ancestor): new_utility, ...} WITHOUT
    mutating node_utils_cache -- the caller decides whether to merge it in (a real commit, once
    a leaf's shadow cost is finalised) or discard it (a bisection probe).
    """
    out = {leaf: V_leaf}
    child_val, child_key = V_leaf, leaf
    for node in reversed(path):
        _, lambda_key, children = node
        lam = 1.0 if lambda_key is None else nest_lambdas[lambda_key]
        scaled = []
        for c in children:
            key = c if isinstance(c, str) else id(c)
            u = child_val if key == child_key else node_utils_cache[key]
            scaled.append(u / lam)
        m = max(scaled)
        u_node = lam * (m + math.log(sum(math.exp(s - m) for s in scaled)))
        node_key = id(node)
        out[node_key] = u_node
        child_val, child_key = u_node, node_key
    return out


def _path_share(path, leaf, fresh_utils, node_utils_cache, nest_lambdas):
    """
    Top-down probability of `leaf`, using `fresh_utils` (from _recompute_path_utilities) for
    nodes on `path` plus `leaf` itself, and node_utils_cache for every sibling encountered along
    the way. Only ever touches `path`'s nodes and their immediate children -- never recurses
    into a sibling's own subtree, since a sibling's already-cached aggregate utility is all
    that's needed to normalise the softmax at each level.
    """
    p = 1.0
    for depth, node in enumerate(path):
        _, lambda_key, children = node
        lam = 1.0 if lambda_key is None else nest_lambdas[lambda_key]
        next_key = id(path[depth + 1]) if depth + 1 < len(path) else leaf
        scaled, target_scaled = [], None
        for c in children:
            key = c if isinstance(c, str) else id(c)
            u = fresh_utils[key] if key in fresh_utils else node_utils_cache[key]
            s = u / lam
            scaled.append(s)
            if key == next_key:
                target_scaled = s
        m = max(scaled)
        p *= math.exp(target_scaled - m) / sum(math.exp(s - m) for s in scaled)
    return p


def _shadow_price_shares(npv, caps, price_lambda, nest_lambdas, mu0=None, max_sweeps=2000, max_bisect=30, tol=1e-5):
    """
    Solve for a shadow cost mu_p >= 0 per powertrain p such that the resulting nested-logit
    share of p never exceeds caps[p], with complementary slackness (mu_p > 0 only where the
    cap binds exactly): mu_p >= 0, share_p(mu) <= caps[p], mu_p * (caps[p] - share_p(mu)) == 0.

    Shares are computed via NEST_TREE (module-level): a McFadden nested logit with utility
    U_n = lambda_n * ln(sum_c exp(U_c / lambda_n)) feeding each node up to its parent (leaf
    utility V_p = price_lambda * (npv_p - mu_p) at the bottom), and P(c|n) = exp(U_c/lambda_n)
    / sum exp(U_c'/lambda_n) distributing probability back down. The root's scale is fixed at
    1.0 (nothing sits above it to rescale against). This convention -- scaling the log-sum by
    lambda_n before handing it to the parent, rather than passing a raw un-scaled inclusive
    value up -- is what makes an all-lambda=1 tree collapse to exactly the flat softmax at any
    depth (verified by hand and by verification/test_limits.py's
    test_all_nest_lambdas_one_matches_flat_mnl): with every lambda=1, U_n = ln(sum_c exp(U_c))
    so exp(U_n) = sum_c exp(U_c) exactly, which is precisely how a flat-MNL denominator builds
    up recursively regardless of tree depth. A singleton nest (e.g. Electric = {be} alone) needs
    no special case either -- U_n = lambda_n * ln(exp(V_be/lambda_n)) = V_be algebraically for
    any lambda_n, so P(be|Electric) = 1 regardless of lambda_n's value.

    Replaces post-hoc clipping: a capped powertrain stays in the choice set at a discounted
    utility (price_lambda * (npv_p - mu_p)) instead of being removed with its excess demand
    redistributed among the survivors. Removing it instead of discounting it would leak a wrong
    (inflated) inclusive value up through its nest, distorting every sibling nest's share --
    this is the actual reason shadow pricing exists once nesting is in play, even though under a
    single flat logit (no nesting) clip-and-rerun and shadow pricing are provably equivalent by
    IIA.

    Solved via Gauss-Seidel sweeps: one powertrain's shadow cost at a time, by bisection,
    holding every other powertrain's shadow cost fixed at its current value, cycling through
    all powertrains repeatedly until complementary slackness holds everywhere to within a
    relative tolerance of tol (default 1e-5 -- 10x tighter than verification/snapshot.py's own
    RTOL=1e-4 materiality threshold, so any leftover solver imprecision stays invisible to what
    the project already treats as "a real change"; convergence doesn't need to be exact, just
    tighter than what anything downstream can detect) -- every uncapped (mu == 0) leaf's share
    must not exceed its cap, and every capped (mu > 0) leaf's share must equal its cap, not
    merely stay under it.
    Checking only feasibility (share <= cap) is not enough: with a warm-started mu (see below),
    an early leaf in the sweep order can be judged against OTHER leaves' still-stale values and
    end up with an unwarranted mu > 0; once those other leaves are corrected later in the same
    sweep, the wrongly-capped leaf's share settles comfortably under its own cap, which looks
    feasible even though it should have relaxed back to mu == 0. Checking tightness for every
    currently-capped leaf, not just feasibility, forces another sweep instead of accepting that
    inconsistent state (an earlier version of this function checked feasibility only and could
    converge to exactly this kind of self-consistent-but-wrong point after a bad warm start).
    Bisection on a single
    coordinate (all else fixed) is used rather than a joint multi-dimensional Newton/fixed-
    point step, because share_p(mu_p), holding everything else fixed, is monotonically
    non-increasing in mu_p -- bisection can never overshoot or oscillate regardless of how
    close a share is to saturating at 0 or 1. This still holds under nesting: raising mu_p
    strictly lowers leaf p's own utility, which strictly lowers its within-nest conditional
    share and non-decreasingly lowers its nest's inclusive value relative to sibling nests: a
    product of factors each non-decreasing (one strictly increasing) in that utility cannot
    increase as the utility falls, at any tree depth. (A joint Newton step, by contrast, uses
    the local slope to extrapolate a step and badly overshoots once a share is near-saturated;
    when several powertrains in the same call are simultaneously and severely capped -- e.g.
    every nascent ZEV powertrain at once in an early year under a strong policy scenario --
    this caused a genuine oscillation, not just slow convergence, in an earlier version of
    this function.) This mirrors the ZEV-mandate credit-price search in Fleet._run, which
    bisects for the same reason: a monotonic sign-only search, not a magnitude-based step, is
    what makes it robust to how steep or saturated the underlying curve is.

    A cap of exactly 0 has no finite mu that achieves it -- handled by permanently excluding
    that powertrain from the choice set for this call. A single remaining (non-zero-cap)
    powertrain has no lever either: with nothing left to redistribute demand to, its share is
    mechanically 1 regardless of its own cap, so bisection is skipped entirely in both cases.

    npv, caps: dict powertrain -> value, same keys. nest_lambdas: dict nest name -> scale
    (params.py: fleet.nest_lambdas), looked up via NEST_TREE's lambda_key at each node. mu0:
    optional dict powertrain -> initial shadow cost, e.g. the converged result from a previous
    call with slightly different npv (the ZEV-mandate credit-price bisection in Fleet._run
    calls this ~30 times per year with only a small shift in npv each time as the credit price
    changes, so warm-starting from the last converged mu turns most sweeps into a 0-1 iteration
    confirmation instead of a search from scratch -- this changes nothing about the answer,
    since the fixed point solved for doesn't depend on the starting guess, only how many sweeps
    it takes to reach it). Returns (shares, mu): dict powertrain -> final share (sums to 1), and
    dict powertrain -> converged shadow cost (for warm-starting a subsequent call).
    """
    powertrains = list(npv)
    cap_arr  = np.array([caps[p] for p in powertrains])
    zero_cap = cap_arr <= 0.0
    if mu0 is None:
        mu = np.zeros(len(powertrains))
    else:
        mu = np.array([max(0.0, mu0.get(p, 0.0)) for p in powertrains])
        mu[zero_cap] = 0.0

    # Prune NEST_TREE down to whatever leaves are actually active for this call, once (zero_cap
    # is fixed for the whole call). Names not present in NEST_TREE (e.g. synthetic test names)
    # fall back to being bare root-level leaves -- since the root's own scale is fixed at 1.0,
    # this reduces to exactly flat MNL among them, matching pre-nesting behavior for callers
    # that don't use real powertrain names.
    active_leaves = {powertrains[i] for i in range(len(powertrains)) if not zero_cap[i]}
    known_active  = active_leaves & _NEST_TREE_LEAVES
    pruned_known  = _prune_tree(NEST_TREE, known_active)
    extra_leaves  = tuple(sorted(active_leaves - _NEST_TREE_LEAVES))
    root_children = (pruned_known[2] if pruned_known else ()) + extra_leaves
    pruned_tree   = ('root', None, root_children) if root_children else None

    def shares_at(mu_vec):
        if pruned_tree is None:
            return np.zeros(len(powertrains))
        V = {p: price_lambda * (npv[p] - m) for p, m in zip(powertrains, mu_vec)}
        node_utils = {}
        _bottom_up_utility(pruned_tree, V, nest_lambdas, node_utils)
        shares = {}
        _top_down_shares(pruned_tree, 1.0, nest_lambdas, node_utils, shares)
        return np.array([shares.get(p, 0.0) for p in powertrains])

    active_idx = [i for i in range(len(powertrains)) if not zero_cap[i]]
    if len(active_idx) > 1:
        cap_total = float(cap_arr[active_idx].sum())
        if cap_total < 1.0 - 1e-6:
            raise ValueError(
                f"_shadow_price_shares: infeasible cap set -- active caps sum to "
                f"{cap_total:.4f} < 1.0 for powertrains "
                f"{[powertrains[i] for i in active_idx]}; no shadow cost can make shares sum "
                f"to 1 while respecting every cap (check exclude_powertrains / "
                f"init_market_limit for this combination). Failing fast rather than burning "
                f"max_sweeps and returning a cap-violating result."
            )

        # Incremental bisection: a probe for leaf i used to call shares_at() -- a FULL tree
        # evaluation -- on every single bisection step, including every OTHER leaf's nest,
        # unaffected by leaf i's own mu. node_utils_cache holds every node's bottom-up utility,
        # kept consistent with the CURRENT mu at all times (every leaf commits into it below,
        # every sweep); a probe for leaf i only recomputes i's own ancestor chain
        # (_leaf_ancestor_path/_recompute_path_utilities/_path_share), reusing the cache for
        # every sibling instead of recomputing it. Measured (cProfile, 2026-07) to be the
        # majority of remaining cost after the plain-Python log-sum-exp rewrite above, since one
        # leaf's bisection makes up to max_bisect probes and this whole function is warm-started
        # across ~30 ZEV-mandate calls/year -- see "Market-share allocation architecture" in
        # CLAUDE.md.
        leaf_path = {powertrains[i]: _leaf_ancestor_path(pruned_tree, powertrains[i]) for i in active_idx}
        node_utils_cache = {}
        V0 = {p: price_lambda * (npv[p] - m) for p, m in zip(powertrains, mu)}
        _bottom_up_utility(pruned_tree, V0, nest_lambdas, node_utils_cache)

        for _ in range(max_sweeps):
            for i in active_idx:
                p, path = powertrains[i], leaf_path[powertrains[i]]

                def probe(mu_p, p=p, path=path):
                    V_leaf = price_lambda * (npv[p] - mu_p)
                    fresh = _recompute_path_utilities(path, p, V_leaf, nest_lambdas, node_utils_cache)
                    return fresh, _path_share(path, p, fresh, node_utils_cache, nest_lambdas)

                fresh0, s0 = probe(0.0)
                if s0 <= cap_arr[i]:
                    mu[i] = 0.0
                    node_utils_cache.update(fresh0)
                    continue
                # Warm-start the bracket from this leaf's own previous-sweep value instead of
                # always redoubling from 1.0 -- once a leaf's shadow cost is roughly right,
                # later sweeps only need to nudge it, not rediscover its whole magnitude.
                lo, hi = 0.0, mu[i] if mu[i] > 0.0 else 1.0
                fresh_hi, s_hi = probe(hi)
                while s_hi > cap_arr[i] and hi < 1e18:
                    hi *= 2.0
                    fresh_hi, s_hi = probe(hi)
                for _ in range(max_bisect):
                    mid = 0.5 * (lo + hi)
                    fresh_mid, s_mid = probe(mid)
                    if s_mid > cap_arr[i]:
                        lo = mid
                    else:
                        hi, fresh_hi = mid, fresh_mid
                mu[i] = hi
                node_utils_cache.update(fresh_hi)

            shares_dict = {}
            _top_down_shares(pruned_tree, 1.0, nest_lambdas, node_utils_cache, shares_dict)
            cur = np.array([shares_dict.get(p, 0.0) for p in powertrains])
            # Complementary slackness, checked properly: an UNCAPPED leaf (mu == 0) just needs
            # feasibility (share <= cap); a CAPPED leaf (mu > 0) must be tight -- its share must
            # equal its cap, not merely stay under it. Checking only feasibility here would miss
            # a leaf that was wrongly capped this sweep from another leaf's stale, not-yet-
            # updated warm-started value (a real failure mode: leaf A's stale mu makes leaf B
            # look artificially over its cap when B is processed early in the sweep, so B picks
            # up an unwarranted mu > 0; once A is corrected later in the sweep B's share settles
            # comfortably under its own cap, which looks "fine" under a feasibility-only check
            # even though B should have relaxed back to mu == 0 -- checking tightness for every
            # capped leaf forces another sweep instead of accepting that inconsistent state).
            max_gap = 0.0
            for i in active_idx:
                if mu[i] > 0.0:
                    gap = abs(cur[i] - cap_arr[i]) / cap_arr[i]
                else:
                    gap = max(0.0, cur[i] - cap_arr[i]) / cap_arr[i]
                max_gap = max(max_gap, gap)
            if max_gap < tol:
                break
        else:
            raise RuntimeError(
                f"_shadow_price_shares did not converge after {max_sweeps} sweeps "
                f"(max cap/complementary-slackness gap {max_gap:.2e}). The cap set was already "
                f"checked feasible (caps sum to >= 1), so this is genuine numerical "
                f"non-convergence, not infeasibility -- investigate rather than letting a "
                f"cap-violating result silently propagate into the ZEV-mandate bisection or a "
                f"Monte Carlo run."
            )

        # node_utils_cache is fully consistent with the final mu (every active leaf committed
        # into it this sweep), so the final shares only need one more top-down pass -- no need
        # to redo the bottom-up work shares_at() would otherwise repeat.
        shares_dict = {}
        _top_down_shares(pruned_tree, 1.0, nest_lambdas, node_utils_cache, shares_dict)
        shares = np.array([shares_dict.get(p, 0.0) for p in powertrains])
    else:
        shares = shares_at(mu)

    total = float(shares.sum())
    if total < 1.0 - 1e-6:
        warnings.warn(
            f"_shadow_price_shares: all powertrains are zero-capped "
            f"({[p for p, c in zip(powertrains, cap_arr) if c <= 0.0]}); returned shares sum "
            f"to {total:.4f}, not 1.0."
        )
    return (
        {p: float(s)  for p, s  in zip(powertrains, shares)},
        {p: float(mp) for p, mp in zip(powertrains, mu)},
    )


class Fleet:
    def __init__(self, params, param_cps, policies=None, exclude_powertrains=(), foresight=0.0):
        """
        foresight -- 0..1, how much of the future fuel-price / FC-replacement-cost
                     trajectory vehicles "know" at construction (see Vehicles docstring).
                     0 (default) = frozen at each cohort's own construction-year price for
                     its whole life, matching how the Paper 1 reference model (old/model_old.py)
                     was actually run in every reported scenario. 1 = perfect foresight of the
                     realized future trajectory. Carbon tax and LCFS always use the actual
                     future policy schedule regardless of this setting (announced/legislated,
                     not a market forecast); initial capital cost is always frozen at
                     construction year; revenue_per_tkm is already backward-looking by
                     construction and has no foresight to toggle.
        """
        self.params = copy.deepcopy(params)
        self.realise_uncertainties(param_cps)
        self.params = convert_to_float32(self.params)
        params = None

        self.K            = list(self.params['vehicles']['types'])
        self.P            = {k: [p for p in self.params['vehicles']['types'][k]['powertrains']
                                 if p not in exclude_powertrains]
                             for k in self.K}
        self.years        = np.arange(START_YEAR, END_YEAR + 1)
        self.price_lambda = float(self.params['fleet']['price_lambda'])
        self.nest_lambdas = {k: float(v) for k, v in self.params['fleet']['nest_lambdas'].items()}
        self.policies     = policies
        self.foresight    = float(foresight)

        # Activity requirement (t-km/year) by vehicle type and calendar year
        init_act   = float(self.params['fleet']['initial_activity'])
        act_growth = float(self.params['fleet']['activity_growth'])
        self.activity_req = {
            (k, t): init_act * (1 + act_growth) ** (t - START_YEAR)
                    * float(self.params['vehicles']['types'][k]['shared']['activity_proportion'])
            for k in self.K for t in self.years
        }

        self.stock          = {}  # (k, p, model_year, calendar_year) -> trucks
        self.vehicles       = {}  # (k, p, model_year) -> Vehicles
        self.market_share   = {}  # (k, p, calendar_year) -> fraction [0, 1]
        self._mu_warm_start = {}  # (k, p) -> last-converged shadow cost, warm-starts the next _calculate_market_share call
        self.penalty_history = {}  # calendar_year -> penalty / penalty_max (ZEV mandate)
        self.cost_per_tkm_history    = {}  # (k, calendar_year) -> realized system+policy cost per t-km
        self.revenue_per_tkm_history = {}  # (k, calendar_year) -> revenue_per_tkm applied that year

        self._build_initial_stock()
        self._run()
        self._aggregate()

    def _make_vehicle(self, k, p, t):
        params = self.select_vehicle_params(k, p, t)
        if self.policies:
            self.policies.pre_apply(params, k=k, p=p, t=t)
        v = Vehicles(params, self.params['fuels'], self.params['vehicles']['costs'], p=p, k=k,
                     foresight=self.foresight)
        if self.policies:
            self.policies.apply(v)
        return v

    def _apply_mandate_credit(self, t, credit_price, target, p_zev, k=None):
        """
        Write ZEV credit value into annual_cost['zev_mandate'][0] for all year-t vehicles
        in vehicle type k (or all k if k is None), then recompute NPV.

        Non-ZEVs always owe their own flat share of the target obligation, valued at the
        market credit_price:
            c_nonzev = credits_per_vehicle[k] * credit_price * target
        The government never pays out more than it collects. Total collected from non-ZEVs
        is the pool = c_nonzev * N_nonzev. ZEVs are paid the same flat market rate for their
        own credits (credits_per_vehicle[k] * credit_price) if the pool covers it; if ZEV
        supply is abundant enough that paying everyone in full would exceed the pool,
        payouts ration down proportionally so total payout never exceeds the pool:
            ration = min(1, pool / payout_if_full) = min(1, target * p_nonzev / p_zev)
            c_zev  = credits_per_vehicle[k] * credit_price * ration
        Net revenue (pool - c_zev * N_zev) is >= 0 at the bisection's converged fixed point
        (within its 1e-4 bracket-width tolerance -- ration is computed from the probe p_zev,
        not the realized stock split, so it can drift marginally negative mid-search): positive
        when ZEVs are undersupplied relative to target, ~0 once ZEV supply is abundant enough
        to exhaust the pool. Also assumes credits_per_vehicle is uniform across k when
        scope='fleet', since ration is computed once from the fleet-wide p_zev but applied with
        each k's own credits_per_vehicle.
        """
        _mandate   = self.policies.zev_mandate
        p_nonzev   = max(1.0 - p_zev, 1e-9)
        p_zev_safe = max(p_zev, 1e-9)
        ration     = min(1.0, target * p_nonzev / p_zev_safe)
        for k_ in ([k] if k is not None else self.K):
            cpv      = _mandate.credits_per_vehicle.get(k_, 0.0)
            c_nonzev = np.float32(cpv * credit_price * target)
            c_zev    = np.float32(cpv * credit_price * ration)
            for p in self.P[k_]:
                if (k_, p, t) not in self.vehicles:
                    continue
                v = self.vehicles[k_, p, t]
                v.annual_cost['zev_mandate'][0] = -c_zev if p in ZEV_POWERTRAINS else c_nonzev
                v._calculate_tco_npv()

    def _build_initial_stock(self):
        """
        Populate the pre-2025 diesel-only fleet so that cumulative activity at START_YEAR
        matches the exogenous activity_req.

        Sizing formula for cohort y (y < START_YEAR):
            stock[k, dice, y, START_YEAR] = activity_req[k, START_YEAR]
                                            x (1 + growth_rate)^(y - START_YEAR)
                                            x survival_rate[START_YEAR - y]
                                            / denom

        denom = sum_a  annual_distance[a] x payload[a]/1000 x survival_rate[a] x (1 + growth_rate)^(-a)
        is the survival-and-growth-weighted t-km per vehicle over a full MAX_AGE lifespan.
        Dividing by denom converts a total activity target into a number of vehicles per cohort.

        The oldest cohort (age MAX_AGE-1 at START_YEAR) is used as the denominator reference
        to stay consistent with the Paper 1 calibration.
        """
        for k in self.K:
            for y in range(START_YEAR - MAX_AGE, START_YEAR):
                self.vehicles[k, 'dice', y] = self._make_vehicle(k, 'dice', y)

            # Fixed reference vehicle (oldest vintage) for denominator -- matches old model
            v_ref   = self.vehicles[k, 'dice', START_YEAR - MAX_AGE]
            surv_r  = v_ref.params['survival_rate']
            denom   = sum(v_ref.annual_distance[a] * float(np.asarray(v_ref.mass['payload'])[a]) / 1000.0
                          * float(surv_r[a]) * (1 + GROWTH_RATE) ** (-a)
                          for a in range(MAX_AGE))

            for y in range(START_YEAR - MAX_AGE + 1, START_YEAR):
                v    = self.vehicles[k, 'dice', y]
                surv = v.params['survival_rate']
                self.stock[k, 'dice', y, START_YEAR] = np.float32(
                    self.activity_req[k, START_YEAR]
                    * (1 + GROWTH_RATE) ** (y - START_YEAR)
                    * float(surv[START_YEAR - y])
                    / denom
                )

    def _system_cost_per_tkm(self, year):
        """
        Realized system+policy cost per t-km for each vehicle type, from the fleet as it
        stands at `year` (all cohorts currently in self.stock/self.vehicles).

        total_cost sums all 8 categories in v.annual_cost (capital, operational, fuel,
        driver, fc_replacements, carbon_tax, lcfs, zev_mandate) rather than referencing
        COST_CATEGORIES directly, so it can't drift out of sync with what annual_cost
        actually contains.

        total_tkm reuses the stock x annual_distance x payload/1000 formula used for
        activity_met elsewhere in this file.

        Called at the top of each _run() iteration with year = t-1.  Not used for
        t == START_YEAR -- see _new_vehicle_cost_per_tkm for why.
        """
        result = {}
        for k in self.K:
            total_cost = 0.0
            total_tkm  = 0.0
            for p in self.P[k]:
                for y in range(year - MAX_AGE + 1, year + 1):
                    if (k, p, y) not in self.vehicles:
                        continue
                    n = self.stock.get((k, p, y, year), 0.0)
                    if n == 0.0:
                        continue
                    v = self.vehicles[k, p, y]
                    a = year - y
                    total_cost += n * sum(c[a] for c in v.annual_cost.values())
                    total_tkm  += n * v.annual_distance[a] * float(np.asarray(v.mass['payload'])[a]) / 1000.0
            result[k] = total_cost / max(total_tkm, 1e-9)
        return result

    def _new_vehicle_cost_per_tkm(self, k):
        """
        Survival-weighted lifetime cost per t-km for a brand-new diesel vehicle built at
        START_YEAR, used only as the revenue bootstrap for t == START_YEAR ("the cost
        for diesel").

        The pre-existing historical stock built by _build_initial_stock() only contains
        ages 1..MAX_AGE-1 at START_YEAR (the oldest cohort is a sizing reference, never
        added to self.stock) -- no cohort is ever at age 0 there, so it carries zero
        capital cost and understates the true cost of a new vehicle.  Every subsequent
        year's _system_cost_per_tkm() naturally includes that year's own new purchases
        (with capital cost) in its cross-sectional blend; using a throwaway brand-new
        vehicle here (same pattern as the LCFS baseline throwaway in __init__) is the
        closest match for year 1, where no such blend yet exists.

        Uses the full lifetime (all ages), weighted by survival_rate but NOT time-
        discounted, matching the undiscounted cash-basis convention of
        _system_cost_per_tkm -- just applied to a single (synthetic) vintage rather
        than a real cross-section of many.  Age 0 alone is not representative: this
        vehicle type's target_distance follows a logistic ramp-up with age (Statistics
        Canada CVS data), so a brand-new vehicle's first-year utilization is well below
        its lifetime average, which would inflate cost/tkm if age 0 were used alone.
        """
        v    = self._make_vehicle(k, 'dice', START_YEAR)
        surv = np.asarray(v.params['survival_rate'])
        total_cost = sum(float((c * surv).sum()) for c in v.annual_cost.values())
        total_tkm  = float((v.annual_distance * np.asarray(v.mass['payload']) / 1000.0 * surv).sum())
        return total_cost / max(total_tkm, 1e-9)

    def _run(self):
        """
        Year-by-year simulation START_YEAR -> END_YEAR.  Each year has three steps:

        0. Revenue update: freight revenue_per_tkm for year t is set to
           fleet.revenue_markup x the realized system+policy cost per t-km from year
           t-1 (_system_cost_per_tkm), or -- for t == START_YEAR, where no simulated
           prior year exists -- x a brand-new diesel vehicle's own age-0 cost per t-km
           (_new_vehicle_cost_per_tkm).  Mutates self.params so that
           select_vehicle_params() picks it up when Step 2 builds this year's vehicles.

        1. Roll-over: surviving vehicles from t-1 advance one year.  Stock is scaled by
           the conditional survival ratio  survival_rate[a] / survival_rate[a-1]  rather
           than the raw survival_rate so that each cohort declines at the correct marginal
           rate.

        2. Build new vehicles: a fresh Vehicles object is created for every (k, p, t)
           combination.  Parameters (prices, efficiencies, etc.) are time-sliced to year t
           by select_vehicle_params().

        3. Market share + new purchases: _calculate_market_share() allocates fractions
           across powertrains; then the shortfall between activity_req and activity_met
           by surviving cohorts is filled by new sales split according to those fractions.
           If a ZEV mandate is active, an outer loop iterates a ZEV credit price and the
           resulting ZEV sales share against each other (apply price -> new market share ->
           new share -> new price -> repeat, damped 0.5/0.5) on year-t vehicles until they
           reach a fixed-point market equilibrium -- economic pressure finding where the
           market settles, not a search forced to hit the target.  Production-cap-bound
           years converge the same way (the bracket still narrows: at high enough price
           the market response saturates below target, and bisection converges to that
           saturation point rather than target itself).  A warning is only emitted on
           true numerical non-convergence (30-iteration limit without the bracket
           narrowing below tolerance) -- solved via bisection on p_zev in [0, 1] rather
           than a damped linear blend, since credit_price(target, p_zev) is monotonically
           decreasing and the market's share response to price is monotonically
           non-decreasing, so their composition minus p_zev is monotonic and has at most
           one root: robust regardless of how steep the credit-price transition is,
           with no oscillation-detection heuristics needed.
        """
        mandate = self.policies.zev_mandate if self.policies else None

        for t in self.years:
            # --- Step 0: update revenue_per_tkm from the prior year's realized cost ---
            if t == START_YEAR:
                cost_per_tkm = {k: self._new_vehicle_cost_per_tkm(k) for k in self.K}
            else:
                cost_per_tkm = self._system_cost_per_tkm(t - 1)
            markup = float(self.params['fleet']['revenue_markup'])
            for k in self.K:
                revenue = np.float32(markup * cost_per_tkm[k])
                self.params['vehicles']['types'][k]['shared']['revenue_per_tkm'] = revenue
                self.cost_per_tkm_history[k, t]    = cost_per_tkm[k]
                self.revenue_per_tkm_history[k, t] = float(revenue)

            # --- Step 1: roll surviving cohorts from t-1 ---
            if t > START_YEAR:
                for k in self.K:
                    for p in self.P[k]:
                        for y in range(t - MAX_AGE + 1, t):
                            if (k, p, y) not in self.vehicles:
                                continue
                            prev = self.stock.get((k, p, y, t - 1), 0.0)
                            if prev == 0.0:
                                continue
                            surv = self.vehicles[k, p, y].params['survival_rate']
                            self.stock[k, p, y, t] = np.float32(prev * float(surv[t - y]) / max(float(surv[t - 1 - y]), 1e-9))

            # --- Step 2: build new vehicles for model year t ---
            for k in self.K:
                for p in self.P[k]:
                    self.vehicles[k, p, t] = self._make_vehicle(k, p, t)

            # --- Step 3: market share + mandate convergence ---
            if mandate and mandate.scope == 'per_k':
                for k in self.K:
                    target       = mandate.target_at(t, k=k)
                    active       = target > 1e-9
                    lo, hi       = 0.0, 1.0
                    p_probe      = 0.0
                    credit_price = 0.0

                    for n in range(30 if active else 1):
                        if active:
                            p_probe      = 0.5 * (lo + hi)
                            credit_price = mandate.credit_price(target, p_probe)
                            self._apply_mandate_credit(t, credit_price, target, p_probe, k=k)
                        self._calculate_market_share(k, t)
                        activity_met = sum(
                            self.stock.get((k, p, y, t), 0.0)
                            * self.vehicles[k, p, y].annual_distance[t - y]
                            * float(np.asarray(self.vehicles[k, p, y].mass['payload'])[t - y]) / 1000.0
                            for p in self.P[k] for y in range(t - MAX_AGE + 1, t)
                            if (k, p, y) in self.vehicles
                        )
                        avg_activity = sum(
                            self.vehicles[k, p, t].annual_distance[0]
                            * float(np.asarray(self.vehicles[k, p, t].mass['payload'])[0]) / 1000.0
                            * self.market_share[k, p, t]
                            for p in self.P[k]
                        )
                        new_sales = max((self.activity_req[k, t] - activity_met) / max(avg_activity, 1.0), 0.0)
                        for p in self.P[k]:
                            self.stock[k, p, t, t] = np.float32(new_sales * self.market_share[k, p, t])

                        if not active:
                            break

                        zev_k    = sum(float(self.stock.get((k, p, t, t), 0.0)) for p in self.P[k] if p in ZEV_POWERTRAINS)
                        total_k  = sum(float(self.stock.get((k, p, t, t), 0.0)) for p in self.P[k])
                        observed = zev_k / max(total_k, 1e-9)

                        if observed > p_probe:
                            lo = p_probe
                        else:
                            hi = p_probe
                        if hi - lo < 1e-4:
                            break  # bracket converged -- market equilibrium (production cap binds if p_zev < target)
                    else:
                        if active:
                            warnings.warn(
                                f"ZEV mandate (per_k={k!r}) did not converge at year {t}: "
                                f"bracket=[{lo:.4f}, {hi:.4f}], target={target:.3f}. "
                                f"Bisection did not narrow below tolerance after 30 iterations."
                            )

            else:
                # Fleet-wide convergence (or no mandate)
                target       = mandate.target_at(t) if mandate else 0.0
                active       = target > 1e-9
                lo, hi       = 0.0, 1.0
                p_probe      = 0.0
                credit_price = 0.0

                for n in range(30 if active else 1):
                    if active:
                        p_probe      = 0.5 * (lo + hi)
                        credit_price = mandate.credit_price(target, p_probe)
                        self._apply_mandate_credit(t, credit_price, target, p_probe)

                    for k in self.K:
                        self._calculate_market_share(k, t)
                        activity_met = sum(
                            self.stock.get((k, p, y, t), 0.0)
                            * self.vehicles[k, p, y].annual_distance[t - y]
                            * float(np.asarray(self.vehicles[k, p, y].mass['payload'])[t - y]) / 1000.0
                            for p in self.P[k] for y in range(t - MAX_AGE + 1, t)
                            if (k, p, y) in self.vehicles
                        )
                        avg_activity = sum(
                            self.vehicles[k, p, t].annual_distance[0]
                            * float(np.asarray(self.vehicles[k, p, t].mass['payload'])[0]) / 1000.0
                            * self.market_share[k, p, t]
                            for p in self.P[k]
                        )
                        new_sales = max((self.activity_req[k, t] - activity_met) / max(avg_activity, 1.0), 0.0)
                        for p in self.P[k]:
                            self.stock[k, p, t, t] = np.float32(new_sales * self.market_share[k, p, t])

                    if not active:
                        break

                    zev_s    = sum(float(self.stock.get((k, p, t, t), 0.0))
                                   for k in self.K for p in self.P[k] if p in ZEV_POWERTRAINS)
                    total_s  = sum(float(self.stock.get((k, p, t, t), 0.0))
                                   for k in self.K for p in self.P[k])
                    observed = zev_s / max(total_s, 1e-9)

                    if observed > p_probe:
                        lo = p_probe
                    else:
                        hi = p_probe
                    if hi - lo < 1e-4:
                        break  # bracket converged -- market equilibrium (production cap binds if p_zev < target)
                else:
                    if active:
                        warnings.warn(
                            f"ZEV mandate (fleet) did not converge at year {t}: "
                            f"bracket=[{lo:.4f}, {hi:.4f}], target={target:.3f}. "
                            f"Bisection did not narrow below tolerance after 30 iterations."
                        )

                if mandate:
                    self.penalty_history[t] = credit_price / mandate.penalty_max

    def _calculate_market_share(self, k, t):
        """
        Nested logit (see NEST_TREE, module-level) with production caps enforced via shadow
        pricing.

        Leaf utility for powertrain p: V(p) = lam x NPV(p), where lam = price_lambda (controls
        sensitivity to NPV differences; higher lam -> winner-takes-all, lam -> 0 -> uniform
        shares within a nest). Nests (Liquid/Conventional/Hydrogen/Electric) aggregate leaf
        utilities via their own scale parameters (self.nest_lambdas, from params.py:
        fleet.nest_lambdas) before the resulting nest choice is itself made via the same
        exp/sum-exp mechanics one level up -- see _shadow_price_shares's docstring for the exact
        McFadden convention and why it reduces to flat MNL when every nest_lambdas value is 1.0.

        Production cap: nascent technologies cannot grow faster than their supply chain allows.
        _market_share_limit() returns the maximum achievable share given last year's share and
        cagr_nacent / cagr_mature parameters, exactly as before. Rather than clipping any
        powertrain that would exceed its cap and re-running the logit over the remaining
        powertrains (the previous mechanism), a shadow cost mu_p >= 0 is solved for every
        powertrain jointly (see _shadow_price_shares) so that the resulting logit share never
        exceeds its cap, with complementary slackness (mu_p > 0 only where the cap binds
        exactly). Diesel is not a special-cased "unconstrained" powertrain -- its cap is simply
        never binding in practice given its init_market_limit, so mu_dice always solves to 0
        via the same mechanism as every other powertrain.

        Warm-started from the last shadow cost found for each (k, p) -- this matters because
        the ZEV-mandate credit-price bisection in Fleet._run calls this method up to ~30 times
        per year with only a small shift in npv each time, so most of those calls now resolve
        in a handful of sweeps instead of searching from mu=0 every time. This changes nothing
        about the result, only how quickly _shadow_price_shares reaches it.
        """
        npv  = {p: self.vehicles[k, p, t].npv for p in self.P[k]}
        caps = {}
        for p in self.P[k]:
            vp      = self.vehicles[k, p, t].params
            prev    = self.market_share.get((k, p, t - 1), 1.0 if p == 'dice' else 0.0)
            caps[p] = _market_share_limit(prev, float(vp['init_market_limit']), float(vp['cagr_nacent']), float(vp['cagr_mature']))

        mu0 = {p: self._mu_warm_start.get((k, p), 0.0) for p in self.P[k]}
        shares, mu = _shadow_price_shares(npv, caps, self.price_lambda, self.nest_lambdas, mu0=mu0)
        for p in self.P[k]:
            self.market_share[k, p, t] = shares[p]
            self._mu_warm_start[k, p]  = mu[p]

    def _aggregate(self):
        """
        Sum stock, fuel, emissions, and system costs across all cohorts for each calendar year.

        Capital costs are accounted at the point of sale (year t = model year y), not amortised,
        to be consistent with how annual_cost['capital'] is structured in Vehicles.
        Operational, fuel, and driver costs are summed age-by-age across all surviving cohorts.
        Emissions (embodied/supply/use) follow the same cohort-age loop.
        """
        T = self.years

        self.total_stock = {
            (k, p, t): np.float32(sum(self.stock.get((k, p, y, t), 0.0) for y in range(t - MAX_AGE + 1, t + 1)))
            for k in self.K for p in self.P[k] for t in T
        }
        self.sales = {
            (k, p, t): self.stock.get((k, p, t, t), np.float32(0))
            for k in self.K for p in self.P[k] for t in T
        }

        # Fuel usage [units/year]
        self.fuel_usage = {}
        # Fleet emissions [kgCO2e/year]
        self.emissions = {k: {'embodied': np.zeros(len(T)), 'supply': np.zeros(len(T)), 'use': np.zeros(len(T))} for k in self.K}
        # System costs [$/year]
        _all_costs   = COST_CATEGORIES['system'] + COST_CATEGORIES['policy']
        _flow_costs  = tuple(c for c in _all_costs if c != 'capital')
        self.system_costs = {k: {c: np.zeros(len(T)) for c in _all_costs} for k in self.K}

        for k in self.K:
            for p in self.P[k]:
                for y in range(START_YEAR - MAX_AGE + 1, END_YEAR + 1):
                    if (k, p, y) not in self.vehicles:
                        continue
                    v = self.vehicles[k, p, y]
                    for i, t in enumerate(T):
                        a = t - y
                        n = self.stock.get((k, p, y, t), 0.0)
                        if not (0 <= a < MAX_AGE) or n == 0.0:
                            continue
                        for f, af in v.annual_fuel.items():
                            key = (k, f, t)
                            self.fuel_usage[key] = self.fuel_usage.get(key, np.float32(0)) + np.float32(n * af[a])
                        self.emissions[k]['embodied'][i] += n * v.embodied[a]
                        self.emissions[k]['supply'][i]   += n * v.emissions_supply[a]
                        self.emissions[k]['use'][i]      += n * v.emissions_use[a]
                        for c in _flow_costs:
                            self.system_costs[k][c][i] += n * v.annual_cost[c][a]
                    # Capital at point of sale
                    if START_YEAR <= y <= END_YEAR:
                        i = int(y - START_YEAR)
                        self.system_costs[k]['capital'][i] += self.stock.get((k, p, y, y), 0.0) * v.capital_total

    def select_vehicle_params(self, k, p, y):
        """
        Build the params dict for a single (vehicle type k, powertrain p, model year y) cohort.

        Three-step merge:
          1. Start with shared params for vehicle type k (base_cost, running_cost, payload, ...).
          2. Overlay powertrain-specific params (|= lets powertrain values win on conflicts).
          3. For each component referenced in the powertrain, fill in shared component specs
             from vehicles.components[type][comp_name] -- but only for keys not already set
             by the powertrain (so per-powertrain overrides are preserved).

        Then set_year() slices all time-varying arrays to scalar values at year y.
        Age-varying arrays (target_distance, drive_cycle, survival_rate) are excluded from
        set_year() because Vehicles uses them indexed by vehicle age, not calendar year.

        No deepcopy is needed because set_year() is non-mutating: it returns new dicts rather
        than modifying its input. Shallow dict() copies are sufficient to create fresh container
        structures; age-varying numpy arrays are shared references (read-only in Vehicles).
        """
        vehicle_params  = dict(self.params['vehicles']['types'][k]['shared'])
        vehicle_params |= dict(self.params['vehicles']['types'][k]['powertrains'][p])
        components = {}
        for comp_name, comp in vehicle_params['components'].items():
            comp_copy  = dict(comp)
            shared_def = self.params['vehicles']['components'][comp['type']][comp_name]
            if isinstance(shared_def, dict) and any(kk in ALL_POWERTRAINS for kk in shared_def):
                shared_def = shared_def.get(p, {})
            comp_copy.update({kk: v for kk, v in shared_def.items() if kk not in comp_copy})
            components[comp_name] = comp_copy
        components['frame'] = {
            'type': 'frame',
            'mass': vehicle_params['frame_mass'],
            'embodied_emissions': self.params['vehicles']['components']['frame']['embodied_emissions'],
        }
        if vehicle_params.get('trailer_mass', 0) > 0:
            components['trailer'] = {
                'type': 'trailer',
                'mass': vehicle_params['trailer_mass'],
                'embodied_emissions': self.params['vehicles']['components']['trailer']['embodied_emissions'],
            }
        components['tire'] = {
            'type': 'tire',
            'mass': vehicle_params['tire_mass'],
            'embodied_emissions': self.params['vehicles']['components']['tire']['embodied_emissions'],
        }
        if vehicle_params.get('trailer_tire_mass', 0) > 0:
            components['trailer_tire'] = {
                'type': 'tire',
                'mass': vehicle_params['trailer_tire_mass'],
                'embodied_emissions': self.params['vehicles']['components']['tire']['embodied_emissions'],
            }
        vehicle_params['components'] = components
        vehicle_params['model_year'] = y
        exclude = {'target_distance', 'drive_cycle', 'survival_rate', 'average_speed', 'payload'}
        for key in list(vehicle_params.keys()):
            if key not in exclude:
                vehicle_params[key] = set_year(vehicle_params[key], year=y)
        return vehicle_params

    def realise_uncertainties(self, param_cps):
        """
        Apply a Monte Carlo draw to every uncertain parameter in the params tree.

        param_cps maps (key_path_tuple) -> cp in [0, 1].  Key paths are produced by
        get_uncertainty_distributions(), which walks the nested params dict and returns
        every leaf that has a 'dist' key.

        set_param() converts the cumulative probability cp to a realised value (scalar
        or time-series array depending on the distribution type).  Because a single cp
        is shared across all parameters in one MC run, correlated quantities -- e.g. the
        same component parameter used by multiple powertrains -- move together.
        """
        for keys, cp in param_cps.items():
            d = self.params
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = set_param(d[keys[-1]], cp=cp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    param_cps = {path: np.float32(0.5) for path, _ in get_uncertainty_distributions(PARAMS)}
    fleet = Fleet(PARAMS, param_cps)

    v = fleet.vehicles['sleeper', 'dice', START_YEAR]
    print('Total mass:               ', v.total_mass, 'kg')
    print('Fuel consumption (age 0): ', v.fuel_consumption['diesel'][0], 'L/km')
    print('Annual distance (age 0):  ', v.annual_distance[0], 'km')
    print('Capital total:            ', v.capital_total, '$')
    print('TCO:                      ', v.tco, '$')
    print('NPV:                      ', v.npv, '$')
    print()
    print('Market share (sleeper, 2025):', {p: round(fleet.market_share.get(('sleeper', p, START_YEAR), 0), 4) for p in fleet.P['sleeper']})
