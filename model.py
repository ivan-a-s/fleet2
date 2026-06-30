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
 Needs changing now:
 - Make sure I am happy with the accessory load and how efficiency is applied.
 - Adjust the activity requrements given the payload changes.

 Needs changing later:
 - Adjust the embodied emission factors for the powertrain components using GREET.
 - Make the git accessible but not all of it to everyone (clean without changing old code).
 - Policy net revenue plots.
 - Fast and slow charge could be merged and have a different way of
   doing the pricing. Sleeper PHEs could use non-depot slow charge.
 - Uncertainty analysis.

 Nice to have (in order of priority):
 - Add a resource-haul vehicle type.
 - Size vehicle components for NPV optimisation?
 - Scale factors to relate cost to scale somehow.
 - Altitude on FC/engine performance and air resistance.
 - Variance in use (logit change like NREL).
 - Make scrappage/usage decisions for vehicles?
 - Hotel load for sleepers and fridge units?

 Checked up to:
  - _calculate_fuel_consumption

"""
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

_SURROGATES = load_model_params('vehicle_modelling/surrogates.json')

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
    """

    def __init__(self, params, fuels, costs, p, k):
        self.params       = params
        self._all_fuels   = fuels                                    # full dict -- needed for en-route fast_charge pricing
        self.fuels        = {f: fuels[f] for f in params['fuels']}
        self.costs        = costs
        self.p            = p
        self.k            = k
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
        self.mass = {'frame': float(self.params['frame_mass'])}
        if float(self.params['trailer_mass']) > 0:
            self.mass['trailer'] = float(self.params['trailer_mass'])
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
            # Battery range degradation (1% per year of age + 0.01% per charge cycle)
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
                          Frame/trailer use the shared embodied_emissions (kgCO2e/kg).
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
        emb = float(p['embodied_emissions'])
        embodied_total = (
            float(p['frame_mass'])
            + float(p.get('trailer_mass', 0)) * float(p.get('trailers_per_truck', 0))
        ) * emb
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
        """Get cost array indexed to operation years (one value per age)."""
        val = self.costs.get(key)
        if val is None:
            return np.zeros(len(self.age))
        if isinstance(val, np.ndarray):
            idx = np.clip(self.operation_years - self._Y_start, 0, len(val) - 1)
            return val[idx].astype(float)
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
        if self.p == 'phe':
            _phe_bfuel = next((f for f in self.fuels if 'charge' in f), None)
            if _phe_bfuel and _phe_bfuel != 'fast_charge':
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
          fuel           -- fuel cost in source units x time-varying price.
                           For slow-charge BETs with en-route fast charging, km driven via
                           en-route stops are re-billed at fast-charge rates with a different
                           efficiency: grid_kWh = km x slow_fc x slow_eff / fast_eff.
          driver         -- wage proportional to target_distance (not actual), because the
                           driver is paid whether or not the vehicle is range-constrained.
          fc_replacements -- time-varying $/kW x stack capacity at each replacement age.

        Revenue is also computed here: annual_distance x payload_tonnes x revenue_per_tkm.
        """
        fuel_cost = np.zeros(len(self.age))
        for f, annual in self.annual_fuel.items():
            cost_arr = np.asarray(self.fuels[f]['cost'])
            idx      = np.clip(self.operation_years - self._Y_start, 0, len(cost_arr) - 1)
            fuel_cost += annual * cost_arr[idx]

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


class Fleet:
    def __init__(self, params, param_cps, policies=None, exclude_powertrains=()):
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
        self.policies     = policies

        # Activity requirement (t-km/year) by vehicle type and calendar year
        init_act   = float(self.params['fleet']['initial_activity'])
        act_growth = float(self.params['fleet']['activity_growth'])
        self.activity_req = {
            (k, t): init_act * (1 + act_growth) ** (t - START_YEAR)
                    * float(self.params['vehicles']['types'][k]['shared']['activity_proportion'])
            for k in self.K for t in self.years
        }

        self.stock        = {}  # (k, p, model_year, calendar_year) -> trucks
        self.vehicles     = {}  # (k, p, model_year) -> Vehicles
        self.market_share = {}  # (k, p, calendar_year) -> fraction [0, 1]

        self._build_initial_stock()
        if self.policies and self.policies.lcfs:
            for k in self.K:
                # Build a temporary diesel vehicle at START_YEAR solely to extract
                # baseline fuel consumption; not stored in self.vehicles.
                self.policies.lcfs.set_baseline_fc(k, self._make_vehicle(k, 'dice', START_YEAR))
        self._run()
        self._aggregate()

    def _make_vehicle(self, k, p, t):
        params = self.select_vehicle_params(k, p, t)
        if self.policies:
            self.policies.pre_apply(params, k=k, p=p, t=t)
        v = Vehicles(params, self.params['fuels'], self.params['vehicles']['costs'], p=p, k=k)
        if self.policies:
            self.policies.apply(v)
        return v

    def _apply_mandate_penalty(self, t, penalty, p_zev, k=None):
        """
        Write ZEV mandate penalty/rebate into annual_cost['zev_mandate'][0] for all
        year-t vehicles in vehicle type k (or all k if k is None), then recompute NPV.

        Non-ZEV powertrains incur `penalty` $/vehicle/yr (at age 0).
        ZEV powertrains receive a rebate scaled so that total payments balance:
            rebate = penalty x (1 - p_zev) / max(p_zev, eps)
        capped at `penalty` to avoid exploding rebates when p_zev is tiny.
        """
        rebate = min(penalty, penalty * max(1.0 - p_zev, 0.0) / max(p_zev, 1e-9))
        for k_ in ([k] if k is not None else self.K):
            for p in self.P[k_]:
                if (k_, p, t) not in self.vehicles:
                    continue
                v = self.vehicles[k_, p, t]
                v.annual_cost['zev_mandate'][0] = np.float32(
                    -rebate if p in ZEV_POWERTRAINS else penalty
                )
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

    def _run(self):
        """
        Year-by-year simulation START_YEAR -> END_YEAR.  Each year has three steps:

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
           If a ZEV mandate is active, an outer convergence loop adjusts a penalty/rebate
           on year-t vehicles until the ZEV share of new sales meets the target (or the
           30-iteration limit is reached, after which a warning is emitted and the last
           iterate is used).  Warm-start carries the converged penalty from the previous
           year to speed convergence.  Oscillation is detected after 5 iterations and
           dampened via bisection of the step.
        """
        mandate = self.policies.zev_mandate if self.policies else None

        # Warm-start: carry last year's converged penalty/ZEV share into next year
        if mandate and mandate.scope == 'per_k':
            warm_pen = {k: 0.0 for k in self.K}
            warm_pzv = {k: 0.0 for k in self.K}
        else:
            warm_pen = 0.0
            warm_pzv = 0.0

        for t in self.years:
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
                    target   = mandate.target_at(t, k=k)
                    active   = target > 1e-9
                    penalty  = warm_pen[k] if active else 0.0
                    p_zev    = warm_pzv[k]
                    prev_pen = 0.0

                    for n in range(30 if active else 1):
                        if active:
                            self._apply_mandate_penalty(t, penalty, p_zev, k=k)
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

                        zev_k   = sum(float(self.stock.get((k, p, t, t), 0.0)) for p in self.P[k] if p in ZEV_POWERTRAINS)
                        total_k = sum(float(self.stock.get((k, p, t, t), 0.0)) for p in self.P[k])
                        p_zev   = zev_k / max(total_k, 1e-9)
                        if p_zev >= target - 1e-3:
                            break

                        raw     = mandate.penalty_max * (target - p_zev) / max(1.0 - p_zev, 1e-9)
                        new_pen = min(0.3 * penalty + 0.7 * raw, mandate.penalty_max)
                        if n >= 5 and (new_pen - penalty) * (penalty - prev_pen) < 0:
                            new_pen = (penalty + new_pen) * 0.5
                        prev_pen = penalty
                        penalty  = new_pen
                        if abs(penalty - prev_pen) < 1.0:
                            break  # penalty converged; production cap is binding -- accept best achievable share
                    else:
                        warnings.warn(
                            f"ZEV mandate (per_k={k!r}) did not converge at year {t}: "
                            f"p_zev={p_zev:.3f} vs target={target:.3f}, "
                            f"penalty=${penalty:,.0f}. Penalty oscillating after 30 iterations."
                        )

                    warm_pen[k] = penalty if active else 0.0
                    warm_pzv[k] = p_zev   if active else 0.0

            else:
                # Fleet-wide convergence (or no mandate)
                target   = mandate.target_at(t) if mandate else 0.0
                active   = target > 1e-9
                penalty  = warm_pen if active else 0.0
                p_zev    = warm_pzv
                prev_pen = 0.0

                for n in range(30 if active else 1):
                    if active:
                        self._apply_mandate_penalty(t, penalty, p_zev)

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

                    zev_s   = sum(float(self.stock.get((k, p, t, t), 0.0))
                                  for k in self.K for p in self.P[k] if p in ZEV_POWERTRAINS)
                    total_s = sum(float(self.stock.get((k, p, t, t), 0.0))
                                  for k in self.K for p in self.P[k])
                    p_zev   = zev_s / max(total_s, 1e-9)
                    if p_zev >= target - 1e-3:
                        break

                    raw     = mandate.penalty_max * (target - p_zev) / max(1.0 - p_zev, 1e-9)
                    new_pen = min(0.3 * penalty + 0.7 * raw, mandate.penalty_max)
                    if n >= 5 and (new_pen - penalty) * (penalty - prev_pen) < 0:
                        new_pen = (penalty + new_pen) * 0.5
                    prev_pen = penalty
                    penalty  = new_pen
                    if abs(penalty - prev_pen) < 1.0:
                        break  # penalty converged; production cap is binding -- accept best achievable share
                else:
                    if active:
                        warnings.warn(
                            f"ZEV mandate (fleet) did not converge at year {t}: "
                            f"p_zev={p_zev:.3f} vs target={target:.3f}, "
                            f"penalty=${penalty:,.0f}. Penalty oscillating after 30 iterations."
                        )

                warm_pen = penalty if active else 0.0
                warm_pzv = p_zev   if active else 0.0

    def _calculate_market_share(self, k, t):
        """
        Multinomial logit with iterative production-cap enforcement.

        Unconstrained logit share for powertrain p:
            share(p) = exp(lam x NPV(p)) / sum_q exp(lam x NPV(q))

        where lam = price_lambda (controls sensitivity to NPV differences; higher lam -> winner-takes-
        all, lam -> 0 -> uniform shares).

        Production cap: nascent technologies cannot grow faster than their supply chain allows.
        _market_share_limit() returns the maximum achievable share given last year's share and
        cagr_nacent / cagr_mature parameters.  If a powertrain's unconstrained share exceeds its
        cap, its share is fixed at the cap and the remaining market is re-allocated by running
        another logit over the unconstrained powertrains.  This repeats until no new caps bind
        (up to 10 iterations, which is always sufficient in practice).
        """
        remaining     = set(self.P[k])
        mkt_remaining = 1.0
        for _ in range(10):
            n_prev = len(remaining)
            denom  = sum(np.exp(self.price_lambda * self.vehicles[k, p, t].npv) for p in remaining)
            for p in list(remaining):
                vp    = self.vehicles[k, p, t].params
                prev  = self.market_share.get((k, p, t - 1), 1.0 if p == 'dice' else 0.0)
                limit = _market_share_limit(prev, float(vp['init_market_limit']), float(vp['cagr_nacent']), float(vp['cagr_mature']))
                logit = mkt_remaining * np.exp(self.price_lambda * self.vehicles[k, p, t].npv) / denom
                if limit <= logit:
                    self.market_share[k, p, t] = limit
                    mkt_remaining -= limit
                    remaining -= {p}
                else:
                    self.market_share[k, p, t] = logit
            if len(remaining) == n_prev:
                break  # Converged -- no new production-limited powertrains

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
