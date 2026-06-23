"""
Multinomial logit fleet adoption model for heavy-duty trucks (HDT) in BC, 2025–2050.

Architecture
------------
Each Monte Carlo run draws one cumulative-probability vector (param_cps) and passes it to
Fleet(), which realises all uncertain parameters at once so shared components (e.g. ICE
efficiency) receive a single consistent draw.

The simulation proceeds in three layers:
  1. Vehicles  — one cohort object per (vehicle type k, powertrain p, model year y).
                 Computes mass, fuel consumption (FASTSim polynomial surrogate), range,
                 annual distance, emissions, capital cost, annual cost, TCO, NPV.
  2. Fleet     — year-by-year roll-over of surviving cohorts, new vehicle creation, and
                 market-share allocation via multinomial logit with production caps.
  3. Aggregate — totals over the stock for fuel use, emissions, and system costs.

To do:
 - Apply some checks (average distance, activity, etc)
 - Market share limit inside fleet.
 - Size vehicle components for NPV optimisation?
 - Altitude on FC/engine performance and air resistance.
 - Make scrappage/usage decisions for vehicles?

 Checked up to:
  - _calculate_fuel_consumption
"""
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
# hice/dhice reuse diesel-engine surrogates; all other powertrains match by name
SURROGATE_NAME = {
    'hice': 'dice',
    'dhice': 'he',
}
# Which component efficiency to pass as peak_eff to the surrogate
EFF_COMPONENT = {
    'dice': 'ice', 'he': 'ice',
    'be':   'motor', 'fc': 'fc',
    'hice': 'ice', 'dhice': 'ice',
}
ZEV_POWERTRAINS     = {'be', 'fc', 'hice'}
HICE_POWERTRAINS    = {'hice', 'dhice'}
CHARGER_POWERTRAINS = {'be'}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_uncertainty_distributions(d, current_path=()):
    """Walk the nested params dict and return all leaf nodes that contain a 'dist' key."""
    paths = []
    if isinstance(d, dict):
        if 'dist' in d:
            paths.append((current_path, d))
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
    Convert a cumulative probability cp ∈ [0,1] to a realised parameter value.
    'linear'  — linearly interpolated array over Y; start and end are themselves distributions.
    'interp'  — piecewise-linear over specified anchor years, realised at each cp.
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
    from this slicing — they vary by vehicle age, not calendar year.
    """
    if isinstance(input_dict, np.ndarray):
        return input_dict[np.where(years == year)[0][0]]
    elif isinstance(input_dict, dict):
        for key, value in input_dict.items():
            input_dict[key] = set_year(value, year, years)
    return input_dict


# ---------------------------------------------------------------------------
# Vehicles class
# ---------------------------------------------------------------------------

class Vehicles:
    """
    Represents a single vehicle cohort (vehicle type k, powertrain p, model year y).

    params — time-sliced vehicle params from Fleet.select_vehicle_params()
    fuels  — full (unsliced) fuel params from PARAMS['fuels'], covering all Y years
    costs  — full (unsliced) vehicle cost arrays from PARAMS['vehicles']['costs']
    p            — powertrain key string, e.g. 'dice', 'be', 'fc'
    k            — vehicle type key string, e.g. 'sleeper', 'day_cab', 'straight'
    """

    def __init__(self, params, fuels, costs, p, k):
        self.params       = params
        self._all_fuels   = fuels                                    # full dict — needed for en-route fast_charge pricing
        self.fuels        = {f: fuels[f] for f in params['fuels']}
        self.costs        = costs
        self.p            = p
        self.k            = k
        self.age          = np.arange(params['max_age'], dtype=int)
        self.model_year   = int(params['model_year'])
        self.operation_years = self.model_year + self.age
        self._Y_start     = START_YEAR - MAX_AGE   # first year in all realized arrays (e.g. 2000)

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
        # TODO: ZEVs receive a GVWL exemption under ZEV mandate policy — disabled until policies are added
        gvwl_increase = 0.0
        self.mass = {'frame': float(self.params['frame_mass'])}
        if float(self.params['trailer_mass']) > 0:
            self.mass['trailer'] = float(self.params['trailer_mass'])
        for comp_name, comp in self.params['components'].items():
            if comp['type'] == 'converter':
                self.mass[comp_name] = float(comp['mass'])
            elif comp['type'] == 'ess':
                # ESS mass = specific_mass (kg per unit capacity) × capacity
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
        # payload_frac is a scalar — it depends only on unloaded_mass vs the reference mass budget.
        # The per-age payload (from drive_cycles) is then scaled by this fraction.
        payload_frac = max(0.0, 1.0 - self.params['p_weighed_out'] * (
            1.0 - (self.params['gvwl'] + gvwl_increase - self.unloaded_mass)
                / (self.params['gvwl'] - self.params['default_unloaded_mass'])
        ))
        self.mass['payload'] = np.asarray(self.params['payload']) * payload_frac
        self.total_mass      = self.unloaded_mass + self.mass['payload']  # age-array

    # -- Fuel consumption ------------------------------------------------------

    def _calculate_fuel_consumption(self):
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
            # Reuses the HEV surrogate (diesel L/km total); split total energy into diesel
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

        Range per ESS = capacity × usable_fraction / fuel_consumption_rate  [km]
        where fuel_consumption is in source units/km (post efficiency correction).
        The binding range is the minimum across all ESS (e.g. diesel tank on a DHICE).

        self.refuel_rate is the effective en-route replenishment rate in the same units
        as fc × speed (energy/hr), used in the time-budget formula in _calculate_annual_distance.
        For H2 tanks it is the pump flow rate (kg/hr).
        For batteries it is the fast-charger wall power × efficiency ratio:
          refuel_rate (kW wall) × fast_eff / slow_eff
        The slow_eff division corrects for fuel_consumption being in wall-plug slow-charge
        basis rather than battery basis, keeping the units commensurable.
        """
        battery_fuel = next((f for f in self.fuels if 'charge' in f), None)
        ESS_FUEL = {
            'diesel_tank': 'diesel',
            'h2_700bar':   'h2',
            'h2_350bar':   'h2',
            'battery':     battery_fuel,
        }
        self.range       = np.full(len(self.age), 1e6)
        self.refuel_rate = 0.0

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
            r = float(comp['capacity']) * float(comp['usable_capacity']) / fc
            self.range = np.minimum(self.range, r)
            if comp_name == 'battery':
                # Battery refuel_rate is kW wall power; apply efficiency ratio to convert
                # to wall-plug slow-charge basis (commensurable with fuel_consumption units)
                rate_raw = float(comp['refuel_rate'])
                if rate_raw > 0 and battery_fuel:
                    fast_eff = float(self._all_fuels['fast_charge']['refuel_efficiency'])
                    slow_eff = float(self.fuels[battery_fuel]['refuel_efficiency'])
                    rate = rate_raw * fast_eff / slow_eff
                else:
                    rate = 0.0
            else:
                rate = float(comp['refuel_rate'])
            if rate > self.refuel_rate:
                self.refuel_rate = rate

    # -- Annual distance -------------------------------------------------------

    def _calculate_annual_distance(self):
        """
        Age-by-age loop: applies battery degradation and range-limited daily distance.

        target_distance (km/year) is converted to a daily working-day target:
          daily_target = annual_km / 365 × (7/5)
        i.e., trucks are assumed to work 5 days out of 7, so each working day covers
        more than 1/365 of the annual distance.

        When daily_target ≤ range, the vehicle drives the full target.  Otherwise it
        can extend its range with a refuelling/recharging stop using a time-budget formula:
          achievable = (time_left - 0.25h) × speed × R / (fc × speed + R)
        where time_left = shortfall / speed (hours that would have been spent driving),
        0.25h is the fixed stop overhead, and R = self.refuel_rate.  This is derived by
        solving simultaneously for stop time and extra distance given a fixed time budget.

        Battery degradation follows a linear capacity-fade model:
          effective_range[a] = range[a] × max(0, 1 − deg_per_year×a − deg_per_cycle×cycles)
        Cycle count accumulates from annual_distance × fuel_consumption / battery_capacity.

        self._enroute_distance tracks km driven via en-route fast charging (non-zero only
        for slow-charge BETs when range < target), used in _calculate_annual_cost to apply
        fast-charge pricing to that portion of electricity consumption.

        Annual distance is converted back from working-day to calendar-year basis:
          annual_km = daily_km × (5/7) × 365
        """
        daily_target = self.params['target_distance'] / 365.0 * 7.0 / 5.0

        battery_comp    = self.params['components'].get('battery')
        battery_fuel    = next((f for f in self.fuels if 'charge' in f), None) if battery_comp else None
        battery_cap     = float(battery_comp['capacity'])      if battery_comp else 0.0
        deg_per_year    = float(battery_comp['deg_per_year'])  if battery_comp else 0.0
        deg_per_cycle   = float(battery_comp['deg_per_cycle']) if battery_comp else 0.0

        primary_fuel = next(iter(self.fuel_consumption))
        range_ = self.range.copy()
        self.annual_distance   = np.zeros(len(self.age))
        self._enroute_distance = np.zeros(len(self.age))
        cycles = 0.0

        for a in self.age:
            # Battery range degradation (1% per year of age + 0.01% per charge cycle)
            if battery_fuel and battery_fuel in self.fuel_consumption:
                range_[a] = self.range[a] * max(0.0, 1.0 - deg_per_year * a - deg_per_cycle * cycles)

            if daily_target[a] <= range_[a]:
                daily      = daily_target[a]
                achievable = 0.0
            else:
                # Estimate extra distance achievable during a refuelling/recharging stop
                shortfall = daily_target[a] - range_[a]
                time_left = shortfall / max(self.average_speed[a], 1.0)
                fc_a = self.fuel_consumption[primary_fuel][a]
                if self.refuel_rate > 0 and fc_a > 0:
                    # Time budget formula: (available_time - 0.25h overhead) × achievable rate
                    achievable = max(0.0,
                        (time_left - 0.25) * self.average_speed[a]
                        * self.refuel_rate / (fc_a * self.average_speed[a] + self.refuel_rate)
                    )
                else:
                    achievable = 0.0
                daily = range_[a] + achievable

            self.annual_distance[a]   = daily      * 5.0 / 7.0 * 365.0
            self._enroute_distance[a] = achievable * 5.0 / 7.0 * 365.0

            if battery_fuel and battery_cap > 0:
                cycles += (self.annual_distance[a]
                           * self.fuel_consumption[battery_fuel][a] / battery_cap)

        self.range = range_  # save degraded-by-age range back to attribute
        self.annual_fuel = {
            f: self.annual_distance * self.fuel_consumption[f]
            for f in self.fuel_consumption
        }

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
          embodied      — manufacturing emissions, assigned entirely to age 0 (kgCO2e).
                          Frame/trailer use a $/kg embodied factor; batteries add a
                          separate kgCO2e/kWh term for their manufacture.
          emissions_supply — upstream (well-to-tank) emissions per year (kgCO2e/yr).
          emissions_use    — tailpipe (tank-to-wheel) emissions per year (kgCO2e/yr).
                             Zero for ZEVs (no combustion at point of use).
        Supply and use emissions scale with annual_fuel, already in source units/km so
        the emissions intensities (kgCO2e per source unit) apply directly.
        """
        p = self.params
        emb = float(p['embodied'])
        embodied_total = (
            float(p['frame_mass'])
            + float(p.get('trailer_mass', 0)) * float(p.get('trailers_per_truck', 0))
        ) * emb
        for comp_name, comp in p['components'].items():
            if comp['type'] == 'ess' and 'embodied_emissions' in comp:
                embodied_total += float(comp['capacity']) * float(comp['embodied_emissions'])
        self.embodied = np.concatenate([[embodied_total], np.zeros(len(self.age) - 1)])

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
          engine     — $/unit: hice/dhice use the hydrogen-engine price, all others diesel engine
          motor      — $/kW × motor capacity
          battery    — $/kWh × battery capacity (also includes ESS for phe/he)
          h2_tank    — $/kg × tank capacity
          fc         — $/kW × fuel-cell capacity
          tank       — $/L × diesel tank capacity
          charger    — depots only (not sleeper — sleeper trucks use en-route charging)
          after_treatment — NOx catalyst; present in data.json for dice, he, phe, dhice

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
        self.capital       = c
        self.capital_total = sum(c.values())

    # -- Annual cost & TCO -----------------------------------------------------

    def _calculate_annual_cost(self):
        """
        Five cost components, all in $/year as age arrays:

          capital        — full purchase price at age 0, zero thereafter.  Placed here
                           (rather than amortised) so _discount() gives the correct NPV.
          operational    — maintenance, tyres, etc. proportional to km driven (actual distance).
          fuel           — fuel cost in source units × time-varying price.
                           For slow-charge BETs with en-route fast charging, km driven via
                           en-route stops are re-billed at fast-charge rates with a different
                           efficiency: grid_kWh = km × slow_fc × slow_eff / fast_eff.
          driver         — wage proportional to target_distance (not actual), because the
                           driver is paid whether or not the vehicle is range-constrained.
          fc_replacements — time-varying $/kW × stack capacity at each replacement age.

        Revenue is also computed here: annual_distance × payload_tonnes × revenue_per_tkm.
        """
        fuel_cost    = np.zeros(len(self.age))
        battery_fuel = next((f for f in self.fuels if 'charge' in f), None)
        for f, annual in self.annual_fuel.items():
            cost_arr = np.asarray(self.fuels[f]['cost'])
            idx      = np.clip(self.operation_years - self._Y_start, 0, len(cost_arr) - 1)
            if f == 'slow_charge' and f == battery_fuel and np.any(self._enroute_distance > 0):
                # En-route km are fast-charged: separate pricing and efficiency
                slow_eff  = float(self.fuels[f].get('refuel_efficiency', 1.0))
                fc_per_km = self.fuel_consumption[f]               # wall-plug kWh/km (slow basis)
                fast_data = self._all_fuels.get('fast_charge', {})
                fast_eff  = float(fast_data.get('refuel_efficiency', 1.0))
                # Grid kWh for en-route km (battery kWh / fast_eff)
                enroute_kwh = self._enroute_distance * fc_per_km * slow_eff / fast_eff
                depot_kwh   = annual - self._enroute_distance * fc_per_km
                fuel_cost  += depot_kwh * cost_arr[idx]
                if fast_data:
                    fast_cost = np.asarray(fast_data['cost'])
                    idx_f     = np.clip(self.operation_years - self._Y_start, 0, len(fast_cost) - 1)
                    fuel_cost += enroute_kwh * fast_cost[idx_f]
            else:
                fuel_cost += annual * cost_arr[idx]

        # FC replacement cost (time-varying $/kW × capacity × replacement events)
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
        }
        self.annual_revenue = (
            self.annual_distance
            * self.mass['payload'] / 1000.0
            * float(self.params['revenue_per_tkm'])
        )

    def _discount(self, annual):
        """
        Survival-weighted net-present value of an annual cost/revenue array.

        NPV = Σ_a  annual[a] × survival_rate[a] / (1 + r)^a

        survival_rate[a] ∈ [0, 1] is the probability the vehicle is still operating at age a,
        so the expectation over the fleet is already embedded here (no separate fleet-level
        discounting is needed).  r = DISCOUNT_RATE from settings.
        """
        return float(np.sum(
            np.asarray(annual)
            * np.asarray(self.params['survival_rate'])
            / (1.0 + DISCOUNT_RATE) ** self.age
        ))

    def _calculate_tco_npv(self):
        """
        TCO = NPV sum of all cost components (capital + operating + fuel + driver + FC stack).
        NPV = NPV(revenue) − TCO.

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
    def __init__(self, params, param_cps, exclude_powertrains=()):
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
        self._run()
        self._aggregate()

    def _make_vehicle(self, k, p, t):
        return Vehicles(
            self.select_vehicle_params(k, p, t),
            self.params['fuels'],
            self.params['vehicles']['costs'],
            p=p, k=k,
        )

    def _build_initial_stock(self):
        """
        Populate the pre-2025 diesel-only fleet so that cumulative activity at START_YEAR
        matches the exogenous activity_req.

        Sizing formula for cohort y (y < START_YEAR):
            stock[k, dice, y, START_YEAR] = activity_req[k, START_YEAR]
                                            × (1 + growth_rate)^(y - START_YEAR)
                                            × survival_rate[START_YEAR - y]
                                            / denom

        denom = Σ_a  annual_distance[a] × payload[a]/1000 × survival_rate[a] × (1 + growth_rate)^(-a)
        is the survival-and-growth-weighted t-km per vehicle over a full MAX_AGE lifespan.
        Dividing by denom converts a total activity target into a number of vehicles per cohort.

        The oldest cohort (age MAX_AGE-1 at START_YEAR) is used as the denominator reference
        to stay consistent with the Paper 1 calibration.
        """
        for k in self.K:
            for y in range(START_YEAR - MAX_AGE, START_YEAR):
                self.vehicles[k, 'dice', y] = self._make_vehicle(k, 'dice', y)

            # Fixed reference vehicle (oldest vintage) for denominator — matches old model
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
        Year-by-year simulation START_YEAR → END_YEAR.  Each year has three steps:

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
        """
        for t in self.years:
            # Roll surviving cohorts from t-1
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

            # Build new vehicles for model year t
            for k in self.K:
                for p in self.P[k]:
                    self.vehicles[k, p, t] = self._make_vehicle(k, p, t)

            # Market share then fill activity gap with new purchases
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

    def _calculate_market_share(self, k, t):
        """
        Multinomial logit with iterative production-cap enforcement.

        Unconstrained logit share for powertrain p:
            share(p) = exp(λ × NPV(p)) / Σ_q exp(λ × NPV(q))

        where λ = price_lambda (controls sensitivity to NPV differences; higher λ → winner-takes-
        all, λ → 0 → uniform shares).

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
                break  # Converged — no new production-limited powertrains

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
        self.system_costs = {k: {c: np.zeros(len(T)) for c in ('capital', 'operational', 'fuel', 'driver')} for k in self.K}

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
                        for c in ('operational', 'fuel', 'driver'):
                            self.system_costs[k][c][i] += n * v.annual_cost[c][a]
                    # Capital at point of sale
                    if START_YEAR <= y <= END_YEAR:
                        i = int(y - START_YEAR)
                        self.system_costs[k]['capital'][i] += self.stock.get((k, p, y, y), 0.0) * v.capital_total

    def select_vehicle_params(self, k, p, y):
        """
        Build the params dict for a single (vehicle type k, powertrain p, model year y) cohort.

        Three-step merge:
          1. Start with shared params for vehicle type k (base_cost, running_cost, payload, …).
          2. Overlay powertrain-specific params (|= lets powertrain values win on conflicts).
          3. For each component referenced in the powertrain, fill in shared component specs
             from vehicles.components[type][comp_name] — but only for keys not already set
             by the powertrain (so per-powertrain overrides are preserved).

        Then set_year() slices all time-varying arrays to scalar values at year y.
        Age-varying arrays (target_distance, drive_cycle, survival_rate) are excluded from
        set_year() because Vehicles uses them indexed by vehicle age, not calendar year.
        """
        vehicle_params  = copy.deepcopy(self.params['vehicles']['types'][k]['shared'])
        vehicle_params |= copy.deepcopy(self.params['vehicles']['types'][k]['powertrains'][p])
        for comp_name, comp in list(vehicle_params['components'].items()):
            shared_def = copy.deepcopy(self.params['vehicles']['components'][comp['type']][comp_name])
            comp.update({kk: v for kk, v in shared_def.items() if kk not in comp})
        vehicle_params['model_year'] = y
        exclude = {'target_distance', 'drive_cycle', 'survival_rate', 'average_speed', 'payload'}
        for key in list(vehicle_params.keys()):
            if key not in exclude:
                vehicle_params[key] = set_year(vehicle_params[key], year=y)
        return vehicle_params

    def realise_uncertainties(self, param_cps):
        """
        Apply a Monte Carlo draw to every uncertain parameter in the params tree.

        param_cps maps (key_path_tuple) → cp ∈ [0, 1].  Key paths are produced by
        get_uncertainty_distributions(), which walks the nested params dict and returns
        every leaf that has a 'dist' key.

        set_param() converts the cumulative probability cp to a realised value (scalar
        or time-series array depending on the distribution type).  Because a single cp
        is shared across all parameters in one MC run, correlated quantities — e.g. the
        same component parameter used by multiple powertrains — move together.
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
    np.random.seed(0)
    inputs_distributions = dict(get_uncertainty_distributions(PARAMS))
    param_cps = dict(zip(inputs_distributions.keys(), np.random.rand(len(inputs_distributions)).astype('float32')))
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
