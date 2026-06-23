""" Model cleaned up.
To do:
 - Does battery degradation apply to PHEVs?
 - Does embodied battery emissions apply to PHEVs or HEVs or FCs?
 - Apply some checks (average distance, activity, etc)
 - Drivers need to be paid during breaks.
 - Market share limit inside fleet.
 - Size vehicle components for NPV optimisation?
   - Combine with improved fuel consumption calculation.
 - Altitude on FC/engine performance and air resistance.
 - Bring the policies into something less annoying.
 - Payload by drivecycle not vehicle type
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

def estimate_fuel_consumption(input_data, model_params):
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
# phe surrogate only has udds_hdt/cruise_hdt files
PHE_DC_MAP = {
    'short_haul': 'udds_hdt', 'regional_haul': 'udds_hdt', 'long_haul': 'cruise_hdt',
}
# Which component efficiency to pass as peak_eff to the surrogate
EFF_COMPONENT = {
    'dice': 'ice', 'he': 'ice',   'phe': 'ice',
    'be':   'motor', 'fc': 'fc',
    'hice': 'ice', 'dhice': 'ice',
}
ZEV_POWERTRAINS          = {'be', 'fc', 'hice'}
HICE_POWERTRAINS         = {'hice', 'dhice'}
AFTERTREATMENT_POWERTRAINS = {'dice', 'he', 'phe', 'dhice'}
CHARGER_POWERTRAINS      = {'be', 'phe'}


# ---------------------------------------------------------------------------
# Helper functions (unchanged)
# ---------------------------------------------------------------------------

def get_uncertainty_distributions(d, current_path=()):
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
    """ Convert a cumulative probability to a realized parameter value (scalar or array over Y). """
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

    params       — time-sliced vehicle params from Fleet.select_vehicle_params()
    fuels        — full (unsliced) fuel params from PARAMS['fuels'], covering all Y years
    drive_cycles — drive cycle metadata from PARAMS['drive_cycles']
    costs        — full (unsliced) vehicle cost arrays from PARAMS['vehicles']['costs']
    p            — powertrain key string, e.g. 'dice', 'be', 'fc'
    k            — vehicle type key string, e.g. 'sleeper', 'day_cab', 'straight'
    """

    def __init__(self, params, fuels, drive_cycles, costs, p, k):
        self.params       = params
        self._all_fuels   = fuels                                    # full dict — needed for en-route fast_charge pricing
        self.fuels        = {f: fuels[f] for f in params['fuels']}
        self.drive_cycles = drive_cycles
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
        gvwl_increase = float(self.params.get('gvwl_increase', 0)) if self.p in ZEV_POWERTRAINS else 0.0
        self.mass = {'frame': float(self.params['frame_mass'])}
        if float(self.params.get('trailer_mass', 0)) > 0:
            self.mass['trailer'] = float(self.params['trailer_mass'])
        for comp_name, comp in self.params['components'].items():
            if comp['type'] == 'converter':
                self.mass[comp_name] = float(comp['mass'])
            elif comp['type'] == 'ess':
                self.mass[comp_name] = float(comp.get('specific_mass', 0)) * float(comp['capacity'])
            elif comp['type'] == 'transmission':
                self.mass[comp_name] = float(comp['mass'])
        self.unloaded_mass = sum(self.mass.values())
        payload_frac = 1.0 - self.params['p_weighed_out'] * (
            1.0 - (self.params['gvwl'] + gvwl_increase - self.unloaded_mass)
                / (self.params['gvwl'] - self.params['default_unloaded_mass'])
        )
        self.mass['payload'] = float(self.params['default_payload']) * max(0.0, payload_frac)
        self.total_mass = self.unloaded_mass + self.mass['payload']

    # -- Fuel consumption ------------------------------------------------------

    def _calculate_fuel_consumption(self):
        surrogate = SURROGATE_NAME.get(self.p, self.p)
        peak_eff  = float(self.params['components'][EFF_COMPONENT[self.p]]['efficiency'])

        self.average_speed    = np.array([self.drive_cycles[dc]['average_speed']
                                          for dc in self.params['drive_cycle']])
        self.fuel_consumption = {f: np.zeros(len(self.age)) for f in self.fuels}

        for dc in np.unique(self.params['drive_cycle']):
            dc_file     = PHE_DC_MAP.get(dc, dc) if surrogate == 'phe' else dc
            model_params = load_model_params(f'drive_cycles/{surrogate}_{dc_file}.json')
            raw          = estimate_fuel_consumption({
                'mass':           self.total_mass,
                'drag_coef':      self.params['drag_coef'],
                'accessory_load': self.params['accessory_load'],
                'peak_eff':       peak_eff,
            }, model_params)
            per_fuel = self._split_surrogate_output(raw)
            mask     = np.array([self.params['drive_cycle'][a] == dc for a in self.age])
            for f, val in per_fuel.items():
                if f in self.fuel_consumption:
                    self.fuel_consumption[f][mask] = val

        # Convert from ESS units/km to source (grid/pump) units/km
        for f in self.fuel_consumption:
            eff = float(self.fuels[f].get('refuel_efficiency', 1.0))
            if eff < 1.0:
                self.fuel_consumption[f] = self.fuel_consumption[f] / eff

    def _split_surrogate_output(self, raw_val):
        """
        Map scalar surrogate output (primary fuel, L/km or kg/km or kWh/km) to a
        per-fuel dict covering all fuels this powertrain uses.
        """
        DIESEL_LHV = 38_600_000.0   # J/L
        H2_LHV    = 120_000_000.0   # J/kg
        ELEC_LHV  = 3_600_000.0     # J/kWh
        fp = self.params['fuels']   # {fuel: {proportion: x}}

        if self.p in ('dice', 'he'):
            # Surrogate gives net diesel L/km (regen accounted for in HEV model)
            return {'diesel': raw_val}

        elif self.p == 'phe':
            # Surrogate (phe_parallel) gives diesel L/km; estimate electricity from proportion
            d_prop = fp.get('diesel',      {}).get('proportion', 0.95)
            e_prop = fp.get('fast_charge', {}).get('proportion', 0.08)
            total  = d_prop + e_prop
            elec_energy = raw_val * DIESEL_LHV * (e_prop / total) / (d_prop / total)
            e_fuel = 'fast_charge' if 'fast_charge' in self.fuels else 'slow_charge'
            return {'diesel': raw_val, e_fuel: elec_energy / ELEC_LHV}

        elif self.p == 'be':
            e_fuel = next(f for f in self.fuels if 'charge' in f)
            return {e_fuel: raw_val}

        elif self.p == 'fc':
            return {'h2': raw_val}

        elif self.p == 'hice':
            # dice surrogate gives L/km diesel; convert to kg/km H2 via LHV
            return {'h2': raw_val * DIESEL_LHV / H2_LHV}

        elif self.p == 'dhice':
            # HEV surrogate gives diesel-equivalent L/km; split by fuel proportion
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
            r = float(comp['capacity']) * float(comp.get('usable_capacity', 1.0)) / fc
            self.range = np.minimum(self.range, r)
            if comp_name == 'battery':
                rate_raw = float(comp.get('charging_rate', 0))
                if rate_raw > 0 and battery_fuel:
                    # charging_rate is fast-charger wall power (kW); convert to slow_charge-basis
                    # units so it is commensurable with fc_a (wall-plug kWh/km) × speed (km/hr)
                    fast_eff = float(self._all_fuels.get('fast_charge', {}).get('refuel_efficiency', 1.0))
                    slow_eff = float(self.fuels[battery_fuel].get('refuel_efficiency', 1.0))
                    rate = rate_raw * fast_eff / slow_eff
                else:
                    rate = 0.0
            else:
                rate = float(comp.get('refuel_rate', 0))
            if rate > self.refuel_rate:
                self.refuel_rate = rate

    # -- Annual distance -------------------------------------------------------

    def _calculate_annual_distance(self):
        """
        Age-by-age loop: applies battery degradation and range-limited daily distance.
        Ported from fleet/model.py Vehicle.calculate_annual_distance().
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
        if self.p in AFTERTREATMENT_POWERTRAINS:
            c['after_treatment'] = self._cap_cost('after_treatment')
        if self.p in CHARGER_POWERTRAINS and self.k != 'sleeper':
            c['charger'] = self._cap_cost('charger_50kw')
        self.capital       = c
        self.capital_total = sum(c.values())

    # -- Annual cost & TCO -----------------------------------------------------

    def _calculate_annual_cost(self):
        # Fuel cost
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
            * float(self.mass['payload']) / 1000.0
            * float(self.params['revenue_per_tkm'])
        )

    def _discount(self, annual):
        return float(np.sum(
            np.asarray(annual)
            * np.asarray(self.params['survival_rate'])
            / (1.0 + DISCOUNT_RATE) ** self.age
        ))

    def _calculate_tco_npv(self):
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

        # Activity requirement (km·t/year) by vehicle type and calendar year
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
            self.params['drive_cycles'],
            self.params['vehicles']['costs'],
            p=p, k=k,
        )

    def _build_initial_stock(self):
        """Pre-START_YEAR fleet: diesel cohorts only. Stock at START_YEAR sized to meet activity."""
        for k in self.K:
            for y in range(START_YEAR - MAX_AGE, START_YEAR):
                self.vehicles[k, 'dice', y] = self._make_vehicle(k, 'dice', y)

            # Fixed reference vehicle (oldest vintage) for denominator — matches old model
            v_ref   = self.vehicles[k, 'dice', START_YEAR - MAX_AGE]
            surv_r  = v_ref.params['survival_rate']
            denom   = sum(v_ref.annual_distance[a] * float(surv_r[a]) * (1 + GROWTH_RATE) ** (-a)
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
        """Year-by-year simulation START_YEAR → END_YEAR."""
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
                    self.stock.get((k, p, y, t), 0.0) * self.vehicles[k, p, y].annual_distance[t - y]
                    for p in self.P[k] for y in range(t - MAX_AGE + 1, t)
                    if (k, p, y) in self.vehicles
                )
                avg_activity = sum(
                    self.vehicles[k, p, t].annual_distance[0] * self.market_share[k, p, t]
                    for p in self.P[k]
                )
                new_sales = max((self.activity_req[k, t] - activity_met) / max(avg_activity, 1.0), 0.0)
                for p in self.P[k]:
                    self.stock[k, p, t, t] = np.float32(new_sales * self.market_share[k, p, t])

    def _calculate_market_share(self, k, t):
        """Multinomial logit with iterative production cap."""
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
        """Sum stock, fuel, emissions, and system costs across all cohorts."""
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
        # Merge shared + powertrain params
        vehicle_params  = copy.deepcopy(self.params['vehicles']['types'][k]['shared'])
        vehicle_params |= copy.deepcopy(self.params['vehicles']['types'][k]['powertrains'][p])
        # Fill in shared component definitions (powertrain-specific values take precedence)
        for comp_name, comp in list(vehicle_params['components'].items()):
            shared_def = copy.deepcopy(self.params['vehicles']['components'][comp['type']][comp_name])
            comp.update({kk: v for kk, v in shared_def.items() if kk not in comp})
        # Time-slice to model year y (target_distance, drive_cycle, survival_rate stay as age arrays)
        vehicle_params['model_year'] = y
        exclude = {'target_distance', 'drive_cycle', 'survival_rate'}
        for key in list(vehicle_params.keys()):
            if key not in exclude:
                vehicle_params[key] = set_year(vehicle_params[key], year=y)
        return vehicle_params

    def realise_uncertainties(self, param_cps):
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
    fleet = Fleet(PARAMS, param_cps, exclude_powertrains=('phe',))

    v = fleet.vehicles['sleeper', 'dice', START_YEAR]
    print('Total mass:               ', v.total_mass, 'kg')
    print('Fuel consumption (age 0): ', v.fuel_consumption['diesel'][0], 'L/km')
    print('Annual distance (age 0):  ', v.annual_distance[0], 'km')
    print('Capital total:            ', v.capital_total, '$')
    print('TCO:                      ', v.tco, '$')
    print('NPV:                      ', v.npv, '$')
    print()
    print('Market share (sleeper, 2025):', {p: round(fleet.market_share.get(('sleeper', p, START_YEAR), 0), 4) for p in fleet.P['sleeper']})
