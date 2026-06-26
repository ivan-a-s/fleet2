""" Just the model, no optimisation.
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
"""
# Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from dataclasses import dataclass, fields
import scipy.stats as stats
import copy
import cProfile
import pstats
import pickle

# Local imports — ensure old/ is on sys.path so interactive window runs work
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_old as d

# Physical parameters
AIR_DENSITY = 1.225  # kg/m^3
GRAVITY = 9.81  # m/s^2

# Sector parameters
MAX_AGE = 25
START_YEAR = 2025
END_YEAR = 2050
PRIVATE_DISCOUNT_RATE = 0.08
GROWTH_RATE = 0.02
ACTIVIY_YEAR_0 = 61_000 * 62_965 * 12.72 # Fleet size (NRCan 2022) * average distance (NRCan 2022) * average payload (mine)
DEFAULT_UNLOADED_MASS = { # Put in data.py
    'Sleeper': 12_938,
    'Day Cab': 12_266,
    'Class-8 Straight': 9_857,
}
GVWL_INCREASES = {
    'Sleeper': 5_000,
    'Day Cab': 3_000,
    'Class-8 Straight': 2_000,
}
SOCIAL_COST_OF_CARBON_0 = 1.4228 * 31 * (1 + 0.03) ** 10
ZEV_POWERTRAINS = {'BE', 'HICE', 'FC'}
NON_ZEV_POWERTRAINS = {'D', 'DHP', 'DHNP', 'DHICE'}
PRICE_LAMBDA = 3e-5


def market_share_limit(prev_share=0,
    market_limit={
        'init': 0.02,
        'CAGR_early': 0.4,
        'CAGR_other': 0.30,
    }):
    prev_share = max([prev_share, market_limit['init']/(1+market_limit['CAGR_early'])])
    if prev_share < 0.15:
        return prev_share*(1+market_limit['CAGR_early'])/(1+GROWTH_RATE)
    else:
        return prev_share*(1+market_limit['CAGR_other'])/(1+GROWTH_RATE)

def get_series(values, value_years, query_years):
    """
    Return values corresponding to query_years, given arrays of known values and their years.
    Performs linear interpolation and clips query years to known range.
    """
    values = np.asarray(values)
    value_years = np.asarray(value_years)
    query_years = np.atleast_1d(query_years)

    # clip query to known range
    query_years = np.clip(query_years, value_years.min(), value_years.max())

    # linear interpolation
    return np.interp(query_years, value_years, values)

def set_constant_params(params):
    def get_uncertainty_distributions_(d, current_path=()):
        paths = []
        if isinstance(d, dict):
            if 'dist' in d and any(k in d for k in ['val', 'min', 'max', 'mode']):
                # This is a numeric distribution dict
                paths.append((current_path, d))
            else:
                for k, v in d.items():
                    paths.extend(get_uncertainty_distributions_(v, current_path + (k,)))
        return paths
    inputs_distributions = dict(get_uncertainty_distributions_(params))
    for key, val in inputs_distributions.items():
        if val.get('dist') == 'const':
            d_ref = params
            for k in key[:-1]:
                d_ref = d_ref[k]
            d_ref[key[-1]] = val['val']

def get_uncertainty_distributions(d, current_path=()):
    paths = []
    if isinstance(d, dict):
        if 'dist' in d:
            # This is a numeric distribution dict
            paths.append((current_path, d))
        else:
            for k, v in d.items():
                paths.extend(get_uncertainty_distributions(v, current_path + (k,)))
    return paths

def set_param_(param, cp):
    # If no cumulative probability given, draw a random one
    if isinstance(param, (int, float, np.floating)):
        return param
    elif param['dist'] == 'const':
        return param['val']
    elif param['dist'] == 'triangle':
        # Triangular distribution: convert mode to c fraction
        c = (param['mode'] - param['min']) / (param['max'] - param['min'])
        return stats.triang.ppf(cp, c, loc=param['min'], scale=param['max'] - param['min'])
    elif param['dist'] == 'uniform':
        return stats.uniform.ppf(cp, loc=param['min'], scale=param['max'] - param['min'])
    else:
        raise ValueError(f"Unknown distribution: {param['dist']}")

def set_param(param, cp=0.5, Y=np.arange(START_YEAR-MAX_AGE, END_YEAR+1)):
    """ Convert a seed to a parameter value. """
    if param['dist'] == 'linear':
        start = set_param_(param['start'], cp)
        end = set_param_(param['end'], cp)
        return np.concatenate([np.ones(MAX_AGE) * start, np.linspace(start, end, len(Y)-MAX_AGE)])
    elif param['dist'] == 'interp':
        # Extract the year-value mappings
        year_keys = sorted([int(k) for k in param.keys() if k.isdigit()])
        values = [set_param_(param[str(y)], cp) for y in year_keys]
        # Interpolate
        interp_vals = np.interp(Y, year_keys, values, left=values[0], right=values[-1])
        return interp_vals
    else:
        return set_param_(param, cp)

def set_all_params(d):
    if isinstance(d, dict):
        if "dist" in d:
            return set_param(d)
        return {k: set_all_params(v) for k, v in d.items()}
    else:
        return d

def realise_uncertainties(d):
    if isinstance(d, dict):
        if "dist" in d and any(k in d for k in ['val', 'min', 'max', 'mode']):
            return set_param_(d)  # No Y passed — uses default
        return {k: realise_uncertainties(v) for k, v in d.items()}
    else:
        return d

def convert_to_float32(d):
    """Recursively convert numeric values and arrays in a nested dict to float32."""
    for k, v in d.items():
        if isinstance(v, dict):
            convert_to_float32(v)  # recurse
        elif isinstance(v, (int, float)):
            d[k] = np.float32(v)
        elif isinstance(v, np.ndarray) and v.dtype != np.float32:
            d[k] = v.astype(np.float32)
    return d

def energy_consumption_fast(v, drive_cycle='long_haul'):
    if drive_cycle == 'long_haul':
        return (
            184468.0062 +
            v.mass.total * 65.678763 +
            v.roll_coefficient * 0 + 
            v.drag_coefficient * 4350611.430231 + 
            v.regen_efficiency * -292181.3162943 +
            v.accessory_load * 47.472692 +
            getattr(v, "motor_size", 0) * -82.872142
        )
    elif drive_cycle == 'regional_haul':
        return (
            100174.0174 +
            v.mass.total * 62.269866 +
            v.roll_coefficient * 0 + 
            v.drag_coefficient * 3771521.906184 + 
            v.regen_efficiency * -145997.061314 +
            v.accessory_load * 51.988641 +
            getattr(v, "motor_size", 0) * -50.706751
        )
    else:
        return (
            31692.4468 +
            v.mass.total * 136.098951 +
            v.roll_coefficient * 0 + 
            v.drag_coefficient * 3849936.569754 + 
            v.regen_efficiency * -39075.473546 +
            v.accessory_load * 96.047578 +
            getattr(v, "motor_size", 0) * -10.420126
        )



class CarbonTax:
    def __init__(self, years=d.PARAMS['Years']['T'], price={'2025': 0, '2030': 0, '2050': 0}):
        self.years = years
        self.price = np.zeros(len(self.years))
        # Interpolate anchor-year targets
        anchor_years = np.array(sorted(int(y) for y in price))
        anchor_vals  = np.array([price[str(y)] for y in anchor_years])
        years_np = np.array(years)
        self.price = np.interp(years_np, anchor_years, anchor_vals)
        self.price[self.years < 2025] = 0

class LCFS:
    def __init__(self, params=d.PARAMS, credit_price=0, start=0.183, end=0.76):
        self.baseline_emissions = params['Fuels']['Diesel']['Emissions Intensity']['Combustion'] + params['Fuels']['Diesel']['Emissions Intensity']['Supply'] # kgCO2/L diesel
        self.baseline_fuel_consumption = {
            'Sleeper': 0.28977364,
            'Day Cab': 0.41699174,
            'Class-8 Straight': 0.32835686,
        } # L/km
        self.credit_price = credit_price
        self.years = params['Years']['Y']
        self.target = np.zeros(len(self.years)) # Reduction compared to baseline (e.g. 0.2 = 20%)
        self.target[self.years < 2025] = 0
        self.target[self.years >= 2025] = np.linspace(start, end, len(self.years[self.years >= 2025]))

class ZEVMandate:
    def __init__(self, years=d.PARAMS['Years']['T'], targets={'2025': 0, '2030': 0, '2050': 0}, penalty=0, rebates=False):
        self.years = years
        self.penalty=penalty
        self.rebates = rebates
        
        anchor_years = np.array(sorted(int(y) for y in targets))
        anchor_vals  = np.array([targets[str(y)] for y in anchor_years])
        self.targets = np.interp(years, anchor_years, anchor_vals)

class AutonomousPermits:
    def __init__(self, permits={
            'D': 0,
            'DHNP': 0,
            'DHP': 0,
            'BE': 0,
            'FC': 0,
            'HICE': 0,
            'DHICE': 0,
        }):
        self.permits = permits


class DriveCycle: # Should factor in vehicle age.
    """ Class to hold the drive cycle data. """
    def __init__(self, path, smooth=True, fast=True, bc_slope=True, traffic_cost=0):
        self.df = pd.read_csv(path)
        self.traffic_cost = traffic_cost
        if fast:
            self.df = pd.read_csv(path)[:5_000]
        self.t = self.df['Time (seconds)'].to_numpy()
        self.v = self.df['Speed (mph)'].to_numpy() * 0.44704  # Convert mph to m/s
        if smooth:
            self.v = savgol_filter(self.v, 61, 1)
        self.v_squared = self.v**2
        self.slope = np.arctan(self.df['Grade (rise/run)'].to_numpy())
        if bc_slope and ('regional_haul' in path or 'long_haul' in path): # Increase the grade to simulate B.C. terrain (up to 8%).
            self.slope = self.slope * 0.08 / max(abs(self.slope))
        self.cos_slope = np.cos(self.slope)
        self.sin_slope = np.sin(self.slope)
        self.key_on = self.df['Key On/Off (1/0)'].to_numpy()
        self.a = np.diff(self.v, prepend=self.v[0])
        self.total_distance = np.sum(self.v) / 1000
        self.average_moving_speed = np.average(self.v[self.v>0])


class Fuel:
    def __init__(self, params, years):
        self.lhv = params['LHV']
        self.units = params['Units']
        self.recharge_efficiency = params['Re-fuel Efficiency']
        self.air_pollution = params['Air Pollution']
        self.years = years
        self.cost = self.Cost(
            fuel=params['Cost']
        )
        self.emissions_intensity = self.EmissionsIntensity(
            supply=np.full(len(years), params['Emissions Intensity']['Supply']),
            combustion=np.full(len(years), params['Emissions Intensity']['Combustion'])
        )
        self.water_usage = params['Water Usage']
        self.electricity_usage = params['Electricity Usage']

    @dataclass
    class Cost:
        fuel: np.ndarray

        @property
        def customer(self):
            return self.fuel

    @dataclass
    class EmissionsIntensity:
        supply: np.ndarray
        combustion: np.ndarray

        @property
        def total(self):
            return self.combustion + self.supply


class Vehicle:
    def __init__(self, params, fuels, p_fuel, drive_cycles, costs, seed=0, policy=None,
            k='Sleeper', p='D', y=2025, max_age=25, T=np.arange(START_YEAR, END_YEAR+1), Y=np.arange(START_YEAR-MAX_AGE, END_YEAR+1),
            discount_rate=PRIVATE_DISCOUNT_RATE, foresight=True,
            ctax=CarbonTax(), lcfs=LCFS(), zev_mandate=ZEVMandate(), autonomous_penetration=0,
            gvwl_increase=False, break_mandate=0.0, accelerated_retirement=False, zev_rebate=0.0,
            **kwargs):
        # Constant parameters
        skip = {'self', 'kwargs', 'params'}  # any locals you want to ignore
        for name, value in locals().items():
            if name not in skip:
                setattr(self, name, value)
        self.F = list(fuels.keys())
        self.A = np.arange(max_age)
        self.p = p
        self.operation_years = np.array(self.A + self.y, dtype=int)

        # Initialise variables from params
        self.set_attributes(params)
        if accelerated_retirement and p == 'D':
                self.survival_rate = self.survival_rate * np.array([0.001 if (a >= 15 and (self.y + a) >= START_YEAR) else 1 for a in range(MAX_AGE)])

        # Vehicle mass (fc/battery replacements)
        if p in ZEV_POWERTRAINS and self.gvwl_increase:
            self.gvwl_increase = GVWL_INCREASES[self.k]
        else:
            self.gvwl_increase = 0
        self.mass = self.Mass(
            frame=self.frame_mass,
            drivetrain=self.drivetrain_mass,
            ess=sum(self.ess_specific_mass[f] * self.fuel_capacity[f] for f in self.F), # hybrids?
            trailer=self.trailer_mass,
            average_payload=self.average_payload,
            gvwl=self.gvwl,
            gvwl_increase=self.gvwl_increase,
            default_unloaded=DEFAULT_UNLOADED_MASS[k]
        )

        # Calculate energy and fuel consumption
        self.average_speed = np.zeros(MAX_AGE)
        energy_consumptions = {}
        for key in np.unique(self.drive_cycle):
            energy_consumptions[key] = energy_consumption_fast(self, key)
        self.energy_consumption = np.zeros(len(self.A), dtype=np.float32)
        for a in self.A: # Remove age dependency and change to drive-cycle dependency.
            self.average_speed[a] = drive_cycles[self.drive_cycle[a]]['average_moving_speed'] * 3.6
            self.energy_consumption[a] = energy_consumptions[self.drive_cycle[a]]
        
        self.fuel_consumption = {}
        for f in self.F:
            self.fuel_consumption[f] = self.energy_consumption * p_fuel[f] / self.efficiency[f] / self.fuels[f].lhv / self.fuels[f].recharge_efficiency

        # Activity and usage
        self.range = (self.fuel_capacity[f] * self.usable_capacity[f]) / self.fuel_consumption[f]
        self.break_mandate = np.array([0 if y<START_YEAR else break_mandate for y in self.operation_years])
        self.range_adder = np.min([self.range, (self.break_mandate * self.refuel_rate[f]) / self.fuel_consumption[f]], axis=0) * (1-self.autonomous_penetration) #technically shouldn't be implemented prior to start year but makes no impact.
        self.range += self.range_adder
        self.calculate_annual_distance()
        self.annual_fuel = {}
        for f in self.F:
            self.annual_fuel[f] = self.annual_distance * self.fuel_consumption[f]

        # Fuel cell replacements
        self.fc_replacements = np.zeros(len(self.A))
        if p == 'FC':
            fc_hours = 0
            for a in self.A:
                fc_hours += self.annual_distance[a] / self.average_speed[a]
                if fc_hours > self.fc_lifetime:
                    self.fc_replacements[a] = 1
                    fc_hours = 0

        # Emissions
        self.emissions = self.Emissions( # Need to make change by year.
            embodied = self.Embodied( # Can calculate these above if I need multiple lines
                frame=np.concatenate([[self.mass.frame * self.embodied], np.zeros(self.max_age-1)]).astype('float32'),
                trailer=np.concatenate([[self.mass.trailer * self.embodied * self.trailers_per_truck], np.zeros(self.max_age-1)]).astype('float32'),
                drivetrain=np.concatenate([[self.mass.drivetrain * self.embodied], np.zeros(self.max_age-1)]).astype('float32'),
                ess=np.concatenate([[sum(self.fuel_capacity[f] * self.ess_embodied[f] for f in self.F)], np.zeros(self.max_age-1)]).astype('float32'), # Battery replacements
            ),
            fuel_combustion = sum(self.annual_fuel[f] * get_series(self.fuels[f].emissions_intensity.combustion, self.fuels[f].years, self.operation_years) for f in self.F).astype('float32'),
            fuel_supply = sum(self.annual_fuel[f] * get_series(self.fuels[f].emissions_intensity.supply, self.fuels[f].years, self.operation_years) for f in self.F).astype('float32'),
            end_of_life=np.zeros(MAX_AGE)
        )

        # Costs
        self.calculate_capital_cost()
        self.zev_rebate = zev_rebate
        self.calculate_annual_cost(foresight=foresight)
        self.tco = self.TCO(
            capital=self.discount(self.annual_cost.capital),
            operational=self.discount(self.annual_cost.operational),
            fuel=self.discount(self.annual_cost.fuel),
            driver=self.discount(self.annual_cost.driver),
            battery_replacements=self.discount(self.annual_cost.battery_replacements),
            fc_replacements=self.discount(self.annual_cost.fc_replacements),
            carbon_tax=self.discount(self.annual_cost.carbon_tax),
            lcfs=self.discount(self.annual_cost.lcfs),
            zev_rebate=self.discount(self.annual_cost.zev_rebate),
        )
        self.annual_activity = self.annual_distance * self.mass.payload/1000
        self.annual_revenue = self.annual_activity * self.revenue_per_tkm
        self.total_revenue = np.float32(self.discount(self.annual_revenue))

        # Plots
        # self.plots = self.Plots(self)

    def set_attributes(self, params):
        temp = {}
        # Constants
        by_year = ['drag_coefficient', 'regen_efficiency', 'fc_lifetime', 'embodied', 'roll_coefficient']
        by_fuel = ['ess_embodied', 'ess_specific_mass', 'refuel_rate', 'efficiency', 'usable_capacity']
        for key, value in params.items():
            if key in by_fuel:
                temp[key] = {}
                for f in self.F:
                    if isinstance(params[key][f], np.ndarray):
                        temp[key][f] = params[key][f][np.argwhere(self.Y==self.y)[0,0]]
                    else:
                        temp[key][f] = params[key][f]
            elif key in by_year and isinstance(value, np.ndarray):
                temp[key] = value[np.argwhere(self.Y==self.y)[0,0]]
            else:
                temp[key] = value
        self.__dict__.update(temp)
        self.params = None

    @dataclass
    class Mass:
        frame: float
        drivetrain: float
        ess: float
        trailer: float
        average_payload: float
        gvwl: float
        gvwl_increase: float = 0
        default_unloaded: float = 15_000
    
        @property
        def unloaded(self):
            return self.frame + self.drivetrain + self.ess + self.trailer
        
        @property
        def payload(self):
            return self.average_payload * (1-0.3*(1-(self.gvwl + self.gvwl_increase - self.unloaded) / (self.gvwl - self.default_unloaded))) # Use diesel mass
            
        @property
        def total(self):
            return self.unloaded + self.payload

    @dataclass
    class Embodied:
        frame: np.ndarray
        trailer: np.ndarray
        drivetrain: np.ndarray
        ess: np.ndarray

        def __post_init__(self):
            for field_name in self.__dataclass_fields__:
                arr = getattr(self, field_name)
                if isinstance(arr, np.ndarray):
                    setattr(self, field_name, arr.astype(np.float32))

        @property
        def annual(self):  # Factor in survival rate.
            return self.frame + self.trailer + self.drivetrain + self.ess
        
        @property
        def lca(self):
            return sum(self.annual)

    @dataclass
    class Emissions:
        embodied: "Vehicle.Embodied"
        fuel_combustion: np.ndarray
        fuel_supply: np.ndarray
        end_of_life: np.ndarray

        def __post_init__(self):
            # convert all fields to float32
            for field_name in self.__dataclass_fields__:
                arr = getattr(self, field_name)
                if isinstance(arr, np.ndarray):
                    setattr(self, field_name, arr.astype(np.float32))

        @property
        def annual_fuel(self):
            return self.fuel_combustion + self.fuel_supply
        
        @property
        def annual(self):
            return self.annual_fuel + self.embodied.annual
        
        @property
        def lca(self):
            return sum(self.annual)

    @dataclass
    class Capital:
        base: np.float32
        engine: np.float32=0
        combustion_transmission: np.float32=0
        after_treatment: np.float32=0
        tank: np.float32=0
        electric_transmission: np.float32=0
        motor: np.float32=0
        battery: np.float32=0
        fc: np.float32=0
        h2_tank: np.float32=0
        charger: np.float32=0

        @property
        def total(self):
            return sum(getattr(self, f.name) for f in fields(self))

    @dataclass
    class AnnualCost:
        capital: np.ndarray
        operational: np.ndarray
        fuel: np.ndarray
        driver: np.ndarray
        battery_replacements: np.ndarray
        fc_replacements: np.ndarray
        carbon_tax: np.ndarray
        lcfs: np.ndarray
        zev_mandate: np.ndarray
        zev_rebate: np.ndarray

        def __post_init__(self):
            # convert all fields to float32
            for field_name in self.__dataclass_fields__:
                arr = getattr(self, field_name)
                if isinstance(arr, np.ndarray):
                    setattr(self, field_name, arr.astype(np.float32))

        @property
        def system(self):
            return self.capital + self.operational + self.fuel + self.driver + self.battery_replacements + self.fc_replacements
        
        @property
        def total(self):
            return np.sum(np.stack([getattr(self, f.name) for f in fields(self)]), axis=0)

    @dataclass
    class TCO:
        capital: np.float32
        operational: np.float32
        fuel: np.float32
        driver: np.float32
        battery_replacements: np.float32
        fc_replacements: np.float32
        carbon_tax: np.float32
        lcfs: np.float32
        zev_mandate: np.float32=0.0
        zev_rebate: np.float32=0.0

        @property
        def total(self):
            return sum(getattr(self, f.name) for f in fields(self))

    @property
    def npv(self):
        return self.total_revenue - self.tco.total

    def discount(self, annual):
        return sum(annual * self.survival_rate / (1 + self.discount_rate) ** self.A)

    def calculate_annual_distance(self): # vectorise
        default = self.target_distance
        self.annual_distance = np.zeros(len(default))
        self.daily_target = default / 365 * 7 / 5

        self.daily_distance = np.zeros(MAX_AGE)
        cycles = 0
        for iAge in range(len(default)):
            # Impact of break mandates
            if self.daily_target[iAge] / self.average_speed[iAge] > 4.5 and self.y + iAge >= START_YEAR:
                self.daily_target[iAge] = max([self.daily_target[iAge] - self.break_mandate[iAge] * self.average_speed[iAge], 4.5 * self.average_speed[iAge]])
            
            # Battery degradation
            if any(f in ['Slow Charge', 'Fast Charge'] for f in self.F):
                range_loss = 0.01 * iAge + 1e-4 * cycles
                self.range[iAge] *= (1-range_loss)
            
            # Impact of range on daily distance
            if self.daily_target[iAge] < self.range[iAge]:
                # Range is not an issue
                self.daily_distance[iAge] = self.daily_target[iAge]
            else:
                # Range is an issue
                # What if they had to re-fuel multiple times?
                shortfall = (self.daily_target[iAge] - self.range[iAge])
                time_left = shortfall / self.average_speed[iAge]
                achievable_distance = max([0, min((time_left-0.25) * self.average_speed[iAge] * self.refuel_rate[f] / (self.fuel_consumption[f][iAge] * self.average_speed[iAge] + self.refuel_rate[f]) for f in self.F)])
                self.daily_distance[iAge] = self.range[iAge] + achievable_distance

            self.annual_distance[iAge] = self.daily_distance[iAge] * 5/7 * 365
            cycles += self.annual_distance[iAge] * sum(self.fuel_consumption[f][iAge] / self.fuel_capacity[f] for f in list(set(['Slow Charge', 'Fast Charge']) & set(self.F)))

    def calculate_capital_cost(self):
        costs = self.costs
        if self.p == 'D':
            self.capital = self.Capital(
                base=costs['base'],
                engine=costs['diesel_engine'],
                combustion_transmission=costs['combustion_transmission'],
                after_treatment=costs['after_treatment'],
                tank=costs['tank'] * self.fuel_capacity['Diesel'],
            )
        elif self.p == 'DHNP':
            self.capital = self.Capital(
                base=costs['base'],
                engine=costs['diesel_engine'],
                combustion_transmission=costs['combustion_transmission'],
                after_treatment=costs['after_treatment'],
                tank=costs['tank'] * self.fuel_capacity['Diesel'],
                electric_transmission=costs['electric_transmission'], # Should have an AC-DC converter
                motor=costs['motor'] * self.motor_size,
                battery=costs['battery'][np.argwhere(self.y == self.Y).flatten()][0] * self.battery_size,
            )
        elif self.p == 'DHP':
            self.capital = self.Capital(
                base=costs['base'],
                engine=costs['diesel_engine'],
                combustion_transmission=costs['combustion_transmission'],
                after_treatment=costs['after_treatment'],
                tank=costs['tank'] * self.fuel_capacity['Diesel'],
                electric_transmission=costs['electric_transmission'], # Should have an AC-DC converter
                motor=costs['motor'] * self.motor_size,
                battery=costs['battery'][np.argwhere(self.y == self.Y).flatten()][0] * self.battery_size,
            )
            if self.k != 'Sleeper':
                self.capital.charger=costs['charger_50kW'][np.argwhere(self.y == self.Y).flatten()][0]/4
        elif self.p == 'BE':
            self.capital = self.Capital(
                base=costs['base'],
                electric_transmission=costs['electric_transmission'],
                motor=costs['motor'] * self.motor_size,
                battery=costs['battery'][np.argwhere(self.y == self.Y).flatten()][0] * self.battery_size,
            )
            if self.k != 'Sleeper':
                self.capital.charger=costs['charger_50kW'][np.argwhere(self.y == self.Y).flatten()][0]
        elif self.p == 'FC':
            self.capital = self.Capital(
                base=costs['base'],
                electric_transmission=costs['electric_transmission'],
                motor=costs['motor'] * self.motor_size,
                fc=costs['fc'][np.argwhere(self.y == self.Y).flatten()][0] * self.fc_size,
                h2_tank=costs['h2_tank'][np.argwhere(self.y == self.Y).flatten()][0] * self.fuel_capacity['Hydrogen'],
            )
        elif self.p == 'HICE': # after treatment?
            self.capital = self.Capital(
                base=costs['base'],
                engine=costs['HICE_engine'][np.argwhere(self.y == self.Y).flatten()][0],
                combustion_transmission=costs['combustion_transmission'],
                h2_tank=costs['h2_tank'][np.argwhere(self.y == self.Y).flatten()][0] * self.fuel_capacity['Hydrogen'],
            )
        elif self.p == 'DHICE':
            self.capital = self.Capital(
                base=costs['base'],
                engine=costs['HICE_engine'][np.argwhere(self.y == self.Y).flatten()][0],
                combustion_transmission=costs['combustion_transmission'],
                h2_tank=costs['h2_tank'][np.argwhere(self.y == self.Y).flatten()][0] * self.fuel_capacity['Hydrogen'],
                after_treatment=costs['after_treatment'],
                tank=costs['tank'] * self.fuel_capacity['Diesel'],
            )

    def calculate_annual_cost(self, foresight=True):
        if foresight:
            self.annual_cost = self.AnnualCost(
                capital=np.array([self.capital.total if a == 0 else 0 for a in self.A]),
                operational=self.annual_distance * self.running_cost,
                fuel=sum(self.annual_fuel[f] * get_series(self.fuels[f].cost.customer, self.fuels[f].years, self.operation_years) for f in self.F),
                driver=self.target_distance * self.driver_cost * (1-self.autonomous_penetration), # account for vehicle speed.
                battery_replacements=np.zeros(self.max_age),
                fc_replacements=self.fc_replacements * getattr(self, "fc_size", 0) * get_series(self.costs['fc'], self.Y, self.operation_years),
                carbon_tax=self.emissions.annual_fuel/1000 * (get_series(self.ctax.price, self.ctax.years, self.operation_years) if self.ctax else 0),
                lcfs=self.annual_distance * (sum(get_series(self.fuels[f].emissions_intensity.total, self.fuels[f].years, self.operation_years) * self.fuel_consumption[f] for f in self.F) - self.lcfs.baseline_emissions * self.lcfs.baseline_fuel_consumption[self.k] * (1-get_series(self.lcfs.target, self.lcfs.years, self.operation_years))) * (self.lcfs.credit_price) / 1000,
                zev_mandate=np.zeros(len(self.A)),
                zev_rebate=np.array([-self.capital.total * self.zev_rebate if a == 0 and self.p else 0 for a in self.A]),
            )
        else: # No foreseight on fuel or FC costs (base year used for estimates).
            self.annual_cost = self.AnnualCost(
                capital=np.array([self.capital.total if a == 0 else 0 for a in self.A]),
                operational=self.annual_distance * self.running_cost,
                fuel=sum(self.annual_fuel[f] * self.fuels[f].cost.customer[self.fuels[f].years == self.y] for f in self.F),
                driver=self.target_distance * self.driver_cost * (1-self.autonomous_penetration), # account for vehicle speed.
                battery_replacements=np.zeros(self.max_age),
                fc_replacements=self.fc_replacements * getattr(self, "fc_size", 0) * self.costs['fc'][self.y==self.Y],
                carbon_tax=self.emissions.annual_fuel/1000 * (get_series(self.ctax.price, self.ctax.years, self.operation_years) if self.ctax else 0),
                lcfs=self.annual_distance * (sum(get_series(self.fuels[f].emissions_intensity.total, self.fuels[f].years, self.operation_years) * self.fuel_consumption[f] for f in self.F) - self.lcfs.baseline_emissions * self.lcfs.baseline_fuel_consumption[self.k] * (1-get_series(self.lcfs.target, self.lcfs.years, self.operation_years))) * (self.lcfs.credit_price) / 1000,
                zev_mandate=np.zeros(len(self.A)),
                zev_rebate=np.array([-self.capital.total * self.zev_rebate if a == 0 and self.p in ZEV_POWERTRAINS else 0 for a in self.A]),
            )

    class Plots:
        def __init__(self, vehicle):
            self.v = vehicle
            self.years = self.v.operation_years
        
        def survival_rate(self):
            plt.figure()
            plt.xlabel('Operation Years')
            plt.ylabel('Survival Rate (%)')
            plt.plot(self.v.operation_years, self.v.survival_rate*100)
        
        def average_speed(self):
            plt.figure()
            plt.xlabel('Operation Years')
            plt.ylabel('Average speed (m/s)')
            plt.plot(self.v.operation_years, self.v.average_speed)
            plt.ylim(0, plt.ylim()[1])
        
        def energy_consumption(self):
            plt.figure()
            plt.xlabel('Operation Years')
            plt.ylabel(f'Energy Consumption (J/km)')
            plt.plot(self.v.operation_years, self.v.energy_consumption)
            plt.ylim(0, plt.ylim()[1])
        
        def fuel_consumption(self):
            plt.figure()
            plt.xlabel('Operation Years')
            plt.ylabel(f'Fuel Consumption ({self.v.fuel.units}/km)')
            plt.plot(self.v.operation_years, self.v.fuel_consumption)
            plt.ylim(0, plt.ylim()[1])
        
        def driving_range(self):
            plt.figure()
            plt.xlabel('Operation Years')
            plt.ylabel(f'Range (km)')
            plt.plot(self.v.operation_years, self.v.range)
            plt.ylim(0, plt.ylim()[1])
        
        def annual_distance(self):
            plt.figure()
            plt.xlabel('Operation Years')
            plt.ylabel(f'Annual distance (km)')
            plt.plot(self.v.operation_years, self.v.target_distance, label='Target')
            plt.plot(self.v.operation_years, self.v.annual_distance, label='Actual')
            plt.ylim(0, plt.ylim()[1])
            plt.legend()

        def emissions(self):
            plt.figure()
            plt.xlabel('Operation Years')
            plt.ylabel(f'Emissions (tCO2e)')
            for f in fields(self.v.emissions):
                values = getattr(self.v.emissions, f.name)
                if f.name == 'embodied':
                    values = values.annual
                plt.plot(self.years, values/1000, label=f.name)
            plt.plot(self.years, self.v.emissions.annual/1000, label='total')
            plt.legend()

        def annual_costs(self):
            plt.figure()
            plt.xlabel('Operation Years')
            plt.ylabel(f'Annual Costs ($ thousand)')
            for f in fields(self.v.annual_cost):
                values = getattr(self.v.annual_cost, f.name)
                plt.plot(self.years, values/1000, label=f.name)
            plt.plot(self.years, self.v.annual_cost.total/1000, label='total')
            plt.legend()


class Fleet:
    def __init__(self, params, uncertain_cps, drive_cycles,
            ctax=CarbonTax(), lcfs=LCFS(), autonomous_permits=AutonomousPermits(), zev_mandate=ZEVMandate(),
            pyrolysis=False, pyrolysis_elec=False, accelerated_retirement=False, foresight=True, gvwl_increase=False,
            break_mandate=0.0, zev_rebate=0.0):
        self.params = copy.deepcopy(params) # To avoid modifying the original data
        self.T = params['Years']['T']
        self.Y = params['Years']['Y']
        self.K = list(params['Vehicles'].keys())
        self.P = {k: list(params['Vehicles'][k]['Powertrains']) for k in self.K}
        self.activity_requirement = {(k, y):
            ACTIVIY_YEAR_0 * (1 + GROWTH_RATE) ** (y-START_YEAR) * self.params['Vehicles'][k]['Shared']['activity_proportion']
            for k in self.K for y in self.Y 
        }
        del params
        self.zev_mandate=zev_mandate

        # Realise uncertainties
        for keys, cp in uncertain_cps.items():
            # Navigate to the parent of the final key
            d = self.params
            for k in keys[:-1]:
                d = d[k]
            last_key = keys[-1]
            # Update the value using set_param_
            if keys in uncertain_cps.keys():
                d[last_key] = set_param(d[last_key], cp=cp)
        # self.params = set_all_params(self.params)
        self.params = convert_to_float32(self.params)

        # Fuels (should be able to handle multiple fuels per vehicle)
        self.fuels = {name: Fuel(fuel_params, self.Y) for name, fuel_params in self.params['Fuels'].items()}

        if pyrolysis:
            self.fuels['Hydrogen'] = self.fuels['Hydrogen (pyrolysis)']
        if pyrolysis_elec:
            self.fuels['Hydrogen'] = self.fuels['Hydrogen (pyrolysis + elec.)']
        # Drive Cycles
        self.drive_cycles = drive_cycles
        self.drive_cycles_small = {key: {'average_moving_speed': self.drive_cycles[key].average_moving_speed} for key in self.drive_cycles.keys()}
        del self.drive_cycles

        # Create vehicles year-by-year to enable implicit behaviour
        self.stock = {
            (k, p, y, t): np.float32(0)
            for t in self.T
            for y in range(t - MAX_AGE + 1, t+1)
            for k in self.K
            for p in self.P[k]
        }
        self.market_share = {
            (k, p, t): 1 if p == 'D' else np.float32(0)
            for t in self.T
            for k in self.K
            for p in self.P[k]
        }
        # Initial fleet
        self.vehicles = {(k, p, y): 
            Vehicle(
                params=self.params['Vehicles'][k]['Shared'] | self.params['Vehicles'][k]['Powertrains'][p]['params'],
                fuels={key: value for key, value in self.fuels.items() if key in self.params['Vehicles'][k]['Powertrains'][p]['fuel']},
                p_fuel=self.params['Vehicles'][k]['Powertrains'][p]['fuel'],
                drive_cycles=self.drive_cycles_small,
                costs=self.params['Vehicles'][k]['Costs'],
                k=k, p=p, y=y,
                ctax=ctax, lcfs=lcfs,
                autonomous_penetration=0,
                foresight=foresight, gvwl_increase=gvwl_increase, break_mandate=break_mandate, accelerated_retirement=accelerated_retirement,
            )
            for k in self.K for p in self.P[k] for y in self.Y if y <= START_YEAR
        }
        # Previous stock
        for k in self.K:
            for y in range(START_YEAR-MAX_AGE+1, START_YEAR):
                self.stock[k, 'D', y, START_YEAR] = self.activity_requirement[k,START_YEAR] * (1+GROWTH_RATE)**(y-START_YEAR) * self.vehicles[k, 'D', y].survival_rate[START_YEAR-y] / \
                    sum(self.vehicles[k, 'D', 2000].annual_activity[a] * self.vehicles[k, 'D', 2000].survival_rate[a] * (1+GROWTH_RATE)**(-a) for a in range(MAX_AGE))
            lcfs.baseline_fuel_consumption[k] = self.vehicles[k, 'D', 2025].fuel_consumption['Diesel'][0]

        # Year-by-year
        penalty = 0
        for t in self.T:
            # Roll-over from previous years
            if t > START_YEAR:
                for k in self.K:
                    for p in self.P[k]:
                        for y in range(t-MAX_AGE+1, t):
                            self.stock[k,p,y,t] = self.stock[k,p,y,t-1] * self.vehicles[k,p,y].survival_rate[t-y] / self.vehicles[k,p,y].survival_rate[t-1-y]
            # Create vehicles for this model year, t.
            for k in self.K:
                for p in self.P[k]:
                    self.vehicles[k,p,t] = Vehicle(
                        params=self.params['Vehicles'][k]['Shared'] | self.params['Vehicles'][k]['Powertrains'][p]['params'],
                        fuels={key: value for key, value in self.fuels.items() if key in self.params['Vehicles'][k]['Powertrains'][p]['fuel']},
                        p_fuel=self.params['Vehicles'][k]['Powertrains'][p]['fuel'],
                        drive_cycles=self.drive_cycles_small,
                        costs=self.params['Vehicles'][k]['Costs'],
                        k=k, p=p, y=t,
                        ctax=ctax, lcfs=lcfs,
                        autonomous_penetration=1/(1 + np.exp(-0.5*(t - self.params['autonomous_t50'])))*autonomous_permits.permits[p],
                        foresight=foresight, gvwl_increase=gvwl_increase, break_mandate=break_mandate, accelerated_retirement=accelerated_retirement,
                        zev_rebate=zev_rebate,
                    )
            # Calculate the market-share for each vehicle type
            N = 1
            target = zev_mandate.targets[t==self.T][0]
            if target != 0:
                N = 100
            penalty=0
            temp=0
            for n in range(N):
                # Market shares
                for k in self.K:
                    self.calculate_market_share(k, t)
                    # Activity requirements
                    average_vehicle_activity = np.sum([self.vehicles[k,p,t].annual_activity[0] * self.vehicles[k,p,t].market_share for p in self.P[k]])
                    activity_met = sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_activity[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t))
                    activity_shortfall = self.activity_requirement[k,t] - activity_met
                    new_purchases = activity_shortfall / average_vehicle_activity
                    for p in self.P[k]:
                        self.stock[k,p,t,t] = (new_purchases * self.vehicles[k,p,t].market_share).astype('float32')
                # ZEV mandate penalty
                if target != 0:
                    p_zev = sum(self.stock[k,p,t,t] for k in self.K for p in ZEV_POWERTRAINS) \
                        / sum(self.stock[k,p,t,t] for k in self.K for p in self.P[k])
                    penalty = penalty*0.5 + zev_mandate.penalty * np.max([0, (target - p_zev)/(1.0-p_zev)])*0.5
                    for k in self.K:
                        for p in NON_ZEV_POWERTRAINS:
                            self.vehicles[k,p,t].tco.zev_mandate = penalty
                            self.vehicles[k,p,t].annual_cost.zev_mandate[0] = penalty
                        for p in ZEV_POWERTRAINS:
                            rebate = min([penalty, penalty * (1 - p_zev) / p_zev])
                            self.vehicles[k,p,t].tco.zev_mandate = -rebate
                            self.vehicles[k,p,t].annual_cost.zev_mandate[0] = -rebate
                    if abs(temp-penalty) < 1:
                        break
                    if n > 50:
                         raise ValueError("ZEV penalty failed to converge!")
                    temp = penalty
        self.total_stock = {
            (k, p, t): sum(self.stock[k,p,y,t] for y in range(t-MAX_AGE+1, t+1)).astype('float32')
            for k in self.K for p in self.P[k] for t in self.T
        }
        self.sales = {
            (k, p, t): self.stock[k,p,t,t] for k in self.K for p in self.P[k] for t in self.T
        }
        # Fuel demand
        self.fuel_usage = {(k,f,t): np.float32(0.0) for k in self.K for f in self.fuels.keys() for t in self.T}
        self.energy_by_fuel = {(k,f,t): np.float32(0.0) for k in self.K for f in self.fuels.keys() for t in self.T}
        for t in self.T:
            for f in self.fuels.keys():
                for k in self.K:
                    for p in self.P[k]:
                        if f in self.params['Vehicles'][k]['Powertrains'][p]['fuel']:
                            self.fuel_usage[k,f,t] += sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_fuel[f][t-y] / self.fuels[f].recharge_efficiency for y in range(t-MAX_AGE+1, t+1)).astype('float32')
                            self.energy_by_fuel[k,f,t] += sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].energy_consumption[t-y] * self.vehicles[k,p,y].annual_distance[t-y] * self.vehicles[k,p,y].p_fuel[f][t-y] for y in range(t-MAX_AGE+1, t+1)).astype('float32')
        self.calculate_emissions()
        self.system_costs = {}
        for k in self.K:
            self.system_costs[k] = self.SystemCosts(
                capital=np.array([sum(self.stock[k,p,t,t] * self.vehicles[k,p,t].capital.total for p in self.P[k]) for t in self.T]).astype('float32'),
                fuel=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_fuel[f][t-y] * self.fuels[f].cost.fuel[t-self.Y[0]] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1) for f in self.vehicles[k,p,y].annual_fuel.keys()) for t in self.T]).astype('float32'),
                operational=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_cost.operational[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1))
                                      + sum(self.stock[k,'FC',y,t] * self.vehicles[k,'FC',y].fc_replacements[t-y] * self.vehicles[k,'FC',y].fc_size * self.vehicles[k,'FC',y].costs['fc'][t-self.Y[0]] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]).astype('float32'),
                driver=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_cost.driver[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]).astype('float32'),
            )
        self.policy_costs = {}
        for k in self.K:
            self.policy_costs[k] = {
                'carbon_tax': np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_cost.carbon_tax[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]).astype('float32'),
                'lcfs': np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_cost.lcfs[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]).astype('float32'),
                'zev_mandate': np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_cost.zev_mandate[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]).astype('float32'),
                'zev_rebate': np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_cost.zev_rebate[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]).astype('float32'),
            }
        self.average_external = self.ExternalCosts(
            accidents=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_distance[t-y] * 0.156 * 1.8361 for k in self.K for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]),
            air_pollution=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_fuel[f][t-y] * self.vehicles[k,p,y].fuels[f].air_pollution[x]/1e6 * self.vehicles[k,p,y].pollution_cost[x] for k in self.K for p in self.P[k] for y in range(t-MAX_AGE+1, t+1) for x in self.vehicles[k,p,y].pollution_cost.keys() for f in self.vehicles[k,p,y].fuels.keys()) for t in self.T]),
            ghg_emissions=np.array([sum(self.emissions[k].total[t==self.T])/1000 * SOCIAL_COST_OF_CARBON_0 * (1 + 0.03) ** (t - START_YEAR) for t in self.T]),
            noise=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_distance[t-y] * 0.03 * 1.8361 for k in self.K for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]),
            congestion=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_distance[t-y] * (0.0115 if k=='Sleeper' else 0.4116 if k=='Class-8 Straight' else 0.0915) for k in self.K for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]),
            habitat_loss=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_distance[t-y] * 0.075 for k in self.K for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]),
        )
        self.marginal_external = self.ExternalCosts(
            accidents=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_distance[t-y] * 0.4002 * 1.8361 for k in self.K for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]),
            air_pollution=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_fuel[f][t-y] * self.vehicles[k,p,y].fuels[f].air_pollution[x]/1e6 * self.vehicles[k,p,y].pollution_cost[x] for k in self.K for p in self.P[k] for y in range(t-MAX_AGE+1, t+1) for x in self.vehicles[k,p,y].pollution_cost.keys() for f in self.vehicles[k,p,y].fuels.keys()) for t in self.T]),
            ghg_emissions=np.array([sum(self.emissions[k].total[t==self.T])/1000 * SOCIAL_COST_OF_CARBON_0 * (1 + 0.03) ** (t - START_YEAR) for t in self.T]),
            noise=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_distance[t-y] * 0.03 * 1.8361 for k in self.K for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]),
            congestion=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_distance[t-y] * (0.347 if k=='Sleeper' else 1.023 if k=='Class-8 Straight' else (0.347+1.023)/2) for k in self.K for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]),
            habitat_loss=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].annual_distance[t-y] * 0.075 for k in self.K for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]),
        )
        # Plotting
        # self.plots = self.Plots(self)

    @dataclass
    class Emissions: # end of life
        embodied: np.ndarray
        fuel_supply: np.ndarray
        fuel_combustion: np.ndarray
        end_of_life: np.ndarray

        @property
        def total(self):
            return self.fuel_supply + self.fuel_combustion + self.embodied + self.end_of_life

    @dataclass
    class SystemCosts:
        capital: np.ndarray
        fuel: np.ndarray
        operational: np.ndarray
        driver: np.ndarray

        @property
        def total(self):
            return sum(getattr(self, f.name) for f in fields(self))

    @dataclass
    class ExternalCosts:
        accidents: np.ndarray
        air_pollution: np.ndarray
        ghg_emissions: np.ndarray
        noise: np.ndarray
        congestion: np.ndarray
        habitat_loss: np.ndarray

    def calculate_market_share(self, k, t):
        not_production_limited = set(self.P[k])
        market_remaining = 1.0
        for i in range(10):
            temp = len(not_production_limited)
            for p in not_production_limited:
                if t == START_YEAR:
                    if p == 'D':
                        prev=1.0
                    else:
                        prev=0.0
                else:
                    prev = self.market_share[k,p,t-1]
                if hasattr(self.vehicles[k,p,t], "market_limit"):
                    limit = market_share_limit(prev, self.vehicles[k,p,t].market_limit)
                else:
                    limit = market_share_limit(prev)
                logit_share = market_remaining * np.exp(PRICE_LAMBDA * (self.vehicles[k, p, t].npv)) / sum(np.exp(PRICE_LAMBDA * self.vehicles[k, p, t].npv) for p in not_production_limited)
                if limit <= logit_share:
                    self.market_share[k,p,t] = limit
                    self.vehicles[k,p,t].market_share = limit
                    not_production_limited = not_production_limited - {p}
                    market_remaining -= limit
                else:
                    self.market_share[k,p,t] = logit_share
                    self.vehicles[k,p,t].market_share = logit_share
            if temp == len(not_production_limited):
                break

    def calculate_emissions(self):
        self.emissions = {}
        for k in self.K:
            self.emissions[k] = self.Emissions(
                embodied=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].emissions.embodied.annual[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]).astype('float32'),
                fuel_supply=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].emissions.fuel_supply[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]).astype('float32'),
                fuel_combustion=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].emissions.fuel_combustion[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]).astype('float32'),
                end_of_life=np.array([sum(self.stock[k,p,y,t] * self.vehicles[k,p,y].emissions.end_of_life[t-y] for p in self.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.T]).astype('float32')
            )

    def prune_memory(self):
        # Prune memory
        keep = ['tco', 'npv', 'total_revenue']
        for v in self.vehicles.values():
            for attr in list(v.__dict__.keys()):
                if attr not in keep:
                    del v.__dict__[attr]

    class Plots:
        def __init__(self, fleet):
            self.fleet = fleet
            self.sample_years = np.array([2025, 2030, 2035, 2040, 2045, 2050])
        
        def annual_distance(self):
            plt.plot()
            plt.title('Annual disance')
            for k in self.fleet.K:
                distance = [sum(fleet.stock[k,p,y,t] * fleet.vehicles[k,p,y].annual_distance[t-y] for p in self.fleet.P[k] for y in range(t-MAX_AGE+1, t+1))/sum(fleet.stock[k,p,y,t] for p in self.fleet.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.fleet.T]
                plt.plot(distance, label=k)
            # Average
            distance = [sum(fleet.stock[k,p,y,t] * fleet.vehicles[k,p,y].annual_distance[t-y] for k in self.fleet.K for p in self.fleet.P[k] for y in range(t-MAX_AGE+1, t+1))/sum(fleet.stock[k,p,y,t] for k in self.fleet.K for p in self.fleet.P[k] for y in range(t-MAX_AGE+1, t+1)) for t in self.fleet.T]
            plt.plot(distance, label='Average')
            plt.legend()

        def vehicle_mass(self):
            for k in self.fleet.K:
                plt.figure()
                plt.title(f'Vehicle mass ({k})')
                width = 0.4
                gap = 0.1
                for y in self.sample_years:
                    for p in self.fleet.P[k]:
                        plt.gca().set_prop_cycle(None)
                        offset = np.argwhere(np.array(self.fleet.P[k]) == p).flatten()[0] * (width + gap) - (width + gap) * (len(self.fleet.P[k]) - 1) / 2
                        m = self.fleet.vehicles[k,p,y].mass
                        plt.bar(y + offset, m.frame, width=width)
                        plt.bar(y + offset, m.drivetrain, bottom=m.frame, width=width)
                        plt.bar(y + offset, m.ess, bottom=m.frame+m.drivetrain, width=width)
                        plt.bar(y + offset, m.trailer, bottom=m.frame+m.drivetrain+m.ess, width=width)
                        plt.bar(y + offset, m.payload, bottom=m.frame+m.drivetrain+m.ess+m.trailer, width=width)
                        plt.text(y + offset, m.total*1.01, p, ha='center', va='bottom', fontsize=8, rotation=90)
                plt.ylim(0, plt.ylim()[1]*1.3) 
                plt.legend(['Frame', 'Drivetrain', 'ESS', 'Trailer', 'Payload'])

        def lca_emissions(self):
            for k in self.fleet.K:
                plt.figure()
                plt.title(f'LCA Emissions ({k})')
                plt.xlabel('Model Year')
                plt.ylabel('Emissions (tCO2)')
                width = 0.4
                gap = 0.1
                for y in self.sample_years:
                    for p in self.fleet.P[k]:
                        plt.gca().set_prop_cycle(None)
                        offset = np.argwhere(np.array(self.fleet.P[k]) == p).flatten()[0] * (width + gap) - (width + gap) * (len(self.fleet.P[k]) - 1) / 2
                        emissions = self.fleet.vehicles[k,p,y].emissions
                        bottom=0
                        for f in fields(emissions):
                            values = getattr(emissions, f.name)
                            if f.name == 'embodied':
                                values = values.annual
                            value = sum(values) / 1000
                            plt.bar(y+offset, value, bottom=bottom, width=width, label=f.name)
                            bottom += value
                        plt.text(y + offset, emissions.lca / 1000 * 1.01, p, ha='center', va='bottom', fontsize=8, rotation=90)
                plt.legend([f.name for f in fields(emissions)], loc='upper left', bbox_to_anchor=(1,1))
                plt.ylim(0, plt.ylim()[1]*1.1)
            
        def capital_cost(self):
            for k in self.fleet.K:
                plt.figure()
                plt.title(f'Capital Cost ({k})')
                width = 0.4
                gap = 0.1
                for y in self.sample_years:
                    for p in self.fleet.P[k]:
                        plt.gca().set_prop_cycle(None)
                        offset = np.argwhere(np.array(self.fleet.P[k]) == p).flatten()[0] * (width + gap) - (width + gap) * (len(self.fleet.P[k]) - 1) / 2
                        c = self.fleet.vehicles[k,p,y].capital
                        bottom=0
                        for f in fields(c):
                            value = getattr(c, f.name)
                            plt.bar(y+offset, value, bottom=bottom, width=width, label=f.name)
                            bottom += value
                        plt.text(y + offset, c.total*1.01, p, ha='center', va='bottom', fontsize=8, rotation=90)
                plt.legend([f.name for f in fields(c)], loc='upper left', bbox_to_anchor=(1,1))
                plt.ylim(0, plt.ylim()[1]*1.1)

        def tco(self):
            for k in self.fleet.K:
                plt.figure()
                plt.title(f'TCO ({k})')
                width = 0.4
                gap = 0.1
                for y in self.sample_years:
                    for p in self.fleet.P[k]:
                        plt.gca().set_prop_cycle(None)
                        offset = np.argwhere(np.array(self.fleet.P[k]) == p).flatten()[0] * (width + gap) - (width + gap) * (len(self.fleet.P[k]) - 1) / 2
                        tco = self.fleet.vehicles[k,p,y].tco
                        bottom=0
                        for f in fields(tco):
                            value = getattr(tco, f.name)
                            plt.bar(y+offset, value, bottom=bottom, width=width, label=f.name)
                            bottom += value
                        plt.text(y + offset, tco.total*1.01, p, ha='center', va='bottom', fontsize=8, rotation=90)
                plt.legend([f.name for f in fields(tco)], loc='upper left', bbox_to_anchor=(1,1))
                plt.ylim(0, plt.ylim()[1]*1.1)

        def total_revenue(self):
            for k in self.fleet.K:
                plt.figure()
                plt.title(f'Total Revenue ({k})')
                plt.xlabel('Years')
                plt.ylabel('$')
                width = 0.4
                gap = 0.1
                for y in self.sample_years:
                    # plt.gca().set_prop_cycle(None)
                    for p in self.fleet.P[k]:
                        offset = np.argwhere(np.array(self.fleet.P[k]) == p).flatten()[0] * (width + gap) - (width + gap) * (len(self.fleet.P[k]) - 1) / 2
                        value = self.fleet.vehicles[k,p,y].total_revenue
                        plt.bar(y + offset, value, width=width, label=p)
                        plt.text(y + offset, value*1.01, p, ha='center', va='bottom', fontsize=8, rotation=90)

        def npv(self):
            for k in self.fleet.K:
                plt.figure()
                plt.title(f'NPV ({k})')
                plt.xlabel('Years')
                plt.ylabel('$')
                width = 0.4
                gap = 0.1
                for y in self.sample_years:
                    # plt.gca().set_prop_cycle(None)
                    for p in self.fleet.P[k]:
                        offset = np.argwhere(np.array(self.fleet.P[k]) == p).flatten()[0] * (width + gap) - (width + gap) * (len(self.fleet.P[k]) - 1) / 2
                        npv = self.fleet.vehicles[k,p,y].npv
                        plt.bar(y + offset, npv, width=width, label=p)
                        plt.text(y + offset, npv*1.01, p, ha='center', va='bottom', fontsize=8, rotation=90)

        def fuel_consumption(self, p=None):
            for k in self.fleet.K:
                for p in self.fleet.P[k]:
                    for f in self.fleet.vehicles[k,p,2025].fuels.keys():
                        plt.figure()
                        values = [self.fleet.vehicles[k,p,t].fuel_consumption[f][0] for t in fleet.T]
                        plt.plot(self.fleet.T, values)
                        plt.ylabel('f')
                        plt.title(f'{k}, {p}, {f}')
        
        def fuel_consumption_age(self, k, p, y):
            plt.figure()
            plt.title(f'Fuel Consumption by Age ({k}, {p})')
            plt.plot(self.fleet.vehicles[k,p,y].A, self.fleet.vehicles[k,p,y].fuel_consumption)
            plt.ylim(0, plt.ylim()[1]*1.1)

        def annual_emissions(self, k, p, y):
            v = self.fleet.vehicles[k, p, y]
            plt.figure()
            plt.title('Annual vehicle emissions')
            plt.plot(v.operation_years, v.emissions.embodied.annual, label='Embodied')
            plt.plot(v.operation_years, v.emissions.fuel_combustion)

        def market_share(self):
            width = 1.0
            for k in self.fleet.K:
                powertrains = self.fleet.P[k]
                colours = {p: plt.cm.tab10(i % 10) for i, p in enumerate(powertrains)}
                plt.figure()
                plt.xlabel("Model Year")
                plt.ylabel("Market Share (%)")
                plt.ylim([0, 140])
                for t in self.fleet.T:
                    bottom=0
                    for p in powertrains:
                        bar = plt.bar(t, self.fleet.vehicles[k,p,t].market_share*100, bottom=bottom, width=width, color=colours[p], label=p)
                        bottom += self.fleet.vehicles[k,p,t].market_share*100
                plt.ylim(0, 100)
                plt.legend(powertrains, loc='upper left', bbox_to_anchor=(1,1))

        def initial_stock(self):
            for k in self.fleet.K:
                plt.figure()
                plt.title(f'Initial Stock {k}')
                plt.xlabel('Year')
                plt.ylabel('Stock')
                for y in range(START_YEAR-MAX_AGE+1, START_YEAR):
                    plt.bar(y, self.fleet.stock[k, 'D', y, START_YEAR], color='blue')

        def stock(self):
            for k in self.fleet.K:
                plt.figure()
                plt.title(f'Stock {k}')
                plt.xlabel('Year')
                plt.ylabel('Stock')
                plt.plot(self.fleet.T, [sum(self.fleet.stock[k,p,y,t] for y in range(t-MAX_AGE+1, t) for p in self.fleet.P[k]) for t in self.fleet.T], label='Total')
                for p in self.fleet.P[k]:
                    plt.plot(self.fleet.T, [sum(self.fleet.stock[k,p,y,t] for y in range(t-MAX_AGE+1, t)) for t in self.fleet.T], label=p)
                plt.legend(loc='upper left', bbox_to_anchor=(1,1))

        def stock_by_age(self):
            fig, ax = plt.subplots(1, len(self.fleet.K))
            for i, k in enumerate(self.fleet.K):
                ax[i].set_xlabel('Year')
                ax[i].set_ylabel('Stock')
                for y in fleet.Y[1:]:
                    T = np.arange(max(START_YEAR, y), min(y+MAX_AGE, END_YEAR))
                    ax[i].plot(T, [sum(self.fleet.stock[k,p,y,t] for p in self.fleet.P[k]) for t in T], label=str(y))
                    # for p in self.fleet.P[k]:
                    #     plt.plot(T, [self.fleet.stock[k,p,y,t] for t in T], label=p)
                # plt.legend(loc='upper left', bbox_to_anchor=(1,1))

        def sales(self):
            for k in self.fleet.K:
                plt.figure()
                plt.title(f'Sales {k}')
                plt.xlabel('Year')
                plt.ylabel('Sales')
                plt.plot(self.fleet.T, [sum(self.fleet.stock[k,p,t,t] for p in self.fleet.P[k]) for t in self.fleet.T], label='Total')
                for p in self.fleet.P[k]:
                    plt.plot(self.fleet.T, [self.fleet.stock[k,p,t,t] for t in self.fleet.T], label=p)
                plt.legend(loc='upper left', bbox_to_anchor=(1,1))

        def zev_penetration(self): # Change to subplots. Add a plot for average penalty.
            plt.figure()
            plt.title(f'ZEV penetration')
            plt.xlabel('Year')
            plt.ylabel('Stock')
            p_zev = np.array([sum(self.fleet.stock[k,p,t,t] for k in self.fleet.K for p in ZEV_POWERTRAINS) / sum(self.fleet.stock[k,p,t,t] for k in self.fleet.K for p in self.fleet.P[k]) for t in self.fleet.T])
            penalty = self.fleet.zev_mandate.penalty
            non_compliance_cost = [self.fleet.vehicles['Sleeper', 'D', t].tco.zev_mandate for t in self.fleet.T]
            compliance_value = [-self.fleet.vehicles['Sleeper', 'BE', t].tco.zev_mandate for t in self.fleet.T]
            per_hdt = non_compliance_cost * (1-p_zev) - compliance_value * p_zev
            plt.plot(self.fleet.T, p_zev*100, label='Average')
            for k in self.fleet.K:
                plt.plot(self.fleet.T, [sum(self.fleet.stock[k,p,t,t] for p in ZEV_POWERTRAINS) / sum(self.fleet.stock[k,p,t,t] for p in self.fleet.P[k])*100 for t in self.fleet.T], label=k)
            if sum(self.fleet.zev_mandate.targets) > 0:
                plt.plot(self.fleet.T, self.fleet.zev_mandate.targets*100, label='Target')
            plt.ylim([0,110])
            plt.legend(loc='upper left', bbox_to_anchor=(1,1))
            plt.figure()
            plt.xlabel('Years')
            plt.ylabel('Compliance cost ($)')
            plt.plot(self.fleet.T, non_compliance_cost, label='Non-ZEV added cost')
            plt.plot(self.fleet.T, compliance_value, label='ZEV added value')
            plt.plot(self.fleet.T, per_hdt, label='Added cost per HDT sale')

            plt.legend()
            
        def emissions(self):
            for k in self.fleet.K:
                plt.figure()
                plt.title(f'Emissions {k}')
                plt.xlabel('Year')
                plt.ylabel('Emissions (MtCO2)')
                emissions = self.fleet.emissions[k]
                for f in fields(emissions):
                    values = getattr(emissions, f.name)
                    plt.plot(self.fleet.T, values/1e9, label=f.name)
                plt.legend()
            plt.figure()
            total = np.zeros(len(values))
            for f in fields(emissions):
                values = np.sum([getattr(self.fleet.emissions[k], f.name) for k in self.fleet.K], axis=0)
                plt.plot(self.fleet.T, values/1e9, label=f.name)
                total += values
            plt.plot(self.fleet.T, total/1e9, label='Total')
            plt.legend()


if __name__ == "__main__":
    n_runs = 1
    params = d.PARAMS
    # set_constant_params(params)
    T = params['Years']['T']
    drive_cycles = {key: DriveCycle(df['path']) for key, df in params['Drive Cycles'].items()}

    inputs_distributions = dict(get_uncertainty_distributions(params))
    uncertain_keys = inputs_distributions.keys()
    np.random.seed(0)
    samples = np.random.rand(n_runs, len(inputs_distributions)).astype('float32')
    uncertain_cps = dict(zip(inputs_distributions.keys(), samples[0]))
    fleet = Fleet(params,
        uncertain_cps=uncertain_cps,
        drive_cycles=drive_cycles,
        # gvwl_increase=True,
        # ctax=CarbonTax(years=params['Years']['Y'], price={'2025': 80, '2030': 170, '2050': 170}),
        # foresight=False,
        # accelerated_retirement=True,
        # break_mandate=0.75,
        # pyrolysis=False,
        # pyrolysis_elec=True,
        # zev_mandate=ZEVMandate(
        #     params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
        #     penalty=30_000,
        #     rebates=False,
        # ),
        # zev_rebate=0.33
    )

    plots = fleet.Plots(fleet)
    # plots.zev_penetration()
    # plots.annual_distance()
    # plots.stock()
    # plots.npv()
    plots.sales()


