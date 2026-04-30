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
 - Pyaload by drivecycle not vehicle type
 - Vehicle costs by type should not all eb sampled separately.
 - Separate classes for ESS and powertrains?
 - Accessory load
 - Should roll coeff vary by number of wheels?
 - Re-calculate mass every year?
"""
import numpy as np
import copy
import scipy.stats as stats
import json
from pprint import pprint

# Running parameters
from data import *
N_RUNS = 1


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

def set_year(input_dict, year=START_YEAR, years=np.arange(START_YEAR-MAX_AGE, END_YEAR+1)):
    if isinstance(input_dict, np.ndarray):
        input_dict = input_dict[np.where(years == year)[0][0]]
    elif isinstance(input_dict, dict):
        for key, value in input_dict.items():
            input_dict[key] = set_year(value, year, years)
    return input_dict


class Vehicles:
    def __init__(self, params, fuels, drive_cycles, p):
        # Add parameters
        self.params = copy.copy(params)
        self.fuels = {key: fuels[key] for key in list(params['fuels'])}
        self.drive_cycles = {str(key): drive_cycles[str(key)] for key in np.unique(params['drive_cycle'])}
        self.p = p
        self.age = np.arange(self.params['max_age'], dtype=int)
        
        # Remove unnecessary data
        del params, fuels, drive_cycles, p

        # Vehicle mass
        self.calculate_mass()

        # Simulate vehicle year-by-year
        self.fuel_consumption = {fuel: np.zeros(len(self.age)) for fuel in self.fuels.keys()}
        self.range = np.zeros(len(self.age))
        for a in self.age:
            # Fuel consumption
            for f in self.fuels.keys():
                self.fuel_consumption[f][a] = self.calculate_fuel_consumption(self.params['drive_cycle'][a], 0.45)
            # Range and activity
            self.range = max((self.params['fuel_capacity'][f] * self.usable_capacity[f][a]) / self.fuel_consumption[f])


    def calculate_fuel_consumption(self, drive_cycle, efficiency):
        # Load file
        path = 'drive_cycles/'+self.p+'_'+drive_cycle+'.json'
        with open(path, 'r') as f:
            model_params =  json.load(f)
        # Features
        base = {
            'mass': self.total_mass,
            'drag_coef': self.params['drag_coef'],
            'accessory_load': self.params['accessory_load'],
            'inv_eff': 1 / efficiency,
        }
        # Estimate fuel consumption
        total = model_params['intercept']
        coefs = model_params['features']
        for name, coef in coefs.items():
            if " " in name:
                v1, v2 = name.split(" ")
                total += (base[v1] * base[v2]) * coef
            else:
                total += base[name] * coef
                
        return total

    def calculate_mass(self):
        self.mass = {
            'frame': self.params['frame_mass'],
        }
        if self.params['trailer_mass'] > 0:
            self.mass['trailer'] = self.params['trailer_mass']
        for key, component in self.params['components'].items():
            if component['type'] == 'converter':
                self.mass[key] = component['mass']
            elif component['type'] == 'ess':
                self.mass[key] = component['mass_per_unit'] * component['capacity']
            elif component['type'] == 'transmission':
                self.mass[key] = component['mass']
            else:
                raise Exception('Unrecognised component type.')
        self.unloaded_mass = sum(v for v in self.mass.values())
        self.mass['payload'] = self.params['default_payload'] * (1 - self.params['p_weighed_out'] * (1 - (self.params['gvwl'] - self.unloaded_mass)/(self.params['gvwl'] - self.params['default_unloaded_mass'])))
        self.total_mass = self.unloaded_mass + self.mass['payload']



class Fleet:
    def __init__(self, params, param_cps, start_year=START_YEAR, end_year=END_YEAR, max_age=25):
        # Copy params and set uncertain inputs.
        self.params = copy.deepcopy(params)
        self.realise_uncertainties(param_cps)
        params = None

        # Sets
        self.model_years = np.arange(start_year-max_age, end_year+1)
        self.vehicle_types = self.params['vehicles']['types'].keys()

        # Apply policies (pyrolysis etc.)

        self.vehicles = {}
        for y in range(start_year-max_age, start_year):
            for k in self.vehicle_types:
                # Vehicle, fuel, and drive-cycle parameters
                vehicle_params = self.select_vehicle_params(copy.deepcopy(self.params['vehicles']), k, 'dice', y)
                fuel_params = self.params['fuels']
                drive_cycle_params = self.params['drive_cycles']
                # Create vehicle
                self.vehicles[k,'dice',y] = Vehicles(vehicle_params, fuel_params, drive_cycle_params, p='dice')

    def select_vehicle_params(self, all_vehicle_params, k, p, y):
        # Shared for that vehicle type
        vehicle_params = all_vehicle_params['types'][k]['shared']
        # Specific to that powertrain
        vehicle_params |= all_vehicle_params['types'][k]['powertrains'][p]
        # Component parameters
        for key, component in list(vehicle_params['components'].items()):
            other = all_vehicle_params['components'][component['type']][key]
            component.update({k: v for k, v in other.items() if k not in component})
        # Set parameters using the model year
        exclude = ['target_distance', 'drive_cycle', 'survival_rate']
        for key, value in vehicle_params.items():
            if key not in exclude:
                vehicle_params[key] = set_year(value, year=y)
        return (vehicle_params)

    def realise_uncertainties(self, param_cps):
        """ Set inputs using input distributions and cumulative probibilities."""
        for keys, cp in param_cps.items():
            # Navigate to the parent of the final key
            d = self.params
            for k in keys[:-1]:
                d = d[k]
            last_key = keys[-1]
            # Update the value using set_param
            if keys in param_cps.keys():
                d[last_key] = set_param(d[last_key], cp=cp)




if __name__ == "__main__":
    params = convert_to_float32(PARAMS)
    # Randomly sample the input distributions.
    np.random.seed(0)
    inputs_distributions = dict(get_uncertainty_distributions(PARAMS))
    uncertain_keys = inputs_distributions.keys()
    samples = np.random.rand(N_RUNS, len(inputs_distributions)).astype('float32')

    # Run the model
    for iRun in range(N_RUNS):
        fleet = Fleet(
            params=params,
            param_cps=dict(zip(inputs_distributions.keys(), samples[iRun])),
        )

    v = fleet.vehicles['sleeper', 'dice', 2020]