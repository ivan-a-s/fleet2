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
"""
import numpy as np
import copy
import scipy.stats as stats
import fastsim as fsim
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
    def __init__(self, params, fuels, drive_cycles):
        # Add parameters
        self.__dict__.update(params)
        self.fuels = {key: fuels[key] for key in list(self.fuels)}
        self.drive_cycles = {str(key): drive_cycles[str(key)] for key in np.unique(self.drive_cycle)}
        del params, fuels, drive_cycles

        # Calculate mass
        self.calculate_mass()

        # Calculate energy and fuel consumption for each drive cycle and energy pathway.
        self.accessory_load = self.accessory_demand # SORT
        self.calculate_fuel_consumption()

    #     x = self.hdt_sim()

    #     # Activity
    #     # self.range
    
    # # def 
    
    # def hdt_sim(self):
    #     # In FASTSim 3, you access Vehicle and Cycle directly from the main module
    #     veh = fsim.Vehicle.from_resource("2021_Class8_Diesel_Truck.csv")
    #     cyc = fsim.Cycle.from_resource("heavy_duty_drive_cycle.csv")
        
    #     # Use walk() instead of sim_drive() for the FASTSim 3 Rust engine
    #     sim = fsim.SimDrive(veh, cyc)
    #     sim.walk() 
        
        # return sim.mpgge

    def calculate_fuel_consumption(self):
        # Energy consumption
        for key, drive_cycle in list(self.drive_cycles.items()):
            drive_cycle['energy_consumption'] = self.calculate_energy_consumption(key)
            drive_cycle['fuel_consumption'] = {}
        # Fuel consumption
        for path, properties in self.energy_pathways.items():
            fuel = properties['fuel']
            properties['efficiency'] = np.prod([self.components[component]['efficiency'] for component in path])
            for key, drive_cycle in list(self.drive_cycles.items()):
                drive_cycle['fuel_consumption'][fuel] = drive_cycle['energy_consumption'] / properties['efficiency'] / self.fuels[fuel]['lhv']

    def calculate_mass(self):
        self.mass = {
            'frame': self.frame_mass,
        }
        if self.trailer_mass > 0:
            self.mass['trailer'] = self.trailer_mass
        for key, component in self.components.items():
            if component['type'] == 'converter':
                self.mass[key] = component['mass']
            elif component['type'] == 'ess':
                self.mass[key] = component['mass_per_unit'] * component['capacity']
            elif component['type'] == 'transmission':
                self.mass[key] = component['mass']
            else:
                raise Exception('Unrecognised component type.')
        self.unloaded_mass = sum(v for v in self.mass.values())
        self.payload = self.default_payload * (1 - self.p_weighed_out * (1 - (self.gvwl - self.unloaded_mass)/(self.gvwl - self.default_unloaded_mass)))
        self.total_mass = self.unloaded_mass + self.payload

    def calculate_energy_consumption(v, drive_cycle='long_haul'):
        if drive_cycle == 'long_haul':
            return (
                184468.0062 +
                v.total_mass * 65.678763 +
                v.roll_coefficient * 0 + 
                v.drag_coefficient * 4350611.430231 + 
                v.regen_efficiency * -292181.3162943 +
                v.accessory_load * 47.472692 +
                getattr(v, "motor_size", 0) * -82.872142
            )
        elif drive_cycle == 'regional_haul':
            return (
                100174.0174 +
                v.total_mass * 62.269866 +
                v.roll_coefficient * 0 + 
                v.drag_coefficient * 3771521.906184 + 
                v.regen_efficiency * -145997.061314 +
                v.accessory_load * 51.988641 +
                getattr(v, "motor_size", 0) * -50.706751
            )
        else:
            return (
                31692.4468 +
                v.total_mass * 136.098951 +
                v.roll_coefficient * 0 + 
                v.drag_coefficient * 3849936.569754 + 
                v.regen_efficiency * -39075.473546 +
                v.accessory_load * 96.047578 +
                getattr(v, "motor_size", 0) * -10.420126
            )



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
                self.vehicles[k,'dice',y] = Vehicles(vehicle_params, fuel_params, drive_cycle_params)

    def select_vehicle_params(self, all_vehicle_params, k, p, y):
        # Shared, powertrain-specific, powertrain component-specific
        vehicle_params = all_vehicle_params['types'][k]['shared']
        vehicle_params |= all_vehicle_params['types'][k]['powertrains'][p]
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