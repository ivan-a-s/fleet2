""" Calculate fuel consumption for a vehicle object.
To do: deal with peak efficiency peoperly.
"""
import fastsim as fsim
import json
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import time

# Load drive-cycle.
def load_drive_cycle(fname="drive_cycles/regional_haul.json"):
    with open(fname, "r") as f:
        cyc_json_str = f.read()
    cyc_dict = json.loads(cyc_json_str)
    return fsim.Cycle.from_pydict(cyc_dict)

DRIVE_CYCLES = {
    # 'short_haul': load_drive_cycle('drive_cycles/short_haul.json'),
    # 'regional_haul': load_drive_cycle('drive_cycles/regional_haul.json'),
    # 'long_haul': load_drive_cycle('drive_cycles/long_haul.json'),
    'udds_hdt': load_drive_cycle('drive_cycles/udds_hdt.json'),
    'cruise_hdt': load_drive_cycle('drive_cycles/cruise_hdt.json'),
    'short_haul': load_drive_cycle('drive_cycles/udds_hdt.json'),
    'regional_haul': load_drive_cycle('drive_cycles/udds_hdt.json'),
    'long_haul': load_drive_cycle('drive_cycles/cruise_hdt.json'),
}

# Create the vehicle.
MAX_ENGINE_POWER = 500
VEHICLES = {
    'dice': { # Adapted Line Haul Conv.
        "name": "hdt_diesel",
        "year": 2025,
        "mass_kilograms": 30_000,
        "pwr_aux_base_watts": 3_000.0,
        "chassis": {
            'cg_height_meters': 1.3, # Centre of gravity
            'drag_coef': 0.55,
            'drive_axle_weight_frac': 0.59,
            'drive_type': 'RWD',
            'frontal_area_square_meters': 9.2,
            'num_wheels': 18,
            'wheel_base_meters': 2.7536,
            'wheel_fric_coef': 0.7,
            'wheel_inertia_kilogram_square_meters': 2.0,
            'wheel_radius_meters': 0.5,
            'wheel_rr_coef': 0.007
        },
        'pt_type': {
            'Conv': {
                'alt_eff': 1.0,
                'fc': {
                    'pwr_idle_fuel_watts': 0.0,
                    'pwr_ramp_lag_seconds': 1.0,
                    'pwr_out_max_init_watts': 200e3,
                    'pwr_out_max_watts': 1000e3,
                    'eff_interp_from_pwr_out': {
                        'data': {
                            'grid': [
                                {
                                    # Original points multiplied by 0.4, then extended to 1.0
                                    'data': [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 1.0],
                                    'dim': [12],
                                    'v': 1
                                }
                            ],
                            'values': {
                                'data': [0.0, 0.25, 0.35, 0.39, 0.42, 0.42, 0.41, 0.40, 0.38, 0.36, 0.34, 0.34],
                                'dim': [12],
                                'v': 1
                            }
                        },
                        'extrapolate': 'Error',
                        'strategy': 'Linear'
                    },
                },
                'fs': {
                    'energy_capacity_joules': 500 * 35.8e6,
                    'pwr_out_max_watts': 5000e3,
                    'pwr_ramp_lag_seconds': 1.0,
                },
                'transmission': { # Transmission efficiency.
                    'eff_interp': 0.95,
                },
            },
        },
    },
    'fc': { # Adapted Toyota Mirai
        'name': 'hdt_fc',
        'year': 2025,
        'mass_kilograms': 30_000,
        'pwr_aux_base_watts': 3_000.0,
        'chassis': {
            'cg_height_meters': 1.3, # Centre of gravity
            'drag_coef': 0.55,
            'drive_axle_weight_frac': 0.59,
            'drive_type': 'RWD',
            'frontal_area_square_meters': 9.2,
            'num_wheels': 18,
            'wheel_base_meters': 2.7536,
            'wheel_fric_coef': 0.7,
            'wheel_inertia_kilogram_square_meters': 2.0,
            'wheel_radius_meters': 0.5,
            'wheel_rr_coef': 0.007
        },
        'pt_type': {
            'HEV': {
                'res': {
                    'pwr_out_max_watts': 1000e3,
                    'energy_capacity_joules': 3.6e6 * 50,
                    'eff_interp': {
                        'Constant': 0.9848857801796105
                    },
                    'min_soc': 0.2,
                    'max_soc': 0.9,
                },
                'fs': {
                    'pwr_out_max_watts': 880e3,
                    'pwr_ramp_lag_seconds': 1.0,
                    'energy_capacity_joules': 2.5e10,
                },
                'fc': {
                    'pwr_out_max_watts': 880e3,
                    'pwr_out_max_init_watts': 25e3,
                    'pwr_ramp_lag_seconds': 1.0,
                    'pwr_idle_fuel_watts': 0.0,
                    'eff_interp_from_pwr_out': {
                        'data': {
                            'grid': [{
                                'v': 1,
                                'dim': [12],
                                'data': [0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0]
                            }],
                            'values': {
                                'v': 1,
                                'dim': [12],
                                'data': [0.1, 0.3, 0.36, 0.45, 0.5, 0.56, 0.58, 0.6, 0.58, 0.57, 0.55, 0.54]
                            }
                        },
                        'strategy': 'Linear',
                        'extrapolate': 'Error'
                    },
                },
                'em': {
                    'eff_interp_achieved': {
                        'data': {
                            'grid': [{
                                'v': 1,
                                'dim': [11],
                                # More resolution at the low end (0% to 20% load)
                                'data': [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
                            }],
                            'values': {
                                'v': 1,
                                'dim': [11],
                                # Efficiency starts low (40-60%) and peaks at 60-80% load
                                'data': [0.45, 0.58, 0.72, 0.82, 0.88, 0.92, 0.94, 0.95, 0.93, 0.91, 0.89]
                            }
                        },
                        'strategy': 'Linear',
                        'extrapolate': 'Error'
                    },
                    'pwr_out_max_watts': 880e3,
                },
                'transmission': {
                    'eff_interp': 0.98,
                },
                'pt_cntrl': {
                    'RGWDB': {
                        'speed_soc_disch_buffer_meters_per_second': 22.352,
                        'speed_soc_disch_buffer_coeff': 1.0,
                        'speed_soc_fc_on_buffer_meters_per_second': 26.8224,
                        'speed_soc_fc_on_buffer_coeff': 1.0,
                        'speed_soc_regen_buffer_meters_per_second': 13.4112,
                        'speed_soc_regen_buffer_coeff': 1.0,
                        'fc_min_time_on_seconds': 5.0,
                        'speed_fc_forced_on_meters_per_second': 13.4112,
                        'frac_pwr_demand_fc_forced_on': 0.6802721088435374,
                        'frac_of_most_eff_pwr_to_run_fc': 1.0,
                    }
                },
            }
        },
    },
    'be': {
        'name': '2022 Ford F-150 Lightning 4WD',
        'doc': 'Generated by cal_and_val.tests.test_f2_to_f3.test_f2_to_f3',
        'year': 2022,
        'mass_kilograms': 30_000,
        'pwr_aux_base_watts': 1_000.0,
        'chassis': {
            'cg_height_meters': 1.3, # Centre of gravity
            'drag_coef': 0.55,
            'drive_axle_weight_frac': 0.59,
            'drive_type': 'RWD',
            'frontal_area_square_meters': 9.2,
            'num_wheels': 18,
            'wheel_base_meters': 2.7536,
            'wheel_fric_coef': 0.7,
            'wheel_inertia_kilogram_square_meters': 2.0,
            'wheel_radius_meters': 0.5,
            'wheel_rr_coef': 0.007
        },
        'pt_type': {
            'BEV': {
                'res': { # Battery
                    'pwr_out_max_watts': 1000e3,
                    'energy_capacity_joules': 10*3.6e9,
                    'eff_interp': {
                        'Constant': 0.9848857801796105
                    },
                    'min_soc': 0.05,
                    'max_soc': 0.98,
                },
                'em': { # Electric machine (motor)
                    'pwr_out_max_watts': 1000e3,
                    'eff_interp_achieved': {
                        'data': {
                            'grid': [{
                                'v': 1,
                                'dim': [11],
                                # More resolution at the low end (0% to 20% load)
                                'data': [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
                            }],
                            'values': {
                                'v': 1,
                                'dim': [11],
                                # Efficiency starts low (40-60%) and peaks at 60-80% load
                                'data': [0.45, 0.58, 0.72, 0.82, 0.88, 0.92, 0.94, 0.95, 0.93, 0.91, 0.89]
                            }
                        },
                        'strategy': 'Linear',
                        'extrapolate': 'Error'
                    },
                },
                'transmission': {
                    'eff_interp': 0.98,
                },
            }
        },
    },
    'phe_parallel': {
        'name': '2016 BMW i3 REx PHEV',
        'year': 2025,
        'mass_kilograms': 30_000,
        'pwr_aux_base_watts': 3_000.0,
        'chassis': {
            'cg_height_meters': 1.3,
            'drag_coef': 0.55,
            'drive_axle_weight_frac': 0.59,
            'drive_type': 'RWD',
            'frontal_area_square_meters': 9.2,
            'num_wheels': 18,
            'wheel_base_meters': 2.7536,
            'wheel_fric_coef': 0.7,
            'wheel_inertia_kilogram_square_meters': 2.0,
            'wheel_radius_meters': 0.5,
            'wheel_rr_coef': 0.007
        },
        'pt_type': {
            'PHEV': {
                'aux_cntrl': 'AuxOnResPriority',
                'em': {
                    'pwr_out_max_watts': 100e3,
                    'eff_interp_achieved': {
                        'data': {
                            'grid': [{
                                'data': [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                                'dim': [11],
                                'v': 1
                            }],
                            'values': {
                                'data': [0.86, 0.86, 0.88, 0.9, 0.91, 0.92, 0.94, 0.95, 0.95, 0.94, 0.93],
                                'dim': [11],
                                'v': 1
                            }},
                        'extrapolate': 'Error',
                        'strategy': 'Linear'},
                },
                'fc': {
                    'pwr_idle_fuel_watts': 0.0,
                    'pwr_out_max_init_watts': 0.25e6,
                    'pwr_out_max_watts': 880e3,
                    'pwr_ramp_lag_seconds': 1.0,
                    'eff_interp_from_pwr_out': {
                        'data': {
                            'grid': [{
                                'data': [0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0],
                                'dim': [12],
                                'v': 1
                            }],
                            'values': {
                                'data': [0.1, 0.12, 0.28, 0.35, 0.375, 0.39, 0.4, 0.4, 0.38, 0.37, 0.36, 0.35],
                                'dim': [12],
                                'v': 1
                            }
                        },
                        'extrapolate': 'Error',
                        'strategy': 'Linear'
                    },
                },
                'fs': { # Fuel capacity and supply rate limits.
                    'energy_capacity_joules': 2.5e10,
                    'pwr_out_max_watts': 880e3,
                    'pwr_ramp_lag_seconds': 1.0,
                },
                'res': {
                    'eff_interp': {
                        'Constant': 0.9848857801796105
                    },
                    'energy_capacity_joules': 100 * 3.6e6,
                    'max_soc': 0.8,
                    'min_soc': 0.2,
                    'pwr_out_max_watts': 100e3,
                },
                'pt_cntrl': {
                    'RGWDB': {
                        'fc_min_time_on_seconds': 5.0,
                        'frac_of_most_eff_pwr_to_run_fc': 1.0,
                        'frac_pwr_demand_fc_forced_on': 0.7894736842105263,
                        'speed_fc_forced_on_meters_per_second': 37.9984,
                        'speed_soc_disch_buffer_coeff': 1.0,
                        'speed_soc_disch_buffer_meters_per_second': 22.352,
                        'speed_soc_fc_on_buffer_coeff': 1.0,
                        'speed_soc_fc_on_buffer_meters_per_second': 26.8224,
                        'speed_soc_regen_buffer_coeff': 1.0,
                        'speed_soc_regen_buffer_meters_per_second': 13.4112,
                    }
                },
                'transmission': {
                    'eff_interp': 0.95,
                },
                'sim_params': {
                    'balance_soc': True,
                    'res_per_fuel_lim': 0.000,
                    'save_soc_bal_iters': False,
                    'soc_balance_iter_err': 5
                },
                'soc_bal_iters': 0,
            }
        },
    },
    'he_parallel': { # Adapted Toyota Mirai
        'name': 'hdt_fc',
        'year': 2025,
        'mass_kilograms': 30_000,
        'pwr_aux_base_watts': 3_000.0,
        'chassis': {
            'cg_height_meters': 1.3, # Centre of gravity
            'drag_coef': 0.55,
            'drive_axle_weight_frac': 0.59,
            'drive_type': 'RWD',
            'frontal_area_square_meters': 9.2,
            'num_wheels': 18,
            'wheel_base_meters': 2.7536,
            'wheel_fric_coef': 0.7,
            'wheel_inertia_kilogram_square_meters': 2.0,
            'wheel_radius_meters': 0.5,
            'wheel_rr_coef': 0.007
        },
        'pt_type': {
            'HEV': {
                'res': {
                    'pwr_out_max_watts': 100e3,
                    'energy_capacity_joules': 3.6e6 * 50,
                    'eff_interp': {
                        'Constant': 0.95
                    },
                    'min_soc': 0.2,
                    'max_soc': 0.9,
                },
                'em': {
                    'eff_interp_achieved': {
                        'data': {
                            'grid': [{
                                'v': 1,
                                'dim': [11],
                                # More resolution at the low end (0% to 20% load)
                                'data': [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
                            }],
                            'values': {
                                'v': 1,
                                'dim': [11],
                                # Efficiency starts low (40-60%) and peaks at 60-80% load
                                'data': [0.45, 0.58, 0.72, 0.82, 0.88, 0.92, 0.94, 0.95, 0.93, 0.91, 0.89]
                            }
                        },
                        'strategy': 'Linear',
                        'extrapolate': 'Error'
                    },
                    'pwr_out_max_watts': 100e3,
                },
                'fs': {
                    'pwr_out_max_watts': 1000e3,
                    'pwr_ramp_lag_seconds': 1.0,
                    'energy_capacity_joules': 2.5e10,
                },
                'fc': {
                    'pwr_idle_fuel_watts': 0.0,
                    'pwr_out_max_init_watts': 200e3,
                    'pwr_out_max_watts': 1000e3,
                    'pwr_ramp_lag_seconds': 1.0,
                    'eff_interp_from_pwr_out': {
                        'data': {
                            'grid': [
                                {
                                    # Original points multiplied by 0.4, then extended to 1.0
                                    'data': [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 1.0],
                                    'dim': [12],
                                    'v': 1
                                }
                            ],
                            'values': {
                                'data': [0.0, 0.25, 0.35, 0.39, 0.42, 0.42, 0.41, 0.40, 0.38, 0.36, 0.34, 0.34],
                                'dim': [12],
                                'v': 1
                            }
                        },
                        'extrapolate': 'Error',
                        'strategy': 'Linear'
                    },
                },
                'transmission': {
                    'eff_interp': 0.95,
                },
                'pt_cntrl': {
                    'RGWDB': {
                        'speed_soc_disch_buffer_meters_per_second': 10.0,
                        'speed_soc_disch_buffer_coeff': 0.5,
                        'speed_soc_fc_on_buffer_meters_per_second': 15.0,
                        'speed_soc_fc_on_buffer_coeff': 0.8,
                        'speed_soc_regen_buffer_meters_per_second': 20.0,
                        'speed_soc_regen_buffer_coeff': 1.0,
                        'fc_min_time_on_seconds': 60.0,
                        'speed_fc_forced_on_meters_per_second': 5.0,
                        'frac_pwr_demand_fc_forced_on': 0.3,
                        'frac_of_most_eff_pwr_to_run_fc': 1.0,
                    }
                },
            }
        },
    },
}

def calculate_fuel_consumption(
        mass=20_000,
        accessory_load=3_000,
        roll_coef=0.0054,
        drag_coef=0.6,
        frontal_area=9.2,
        peak_eff=0.4,
        fuel_capacity=500,
        fuel_lhv=35.8e6,
        veh_type='dice',
        cyc=DRIVE_CYCLES['long_haul'],
        ):
    veh_dict = copy.deepcopy(VEHICLES[veh_type])
    if veh_type == 'dice':
        # Scale efficiency
        eff_data = veh_dict['pt_type']['Conv']['fc']['eff_interp_from_pwr_out']['data']['values']['data']
        scaled_eff_data = [val * peak_eff/max(eff_data) for val in eff_data]
        veh_dict['pt_type']['Conv']['fc']['eff_interp_from_pwr_out']['data']['values']['data'] = scaled_eff_data
        veh_dict['pt_type']['Conv']['fs']['energy_capacity_joules'] = fuel_capacity * fuel_lhv
    if veh_type == 'he_parallel':
        # Scale efficiency
        eff_data = veh_dict['pt_type']['HEV']['fc']['eff_interp_from_pwr_out']['data']['values']['data']
        scaled_eff_data = [val * peak_eff/max(eff_data) for val in eff_data]
        veh_dict['pt_type']['HEV']['fc']['eff_interp_from_pwr_out']['data']['values']['data'] = scaled_eff_data
        veh_dict['pt_type']['HEV']['fs']['energy_capacity_joules'] = fuel_capacity * fuel_lhv
    if veh_type == 'phe_parallel':
        pass
    if veh_type == 'phe_series':
        pass
    if veh_type == 'fc':
        # Scale efficiency
        eff_data = veh_dict['pt_type']['HEV']['fc']['eff_interp_from_pwr_out']['data']['values']['data']
        scaled_eff_data = [val * peak_eff/max(eff_data) for val in eff_data]
        veh_dict['pt_type']['HEV']['fc']['eff_interp_from_pwr_out']['data']['values']['data'] = scaled_eff_data
        veh_dict['pt_type']['HEV']['fs']['energy_capacity_joules'] = fuel_capacity * fuel_lhv
    if veh_type == 'be':
        eff_data = veh_dict['pt_type']['BEV']['em']['eff_interp_achieved']['data']['values']['data']
        scaled_eff_data = [val * peak_eff/max(eff_data) for val in eff_data]
        veh_dict['pt_type']['BEV']['em']['eff_interp_achieved']['data']['values']['data'] = scaled_eff_data
        veh_dict['pt_type']['BEV']['res']['energy_capacity_joules'] = fuel_capacity * fuel_lhv * 100

    # Replace values
    veh_dict['mass_kilograms'] = mass
    veh_dict['pwr_aux_base_watts'] = accessory_load
    veh_dict['chassis']['drag_coef'] = drag_coef
    veh_dict['chassis']['frontal_area_square_meters'] = frontal_area
    veh_dict['chassis']['wheel_rr_coef'] = roll_coef

    # Simulate vehicle
    veh = fsim.Vehicle.from_pydict(veh_dict)
    sim = fsim.SimDrive(veh, cyc)
    sim.walk()
    res = sim.to_pydict()

    # Calculate fuel consumption
    dist_km = res['veh']['state']['dist_meters'] / 1000
    fuel_consumption = {}
    if veh_type == 'dice':
        fuel_consumption['Diesel'] = res['veh']['pt_type']['Conv']['fc']['state']['energy_fuel_joules'] / 35.8e6 / dist_km
    if veh_type == 'he_parallel':
        fuel_consumption['Diesel'] = res['veh']['pt_type']['HEV']['fc']['state']['energy_fuel_joules'] / 35.8e6 / dist_km
    if veh_type == 'phe_parallel':
        fuel_consumption['Diesel'] = res['veh']['pt_type']['PHEV']['fc']['state']['energy_fuel_joules'] / 35.8e6 / dist_km
        fuel_consumption['Electricity'] = res['veh']['pt_type']['PHEV']['res']['state']['energy_out_chemical_joules'] / 3.6e6 / dist_km
    if veh_type == 'phe_series':
        fuel_consumption['Diesel'] = res['veh']['pt_type']['PHEV']['fc']['state']['energy_fuel_joules'] / 35.8e6 / dist_km
        fuel_consumption['Electricity'] = res['veh']['pt_type']['PHEV']['res']['state']['energy_out_chemical_joules'] / 3.6e6 / dist_km
    if veh_type == 'fc':
        fuel_consumption['Hydrogen'] = res['veh']['pt_type']['HEV']['fc']['state']['energy_fuel_joules'] / 120e6 / dist_km
    if veh_type == 'be':
        fuel_consumption['Electricity'] = res['veh']['pt_type']['BEV']['res']['state']['energy_out_electrical_joules'] / 3.6e6 / dist_km
    return fuel_consumption

def analyze_test_scheme(scheme, n_train=500, n_test=50):
    ranges = scheme['ranges']
    fixed = scheme['fixed']
    labels = list(ranges.keys())
    
    # 1. LHS Sampling
    sampler = qmc.LatinHypercube(d=len(labels))
    sample_raw = sampler.random(n=n_train)
    l_bounds = [v[0] for v in ranges.values()]
    u_bounds = [v[1] for v in ranges.values()]
    X_train_raw = qmc.scale(sample_raw, l_bounds, u_bounds)
    df_train = pd.DataFrame(X_train_raw, columns=labels)
    
    # 2. Physics-Informed Transformation
    df_train['inv_eff'] = 1 / df_train['peak_eff']
    labels_for_reg = [l for l in labels if l != 'peak_eff'] + ['inv_eff']
    
    y_train = []
    for _, row in df_train[labels].iterrows():
        # Combine parameters and remove non-physics keys
        params = {**row.to_dict(), **fixed}
        
        # Execute simulation - will crash here if physics or inputs fail
        res_dict = calculate_fuel_consumption(**params)
        
        # Extract the first value from the dictionary
        val = next(iter(res_dict.values()))
        y_train.append(val)
    
    # 3. Polynomial Regression
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly_train = poly.fit_transform(df_train[labels_for_reg])
    feature_names = poly.get_feature_names_out(labels_for_reg)
    
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    
    # 4. Validation on Random Points
    val_data = []
    for _ in range(n_test):
        s = {k: np.random.uniform(v[0], v[1]) for k, v in ranges.items()}
        v_params = {**s, **fixed}
        
        actual_dict = calculate_fuel_consumption(**v_params)
        actual_val = next(iter(actual_dict.values()))
        
        s_trans = s.copy()
        s_trans['inv_eff'] = 1 / s_trans.pop('peak_eff')
        X_val_poly = poly.transform(pd.DataFrame([s_trans]))
        pred = model.predict(X_val_poly)[0]
        
        val_data.append({'actual': actual_val, 'pred': pred})
    
    df_val = pd.DataFrame(val_data)
    
    return {
        'r2': r2_score(df_val['actual'], df_val['pred']),
        'mape': mean_absolute_percentage_error(df_val['actual'], df_val['pred']) * 100,
        'model': model,
        'feature_names': feature_names,
        'intercept': model.intercept_
    }

def print_results(results):
    print(f"\n--- Model Performance ---")
    print(f"R2 Score: {results['r2']:.6f}")
    print(f"Mean Absolute Error: {results['mape']:.4f}%")
    print(f"Intercept (L/100km): {results['intercept']:.4f}")
    print("\n--- Coefficients Table (for Paper 2) ---")
    coef_df = pd.DataFrame({
        'Feature': results['feature_names'],
        'Coefficient': results['model'].coef_
    }).sort_values(by='Coefficient', ascending=False, key=abs)

    print(coef_df.to_string(index=False))

def save_model_to_json(results, filename='model_params.json'):
    model = results['model']
    params = {
        'features': dict(zip(results['feature_names'], model.coef_.tolist())),
        'intercept': float(model.intercept_),
        'r2': float(results['r2']),
        'mape': float(results['mape'])
    }
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)

def load_model_params(json_path='model_params.json'):
    with open(json_path, 'r') as f:
        return json.load(f)

def estimate_fuel_consumption(input_data, model_params):
    # 1. Base physics terms
    base = {
        'mass': input_data['mass'],
        'drag_coef': input_data['drag_coef'],
        'accessory_load': input_data['accessory_load'],
        'inv_eff': 1 / input_data['peak_eff']
    }
    
    total = model_params['intercept']
    coefs = model_params['features']
    
    # 2. Iterate through the JSON keys to ensure perfect alignment
    for name, coef in coefs.items():
        if " " in name:
            # It's an interaction term like 'mass drag_coef'
            v1, v2 = name.split(" ")
            total += (base[v1] * base[v2]) * coef
        else:
            # It's a base term
            total += base[name] * coef
            
    return total

SCHEMES = {
    'dice': {
        'ranges': {
            'mass': (5_000, 40_000),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.35, 0.6),             # decimal
            'accessory_load': (1_000, 7_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'fuel_capacity': 500,               # Liters
            'fuel_lhv': 35.8e6,                 # J/L (Diesel)
            'veh_type': 'dice',
            'cyc': DRIVE_CYCLES['cruise_hdt'],
        }
    },
    'fc': {
        'ranges': {
            'mass': (6_000, 41_000),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.4, 0.8),             # decimal
            'accessory_load': (1_000, 7_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'fuel_capacity': 500,               # Liters
            'fuel_lhv': 120e6,                 # J/L (Diesel)
            'veh_type': 'fc',
            'cyc': DRIVE_CYCLES['long_haul'],
        }
    },
    'be': {
        'ranges': {
            'mass': (9_000, 44_000),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.4, 0.8),             # decimal
            'accessory_load': (1_000, 7_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'fuel_capacity': 1000,               # Liters
            'fuel_lhv': 3.6e6,                 # J/L (Diesel)
            'veh_type': 'be',
            'cyc': DRIVE_CYCLES['long_haul'],
        }
    },
    'phe_parallel': {
        'ranges': {
            'mass': (7_000, 42_000),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.3, 0.6),             # decimal
            'accessory_load': (1_000, 7_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'fuel_capacity': 1000,               # Liters
            'fuel_lhv': 3.6e6,                 # J/L (Diesel)
            'veh_type': 'phe_parallel',
            'cyc': DRIVE_CYCLES['long_haul'],
        }
    },
    'he_parallel': {
        'ranges': {
            'mass': (5_500, 40_500),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.35, 0.6),             # decimal
            'accessory_load': (1_000, 7_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'fuel_capacity': 500,               # Liters
            'fuel_lhv': 35.8e6,                 # J/L (Diesel)
            'veh_type': 'he_parallel',
            'cyc': DRIVE_CYCLES['cruise_hdt'],
        }
    },
}


if __name__ == '__main__':
    # Vehicles
    ps = ['dice', 'he_parallel', 'be', 'fc']
    # ps = ['dice', 'he_parallel']
    # ps = ['dice']
    # ps = ['he_parallel']
    # ps = ['be']
    # Drive cycles
    dc = 'udds_hdt'
    # dcs = ['udds_hdt', 'cruise_hdt']
    dcs = ['short_haul', 'regional_haul', 'long_haul']
    dc = 'long_haul'
    for dc in dcs:
        for p in ps:
            fname = 'drive_cycles/' + p + '_' + dc + '.json'
            scheme = SCHEMES[p]
            scheme['fixed']['cyc'] = DRIVE_CYCLES[dc]
            results = analyze_test_scheme(scheme, n_train=100, n_test=20)
            # print_results(results)
            save_model_to_json(results, filename=fname)
            
            model_params = load_model_params(fname)
            my_truck = {
                key: np.mean(val) for key, val in scheme['ranges'].items()
            }
            t0 = time.time()
            test = estimate_fuel_consumption(my_truck, model_params)
            t1 = time.time()
            var = list(calculate_fuel_consumption(**{**scheme['fixed'], **my_truck}).values())[0]
            t2 = time.time()
            print(f"{p}, {dc}:\n {test:.5f}")#\n {var:.5f}")
            print(f'model: {(t1-t0 + 1e-10)/(t2-t1 + 1e-10)}')

    # veh = fsim.Vehicle.from_pydict(VEHICLES[p])
    # cyc = DRIVE_CYCLES[dc]
    # sim = fsim.SimDrive(veh, cyc)
    # # sim_dict = sim.to_pydict()
    # # sim = fsim.SimDrive.from_pydict(sim_dict)
    # fuel_consumption = calculate_fuel_consumption(**{**scheme['fixed'], **my_truck})
    # print(fuel_consumption)
    # sim.walk()
    # res = sim.to_pydict()

    # veh_dict = veh.to_pydict()
    
    # # Fuel converter power curve
    # x = veh_dict['pt_type']['Conv']['fc']['eff_interp_from_pwr_out']['data']['grid'][0]['data']
    # y = veh_dict['pt_type']['Conv']['fc']['eff_interp_from_pwr_out']['data']['values']['data']
    # plt.plot(x, y)
    # plt.plot(cyc.to_pydict()['speed_meters_per_second'])