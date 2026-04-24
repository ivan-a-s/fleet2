""" Calculate fuel consumption for a vehicle object.
To do: deal with peak efficiency peoperly.
"""
import fastsim as fsim
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Load drive-cycle.
def load_drive_cycle(fname="drive_cycles/regional_haul.json"):
    with open(fname, "r") as f:
        cyc_json_str = f.read()
    cyc_dict = json.loads(cyc_json_str)
    return fsim.Cycle.from_pydict(cyc_dict)

DRIVE_CYCLES = {
    'short_haul': load_drive_cycle('drive_cycles/short_haul.json'),
    'regional_haul': load_drive_cycle('drive_cycles/regional_haul.json'),
    'long_haul': load_drive_cycle('drive_cycles/long_haul.json'),
    'udds_hdt': load_drive_cycle('drive_cycles/udds_hdt.json'),
    'cruise_hdt': load_drive_cycle('drive_cycles/cruise_hdt.json'),
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
                    'pwr_out_max_init_watts': 100e3,
                    'pwr_out_max_watts': 800e3,
                    'pwr_ramp_lag_seconds': 1.0,
                    'eff_interp_from_pwr_out': {
                        'data': {
                            'grid': [
                                {
                                    'data': [0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0],
                                    'dim': [12],
                                    'v': 1
                                }
                            ],
                            'values': {
                                'data': [0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34],
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
                    'pwr_out_max_watts': 1_000e3,
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
                    'pwr_out_max_watts': 500e3,
                    'energy_capacity_joules': 3.6e6 * 50,
                    'eff_interp': {
                        'Constant': 0.9848857801796105
                    },
                    'min_soc': 0.2,
                    'max_soc': 0.9,
                },
                'fs': {
                    'pwr_out_max_watts': 1_000e3,
                    'pwr_ramp_lag_seconds': 1.0,
                    'energy_capacity_joules': 2.5e10,
                },
                'fc': {
                    'pwr_out_max_watts': 500e3,
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
                                'data': [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
                            }],
                            'values': {
                                'v': 1,
                                'dim': [11],
                                'data': [0.86, 0.86, 0.88, 0.9, 0.91, 0.92, 0.94, 0.95, 0.95, 0.94, 0.93]
                            }
                        },
                        'strategy': 'Linear',
                        'extrapolate': 'Error'
                    },
                    'pwr_out_max_watts': 500e3,
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
                    'pwr_out_max_watts': 500e3,
                    'energy_capacity_joules': 10*3.6e9,
                    'eff_interp': {
                        'Constant': 0.9848857801796105
                    },
                    'min_soc': 0.05,
                    'max_soc': 0.98,
                },
                'em': { # Electric machine (motor)
                    'pwr_out_max_watts': 500e3,
                    'eff_interp_achieved': {
                        'data': {
                            'grid': [{
                                'v': 1,
                                'dim': [11],
                                'data': [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
                            }],
                            'values': {
                                'v': 1,
                                'dim': [11],
                                'data': [0.84, 0.86, 0.88, 0.9, 0.91, 0.92, 0.94, 0.95, 0.95, 0.94, 0.93]
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
                    'pwr_out_max_watts': 50e3,
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
                    'pwr_out_max_watts': 500e3,
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
                    'pwr_out_max_watts': 5e6,
                    'pwr_ramp_lag_seconds': 1.0,
                },
                'res': {
                    'eff_interp': {
                        'Constant': 0.9848857801796105
                    },
                    'energy_capacity_joules': 100 * 3.6e6,
                    'max_soc': 0.9,
                    'min_soc': 0.2,
                    'pwr_out_max_watts': 50e3,
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
                    'eff_interp': 0.98,
                },
                'sim_params': {
                    'balance_soc': False,
                    'res_per_fuel_lim': 0.005,
                    'save_soc_bal_iters': False,
                    'soc_balance_iter_err': 5
                },
                'soc_bal_iters': 0,
            }
        },
    },
    'phe_series': {
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
                    'pwr_out_max_watts': 500e3,
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
                    'pwr_out_max_watts': 100e3,
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
                    'pwr_out_max_watts': 5e6,
                    'pwr_ramp_lag_seconds': 1.0,
                },
                'res': {
                    'eff_interp': {
                        'Constant': 0.9848857801796105
                    },
                    'energy_capacity_joules': 100 * 3.6e6,
                    'max_soc': 0.9,
                    'min_soc': 0.2,
                    'pwr_out_max_watts': 400e3,
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
                    'eff_interp': 0.98,
                },
                'sim_params': {
                    'balance_soc': False,
                    'res_per_fuel_lim': 0.005,
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
                    'pwr_out_max_watts': 500e3,
                    'energy_capacity_joules': 3.6e6 * 50,
                    'eff_interp': {
                        'Constant': 0.9848857801796105
                    },
                    'min_soc': 0.2,
                    'max_soc': 0.9,
                },
                'fs': {
                    'pwr_out_max_watts': 1_000e3,
                    'pwr_ramp_lag_seconds': 1.0,
                    'energy_capacity_joules': 2.5e10,
                },
                'fc': {
                    'pwr_idle_fuel_watts': 0.0,
                    'pwr_out_max_init_watts': 100e3,
                    'pwr_out_max_watts': 500e3,
                    'pwr_ramp_lag_seconds': 1.0,
                    'eff_interp_from_pwr_out': {
                        'data': {
                            'grid': [
                                {
                                    'data': [0.0, 0.005, 0.015, 0.04, 0.06, 0.1, 0.14, 0.2, 0.4, 0.6, 0.8, 1.0],
                                    'dim': [12],
                                    'v': 1
                                }
                            ],
                            'values': {
                                'data': [0.10, 0.14, 0.20, 0.26, 0.32, 0.39, 0.41, 0.42, 0.41, 0.38, 0.36, 0.34],
                                'dim': [12],
                                'v': 1
                            }
                        },
                        'extrapolate': 'Error',
                        'strategy': 'Linear'
                    },
                },
                'em': {
                    'eff_interp_achieved': {
                        'data': {
                            'grid': [{
                                'v': 1,
                                'dim': [11],
                                'data': [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
                            }],
                            'values': {
                                'v': 1,
                                'dim': [11],
                                'data': [0.86, 0.86, 0.88, 0.9, 0.91, 0.92, 0.94, 0.95, 0.95, 0.94, 0.93]
                            }
                        },
                        'strategy': 'Linear',
                        'extrapolate': 'Error'
                    },
                    'pwr_out_max_watts': 500e3,
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
}

def calculate_fuel_consumption(
        mass=20_000,
        accessory_load=3_000,
        roll_coef=0.0054,
        drag_coef=0.6,
        frontal_area=9.2,
        peak_eff=0.4,
        regen_eff=0,
        fuel_capacity=500,
        fuel_lhv=35.8e6,
        veh_type='dice',
        cyc=DRIVE_CYCLES['long_haul'],
        ):
    if veh_type == 'dice':
        veh_dict = VEHICLES['dice']
        # Scale efficiency
        eff_data = veh_dict['pt_type']['Conv']['fc']['eff_interp_from_pwr_out']['data']['values']['data']
        scaled_eff_data = [val * peak_eff/max(eff_data) for val in eff_data]
        veh_dict['pt_type']['Conv']['fc']['eff_interp_from_pwr_out']['data']['values']['data'] = scaled_eff_data
        veh_dict['pt_type']['Conv']['fs']['energy_capacity_joules'] = fuel_capacity * fuel_lhv
    if veh_type == 'he_parallel':
        veh_dict = VEHICLES['he_parallel']
        # Scale efficiency
        eff_data = veh_dict['pt_type']['HEV']['fc']['eff_interp_from_pwr_out']['data']['values']['data']
        scaled_eff_data = [val * peak_eff/max(eff_data) for val in eff_data]
        veh_dict['pt_type']['HEV']['fc']['eff_interp_from_pwr_out']['data']['values']['data'] = scaled_eff_data
        veh_dict['pt_type']['HEV']['fs']['energy_capacity_joules'] = fuel_capacity * fuel_lhv
    if veh_type == 'phe_parallel':
        veh_dict = VEHICLES['phe_parallel']
    if veh_type == 'phe_series':
        veh_dict = VEHICLES['phe_series']
    if veh_type == 'fc':
        veh_dict = VEHICLES['fc']
        # Scale efficiency
        eff_data = veh_dict['pt_type']['HEV']['fc']['eff_interp_from_pwr_out']['data']['values']['data']
        scaled_eff_data = [val * peak_eff/max(eff_data) for val in eff_data]
        veh_dict['pt_type']['HEV']['fc']['eff_interp_from_pwr_out']['data']['values']['data'] = scaled_eff_data
        veh_dict['pt_type']['HEV']['fs']['energy_capacity_joules'] = fuel_capacity * fuel_lhv
    if veh_type == 'be':
        veh_dict = VEHICLES['be']
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
        fuel_consumption['Electricity'] = res['veh']['pt_type']['BEV']['res']['state']['energy_out_electrical_joules'] / 35.8e6 / dist_km
    return fuel_consumption


def analyze_test_scheme(scheme_name, n_train=500, n_test=50):
    scheme = SCHEMES[scheme_name]
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
    
    print(f"Running {n_train} LHS iterations for {scheme_name}...")
    
    y_train = []
    for _, row in df_train[labels].iterrows():
        # Combine parameters and remove non-physics keys
        params = {**row.to_dict(), **fixed}
        params.pop('fuels', None)
        params.pop('veh_type', None)
        
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
        v_params.pop('fuels', None)
        v_params.pop('veh_type', None)
        
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

SCHEMES = {
    'dice': {
        'ranges': {
            'mass': (5_000, 40_000),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.35, 0.6),             # decimal
            'accessory_load': (1_000, 5_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'regen_eff': 0,                     # Fixed for 'dice' (Conventional Diesel)
            'fuel_capacity': 500,               # Liters
            'fuel_lhv': 35.8e6,                 # J/L (Diesel)
            'veh_type': 'dice',
            'cyc': DRIVE_CYCLES['long_haul'],
        }
    },
    'fc': {
        'ranges': {
            'mass': (5_000, 40_000),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.4, 0.8),             # decimal
            'accessory_load': (1_000, 7_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'regen_eff': 0,                     # Fixed for 'dice' (Conventional Diesel)
            'fuel_capacity': 500,               # Liters
            'fuel_lhv': 120e6,                 # J/L (Diesel)
            'veh_type': 'fc',
            'cyc': DRIVE_CYCLES['long_haul'],
        }
    },
    'be': {
        'ranges': {
            'mass': (5_000, 40_000),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.4, 0.8),             # decimal
            'accessory_load': (1_000, 7_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'regen_eff': 0,                     # Fixed for 'dice' (Conventional Diesel)
            'fuel_capacity': 1000,               # Liters
            'fuel_lhv': 3.6e6,                 # J/L (Diesel)
            'veh_type': 'be',
            'cyc': DRIVE_CYCLES['long_haul'],
        }
    },
    'phe_parallel': {
        'ranges': {
            'mass': (5_000, 40_000),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.3, 0.6),             # decimal
            'accessory_load': (1_000, 7_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'regen_eff': 0,                     # Fixed for 'dice' (Conventional Diesel)
            'fuel_capacity': 1000,               # Liters
            'fuel_lhv': 3.6e6,                 # J/L (Diesel)
            'veh_type': 'phe_series',
            'cyc': DRIVE_CYCLES['long_haul'],
        }
    },
    'phe_series': {
        'ranges': {
            'mass': (5_000, 40_000),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.3, 0.6),             # decimal
            'accessory_load': (1_000, 7_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'regen_eff': 0,                     # Fixed for 'dice' (Conventional Diesel)
            'fuel_capacity': 1000,               # Liters
            'fuel_lhv': 3.6e6,                 # J/L (Diesel)
            'veh_type': 'phe_series',
            'cyc': DRIVE_CYCLES['long_haul'],
        }
    },
    'he_parallel': {
        'ranges': {
            'mass': (5_000, 40_000),           # kg (extended to 45k for BC limits)
            'drag_coef': (0.2, 0.7),           # dimensionless
            'peak_eff': (0.4, 0.8),             # decimal
            'accessory_load': (1_000, 7_000),   # Watts
        },
        'fixed': {
            'roll_coef': 0.0054,        # dimensionless (standard vs low-rolling)
            'frontal_area': 9.2,        # m^2 (captures different trailer heights)
            'regen_eff': 0,                     # Fixed for 'dice' (Conventional Diesel)
            'fuel_capacity': 1000,               # Liters
            'fuel_lhv': 3.6e6,                 # J/L (Diesel)
            'veh_type': 'phe_series',
            'cyc': DRIVE_CYCLES['long_haul'],
        }
    },
}


if __name__ == '__main__':
    # Execution and Reporting
    results = analyze_test_scheme('dice', n_train=20, n_test=10)
    # print_results(results)

    # Test vehicles
    # cyc = fsim.Cycle.from_resource('hwfet.csv')
    # cyc = fsim.Cycle.from_resource('udds.csv')
    # cyc = DRIVE_CYCLES['short_haul']
    # cyc = DRIVE_CYCLES['long_haul']
    cyc = DRIVE_CYCLES['udds_hdt']
    cyc = DRIVE_CYCLES['cruise_hdt']

    # DICE
    fuel_consumption = calculate_fuel_consumption(accessory_load=3_000, mass=30_000, peak_eff=0.45, fuel_capacity=500, fuel_lhv=35.8e6, veh_type='dice', cyc=cyc)
    print(fuel_consumption)
    
    # BE
    fuel_consumption = calculate_fuel_consumption(accessory_load=6_000, mass=35_000, peak_eff=0.9, fuel_capacity=1000, fuel_lhv=3.6e6, veh_type='be', cyc=cyc)
    print(fuel_consumption)
    
    # FC
    fuel_consumption = calculate_fuel_consumption(accessory_load=3_000, mass=30_000, peak_eff=0.6, fuel_capacity=80, fuel_lhv=120e6, veh_type='fc', cyc=cyc)
    print(fuel_consumption)
    

    # PHE-Parallel
    fuel_consumption = calculate_fuel_consumption(accessory_load=3_000, mass=30_000, peak_eff=0.45, fuel_capacity=500, fuel_lhv=35.8e6, veh_type='phe_parallel', cyc=cyc)
    print(fuel_consumption)

    # PHE-Series
    fuel_consumption = calculate_fuel_consumption(accessory_load=3_000, mass=30_000, peak_eff=0.45, fuel_capacity=500, fuel_lhv=35.8e6, veh_type='phe_series', cyc=cyc)
    print(fuel_consumption)

    # HE-Parallel
    veh = fsim.Vehicle.from_pydict(VEHICLES['he_parallel'])
    sim = fsim.SimDrive(veh, cyc)
    sim.walk()
    res = sim.to_pydict()
    fuel_consumption = calculate_fuel_consumption(accessory_load=3_000, mass=30_000, peak_eff=0.45, fuel_capacity=500, fuel_lhv=35.8e6, veh_type='he_parallel', cyc=cyc)
    print(fuel_consumption)

    # Plot
    plt.plot(cyc.to_pydict()['speed_meters_per_second'])