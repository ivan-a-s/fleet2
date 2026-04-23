""" Calculate fuel consumption for a vehicle object. """
import fastsim as fsim
import json
import numpy as np
import pandas as pd
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
}

# Create the vehicle.
VEHICLES = {
    'dice': {
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
                    'pwr_idle_fuel_watts': 0.0,
                    'pwr_out_max_init_watts': 0.25e6,
                    'pwr_out_max_watts': 1e6,
                    'pwr_ramp_lag_seconds': 1.0,
                },
                'fs': { # Fuel capacity and supply rate limits.
                    'energy_capacity_joules': 2.5e10,
                    'pwr_out_max_watts': 5e6,
                    'pwr_ramp_lag_seconds': 1.0,
                },
                'transmission': { # Transmission efficiency.
                    'eff_interp': 0.875,
                },
            },
        },
    },
    'fc': {
        'name': 'hdt_fc',
        'year': 2025,
        'mass_kilograms': 30_000,
        'pwr_aux_base_watts': 3_000.0,
        'pt_type': {
            'HEV': {
                'res': {
                    'pwr_out_max_watts': 200e3,
                    'energy_capacity_joules': 180e6,
                    'eff_interp': {
                        'Constant': 0.9848857801796105
                    },
                    'min_soc': 0.2,
                    'max_soc': 0.9,
                },
                'fs': {
                    'pwr_out_max_watts': 5e6,
                    'pwr_ramp_lag_seconds': 1.0,
                    'energy_capacity_joules': 2.5e10,
                },
                'fc': {
                    'pwr_out_max_watts': 1_000e3,
                    'pwr_out_max_init_watts': 25e3,
                    'pwr_ramp_lag_seconds': 1.0,
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
                    'pwr_idle_fuel_watts': 0.0,
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
                    'eff_interp_at_max_input': {
                        'data': {
                            'grid': [{
                                'v': 1,
                                'dim': [11],
                                'data': [0.0, 0.0232558, 0.045454545, 0.066667, 0.08791, 0.108695, 0.212765, 0.421052, 0.631578, 0.851063, 1.075268]
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
                    'pwr_out_max_watts': 1_000e3,
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
    if veh_type == 'fc':
        veh_dict = VEHICLES['fc']
        # Scale efficiency
        eff_data = veh_dict['pt_type']['HEV']['fc']['eff_interp_from_pwr_out']['data']['values']['data']
        scaled_eff_data = [val * peak_eff/max(eff_data) for val in eff_data]
        veh_dict['pt_type']['HEV']['fc']['eff_interp_from_pwr_out']['data']['values']['data'] = scaled_eff_data
        veh_dict['pt_type']['HEV']['fs']['energy_capacity_joules'] = fuel_capacity * fuel_lhv
    
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
    if veh_type == 'dice':
        fuel_J = res['veh']['pt_type']['Conv']['fc']['state']['energy_fuel_joules']
    if veh_type == 'fc':
        fuel_J = res['veh']['pt_type']['HEV']['fc']['state']['energy_fuel_joules']
    return (fuel_J / fuel_lhv) / dist_km * 100


SCHEMES = {
    'sleeper_dice': {
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
    'sleeper_fc': {
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
    }
}


def analyze_test_scheme_v3(scheme_name, n_train=500, n_test=50):
    scheme = SCHEMES[scheme_name]
    ranges = scheme['ranges']
    fixed = scheme['fixed']
    labels = list(ranges.keys())
    
    # 1. LHS Sampling using (min, max)
    sampler = qmc.LatinHypercube(d=len(labels))
    sample_raw = sampler.random(n=n_train)
    
    # Scale LHS: bounds are now simply the first and second elements of the tuple
    l_bounds = [v[0] for v in ranges.values()]
    u_bounds = [v[1] for v in ranges.values()]
    X_train_raw = qmc.scale(sample_raw, l_bounds, u_bounds)
    df_train = pd.DataFrame(X_train_raw, columns=labels)
    
    # 2. Physics-Informed Transformation: Linearize Efficiency
    df_train['inv_eff'] = 1 / df_train['peak_eff']
    # We use 'inv_eff' for the regression, but 'peak_eff' for the FASTSim call
    labels_for_reg = [l for l in labels if l != 'peak_eff'] + ['inv_eff']
    
    print(f"Running {n_train} LHS iterations for {scheme_name}...")
    y_train = []
    for _, row in df_train[labels].iterrows():
        params = {**row.to_dict(), **fixed}
        y_train.append(calculate_fuel_consumption(**params))
    
    # 3. Polynomial Regression (Interactions Only)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly_train = poly.fit_transform(df_train[labels_for_reg])
    feature_names = poly.get_feature_names_out(labels_for_reg)
    
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    
    # 4. Validation on Random Points
    val_data = []
    for _ in range(n_test):
        s = {k: np.random.uniform(v[0], v[1]) for k, v in ranges.items()}
        actual = calculate_fuel_consumption(**{**s, **fixed})
        
        s_trans = s.copy()
        s_trans['inv_eff'] = 1 / s_trans.pop('peak_eff')
        X_val_poly = poly.transform(pd.DataFrame([s_trans]))
        pred = model.predict(X_val_poly)[0]
        val_data.append({'actual': actual, 'pred': pred})
    
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

if __name__ == '__main__':
    # Execution and Reporting
    results = analyze_test_scheme_v3('sleeper_fc', n_train=50)
    print_results(results)

    # Fuel cell
    veh = fsim.Vehicle.from_pydict(VEHICLES['fc'])
    cyc = DRIVE_CYCLES['long_haul']
    sim = fsim.SimDrive(veh, cyc)
    sim.walk()
    res = sim.to_pydict()
    dist_km = res['veh']['state']['dist_meters'] / 1000
    fuel = res['veh']['pt_type']['HEV']['fc']['state']['energy_fuel_joules'] / 120e6
    print(fuel/dist_km*100)
    print(calculate_fuel_consumption(peak_eff=0.6, fuel_capacity=80, fuel_lhv=120e6, veh_type='fc'))
