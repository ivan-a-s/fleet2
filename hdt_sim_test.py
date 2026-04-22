import fastsim as fsim
from scipy.signal import savgol_filter
import pandas as pd

# Load vehicle
# veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
# # Re-configure
# veh_dict = veh.to_pydict()
# veh_dict['mass_kilograms'] = 20000
# veh_dict['chassis']['drag_coef'] = 0.6
# veh_dict['chassis']['frontal_area_square_meters'] = 10
# veh_dict['chassis']['wheel_rr_coef'] = 0.006
# veh_dict['pt_type']['Conv']['fc']['pwr_out_max_watts'] = 1000_000
# veh_dict['pt_type']['Conv']['fc']['pwr_ramp_lag_seconds'] = 2.0
# veh_dict['pt_type']['Conv']['transmission']['eff_interp'] = 0.92
# veh_dict['chassis']['wheel_radius_meters'] = 0.5
# veh_dict['chassis']['num_wheels'] = 10
# # Re-load vehicle
# veh = fsim.Vehicle.from_pydict(veh_dict)
# print(veh.state.pwr_prop_fwd_max_watts)


import fastsim as fsim

import fastsim as fsim

veh_dict = {
    "name": "Custom HDT",
    "year": 2012,

    "mass_kilograms": 30_000,
    "pwr_aux_base_watts": 3_000.0,
    "save_interval": 1,

    "chassis": {
        'cg_height_meters': 1.3, # Centre of gravity
        'drag_coef': 0.55,
        'drive_axle_weight_frac': 0.5,
        'drive_type': 'RWD',
        'frontal_area_square_meters': 9.2,
        'num_wheels': 18,
        'wheel_base_meters': 6,
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
                                'data': [
                                    0.0, 0.005, 0.015, 0.04,
                                    0.06, 0.1, 0.14, 0.2,
                                    0.4, 0.6, 0.8, 1.0
                                ],
                                'dim': [12],
                                'v': 1
                            }
                        ],
                        'values': {
                            'data': [
                                0.10, 0.15, 0.22, 0.30,
                                0.36, 0.40, 0.42, 0.44,
                                0.44, 0.43, 0.41, 0.38
                            ],
                            'dim': [12],
                            'v': 1
                        }
                    },
                    'extrapolate': 'Error',
                    'strategy': 'Linear'
                },
                'pwr_idle_fuel_watts': 0.0,
                'pwr_out_max_init_watts': 100_000.0,
                'pwr_out_max_watts': 1_500_000.0,
                'pwr_ramp_lag_seconds': 2.0,
                'save_interval': 1,
                'thrml': 'None'
            },
            'fs': {
                'energy_capacity_joules': 2.5e10,
                'pwr_out_max_watts': 2e6,
                'pwr_ramp_lag_seconds': 1.0,
            },
            'transmission': {
                'eff_interp': 0.875,
                'save_interval': 1,
            },
        },
    },
}

veh = fsim.Vehicle.from_pydict(veh_dict)

# Load drive-cycle
cyc = fsim.Cycle.from_resource("hwfet.csv")
# cyc = fsim.Cycle.from_file("drive_cycles/Fleet DNA Long-Haul Representative_.csv")




df = pd.read_csv("drive_cycles/Fleet DNA Long-Haul Representative_.csv")

# --- rename ---
df = df.rename(columns={
    "Time (seconds)": "time_seconds",
    "Speed (mph)": "speed_mph",
    "Grade (rise/run)": "grade"
})

# --- convert units ---
df["speed_meters_per_second"] = df["speed_mph"] * 0.44704

# --- Savitzky-Golay smoothing ---
# NOTE: window_length must be odd → 61 instead of 60
df["speed_meters_per_second"] = savgol_filter(
    df["speed_meters_per_second"],
    window_length=61,   # closest valid odd number to 60
    polyorder=3
)

# --- keep only FASTSim-required columns ---
cyc_df = df[["time_seconds", "speed_meters_per_second", "grade"]]

# --- convert to dict-of-lists (FASTSim format) ---
cyc_dict = {
    "time_seconds": cyc_df["time_seconds"].tolist(),
    "speed_meters_per_second": cyc_df["speed_meters_per_second"].tolist(),
    "grade": cyc_df["grade"].tolist()
}

df["speed_meters_per_second"] = (
    df["speed_meters_per_second"]
    .rolling(window=5, center=True)
    .mean()
)

# --- build cycle ---
cyc = fsim.Cycle.from_pydict(cyc_dict)






sim = fsim.SimDrive(veh, cyc)
sim.walk()
df = sim.to_dataframe()

fuel_J = df['veh.pt_type.Conv.fc.history.energy_fuel_joules'][-1]
distance_km = df['veh.history.dist_meters'][-1] / 1000

energy_consumption = fuel_J / distance_km / 1000
fuel_consumption = energy_consumption / 35.8e3
print(fuel_consumption)
