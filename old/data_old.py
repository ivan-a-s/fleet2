""" Parameters for the HDT adoption model.
To do:
 - Make certain parameters shared.
 - Efficiency boost for long haul.
 - Add diesel tank mass.
 - Same seed for all ICE engines
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

MAX_AGE = 25
AIR_DENSITY = 1.225  # kg/m^3
GRAVITY = 9.81  # m/s^2
START_YEAR = 2025
END_YEAR = 2050
DISCOUNT_RATE = 0.08
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


POWERTRAINS = {
    'ICE': {
        'dist': 'interp',
        '2000': {'dist': 'const', 'val': 0.40}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
        '2010': {'dist': 'const', 'val': 0.43}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf?utm_source=chatgpt.com
        '2025': {'dist': 'const', 'val': 0.46}, # US DOE Baseline
        '2030': {'dist': 'uniform', 'min': 0.49, 'max': 0.51}, # US DOE 2030 Target
        '2050': {'dist': 'uniform', 'min': 0.51, 'max': 0.54}, # US DOE Ultimate Target
    },
    'BE': 0.875,
    'FC': {
        'dist': 'interp',
        '2025': {'dist': 'const', 'val': 0.59}, # US DOE Baseline
        '2030': {'dist': 'uniform', 'min': 0.59, 'max': 0.63}, # US DOE 2030 Target
        '2050': {'dist': 'uniform', 'min': 0.63, 'max': 0.66}, # US DOE Ultimate Target
    },
}



PARAMS = {
    "Years": {
        "T": np.arange(START_YEAR, END_YEAR + 1),
        "Y": np.arange(START_YEAR - MAX_AGE, END_YEAR + 1),
    },
    'Drive Cycles': {
        'long_haul': {
            'path': 'param_estimation/energy_consumption/Fleet DNA Long-Haul Representative_.csv',
        },
        'regional_haul': {
           'path': 'param_estimation/energy_consumption/Fleet DNA Regional-Haul Representative_.csv',
        },
        'short_haul': {
            'path': 'param_estimation/energy_consumption/Fleet DNA Local Delivery Representative_.csv',
        }
    },
    "Fuels": { # Recharge efficiency?
        "diesel": {
            "Units": "L",
            "LHV": 38.6e6,
            "Emissions Intensity": { # GHGenius 502c
                'Supply': 0.88,
                'Combustion': 2.52,
            },
            "Re-fuel Efficiency": 1.0,
            "Air Pollution": { # g/kg
                'NOX': 4,
                'NMVC': 0.5,
                'SO2': 0.03,
                'PM': 0.1,
                'PMNE': 0.6,
            },
            "Cost": { # CER end-use + 5% GST.
                'dist': 'interp',
                '2025': {'dist': 'uniform', 'min': 1.67, 'max': 1.73},
                '2030': {'dist': 'uniform', 'min': 1.46, 'max': 1.81},
                '2035': {'dist': 'uniform', 'min': 1.65, 'max': 1.91},
                '2040': {'dist': 'uniform', 'min': 1.74, 'max': 2.08},
                '2045': {'dist': 'uniform', 'min': 1.71, 'max': 2.18},
                '2050': {'dist': 'uniform', 'min': 1.67, 'max': 2.15},
            },
            "Water Usage": 2.95, # GREET1
            "Electricity Usage": 0.1, # GREET1
        },
        "Hydrogen": { # (air pollution factors, cost?)
            "Units": "kg",
            "LHV": 120e6,
            "Emissions Intensity": {
                'Supply': 0.57, # Electricity --> H2 (57.5 kWh * 0.0099 kgCO2/kWh)
                'Combustion': 0,
            },
            "Re-fuel Efficiency": 1.0,
            "Air Pollution": { # HICE!
                'NOX': 0,
                'NMVC': 0,
                'SO2': 0,
                'PM': 0,
                'PMNE': 0,
            },
            "Cost": { # Electrolysis
                'dist': 'interp',
                '2025': {'dist': 'const', 'val': 14.97}, # 5t/d + delivery
                '2030': {'dist': 'uniform', 'min': 8.40, 'max': 11.13}, # 5 t/d on-site; 50 t/d + 3.6 delivery
                '2035': {'dist': 'uniform', 'min': 6.55, 'max': 10.56}, # 50 t/d + 1.6 delivery; 5 t/d on-site
                '2040': {'dist': 'uniform', 'min': 5.25, 'max': 6.34}, # 300 t/d + delivery; 50 t/d + delivery
                '2045': {'dist': 'uniform', 'min': 5.20, 'max': 6.13}, # 300 t/d + delivery; 50 t/d + delivery
                '2050': {'dist': 'uniform', 'min': 5.15, 'max': 5.94}, # 300 t/d + delivery; 50 t/d + delivery
            },
            "Water Usage": {'dist': 'uniform', 'min': 95.6, 'max': 121.4}, # IRENA water for electrolysis + water for electricity
            "Electricity Usage": 55,
        },
        "Hydrogen (pyrolysis)": { # Production 5t/d: $5.65, 50 t/d: $3.39, 300 t/d: $2.71 (2030 CAD so * 0.89) CICE (used for scale)
            "Units": "kg", # Compression $0.8/kg and conditioning 0.1 # Delivery (3.2 - 1.6)
            "LHV": 120e6, # 2% annual production cost reduction
            "Emissions Intensity": { # UVic Paper, cost and emissions intensity of hydrogen from thermal pyrolysis of natural gas in BC,
                'Supply': 4.14,
                'Combustion': 0, # NG supply leakage 0.2-0.42%, elec 0.009-0.015, burned NG to CO2 (*44/16) (NG emission factor 28-32)
            },
            "Re-fuel Efficiency": 1.0,
            "Air Pollution": {
                'NOX': 0,
                'NMVC': 0,
                'SO2': 0,
                'PM': 0,
                'PMNE': 0,
            },
            "Cost": { # CICE
                'dist': 'interp',
                '2025': {'dist': 'const', 'val': 11.05}, # 5t/d + delivery
                '2030': {'dist': 'uniform', 'min': 7.24, 'max': 7.38}, # 5 t/d on-site; 50 t/d + 3.6 delivery
                '2035': {'dist': 'uniform', 'min': 5.41, 'max': 6.96}, # 50 t/d + 1.6 delivery; 5 t/d on-site
                '2040': {'dist': 'uniform', 'min': 4.18, 'max': 5.21}, # 300 t/d + delivery; 50 t/d + delivery
                '2045': {'dist': 'uniform', 'min': 2.91, 'max': 5.02}, # 300 t/d + delivery with carbon black; 50 t/d + delivery
                '2050': {'dist': 'uniform', 'min': 2.80, 'max': 4.85}, # 300 t/d + delivery with carbon black; 50 t/d + delivery
            },
            "Water Usage": {'dist': 'uniform', 'min': 4.71, 'max': 5.66}, # 1UVic
            "Electricity Usage": 2.12, # UVic
        },
        "Hydrogen (pyrolysis + elec.)": { # Production 1.85 (compression same) * regular pyrolysis
            "Units": "kg",
            "LHV": 120e6,
            "Emissions Intensity": { # Could potentially change over time
                'Supply': 1.92, # UVic + methane supply chain (Seymour et al., 2024)
                'Combustion': 0,
            },
            "Re-fuel Efficiency": 1.0,
            "Air Pollution": {
                'NOX': 0,
                'NMVC': 0,
                'SO2': 0,
                'PM': 0,
                'PMNE': 0,
            },
            "Cost": {
                'dist': 'interp',
                '2025': {'dist': 'const', 'val': 15.85}, # 5t/d + delivery
                '2030': {'dist': 'uniform', 'min': 9.90, 'max': 11.81}, # 5 t/d on-site; 50 t/d + 3.6 delivery
                '2035': {'dist': 'uniform', 'min': 7.86, 'max': 11.05}, # 50 t/d + 1.6 delivery; 5 t/d on-site
                '2040': {'dist': 'uniform', 'min': 5.60, 'max': 7.48}, # 300 t/d + delivery; 50 t/d + delivery
                '2045': {'dist': 'uniform', 'min': 4.75, 'max': 7.12}, # 300 t/d + delivery with carbon black; 50 t/d + delivery
                '2050': {'dist': 'uniform', 'min': 4.55, 'max': 6.81}, # 300 t/d + delivery with carbon black; 50 t/d + delivery
            },
            "Water Usage": {'dist': 'uniform', 'min': 16.3, 'max': 20.9}, # Extra 7.5-10 cooling
            "Electricity Usage": 10.23,
        },
        "Fast Charge": { # charging efficiency
            "Units": "kWh",
            "LHV": 3.6e6,
            "Emissions Intensity": {
                'Supply': 0.0099,
                'Combustion': 0,
            },
            "Re-fuel Efficiency": 0.86,
            "Air Pollution": {
                'NOX': 0,
                'NMVC': 0,
                'SO2': 0,
                'PM': 0,
                'PMNE': 0,
            },
            "Cost": { # BC Hydro fleet rate.
                'dist': 'interp',
                '2025': {'dist': 'uniform', 'min': 0.360, 'max': 0.361},
                '2030': {'dist': 'uniform', 'min': 0.368, 'max': 0.372},
                '2035': {'dist': 'uniform', 'min': 0.374, 'max': 0.385},
                '2040': {'dist': 'uniform', 'min': 0.382, 'max': 0.397},
                '2045': {'dist': 'uniform', 'min': 0.401, 'max': 0.407},
                '2050': {'dist': 'uniform', 'min': 0.412, 'max': 0.418},
            },
            "Water Usage": {'dist': 'uniform', 'min': 1.43, 'max': 1.88},
            "Electricity Usage": 1.0,
        },
        "Slow Charge": {
            "Units": "kWh",
            "LHV": 3.6e6,
            "Emissions Intensity": {
                'Supply': 0.0099,
                'Combustion': 0,
            },
            "Re-fuel Efficiency": 0.95,
            "Air Pollution": {
                'NOX': 0,
                'NMVC': 0,
                'SO2': 0,
                'PM': 0,
                'PMNE': 0,
            },
            "Cost": { # BC Hydro fleet rate.
                'dist': 'interp',
                '2025': {'dist': 'uniform', 'min': 0.102, 'max': 0.103},
                '2030': {'dist': 'uniform', 'min': 0.105, 'max': 0.106},
                '2035': {'dist': 'uniform', 'min': 0.106, 'max': 0.110},
                '2040': {'dist': 'uniform', 'min': 0.109, 'max': 0.113},
                '2045': {'dist': 'uniform', 'min': 0.114, 'max': 0.116},
                '2050': {'dist': 'uniform', 'min': 0.117, 'max': 0.119},
            },
            "Water Usage": {'dist': 'uniform', 'min': 1.43, 'max': 1.88},
            "Electricity Usage": 1.0,
        }
    },
    'autonomous_t50': 2040,
    'powertrains': {
        'ice': { # Same seed for all ICE engines
            'mass': 1857,
            'efficiency': {
                'dist': 'interp',
                '2000': {'dist': 'const', 'val': 0.40}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
                '2010': {'dist': 'const', 'val': 0.43}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf?utm_source=chatgpt.com
                '2025': {'dist': 'const', 'val': 0.46}, # US DOE Baseline
                '2030': {'dist': 'uniform', 'min': 0.49, 'max': 0.51}, # US DOE 2030 Target
                '2050': {'dist': 'uniform', 'min': 0.51, 'max': 0.54}, # US DOE Ultimate Target
            },
            'cagr_nacent': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
            'cagr_mature': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
        },
        'be': 0.875,
        'fc': {
            'dist': 'interp',
            '2025': {'dist': 'const', 'val': 0.59}, # US DOE Baseline
            '2030': {'dist': 'uniform', 'min': 0.59, 'max': 0.63}, # US DOE 2030 Target
            '2050': {'dist': 'uniform', 'min': 0.63, 'max': 0.66}, # US DOE Ultimate Target
        },
    },
    'esss': {
        'diesel_tank': {
            'ess_specific_mass': 0,
            'ess_specific_embodied': 0,
            'ess_usable_capacity': 0.95,
            'ess_refuel_rate': 6000,
        },
        'battery': {
            'specific_mass': { # (Haghbin et al., 2025; Jose et al., 2025)
                'dist': 'interp',
                '2025': {'dist': 'const', 'val': 6},
                '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
                '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
            },
            'embodied_emissions': { # (Xu et al., 2022)
                'dist': 'linear',
                'start': {'dist': 'const', 'val': 87},
                'end': {'dist': 'uniform', 'min': 20, 'max': 40},
            },
        },
        'h2_300pa': {
            'specific_mass': {
                'dist': 'interp',
                '2025': 18.5,
                '2050': 15.1,
            },
            'embodied_emissions': {
                'dist': 'linear',
                'start': {'dist': 'const', 'val': 40.7},
                'end': {'dist': 'triangle', 'min': 13.6, 'mode': 25.7, 'max': 33.2},
            },
        }
    },
    'Vehicles': { # Need to add straight trucks and hybrids.
        'Sleeper': {
            'Shared': {
                'activity_proportion': 0.77, # maybe wrong since Sleepers don't exclusively do long-haul?
                'average_payload': 16_000,
                'target_distance': (208_714-17_478) / (1 + np.exp(0.372*(np.arange(MAX_AGE) - 7.62))) + 17_478,
                'revenue_per_tkm': 0.10, # Uncertainty?
                'trailers_per_truck': 3,
                'gvwl': 53_500, # kg
                'drive_cycle': [
                    'long_haul' 
                    if y < 10 
                    else 'regional_haul'
                    for y in range(MAX_AGE)
                ],
                'survival_rate': np.linspace(1, 0, MAX_AGE+1)[:-1],
                'frontal_area': 9.2, # m^2
                'roll_coefficient': 0.0054,
                'frame_mass': 6_052, # kg
                'trailer_mass': 5_029, # kg
                'mass_correction': 0.2,
                'transmission_efficiency': 0.95,
                'embodied': {
                    'dist': 'interp',
                    '2025': {'dist': 'const', 'val': 2.2}, # World Steel
                    '2050': {'dist': 'triangle', 'min': 0.9, 'mode': 1.7, 'max': 2.2}, # IEA SDS, IEA STEPS, IEA 40% from material efficiency.
                },
                'driver_cost': 0.38, # $/km
                'pollution_cost': { # Rural CE Delft B.C. values per tonne.
                    'NOX': 26_058,
                    'NMVC': 2_428,
                    'SO2': 25_542,
                    'PM': 144_770, # These are in Euro 2016 currency, change.
                    'PMNE': 46_120,
                },
                'drag_coefficient': { # US DoE Targets
                    'dist': 'interp',
                    '2000': {'dist': 'const', 'val': 0.7}, # Old standard
                    '2010': {'dist': 'const', 'val': 0.6},
                    '2025': {'dist': 'const', 'val': 0.49}, # US DOE Baseline
                    '2030': {'dist': 'uniform', 'min': 0.34, 'max': 0.43}, # US DOE 2030 Target
                    '2050': {'dist': 'uniform', 'min': 0.30, 'max': 0.41}, # US DOE Ultimate Target
                },
                'accessory_demand': 4_250 * 0.46,
            },
            'Powertrains': {
                'D': {
                    'fuel': 'diesel',
                    'powertrain': 'ice',
                    'ess': 'diesel_tank',
                    'fuel_capacity': 500,
                    'regen_efficiency': 0,
                    'running_cost': 0.17,
                    'init_market_limit': 1.0,
                },
                # 'DHNP': { # Need to add battery for weight and emissions.
                #     'fuel': {
                #         'Diesel': np.ones(MAX_AGE) * 0.999,
                #         'Slow Charge': np.ones(MAX_AGE) * 0.001,
                #     },
                #     'params': {
                #         'drivetrain_mass': 1_974,
                #         'motor_size': 220,
                #         'ess_specific_mass': {
                #             'Diesel': 0, # included in powertrain mass.
                #             'Slow Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 6},
                #                 '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
                #                 '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
                #             }
                #         },
                #         'ess_embodied': {
                #             'Diesel': 0,
                #             'Slow Charge': { # (Xu et al., 2022)
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 87},
                #                 'end': {'dist': 'uniform', 'min': 20, 'max': 40},
                #             }
                #         },
                #         'efficiency': {
                #             'Diesel': { # US DoE Targets
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 0.46}, # US DOE Baseline
                #                 '2030': {'dist': 'uniform', 'min': 0.49, 'max': 0.51}, # US DOE 2030 Target
                #                 '2050': {'dist': 'uniform', 'min': 0.51, 'max': 0.54}, # US DOE Ultimate Target
                #             },
                #             'Slow Charge': 0.875,
                #         },
                #         'regen_efficiency': 0.71,
                #         'accessory_load': 3_400,
                #         'fuel_capacity': {
                #             'Diesel': 500,
                #             'Slow Charge': 10
                #         },
                #         'battery_size': 10,
                #         'usable_capacity': {
                #             'Diesel': 0.95,
                #             'Slow Charge': 0.95
                #         },
                #         'refuel_rate': {
                #             'Diesel': 6000,
                #             'Slow Charge': { # Not relevant.
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 250},
                #                 'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
                #             }
                #         },
                #         'running_cost': 0.17,
                #         'market_limit': {
                #             'init': 0.02,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
                #         }
                #     },
                # },
                # 'DHP': { # Need to add battery for weight and emissions.
                #     'fuel': { # make sure this is appropriate.
                #         'Diesel': np.ones(MAX_AGE) * 0.95,
                #         'Fast Charge': np.ones(MAX_AGE) * 0.08,
                #     },
                #     'params': {
                #         'drivetrain_mass': 1_974,
                #         'motor_size': 220,
                #         'ess_specific_mass': {
                #             'Diesel': 0, # included in powertrain mass.
                #             'Fast Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 6},
                #                 '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
                #                 '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
                #             }
                #         },
                #         'ess_embodied': {
                #             'Diesel': 0,
                #             'Fast Charge': { # (Xu et al., 2022)
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 87},
                #                 'end': {'dist': 'uniform', 'min': 20, 'max': 40},
                #             }
                #         },
                #         'efficiency': {
                #             'Diesel': { # US DoE Targets
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 0.46}, # US DOE Baseline
                #                 '2030': {'dist': 'uniform', 'min': 0.49, 'max': 0.51}, # US DOE 2030 Target
                #                 '2050': {'dist': 'uniform', 'min': 0.51, 'max': 0.54}, # US DOE Ultimate Target
                #             },
                #             'Fast Charge': 0.875,
                #         },
                #         'regen_efficiency': 0.71,
                #         'accessory_load': 3_400, # maybe lower if electrified.
                #         'fuel_capacity': {
                #             'Diesel': 500,
                #             'Fast Charge': 100
                #         },
                #         'battery_size': 100, # This shouldn't be needed anymore.
                #         'usable_capacity': {
                #             'Diesel': 0.95,
                #             'Fast Charge': 0.95
                #         },
                #         'refuel_rate': {
                #             'Diesel': 6000,
                #             'Fast Charge': {
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 250},
                #                 'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
                #             }
                #         },
                #         'running_cost': 0.17,
                #         'market_limit': {
                #             'init': 0.02,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
                #         }
                #     },
                # },
                # 'BE': {
                #     'fuel': {'Fast Charge': np.ones(MAX_AGE)},
                #     'params': {
                #         'drivetrain_mass': 1_115,
                #         'motor_size': 880,
                #         'ess_specific_mass': {
                #             'Fast Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 6},
                #                 '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
                #                 '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
                #             }
                #         },
                #         'ess_embodied': {
                #             'Fast Charge': { # (Xu et al., 2022)
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 87},
                #                 'end': {'dist': 'uniform', 'min': 20, 'max': 40},
                #             }
                #         },
                #         'efficiency': {
                #             'Fast Charge': 0.875,
                #         },
                #         'regen_efficiency': 0.71,
                #         'accessory_load': 6_900, # weather impacts
                #         'fuel_capacity': {'Fast Charge': 1000},
                #         'battery_size': 1000,
                #         'usable_capacity': {
                #             'Fast Charge': {
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 0.8},
                #                 '2030': {'dist': 'uniform', 'min': 0.85, 'max': 0.90},
                #                 '2050': {'dist': 'uniform', 'min': 0.9, 'max': 0.95},
                #             },
                #         },
                #         'refuel_rate': { # Do I account for efficiency in re-fuel time?
                #             'Fast Charge': {
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 250},
                #                 'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
                #             }
                #         },
                #         'running_cost': 0.12,
                #         'market_limit': {
                #             'init': 0.02,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.35, 'max': 0.49},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.25, 'max': 0.29},
                #         }
                #     },
                # },
                # 'FC': {
                #     'fuel': {
                #         'Hydrogen': np.ones(MAX_AGE) * 0.999,
                #         'Slow Charge': np.ones(MAX_AGE) * 0.001,
                #     },
                #     'params': {
                #         'drivetrain_mass': 1_999, # Fuel cell weight? 1.818 * 880 (Lajevardi)
                #         'motor_size': 880,
                #         'fc_size': 880,
                #         'ess_specific_mass': {
                #             'Hydrogen': {
                #                 'dist': 'interp',
                #                 '2025': 18.5,
                #                 '2050': 15.1,
                #             },
                #             'Slow Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 6},
                #                 '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
                #                 '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
                #             }
                #         },
                #         'ess_embodied': {
                #             'Hydrogen': {
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 40.7},
                #                 'end': {'dist': 'triangle', 'min': 13.6, 'mode': 25.7, 'max': 33.2},
                #             },
                #             'Slow Charge': { # (Xu et al., 2022)
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 87},
                #                 'end': {'dist': 'uniform', 'min': 20, 'max': 40},
                #             }
                #         },
                #         'efficiency': { # US DoE Targets
                #             'Hydrogen': {
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 0.59}, # US DOE Baseline
                #                 '2030': {'dist': 'uniform', 'min': 0.59, 'max': 0.63}, # US DOE 2030 Target
                #                 '2050': {'dist': 'uniform', 'min': 0.63, 'max': 0.66}, # US DOE Ultimate Target
                #             },
                #             'Slow Charge': 0.875,
                #         },
                #         'regen_efficiency': 0.71,
                #         'fc_lifetime': { # hours, US DoE Targets
                #             'dist': 'interp',
                #             '2025': {'dist': 'const', 'val': 20_000}, # Ballard tech doc, The FCmove-XD
                #             '2030': {'dist': 'uniform', 'min': 22_500, 'max': 27_500}, # US DOE 2030 Target
                #             '2050': {'dist': 'uniform', 'min': 27_500, 'max': 32_500}, # US DOE Ultimate Target
                #         },
                #         'accessory_load': 3_400,
                #         'fuel_capacity': {
                #             'Hydrogen': 80,
                #             'Slow Charge': 5,
                #         },
                #         'battery_size': 5, # small?
                #         'usable_capacity': {
                #             'Hydrogen': 0.9,
                #             'Slow Charge': 0.95
                #         },
                #         'refuel_rate': {'Hydrogen': 480,
                #             'Slow Charge': { # Not relevant.
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 250},
                #                 'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
                #             },
                #         },
                #         'running_cost': 0.14,
                #         'market_limit': {
                #             'init': 0.02,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.35, 'max': 0.49},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.25, 'max': 0.29},
                #         }
                #     },
                # },
                # 'HICE': { # add hybrid version?
                #     'fuel': {'Hydrogen': np.ones(MAX_AGE)},
                #     'params': {
                #         'drivetrain_mass': 1_857,
                #         'ess_specific_mass': {
                #             'Hydrogen': {
                #                 'dist': 'interp',
                #                 '2025': 18.5,
                #                 '2050': 15.1,
                #             },
                #         },
                #         'ess_embodied': {
                #             'Hydrogen': {
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 40.7},
                #                 'end': {'dist': 'triangle', 'min': 13.6, 'mode': 25.7, 'max': 33.2},
                #             },
                #         },
                #         'efficiency': { # US DoE Targets for Diesel
                #             'Hydrogen': {
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 0.46}, # US DOE Baseline
                #                 '2030': {'dist': 'uniform', 'min': 0.49, 'max': 0.51}, # US DOE 2030 Target
                #                 '2050': {'dist': 'uniform', 'min': 0.51, 'max': 0.54}, # US DOE Ultimate Target
                #             },
                #         },
                #         'regen_efficiency': 0,
                #         'accessory_load': 4_250,
                #         'fuel_capacity': {'Hydrogen': 80},
                #         'usable_capacity': {'Hydrogen': 0.9},
                #         'refuel_rate': {'Hydrogen': 480},
                #         'running_cost': 0.17,
                #         'market_limit': {
                #             'init': 0.02,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
                #         }
                #     },
                # },
                # 'DHICE': { # add hybrid version?
                    # 'fuel': {'Diesel': np.ones(MAX_AGE)*0.75, 'Hydrogen': np.ones(MAX_AGE)*0.25},
                    # 'params': {
                    #     'drivetrain_mass': 1_817,
                    #     'ess_specific_mass': {
                    #         'Diesel': 0, #included in powertrain
                    #         'Hydrogen': {
                    #             'dist': 'interp',
                    #             '2025': 16.0,
                    #             '2050': 13.0,
                    #         },
                    #     },
                    #     'ess_embodied': {
                    #         'Diesel': 0,
                    #         'Hydrogen': {
                    #             'dist': 'linear',
                    #             'start': {'dist': 'const', 'val': 35.2},
                    #             'end': {'dist': 'triangle', 'min': 11.7, 'mode': 22.1, 'max': 28.6},
                    #         },
                    #     },
                    #     'efficiency': { # US DoE Targets for Diesel
                    #         'Diesel': { # US DoE Targets
                    #             'dist': 'interp',
                    #             '2025': {'dist': 'const', 'val': 0.46}, # US DOE Baseline
                    #             '2030': {'dist': 'uniform', 'min': 0.49, 'max': 0.51}, # US DOE 2030 Target
                    #             '2050': {'dist': 'uniform', 'min': 0.51, 'max': 0.54}, # US DOE Ultimate Target
                    #         },
                    #         'Hydrogen': {
                    #             'dist': 'interp',
                    #             '2025': {'dist': 'const', 'val': 0.46}, # US DOE Baseline
                    #             '2030': {'dist': 'uniform', 'min': 0.49, 'max': 0.51}, # US DOE 2030 Target
                    #             '2050': {'dist': 'uniform', 'min': 0.51, 'max': 0.54}, # US DOE Ultimate Target
                    #         },
                    #     },
                    #     'regen_efficiency': 0,
                    #     'accessory_load': 4_250,
                    #     'fuel_capacity': {
                    #         'Diesel': 500,
                    #         'Hydrogen': 40
                    #     },
                    #     'usable_capacity': {
                    #         'Diesel': 0.95,
                    #         'Hydrogen': 0.9
                    #     },
                    #     'refuel_rate': {
                    #         'Diesel': 6000,
                    #         'Hydrogen': 480
                    #     },
                    #     'running_cost': 0.17,
                    #     'market_limit': {
                    #         'init': 0.02,
                    #         'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
                    #         'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
                    #     }
                    # },
                # },
            },
            'Costs': { # 1.573 USD 2020 (Lajevardi 2019) -> CAD 2024, don't bother with trailers (battery and FC should progress with time) break down in capital plot
                'base': 163_000,
                'diesel_engine': 35_000,
                'combustion_transmission': 13_700,
                'electric_transmission': 3_200,
                'after_treatment': 12_000,
                'tank': 9.44, # $/L
                'battery': {
                    'dist': 'linear',
                    'start': {'dist': 'const', 'val': 159}, # USD $115/kWh Bloomberg * 1.37
                    'end': {'dist': 'triangle', 'min': 81, 'mode': 101, 'max': 157}, # Lukas Mauler 2021
                },
                'h2_tank': { # Maybe ought be different for 700 bar and 350 bar.
                    'dist': 'interp',
                    '2025': {'dist': 'uniform', 'min': 625, 'max': 686}, # 2023 US DoE per 100k/yr production
                    '2030': {'dist': 'uniform', 'min': 498, 'max': 625}, # US DOE 2030 Target
                    '2050': {'dist': 'uniform', 'min': 442, 'max': 498}, # US DOE Ultimate Target
                },
                'motor': 40, # $/kW,
                'fc': {
                    'dist': 'interp',
                    '2025': {'dist': 'uniform', 'min': 357, 'max': 500}, # 2025 US DoE per 1k/yr production (264 at 100k/yr)
                    '2030': {'dist': 'uniform', 'min': 146, 'max': 285}, # US DOE 2030 Target at 1k-10k
                    '2040': {'dist': 'uniform', 'min': 100, 'max': 201}, # US DOE 2030 Target at 10k-100k
                    '2050': {'dist': 'uniform', 'min': 79, 'max': 146}, # US DOE Ultimate Target 100k lower to 2030 Target 10k higher
                },
                'HICE_engine': {
                    'dist': 'interp',
                    '2025': {'dist': 'const', 'val': 43_750},
                    '2050': {'dist': 'const', 'val': 35_000},
                },
            } # Should evolve with time and scale.
        },
        # 'Day Cab': {
        #     'Shared': {
        #         'activity_proportion': 0.17,
        #         'average_payload': 10_000,
        #         'target_distance': (137_922-11_551) / (1 + np.exp(0.372*(np.arange(MAX_AGE) - 7.62))) + 11_551,
        #         'revenue_per_tkm': 0.17, # Uncertainty?
        #         'trailers_per_truck': 3,
        #         'gvwl': 53_500, # kg
        #         'drive_cycle': [
        #             'regional_haul' for y in range(MAX_AGE)
        #         ],
        #         'survival_rate': np.linspace(1, 0, MAX_AGE+1)[:-1],
        #         'frontal_area': 9.2, # m^2
        #         'roll_coefficient': 0.0054,
        #         'frame_mass': 5_380, # kg
        #         'trailer_mass': 5_029, # kg
        #         'mass_correction': 0.2, # Should change by drivetrain
        #         'transmission_efficiency': 0.95,
        #         'embodied': {
        #             'dist': 'interp',
        #             '2025': {'dist': 'const', 'val': 2.2}, # World Steel
        #             '2050': {'dist': 'triangle', 'min': 0.9, 'mode': 1.7, 'max': 2.2}, # IEA SDS, IEA STEPS, IEA 40% from material efficiency.
        #         },
        #         'driver_cost': 0.48, # $/km
        #         'pollution_cost': { # Average rural & urban CE Delft B.C. values per tonne.
        #             'NOX': (26_058 + 44_051)/2,
        #             'NMVC': 2_482,
        #             'SO2': 25_542,
        #             'PM': (144_770 + 254_383)/2,
        #             'PMNE': 46_120,
        #         },
        #         'drag_coefficient': { # US DoE Targets
        #             'dist': 'interp',
        #             '2000': {'dist': 'const', 'val': 0.7}, # Old standard
        #             '2010': {'dist': 'const', 'val': 0.6},
        #             '2025': {'dist': 'const', 'val': 0.49}, # US DOE Baseline
        #             '2030': {'dist': 'uniform', 'min': 0.34, 'max': 0.43}, # US DOE 2030 Target
        #             '2050': {'dist': 'uniform', 'min': 0.30, 'max': 0.41}, # US DOE Ultimate Target
        #         },
        #     },
        #     'Powertrains': {
        #         'D': {
        #             'fuel': {'Diesel': np.ones(MAX_AGE)},
        #             'params': {
        #                 'drivetrain_mass': 1857,
        #                 'ess_specific_mass': {
        #                     'Diesel': 0, # Included in powertrain
        #                 },
        #                 'efficiency': {
        #                     'Diesel': { # 5% cut
        #                         'dist': 'interp',
        #                         '2000': {'dist': 'const', 'val': 0.35}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
        #                         '2010': {'dist': 'const', 'val': 0.38}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf
        #                         '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
        #                         '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
        #                         '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
        #                     },
        #                 },
        #                 'regen_efficiency': 0,
        #                 'accessory_load': 4_250,
        #                 'fuel_capacity': {
        #                     'Diesel': 500
        #                 },
        #                 'ess_embodied': { # Irrelevant
        #                     'Diesel': 0,
        #                 },
        #                 'usable_capacity': {
        #                     'Diesel': 0.95
        #                 },
        #                 'refuel_rate': {
        #                     'Diesel': 6000
        #                 },
        #                 'running_cost': 0.17,
        #                 'market_limit': {
        #                     'init': 1.0,
        #                     'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
        #                     'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
        #                 }
        #             },
        #         },
        #         'DHNP': { # Need to add battery for weight and emissions.
        #             'fuel': {
        #                 'Diesel': np.ones(MAX_AGE) * 0.999,
        #                 'Slow Charge': np.ones(MAX_AGE) * 0.001,
        #             },
        #             'params': {
        #                 'drivetrain_mass': 1_974,
        #                 'motor_size': 220,
        #                 'ess_specific_mass': {
        #                     'Diesel': 0, # included in powertrain mass.
        #                     'Slow Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
        #                         'dist': 'interp',
        #                         '2025': {'dist': 'const', 'val': 6},
        #                         '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
        #                         '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
        #                     }
        #                 },
        #                 'ess_embodied': {
        #                     'Diesel': 0,
        #                     'Slow Charge': { # (Xu et al., 2022)
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 87},
        #                         'end': {'dist': 'uniform', 'min': 20, 'max': 40},
        #                     }
        #                 },
        #                 'efficiency': {
        #                     'Diesel': { # 5% cut
        #                         'dist': 'interp',
        #                         '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
        #                         '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
        #                         '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
        #                     },
        #                     'Slow Charge': 0.875,
        #                 },
        #                 'regen_efficiency': 0.71,
        #                 'accessory_load': 3_400,
        #                 'fuel_capacity': {
        #                     'Diesel': 500,
        #                     'Slow Charge': 5
        #                 },
        #                 'battery_size': 5,
        #                 'usable_capacity': {
        #                     'Diesel': 0.95,
        #                     'Slow Charge': 0.95
        #                 },
        #                 'refuel_rate': {
        #                     'Diesel': 6000,
        #                     'Slow Charge': {
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 250},
        #                         'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
        #                     }
        #                 },
        #                 'running_cost': 0.17,
        #                 'market_limit': {
        #                     'init': 0.02,
        #                     'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
        #                     'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
        #                 }
        #             },
        #         },
        #         'DHP': {
        #             'fuel': {
        #                 'Diesel': np.ones(MAX_AGE) * 0.84,
        #                 'Slow Charge': np.ones(MAX_AGE) * 0.16,
        #             },
        #             'params': {
        #                 'drivetrain_mass': 1_974,
        #                 'motor_size': 220,
        #                 'ess_specific_mass': {
        #                     'Diesel': 0, # included in powertrain mass.
        #                     'Slow Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
        #                         'dist': 'interp',
        #                         '2025': {'dist': 'const', 'val': 6},
        #                         '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
        #                         '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
        #                     }
        #                 },
        #                 'ess_embodied': {
        #                     'Diesel': 0,
        #                     'Slow Charge': { # (Xu et al., 2022)
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 87},
        #                         'end': {'dist': 'uniform', 'min': 20, 'max': 40},
        #                     }
        #                 },
        #                 'efficiency': {
        #                     'Diesel': { # 5% cut
        #                         'dist': 'interp',
        #                         '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
        #                         '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
        #                         '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
        #                     },
        #                     'Slow Charge': 0.875,
        #                 },
        #                 'regen_efficiency': 0.71,
        #                 'accessory_load': 3_400,
        #                 'fuel_capacity': {
        #                     'Diesel': 500,
        #                     'Slow Charge': 100
        #                 },
        #                 'battery_size': 100,
        #                 'usable_capacity': {
        #                     'Diesel': 0.95,
        #                     'Slow Charge': 0.95
        #                 },
        #                 'refuel_rate': {
        #                     'Diesel': 6000,
        #                     'Slow Charge': {
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 250},
        #                         'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
        #                     }
        #                 },
        #                 'running_cost': 0.17,
        #                 'market_limit': {
        #                     'init': 0.02,
        #                     'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
        #                     'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
        #                 }
        #             },
        #         },
        #         'BE': {
        #             'fuel': {'Slow Charge': np.ones(MAX_AGE)},
        #             'params': { # battery replacement costs and frequency
        #                 'drivetrain_mass': 1_115,
        #                 'motor_size': 880,
        #                 'ess_specific_mass': {
        #                     'Slow Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
        #                         'dist': 'interp',
        #                         '2025': {'dist': 'const', 'val': 6},
        #                         '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
        #                         '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
        #                     }
        #                 },
        #                 'ess_embodied': {
        #                     'Slow Charge': { # (Xu et al., 2022)
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 87},
        #                         'end': {'dist': 'uniform', 'min': 20, 'max': 40},
        #                     }
        #                 },
        #                 'efficiency': {
        #                     'Slow Charge': 0.875,
        #                 },
        #                 'regen_efficiency': 0.71,
        #                 'accessory_load': 6_900, # weather impacts
        #                 'fuel_capacity': {'Slow Charge': 600},
        #                 'battery_size': 600,
        #                 'usable_capacity': {
        #                     'Slow Charge': {
        #                         'dist': 'interp',
        #                         '2025': {'dist': 'const', 'val': 0.8}, # US DOE Baseline
        #                         '2030': {'dist': 'uniform', 'min': 0.85, 'max': 0.90}, # US DOE 2030 Target
        #                         '2050': {'dist': 'uniform', 'min': 0.9, 'max': 0.95}, # US DOE Ultimate Target
        #                     },
        #                 },
        #                 'refuel_rate': { # Do I account for efficiency in re-fuel time?
        #                     'Slow Charge': {
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 250},
        #                         'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
        #                     }
        #                 },
        #                 'running_cost': 0.12,
        #                 'market_limit': {
        #                     'init': 0.02,
        #                     'CAGR_early': {'dist': 'uniform', 'min': 0.35, 'max': 0.49},
        #                     'CAGR_other': {'dist': 'uniform', 'min': 0.25, 'max': 0.29},
        #                 }
        #             },
        #         },
        #         'FC': {
        #             'fuel': {'Hydrogen': np.ones(MAX_AGE)},
        #             'params': {
        #                 'drivetrain_mass': 1_999,
        #                 'motor_size': 880,
        #                 'fc_size': 880,
        #                 'ess_specific_mass': {
        #                     'Hydrogen': {
        #                         'dist': 'interp',
        #                         '2025': 16.0,
        #                         '2050': 13.0,
        #                     },
        #                     'Slow Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
        #                         'dist': 'interp',
        #                         '2025': {'dist': 'const', 'val': 6},
        #                         '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
        #                         '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
        #                     }
        #                 },
        #                 'ess_embodied': {
        #                     'Diesel': 0,
        #                     'Hydrogen': {
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 35.2},
        #                         'end': {'dist': 'triangle', 'min': 11.7, 'mode': 22.1, 'max': 28.6},
        #                     },
        #                     'Slow Charge': { # (Xu et al., 2022)
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 87},
        #                         'end': {'dist': 'uniform', 'min': 20, 'max': 40},
        #                     }
        #                 },
        #                 'efficiency': { # US DoE Targets
        #                     'Hydrogen': {
        #                         'dist': 'interp',
        #                         '2025': {'dist': 'const', 'val': 0.59}, # US DOE Baseline
        #                         '2030': {'dist': 'uniform', 'min': 0.59, 'max': 0.63}, # US DOE 2030 Target
        #                         '2050': {'dist': 'uniform', 'min': 0.63, 'max': 0.66}, # US DOE Ultimate Target
        #                     },
        #                     'Slow Charge': 0.875,
        #                 },
        #                 'regen_efficiency': 0.71,
        #                 'fc_lifetime': { # hours, US DoE Targets
        #                     'dist': 'interp',
        #                     '2025': {'dist': 'const', 'val': 20_000}, # Ballard tech doc, The FCmove-XD
        #                     '2030': {'dist': 'triangle', 'min': 22_500, 'mode': 25_000, 'max': 27_500}, # US DOE 2030 Target
        #                     '2050': {'dist': 'triangle', 'min': 27_500, 'mode': 30_000, 'max': 32_500}, # US DOE Ultimate Target
        #                 },
        #                 'accessory_load': 3_400,
        #                 'fuel_capacity': {
        #                     'Hydrogen': 50,
        #                     'Slow Charge': 5,
        #                 },
        #                 'battery_size': 5, # small?
        #                 'usable_capacity': {
        #                     'Hydrogen': 0.9,
        #                     'Slow Charge': 0.95
        #                 },
        #                 'refuel_rate': {'Hydrogen': 480,
        #                     'Slow Charge': { # Not relevant.
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 250},
        #                         'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
        #                     },
        #                 },
        #                 'running_cost': 0.14,
        #                 'market_limit': {
        #                     'init': 0.02,
        #                     'CAGR_early': {'dist': 'uniform', 'min': 0.35, 'max': 0.49},
        #                     'CAGR_other': {'dist': 'uniform', 'min': 0.25, 'max': 0.29},
        #                 }
        #             },
        #         },
        #         'HICE': { # add hybrid version?
        #             'fuel': {'Hydrogen': np.ones(MAX_AGE)},
        #             'params': {
        #                 'drivetrain_mass': 1_857,
        #                 'ess_specific_mass': {
        #                     'Hydrogen': {
        #                         'dist': 'interp',
        #                         '2025': 16.0,
        #                         '2050': 13.0,
        #                     },
        #                 },
        #                 'ess_embodied': {
        #                     'Hydrogen': {
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 35.2},
        #                         'end': {'dist': 'triangle', 'min': 11.7, 'mode': 22.1, 'max': 28.6},
        #                     },
        #                 },
        #                 'efficiency': { # DOE targets
        #                     'Hydrogen': { # 5% cut
        #                         'dist': 'interp',
        #                         '2000': {'dist': 'const', 'val': 0.35}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
        #                         '2010': {'dist': 'const', 'val': 0.38}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf
        #                         '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
        #                         '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
        #                         '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
        #                     },
        #                 },
        #                 'regen_efficiency': 0,
        #                 'accessory_load': 4_250,
        #                 'fuel_capacity': {'Hydrogen': 50},
        #                 'usable_capacity': {'Hydrogen': 0.9},
        #                 'refuel_rate': {'Hydrogen': 480},
        #                 'running_cost': 0.17,
        #                 'market_limit': {
        #                     'init': 0.02,
        #                     'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
        #                     'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
        #                 }
        #             },
        #         },
        #         'DHICE': { # add hybrid version?
        #             'fuel': {'Diesel': np.ones(MAX_AGE)*0.75, 'Hydrogen': np.ones(MAX_AGE)*0.25},
        #             'params': {
        #                 'drivetrain_mass': 1_857,
        #                 'ess_specific_mass': {
        #                     'Diesel': 0,
        #                     'Hydrogen': {
        #                         'dist': 'interp',
        #                         '2025': 16.0,
        #                         '2050': 13.0,
        #                     },
        #                 },
        #                 'ess_embodied': {
        #                     'Diesel': 0,
        #                     'Hydrogen': {
        #                         'dist': 'linear',
        #                         'start': {'dist': 'const', 'val': 35.2},
        #                         'end': {'dist': 'triangle', 'min': 11.7, 'mode': 22.1, 'max': 28.6},
        #                     },
        #                 },
        #                 'efficiency': { # US DoE Targets for Diesel
        #                     'Diesel': { # 5% cut
        #                         'dist': 'interp',
        #                         '2000': {'dist': 'const', 'val': 0.35}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
        #                         '2010': {'dist': 'const', 'val': 0.38}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf
        #                         '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
        #                         '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
        #                         '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
        #                     },
        #                     'Hydrogen': { # 5% cut
        #                         'dist': 'interp',
        #                         '2000': {'dist': 'const', 'val': 0.35}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
        #                         '2010': {'dist': 'const', 'val': 0.38}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf
        #                         '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
        #                         '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
        #                         '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
        #                     },
        #                 },
        #                 'regen_efficiency': 0,
        #                 'accessory_load': 4_250,
        #                 'fuel_capacity': {
        #                     'Diesel': 500,
        #                     'Hydrogen': 20
        #                 },
        #                 'usable_capacity': {
        #                     'Diesel': 0.95,
        #                     'Hydrogen': 0.9
        #                 },
        #                 'refuel_rate': {
        #                     'Diesel': 6000,
        #                     'Hydrogen': 480
        #                 },
        #                 'running_cost': 0.17,
        #                 'market_limit': {
        #                     'init': 0.02,
        #                     'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
        #                     'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
        #                 }
        #             },
        #         },
        #     },
        #     'Costs': { # 1.573 USD 2020 (Lajevardi 2019) -> CAD 2024, don't bother with trailers (battery and FC should progress with time) break down in capital plot
        #         'base': 153_000,
        #         'diesel_engine': 35_000,
        #         'combustion_transmission': 13_700,
        #         'electric_transmission': 3_200,
        #         'after_treatment': 12_000,
        #         'tank': 9.44, # $/L
        #         'battery': {
        #             'dist': 'linear',
        #             'start': {'dist': 'const', 'val': 159}, # USD $115/kWh Bloomberg * 1.37
        #             'end': {'dist': 'triangle', 'min': 81, 'mode': 101, 'max': 157}, # Lukas Mauler 2021
        #         },
        #         'h2_tank': { # Maybe ought be different for 700 bar and 350 bar.
        #             'dist': 'interp',
        #             '2025': {'dist': 'uniform', 'min': 625, 'max': 686}, # 2023 US DoE per 100k/yr production
        #             '2030': {'dist': 'uniform', 'min': 498, 'max': 625}, # US DOE 2030 Target
        #             '2050': {'dist': 'uniform', 'min': 442, 'max': 498}, # US DOE Ultimate Target
        #         },
        #         'motor': 40, # $/kW,
        #         'fc': {
        #             'dist': 'interp',
        #             '2025': {'dist': 'uniform', 'min': 357, 'max': 500}, # 2025 US DoE per 1k/yr production (264 at 100k/yr)
        #             '2030': {'dist': 'uniform', 'min': 146, 'max': 285}, # US DOE 2030 Target at 1k-10k
        #             '2040': {'dist': 'uniform', 'min': 100, 'max': 201}, # US DOE 2030 Target at 10k-100k
        #             '2050': {'dist': 'uniform', 'min': 79, 'max': 146}, # US DOE Ultimate Target 100k lower to 2030 Target 10k higher
        #         },
        #         'HICE_engine': {
        #             'dist': 'interp',
        #             '2025': {'dist': 'const', 'val': 43_750},
        #             '2050': {'dist': 'const', 'val': 35_000},
        #         },
        #         'charger_50kW': {
        #             'dist': 'interp',
        #             '2025': {'dist': 'const', 'val': 60_000},
        #             '2030': {'dist': 'uniform', 'min': 50_000, 'max': 54_000},
        #             '2040': {'dist': 'uniform', 'min': 44_000, 'max': 49_000},
        #             '2050': {'dist': 'uniform', 'min': 30_000, 'max': 37_000},
        #         },
        #     }
        # },
        # 'Class-8 Straight': {
            # 'Shared': {
            #     'activity_proportion': 0.06,
            #     'average_payload': 4_000,
            #     'target_distance': (49_003-4_104) / (1 + np.exp(0.372*(np.arange(MAX_AGE) - 7.62))) + 4_104,
            #     'revenue_per_tkm': 0.60, # Uncertainty?
            #     'trailers_per_truck': 0,
            #     'gvwl': 24_250, # kg
            #     'drive_cycle': [
            #         'short_haul' for y in range(MAX_AGE)
            #     ],
            #     'survival_rate': np.linspace(1, 0, MAX_AGE+1)[:-1],
            #     'frontal_area': 9.2, # m^2
            #     'roll_coefficient': 0.0054,
            #     'frame_mass': 8_000, # kg
            #     'trailer_mass': 0, # kg
            #     'mass_correction': 0.2,
            #     'transmission_efficiency': 0.95,
            #     'embodied': 2, # Make change over time
            #     'driver_cost': 0.50, # $/km (should change to hourly)
            #     'pollution_cost': { # Urban CE Delft B.C. values per tonne.
            #         'NOX': 44_051,
            #         'NMVC': 2_482,
            #         'SO2': 25_542,
            #         'PM': 254_383,
            #         'PMNE': 46_120,
            #     },
            #     'drag_coefficient': {
            #         'dist': 'interp',
            #         '2000': {'dist': 'const', 'val': 0.75},
            #         '2025': {'dist': 'const', 'val': 0.65},
            #         '2050': {'dist': 'uniform', 'min': 0.50, 'max': 0.60},
            #     },
            # },
            # 'Powertrains': {
                # 'D': {
                #     'fuel': {'Diesel': np.ones(MAX_AGE)},
                #     'params': {
                #         'drivetrain_mass': 1857,
                #         'ess_specific_mass': {
                #             'Diesel': 0,
                #         },
                #         'efficiency': {
                #             'Diesel': { # 5% cut
                #                 'dist': 'interp',
                #                 '2000': {'dist': 'const', 'val': 0.35}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
                #                 '2010': {'dist': 'const', 'val': 0.38}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf?utm_source=chatgpt.com
                #                 '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
                #                 '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
                #                 '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
                #             },
                #         },
                #         'regen_efficiency': 0,
                #         'accessory_load': 4_250,
                #         'fuel_capacity': {
                #             'Diesel': 250
                #         },
                #         'ess_embodied': {
                #             'Diesel': 0,
                #         },
                #         'usable_capacity': {
                #             'Diesel': 0.95
                #         },
                #         'refuel_rate': {
                #             'Diesel': 6000
                #         },
                #         'running_cost': 0.17,
                #         'market_limit': {
                #             'init': 1.0,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
                #         }
                #     },
                # },
                # 'DHNP': { # Need to add battery for weight and emissions.
                #     'fuel': {
                #         'Diesel': np.ones(MAX_AGE) * 0.999,
                #         'Slow Charge': np.ones(MAX_AGE) * 0.001,
                #     },
                #     'params': {
                #         'drivetrain_mass': 1_974,
                #         'motor_size': 220,
                #         'ess_specific_mass': {
                #             'Diesel': 0,
                #             'Slow Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 6},
                #                 '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
                #                 '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
                #             }
                #         },
                #         'ess_embodied': {
                #             'Diesel': 0,
                #             'Slow Charge': { # (Xu et al., 2022)
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 87},
                #                 'end': {'dist': 'uniform', 'min': 20, 'max': 40},
                #             }
                #         },
                #         'efficiency': {
                #             'Diesel': { # 5% cut
                #                 'dist': 'interp',
                #                 '2000': {'dist': 'const', 'val': 0.35}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
                #                 '2010': {'dist': 'const', 'val': 0.38}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf
                #                 '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
                #                 '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
                #                 '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
                #             },
                #             'Slow Charge': 0.875,
                #         },
                #         'regen_efficiency': 0.71,
                #         'accessory_load': 3_400,
                #         'fuel_capacity': {
                #             'Diesel': 500,
                #             'Slow Charge': 5
                #         },
                #         'battery_size': 5,
                #         'usable_capacity': {
                #             'Diesel': 0.95,
                #             'Slow Charge': 0.95
                #         },
                #         'refuel_rate': {
                #             'Diesel': 6000,
                #             'Slow Charge': { # Irrelevant
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 250},
                #                 'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
                #             }
                #         },
                #         'running_cost': 0.17,
                #         'market_limit': {
                #             'init': 0.02,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
                #         }
                #     },
                # },
                # 'DHP': {
                #     'fuel': {
                #         'Diesel': np.ones(MAX_AGE) * 0.70,
                #         'Slow Charge': np.ones(MAX_AGE) * 0.30,
                #     },
                #     'params': {
                #         'drivetrain_mass': 1_974, # wrong cuz motor smaller
                #         'motor_size': 220,
                #         'ess_specific_mass': {
                #             'Diesel': 0,
                #             'Slow Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 6},
                #                 '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
                #                 '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
                #             }
                #         },
                #         'ess_embodied': {
                #             'Diesel': 0,
                #             'Slow Charge': { # (Xu et al., 2022)
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 87},
                #                 'end': {'dist': 'uniform', 'min': 20, 'max': 40},
                #             }
                #         },
                #         'efficiency': {
                #             'Diesel': { # 5% cut
                #                 'dist': 'interp',
                #                 '2000': {'dist': 'const', 'val': 0.35}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
                #                 '2010': {'dist': 'const', 'val': 0.38}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf
                #                 '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
                #                 '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
                #                 '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
                #             },
                #             'Slow Charge': 0.875,
                #         },
                #         'regen_efficiency': 0.71,
                #         'accessory_load': 3_400,
                #         'fuel_capacity': {
                #             'Diesel': 500,
                #             'Slow Charge': 100
                #         },
                #         'battery_size': 100, # This shouldn't be needed anymore.
                #         'usable_capacity': {
                #             'Diesel': 0.95,
                #             'Slow Charge': 0.95
                #         },
                #         'refuel_rate': {
                #             'Diesel': 6000,
                #             'Slow Charge': {
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 250},
                #                 'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
                #             }
                #         },
                #         'running_cost': 0.17,
                #         'market_limit': {
                #             'init': 0.02,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
                #         }
                #     },
                # },
                # 'BE': {
                #     'fuel': {'Slow Charge': np.ones(MAX_AGE)},
                #     'params': { # battery replacement costs and frequency
                #         'drivetrain_mass': 1_115,
                #         'motor_size': 880,
                #         'ess_specific_mass': {
                #             'Slow Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 6},
                #                 '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
                #                 '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
                #             }
                #         },
                #         'ess_embodied': {
                #             'Slow Charge': { # (Xu et al., 2022)
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 87},
                #                 'end': {'dist': 'uniform', 'min': 20, 'max': 40},
                #             }
                #         },
                #         'efficiency': {
                #             'Slow Charge': 0.875,
                #         },
                #         'regen_efficiency': 0.71,
                #         'accessory_load': 6_900, # weather impacts
                #         'fuel_capacity': {'Slow Charge': 500},
                #         'battery_size': 500,
                #         'usable_capacity': {
                #             'Slow Charge': {
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 0.8}, # US DOE Baseline
                #                 '2030': {'dist': 'uniform', 'min': 0.85, 'max': 0.90}, # US DOE 2030 Target
                #                 '2050': {'dist': 'uniform', 'min': 0.9, 'max': 0.95}, # US DOE Ultimate Target
                #             },
                #         },
                #         'refuel_rate': { # For en route only (this is why it is not slow).
                #             'Slow Charge': {
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 250},
                #                 'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
                #             }
                #         },
                #         'running_cost': 0.12,
                #         'market_limit': {
                #             'init': 0.02,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.35, 'max': 0.49},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.25, 'max': 0.29},
                #         }
                #     },
                # },
                # 'FC': {
                #     'fuel': {'Hydrogen': np.ones(MAX_AGE)},
                #     'params': {
                #         'drivetrain_mass': 1_999,
                #         'motor_size': 880,
                #         'fc_size': 880,
                #         'ess_specific_mass': {
                #             'Hydrogen': {
                #                 'dist': 'interp',
                #                 '2025': 16.0,
                #                 '2050': 13.0,
                #             },
                #             'Slow Charge': { # (Haghbin et al., 2025; Jose et al., 2025)
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 6},
                #                 '2035': {'dist': 'uniform', 'min': 3.34, 'max': 5.57}, # Solid state
                #                 '2050': {'dist': 'triangle', 'min': 1.74, 'mode': 3.34, 'max': 5.57}, # Possible advanced chemistries
                #             }
                #         },
                #         'ess_embodied': {
                #             'Diesel': 0,
                #             'Hydrogen': {
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 35.2},
                #                 'end': {'dist': 'triangle', 'min': 11.7, 'mode': 22.1, 'max': 28.6},
                #             },
                #             'Slow Charge': { # (Xu et al., 2022)
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 87},
                #                 'end': {'dist': 'uniform', 'min': 20, 'max': 40},
                #             }
                #         },
                #         'efficiency': { # US DoE Targets
                #             'Hydrogen': {
                #                 'dist': 'interp',
                #                 '2025': {'dist': 'const', 'val': 0.59}, # US DOE Baseline
                #                 '2030': {'dist': 'uniform', 'min': 0.59, 'max': 0.63}, # US DOE 2030 Target
                #                 '2050': {'dist': 'uniform', 'min': 0.63, 'max': 0.66}, # US DOE Ultimate Target
                #             },
                #             'Slow Charge': 0.875,
                #         },
                #         'regen_efficiency': 0.71,
                #         'fc_lifetime': { # hours, US DoE Targets
                #             'dist': 'interp',
                #             '2025': {'dist': 'const', 'val': 20_000}, # Ballard tech doc, The FCmove-XD
                #             '2030': {'dist': 'triangle', 'min': 22_500, 'mode': 25_000, 'max': 27_500}, # US DOE 2030 Target
                #             '2050': {'dist': 'triangle', 'min': 27_500, 'mode': 30_000, 'max': 32_500}, # US DOE Ultimate Target
                #         },
                #         'accessory_load': 3_400,
                #         'fuel_capacity': {
                #             'Hydrogen': 50,
                #             'Slow Charge': 5,
                #         },
                #         'battery_size': 5, # small?
                #         'usable_capacity': {
                #             'Hydrogen': 0.9,
                #             'Slow Charge': 0.95
                #         },
                #         'refuel_rate': {'Hydrogen': 480,
                #             'Slow Charge': { # Not relevant.
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 250},
                #                 'end': {'dist': 'triangle', 'min': 250, 'mode': 500, 'max': 750},
                #             },
                #         },
                #         'running_cost': 0.14,
                #         'market_limit': {
                #             'init': 0.02,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.35, 'max': 0.49},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.25, 'max': 0.29},
                #         }
                #     },
                # },
                # 'HICE': { # add hybrid version?
                #     'fuel': {'Hydrogen': np.ones(MAX_AGE)},
                #     'params': {
                #         'drivetrain_mass': 1_857,
                #         'ess_specific_mass': {
                #             'Hydrogen': {
                #                 'dist': 'interp',
                #                 '2025': 16.0,
                #                 '2050': 13.0,
                #             },
                #         },
                #         'ess_embodied': {
                #             'Hydrogen': {
                #                 'dist': 'linear',
                #                 'start': {'dist': 'const', 'val': 35.2},
                #                 'end': {'dist': 'triangle', 'min': 11.7, 'mode': 22.1, 'max': 28.6},
                #             },
                #         },
                #         'efficiency': { # DOE targets
                #             'Hydrogen': { # 5% cut
                #                 'dist': 'interp',
                #                 '2000': {'dist': 'const', 'val': 0.35}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
                #                 '2010': {'dist': 'const', 'val': 0.38}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf
                #                 '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
                #                 '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
                #                 '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
                #             },
                #         },
                #         'regen_efficiency': 0,
                #         'accessory_load': 3_400,
                #         'fuel_capacity': {'Hydrogen': 50},
                #         'usable_capacity': {'Hydrogen': 0.9},
                #         'refuel_rate': {'Hydrogen': 480},
                #         'running_cost': 0.17,
                #         'market_limit': {
                #             'init': 0.02,
                #             'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
                #             'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
                #         }
                #     },
                # },
                # 'DHICE': { # add hybrid version?
                    # 'fuel': {'Diesel': np.ones(MAX_AGE)*0.75, 'Hydrogen': np.ones(MAX_AGE)*0.25},
                    # 'params': {
                    #     'drivetrain_mass': 1_817,
                    #     'ess_specific_mass': {
                    #         'Diesel': 0,
                    #         'Hydrogen': {
                    #             'dist': 'interp',
                    #             '2025': 16.0,
                    #             '2050': 13.0,
                    #         },
                    #     },
                    #     'ess_embodied': {
                    #         'Diesel': 0,
                    #         'Hydrogen': {
                    #             'dist': 'linear',
                    #             'start': {'dist': 'const', 'val': 35.2},
                    #             'end': {'dist': 'triangle', 'min': 11.7, 'mode': 22.1, 'max': 28.6},
                    #         },
                    #     },
                    #     'efficiency': { # US DoE Targets for Diesel
                    #         'Diesel': { # 5% cut
                    #             'dist': 'interp',
                    #             '2000': {'dist': 'const', 'val': 0.35}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
                    #             '2010': {'dist': 'const', 'val': 0.38}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf
                    #             '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
                    #             '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
                    #             '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
                    #         },
                    #         'Hydrogen': { # 5% cut
                    #             'dist': 'interp',
                    #             '2000': {'dist': 'const', 'val': 0.35}, # Peak 42 https://www.nationalacademies.org/read/13288/chapter/5
                    #             '2010': {'dist': 'const', 'val': 0.38}, # Peak 44.8 https://theicct.org/sites/default/files/publications/ICCT_EU-HDV-tech-2025-30_20180116.pdf
                    #             '2025': {'dist': 'const', 'val': 0.41}, # US DOE Baseline
                    #             '2030': {'dist': 'uniform', 'min': 0.44, 'max': 0.46}, # US DOE 2030 Target
                    #             '2050': {'dist': 'uniform', 'min': 0.46, 'max': 0.49}, # US DOE Ultimate Target
                    #         },
                    #     },
                    #     'regen_efficiency': 0,
                    #     'accessory_load': 4_250,
                    #     'fuel_capacity': {
                    #         'Diesel': 500,
                    #         'Hydrogen': 20
                    #     },
                    #     'usable_capacity': {
                    #         'Diesel': 0.95,
                    #         'Hydrogen': 0.9
                    #     },
                    #     'refuel_rate': {
                    #         'Diesel': 6000,
                    #         'Hydrogen': 480
                    #     },
                    #     'running_cost': 0.14,
                    #     'market_limit': {
                    #         'init': 0.02,
                    #         'CAGR_early': {'dist': 'uniform', 'min': 0.44, 'max': 0.54},
                    #         'CAGR_other': {'dist': 'uniform', 'min': 0.34, 'max': 0.38},
                    #     }
                    # },
            #     },
            # },
            # 'Costs': { # 1.573 USD 2020 (Lajevardi 2019) -> CAD 2024, don't bother with trailers (battery and FC should progress with time) break down in capital plot
            #     'base': 105_000,
            #     'diesel_engine': 35_000,
            #     'combustion_transmission': 13_700,
            #     'electric_transmission': 3_200,
            #     'after_treatment': 12_000,
            #     'tank': 9.44, # $/L
            #     'battery': {
            #         'dist': 'linear',
            #         'start': {'dist': 'const', 'val': 159}, # USD $115/kWh Bloomberg * 1.37
            #         'end': {'dist': 'triangle', 'min': 81, 'mode': 101, 'max': 157}, # Lukas Mauler 2021
            #     },
            #     'h2_tank': { # Maybe ought be different for 700 bar and 350 bar.
            #         'dist': 'interp',
            #         '2025': {'dist': 'uniform', 'min': 625, 'max': 686}, # 2023 US DoE per 100k/yr production
            #         '2030': {'dist': 'uniform', 'min': 498, 'max': 625}, # US DOE 2030 Target
            #         '2050': {'dist': 'uniform', 'min': 442, 'max': 498}, # US DOE Ultimate Target
            #     },
            #     'motor': 40, # $/kW,
            #     'fc': {
            #         'dist': 'interp',
            #         '2025': {'dist': 'uniform', 'min': 357, 'max': 500}, # 2025 US DoE per 1k/yr production (264 at 100k/yr)
            #         '2030': {'dist': 'uniform', 'min': 146, 'max': 285}, # US DOE 2030 Target at 1k-10k
            #         '2040': {'dist': 'uniform', 'min': 100, 'max': 201}, # US DOE 2030 Target at 10k-100k
            #         '2050': {'dist': 'uniform', 'min': 79, 'max': 146}, # US DOE Ultimate Target 100k lower to 2030 Target 10k higher
            #     },
            #     'HICE_engine': {
            #         'dist': 'interp',
            #         '2025': {'dist': 'const', 'val': 43_750},
            #         '2050': {'dist': 'const', 'val': 35_000},
            #     },
            #     'charger_50kW': {
            #         'dist': 'interp',
            #         '2025': {'dist': 'const', 'val': 60_000},
            #         '2030': {'dist': 'uniform', 'min': 50_000, 'max': 54_000},
            #         '2040': {'dist': 'uniform', 'min': 44_000, 'max': 49_000},
            #         '2050': {'dist': 'uniform', 'min': 30_000, 'max': 37_000},
            #     },
            # } 
        # },
    },
}



