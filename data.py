""" Parameters for the HDT adoption model.
To do:
 - Make certain parameters shared.
 - Efficiency boost for long haul.
 - Add diesel tank mass.
 - Same seed for all ICE engines
 - Regen capacity
"""
import numpy as np

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



PARAMS = {
    'drive_cycles': {
        'long_haul': {
            'path': 'param_estimation/energy_consumption/Fleet DNA Long-Haul Representative_.csv',
            'average_speed': 88.12 # km/hr (average moving speed)
        },
        'regional_haul': {
           'path': 'param_estimation/energy_consumption/Fleet DNA Regional-Haul Representative_.csv',
            'average_speed': 63.94 # km/hr (average moving speed)
        },
        'short_haul': {
            'path': 'param_estimation/energy_consumption/Fleet DNA Local Delivery Representative_.csv',
            'average_speed': 61.15 # km/hr (average moving speed)
        }
    },
    "fuels": { # Recharge efficiency?
        "diesel": {
            "units": "L",
            "lhv": 38.6e6,
            "emissions_intensity": { # GHGenius 502c
                'supply': 0.88,
                'use': 2.52,
            },
            "refuel_efficiency": 1.0,
            "cost": { # CER end-use + 5% GST.
                'dist': 'interp',
                '2025': {'dist': 'uniform', 'min': 1.67, 'max': 1.73},
                '2030': {'dist': 'uniform', 'min': 1.46, 'max': 1.81},
                '2035': {'dist': 'uniform', 'min': 1.65, 'max': 1.91},
                '2040': {'dist': 'uniform', 'min': 1.74, 'max': 2.08},
                '2045': {'dist': 'uniform', 'min': 1.71, 'max': 2.18},
                '2050': {'dist': 'uniform', 'min': 1.67, 'max': 2.15},
            },
            "water_intensity": 2.95, # GREET1
            "electricity_intensity": 0.1, # GREET1
        },
        "h2": { # (air pollution factors, cost?)
            "units": "kg",
            "lhv": 120e6,
            "emissions_intensity": {
                'supply': 0.57, # Electricity --> H2 (57.5 kWh * 0.0099 kgCO2/kWh)
                'use': 0,
            },
            "refuel_efficiency": 1.0,
            "cost": { # Electrolysis
                'dist': 'interp',
                '2025': {'dist': 'const', 'val': 14.97}, # 5t/d + delivery
                '2030': {'dist': 'uniform', 'min': 8.40, 'max': 11.13}, # 5 t/d on-site; 50 t/d + 3.6 delivery
                '2035': {'dist': 'uniform', 'min': 6.55, 'max': 10.56}, # 50 t/d + 1.6 delivery; 5 t/d on-site
                '2040': {'dist': 'uniform', 'min': 5.25, 'max': 6.34}, # 300 t/d + delivery; 50 t/d + delivery
                '2045': {'dist': 'uniform', 'min': 5.20, 'max': 6.13}, # 300 t/d + delivery; 50 t/d + delivery
                '2050': {'dist': 'uniform', 'min': 5.15, 'max': 5.94}, # 300 t/d + delivery; 50 t/d + delivery
            },
            "water_intensity": {'dist': 'uniform', 'min': 95.6, 'max': 121.4}, # IRENA water for electrolysis + water for electricity
            "electricity_intensity": 55,
        },
        "h2_p": { # Production 5t/d: $5.65, 50 t/d: $3.39, 300 t/d: $2.71 (2030 CAD so * 0.89) CICE (used for scale)
            "units": "kg", # Compression $0.8/kg and conditioning 0.1 # Delivery (3.2 - 1.6)
            "lhv": 120e6, # 2% annual production cost reduction
            "emissions_intensity": { # UVic Paper, cost and emissions intensity of hydrogen from thermal pyrolysis of natural gas in BC,
                'supply': 4.14,
                'use': 0, # NG supply leakage 0.2-0.42%, elec 0.009-0.015, burned NG to CO2 (*44/16) (NG emission factor 28-32)
            },
            "refuel_efficiency": 1.0,
            "cost": { # CICE
                'dist': 'interp',
                '2025': {'dist': 'const', 'val': 11.05}, # 5t/d + delivery
                '2030': {'dist': 'uniform', 'min': 7.24, 'max': 7.38}, # 5 t/d on-site; 50 t/d + 3.6 delivery
                '2035': {'dist': 'uniform', 'min': 5.41, 'max': 6.96}, # 50 t/d + 1.6 delivery; 5 t/d on-site
                '2040': {'dist': 'uniform', 'min': 4.18, 'max': 5.21}, # 300 t/d + delivery; 50 t/d + delivery
                '2045': {'dist': 'uniform', 'min': 2.91, 'max': 5.02}, # 300 t/d + delivery with carbon black; 50 t/d + delivery
                '2050': {'dist': 'uniform', 'min': 2.80, 'max': 4.85}, # 300 t/d + delivery with carbon black; 50 t/d + delivery
            },
            "water_intensity": {'dist': 'uniform', 'min': 4.71, 'max': 5.66}, # 1UVic
            "electricity_intensity": 2.12, # UVic
        },
        "h2_pe": { # Production 1.85 (compression same) * regular pyrolysis
            "units": "kg",
            "lhv": 120e6,
            "emissions_intensity": { # Could potentially change over time
                'supply': 1.92, # UVic + methane supply chain (Seymour et al., 2024)
                'use': 0,
            },
            "refuel_efficiency": 1.0,
            "cost": {
                'dist': 'interp',
                '2025': {'dist': 'const', 'val': 15.85}, # 5t/d + delivery
                '2030': {'dist': 'uniform', 'min': 9.90, 'max': 11.81}, # 5 t/d on-site; 50 t/d + 3.6 delivery
                '2035': {'dist': 'uniform', 'min': 7.86, 'max': 11.05}, # 50 t/d + 1.6 delivery; 5 t/d on-site
                '2040': {'dist': 'uniform', 'min': 5.60, 'max': 7.48}, # 300 t/d + delivery; 50 t/d + delivery
                '2045': {'dist': 'uniform', 'min': 4.75, 'max': 7.12}, # 300 t/d + delivery with carbon black; 50 t/d + delivery
                '2050': {'dist': 'uniform', 'min': 4.55, 'max': 6.81}, # 300 t/d + delivery with carbon black; 50 t/d + delivery
            },
            "water_intensity": {'dist': 'uniform', 'min': 16.3, 'max': 20.9}, # Extra 7.5-10 cooling
            "electricity_intensity": 10.23,
        },
        "fast_charge": { # charging efficiency
            "units": "kWh",
            "lhv": 3.6e6,
            "emissions_intensity": {
                'supply': 0.0099,
                'use': 0,
            },
            "refuel_efficiency": 0.86,
            "cost": { # BC Hydro fleet rate.
                'dist': 'interp',
                '2025': {'dist': 'uniform', 'min': 0.360, 'max': 0.361},
                '2030': {'dist': 'uniform', 'min': 0.368, 'max': 0.372},
                '2035': {'dist': 'uniform', 'min': 0.374, 'max': 0.385},
                '2040': {'dist': 'uniform', 'min': 0.382, 'max': 0.397},
                '2045': {'dist': 'uniform', 'min': 0.401, 'max': 0.407},
                '2050': {'dist': 'uniform', 'min': 0.412, 'max': 0.418},
            },
            "water_intensity": {'dist': 'uniform', 'min': 1.43, 'max': 1.88},
            "electricity_intensity": 1.0,
        },
        "slow_charge": {
            "units": "kWh",
            "lhv": 3.6e6,
            "emissions_intensity": {
                'supply': 0.0099,
                'use': 0,
            },
            "refuel_efficiency": 0.95,
            "cost": { # BC Hydro fleet rate.
                'dist': 'interp',
                '2025': {'dist': 'uniform', 'min': 0.102, 'max': 0.103},
                '2030': {'dist': 'uniform', 'min': 0.105, 'max': 0.106},
                '2035': {'dist': 'uniform', 'min': 0.106, 'max': 0.110},
                '2040': {'dist': 'uniform', 'min': 0.109, 'max': 0.113},
                '2045': {'dist': 'uniform', 'min': 0.114, 'max': 0.116},
                '2050': {'dist': 'uniform', 'min': 0.117, 'max': 0.119},
            },
            "water_intensity": {'dist': 'uniform', 'min': 1.43, 'max': 1.88},
            "electricity_intensity": 1.0,
        }
    },
    'autonomous_t50': 2040,
    'vehicles': { # Need to add straight trucks and hybrids.
        'components': {
            'converter': {
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
                'motor': 0.875,
                'fc': {
                    'dist': 'interp',
                    '2025': {'dist': 'const', 'val': 0.59}, # US DOE Baseline
                    '2030': {'dist': 'uniform', 'min': 0.59, 'max': 0.63}, # US DOE 2030 Target
                    '2050': {'dist': 'uniform', 'min': 0.63, 'max': 0.66}, # US DOE Ultimate Target
                },
            },
            'ess': {
                'diesel_tank': {
                    'mass_per_unit': 0,
                    'embodied_per_unit': 0,
                    'usable_capacity': 0.95,
                    'refuel_rate': 6000,
                    'efficiency': 1.0,
                    'efficiency_deg': 0.0,
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
                },
            },
            'transmission': {
                'combustion_transmission': {
                    'efficiency': 0.95,
                    'mass': 0,
                }
            },
        },
        'types': {
            'sleeper': {
                'shared': {
                    'activity_proportion': 0.77, # maybe wrong since Sleepers don't exclusively do long-haul?
                    'default_payload': 16_000,
                    'default_unloaded_mass': 12_938,
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
                    'embodied': {
                        'dist': 'interp',
                        '2025': {'dist': 'const', 'val': 2.2}, # World Steel
                        '2050': {'dist': 'triangle', 'min': 0.9, 'mode': 1.7, 'max': 2.2}, # IEA SDS, IEA STEPS, IEA 40% from material efficiency.
                    },
                    'driver_cost': 0.38, # $/km
                    'drag_coefficient': { # US DoE Targets
                        'dist': 'interp',
                        '2000': {'dist': 'const', 'val': 0.7}, # Old standard
                        '2010': {'dist': 'const', 'val': 0.6},
                        '2025': {'dist': 'const', 'val': 0.49}, # US DOE Baseline
                        '2030': {'dist': 'uniform', 'min': 0.34, 'max': 0.43}, # US DOE 2030 Target
                        '2050': {'dist': 'uniform', 'min': 0.30, 'max': 0.41}, # US DOE Ultimate Target
                    },
                    'accessory_demand': 4_250 * 0.46,
                    'p_weighed_out': 0.3,
                },
                'powertrains': {
                    'dice': {
                        'components':{
                            'diesel_tank': {
                                'type': 'ess',
                                'capacity': 500
                            },
                            'ice': {
                                'type': 'converter',
                            },
                            'combustion_transmission': {
                                'type': 'transmission',
                            },
                        },
                        'fuels': ['diesel'],
                        'energy_pathways': {
                            ('diesel_tank', 'ice', 'combustion_transmission'): {
                                'energy_proportion': 1.0,
                                'fuel': 'diesel',
                            },
                        },
                        'regen_efficiency': 0,
                        'running_cost': 0.17,
                        'init_market_limit': 1.0,
                    },
                },
            },
        }, # Deal with costs
        'costs': { # 1.573 USD 2020 (Lajevardi 2019) -> CAD 2024, don't bother with trailers (battery and FC should progress with time) break down in capital plot
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
}



