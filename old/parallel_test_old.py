"""
To do:
 - Trailer emissions?
"""
import numpy as np
from multiprocessing import Pool
import pickle
from model_old import *
import copy

# Policy packages
params = d.PARAMS
policy_packages = {
    # 'Base': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Pyrolysis': False,
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Carbon Tax': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 95, '2030': 170, '2050': 170}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Pyrolysis': False,
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'LCFS': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=300, start=0.183, end=0.76),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Pyrolysis': False,
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'ZEV Mandate': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    #         penalty=30_000,
    #         rebates=False,
    #     ),
    #     'Pyrolysis': False,
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'ZEV Mandate (Strong)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    #         penalty=100_000,
    #         rebates=False,
    #     ),
    #     'Pyrolysis': False,
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Accelerated Retirement': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': True,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    'Foresight': {
        'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
        'LCFS': LCFS(params, credit_price=0, start=0.0, end=0.0),
        'Autonomous Permits': AutonomousPermits({
            'D': 0,
            'DHNP': 0,
            'DHP': 0,
            'BE': 0,
            'FC': 0,
            'HICE': 0,
            'DHICE': 0,
        }),
        'ZEV Mandate': ZEVMandate(
            params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
            penalty=0,
            rebates=False,
        ),
        'Accelerated Retirement': False,
        'Foresight': True,
        'ZEV GVWL Increase': False,
        'Break Mandate': 0.0,
        'Pyrolysis': False,
        'Electrified Pyrolysis': False,
        'ZEV Rebate': 0.0,
    },
    # 'ZEV GVWL Increase': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': True,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Break Mandate': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.75,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'ZEV Rebate': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.33,
    # },
    # 'Policy Package': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=300, start=0.183, end=0.76),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    #         penalty=30_000,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': True,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Policy Package (no LCFS)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    #         penalty=30_000,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': True,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Policy Package (No ZEVM)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=300, start=0.183, end=0.76),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.0, '2040': 0.0, '2050': 0.0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': True,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Policy Package (no ZEV GVWL Increase)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=300, start=0.183, end=0.76),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    #         penalty=30_000,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Autonomous Permits (Base)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 1,
    #         'FC': 1,
    #         'HICE': 1,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Pyrolysis': False,
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Autonomous permits (PP)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=300, start=0.183, end=0.76),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 1,
    #         'FC': 1,
    #         'HICE': 1,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    #         penalty=30_000,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': True,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Base (P)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': True,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Base (EP)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': True,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Foresight (PP)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=300, start=0.183, end=0.76),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    #         penalty=30_000,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': True,
    #     'ZEV GVWL Increase': True,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Foresight (P)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': True,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': True,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Foresight (EP)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=0, start=0.0, end=0.0),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0, '2040': 0, '2050': 0},
    #         penalty=0,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': True,
    #     'ZEV GVWL Increase': False,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': True,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Policy Package (P)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=300, start=0.183, end=0.76),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    #         penalty=30_000,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': True,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': True,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Policy Package (EP)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=300, start=0.183, end=0.76),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    #         penalty=30_000,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': False,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': True,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': True,
    #     'ZEV Rebate': 0.0,
    # },
    # 'Accelerated Retirement (PP)': {
    #     'Carbon Tax': CarbonTax(years=params['Years']['Y'], price={'2025': 0, '2030': 0, '2050': 0}),
    #     'LCFS': LFCS(params, credit_price=300, start=0.183, end=0.76),
    #     'Autonomous Permits': AutonomousPermits({
    #         'D': 0,
    #         'DHNP': 0,
    #         'DHP': 0,
    #         'BE': 0,
    #         'FC': 0,
    #         'HICE': 0,
    #         'DHICE': 0,
    #     }),
    #     'ZEV Mandate': ZEVMandate(
    #         params['Years']['T'], targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
    #         penalty=30_000,
    #         rebates=False,
    #     ),
    #     'Accelerated Retirement': True,
    #     'Foresight': False,
    #     'ZEV GVWL Increase': True,
    #     'Break Mandate': 0.0,
    #     'Pyrolysis': False,
    #     'Electrified Pyrolysis': False,
    #     'ZEV Rebate': 0.0,
    # },
}

def merge_outputs(items):
    """items is a list of same-shaped dicts or terminal arrays."""
    # If the items are dicts, recurse
    if isinstance(items[0], dict):
        return {
            k: merge_outputs([item[k] for item in items])
            for k in items[0].keys()
        }
    else:
        # Terminal: list of arrays → 2D numpy array
        return np.stack(items)
    
def add_totals(d, skip_keys=[]):
    if not isinstance(d, dict):
        return d  # leaf array, nothing to do

    # First recurse into children
    for k, v in d.items():
        if k not in skip_keys:
            d[k] = add_totals(v)

    # Now check this level for adding 'Total'
    if "Total" not in d:
        # collect numeric leaf arrays
        arrays = []
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                arrays.append(v)

        # Only add Total if there are at least 1–2 arrays
        if arrays:
            d["Total"] = sum(arrays)

    return d

def init_worker(
    _params,
    _drive_cycles,
    _input_distributions,
    _samples,
    _ctax,
    _lcfs,
    _autonomous_permits,
    _zev_mandate,
    _pyrolysis,
    _pyrolysis_elec,
    _accelerated_retirement,
    _foresight,
    _gvwl_increase,
    _break_mandate,
    _zev_rebate
):

    global params, drive_cycles, inputs_distributions, samples
    global ctax, lcfs, autonomous_permits, zev_mandate
    global pyrolysis, pyrolysis_elec, accelerated_retirement, foresight
    global gvwl_increase, break_mandate, zev_rebate

    (
        params,
        drive_cycles,
        inputs_distributions,
        samples,
        ctax,
        lcfs,
        autonomous_permits,
        zev_mandate,
        pyrolysis,
        pyrolysis_elec,
        accelerated_retirement,
        foresight,
        gvwl_increase,
        break_mandate,
        zev_rebate
    ) = (
        _params,
        _drive_cycles,
        _input_distributions,
        _samples,
        _ctax,
        _lcfs,
        _autonomous_permits,
        _zev_mandate,
        _pyrolysis,
        _pyrolysis_elec,
        _accelerated_retirement,
        _foresight,
        _gvwl_increase,
        _break_mandate,
        _zev_rebate
    )

def one_run(iRun):
    uncertain_cps = dict(zip(inputs_distributions.keys(), samples[iRun]))
    fleet = Fleet(
        params=params,
        uncertain_cps=uncertain_cps,
        drive_cycles=drive_cycles,
        ctax=ctax,
        lcfs=lcfs,
        autonomous_permits=autonomous_permits,
        zev_mandate=zev_mandate,
        pyrolysis=pyrolysis,
        pyrolysis_elec=pyrolysis_elec,
        accelerated_retirement=accelerated_retirement,
        foresight=foresight,
        gvwl_increase=gvwl_increase,
        break_mandate=break_mandate,
        zev_rebate=zev_rebate
        )
    output = {
        'Emissions': {
            k: {
                'Fuel Combustion': fleet.emissions[k].fuel_combustion / 1e9,
                'Fuel Supply': fleet.emissions[k].fuel_supply / 1e9,
                'Embodied': fleet.emissions[k].embodied / 1e9,
            } for k in fleet.K
        },
        'Cost': {
            k: {
                'Capital': fleet.system_costs[k].capital,
                'Fuel': fleet.system_costs[k].fuel,
                'Driver': fleet.system_costs[k].driver,
                'Operational': fleet.system_costs[k].operational,
            } for k in fleet.K
        },
        'Policy cost': {
            k: {
                'Carbon tax': fleet.policy_costs[k]['carbon_tax'],
                'LCFS': fleet.policy_costs[k]['lcfs'],
                'ZEV mandate': fleet.policy_costs[k]['zev_mandate'],
                'ZEV rebate': fleet.policy_costs[k]['zev_rebate'],
            } for k in fleet.K
        },
        'Activity': {
            k: np.array([fleet.activity_requirement[k, t] for t in fleet.T]) for k in fleet.K
        },
        'Stock': {
            k: {
                p: np.array([fleet.total_stock[k,p,t] for t in fleet.T])/1000 for p in fleet.P[k]
            } for k in fleet.K
        },
        'Sales': {
            k: {
                p: np.array([fleet.sales[k,p,t] for t in fleet.T])/1000 for p in fleet.P[k]
            } for k in fleet.K
        },
        'Fuel Energy': { # Useful energy
            k: {
                f: np.array([fleet.energy_by_fuel[k,f,t] for t in fleet.T]) for f in fleet.fuels.keys()
            } for k in fleet.K
        },
        'Fuel Usage': {
            k: {
                f: np.array([fleet.fuel_usage[k,f,t] for t in fleet.T]) for f in fleet.fuels.keys()
            } for k in fleet.K
        },
        'Water Usage': {
            k: {
                f: np.array([fleet.fuel_usage[k,f,t] * fleet.fuels[f].water_usage for t in fleet.T]) for f in fleet.fuels.keys()
            } for k in fleet.K
        },
        'Electricity Usage': {
            k: {
                f: np.array([fleet.fuel_usage[k,f,t] * fleet.fuels[f].electricity_usage for t in fleet.T]) for f in fleet.fuels.keys()
            } for k in fleet.K
        },
        'Capital': {
            k: {
                p: {
                    'Base': np.array([fleet.vehicles[k,p,t].capital.base for t in fleet.T])/1e3,
                    'Engine': np.array([fleet.vehicles[k,p,t].capital.engine for t in fleet.T])/1e3,
                    'Combustion Transmission': np.array([fleet.vehicles[k,p,t].capital.combustion_transmission for t in fleet.T])/1e3,
                    'After Treatment': np.array([fleet.vehicles[k,p,t].capital.after_treatment for t in fleet.T])/1e3,
                    'Tank': np.array([fleet.vehicles[k,p,t].capital.tank for t in fleet.T])/1e3,
                    'Electric Transmission': np.array([fleet.vehicles[k,p,t].capital.electric_transmission for t in fleet.T])/1e3,
                    'Motor': np.array([fleet.vehicles[k,p,t].capital.motor for t in fleet.T])/1e3,
                    'Battery': np.array([fleet.vehicles[k,p,t].capital.battery for t in fleet.T])/1e3,
                    'Fuel Cell': np.array([fleet.vehicles[k,p,t].capital.fc for t in fleet.T])/1e3,
                    'Hygrogen Storage': np.array([fleet.vehicles[k,p,t].capital.h2_tank for t in fleet.T])/1e3,
                    'Charger': np.array([fleet.vehicles[k,p,t].capital.charger for t in fleet.T])/1e3,
                } for p in fleet.P[k]
            } for k in fleet.K
        },
        'TCO': {
            k: {
                p: {
                    'Capital': -np.array([fleet.vehicles[k,p,t].tco.capital for t in fleet.T])/1e6,
                    'Operational': -np.array([fleet.vehicles[k,p,t].tco.operational for t in fleet.T])/1e6,
                    'Fuel': -np.array([fleet.vehicles[k,p,t].tco.fuel for t in fleet.T])/1e6,
                    'Driver': -np.array([fleet.vehicles[k,p,t].tco.driver for t in fleet.T])/1e6,
                    'FC Replacements': -np.array([fleet.vehicles[k,p,t].tco.fc_replacements for t in fleet.T])/1e6,
                    'Carbon Tax': -np.array([fleet.vehicles[k,p,t].tco.carbon_tax for t in fleet.T])/1e6,
                    'LCFS': -np.array([fleet.vehicles[k,p,t].tco.lcfs for t in fleet.T])/1e6,
                    'ZEV Mandate': -np.array([fleet.vehicles[k,p,t].tco.zev_mandate for t in fleet.T])/1e6,
                    'ZEV Rebate': -np.array([fleet.vehicles[k,p,t].tco.zev_rebate for t in fleet.T])/1e6,
                    'Revenue': np.array([fleet.vehicles[k,p,t].total_revenue for t in fleet.T])/1e6,

                } for p in fleet.P[k]
            } for k in fleet.K
        },
        'LCA': {
            k: {
                p: { # Consider the trailer?
                    'Frame': np.array([sum(fleet.vehicles[k,p,t].emissions.embodied.frame) for t in fleet.T])/1e3,
                    'Drivetrain': np.array([sum(fleet.vehicles[k,p,t].emissions.embodied.drivetrain) for t in fleet.T])/1e3,
                    'ESS': np.array([sum(fleet.vehicles[k,p,t].emissions.embodied.ess) for t in fleet.T])/1e3,
                    'Fuel Supply': np.array([sum(fleet.vehicles[k,p,t].emissions.fuel_supply) for t in fleet.T])/1e3,
                    'Fuel Usage': np.array([sum(fleet.vehicles[k,p,t].emissions.fuel_combustion) for t in fleet.T])/1e3,
                } for p in fleet.P[k]
            } for k in fleet.K
        },
        'External cost': {
            'Accident': fleet.average_external.accidents,
            'Air pollution': fleet.average_external.air_pollution,
            'GHG emissions': fleet.average_external.ghg_emissions,
            'Noise': fleet.average_external.noise,
            'Congestion': fleet.average_external.congestion,
            'Habitat loss': fleet.average_external.habitat_loss,
        },
    }
    return output


class Plotting:
    def __init__(self, sample_years=np.array([2025, 2030, 2035, 2040, 2045, 2050])):
        self.T = np.arange(START_YEAR, END_YEAR+1)
        self.sample_years = sample_years

    @staticmethod
    def sum_by_inner(result):
        summed = {}
        for k, subdict in result.items():
            total_arr = None
            for cat, values in subdict.items():
                arr = np.asarray(values)
                if total_arr is None:
                    total_arr = arr.copy()
                else:
                    total_arr += arr
            summed[k] = total_arr
        return summed

    @staticmethod
    def sum_by_outer(result):
        summed = {}
        for k, subdict in result.items():
            for cat, values in subdict.items():
                arr = np.asarray(values)  # turns list into array
                if cat not in summed:
                    summed[cat] = arr.copy()
                else:
                    summed[cat] += arr
        return summed

    def plot_by_inner(self, result, x_label=None, y_label=None, add_total=False):
        fig, ax = plt.subplots(1,1, figsize=(4,3), dpi=300)
        result = self.sum_by_outer(result)
        self.plot_lines(self.T, result, ax, x_label, y_label, add_total)
        axes = np.atleast_1d(ax)
        # axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.legend()
        return fig, axes
    
    def plot_by_outer(self,result,x_label=None,y_label=None, add_total=False):
        fig,ax=plt.subplots(1,1, figsize=(4,3), dpi=300)
        result=self.sum_by_inner(result)
        self.plot_lines(self.T,result,ax,x_label,y_label, add_total)
        ax.legend(bbox_to_anchor=(1.05,1),loc='upper left',borderaxespad=0.)
        return fig,np.atleast_1d(ax)

    def plot_by_both(self, result, x_label=None, y_label=None, add_total=False):
        n = len(result.keys())
        fig, axes = plt.subplots(1, n, figsize=(3.333*n, 3), sharex=True, sharey=True, constrained_layout=True, dpi=300)
        axes = np.atleast_1d(axes)
        all_handles, all_labels = [], []
        global_max = 0
        for i, k in enumerate(result.keys()):
            ax = axes[i]
            local_max = self.plot_lines(self.T, result[k], ax, x_label, y_label, add_total)
            global_max = max(global_max, local_max)
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in all_labels:
                    all_handles.append(h); all_labels.append(l)
            ax.set_title(k)
        for ax in axes: ax.set_ylim(0, global_max*1.1)
        axes[-1].legend(handles=all_handles, labels=all_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        return fig, axes

    @staticmethod
    def plot_lines(t,result,ax,x_label=None,y_label=None,add_total=False):
        local_max=0
        for k,v in result.items():
            p25,p75=np.percentile(v,[5,95],axis=0)
            mean = np.mean(v, axis=0)
            ax.fill_between(t,p25,p75,alpha=0.2)
            ax.plot(t,mean,label=k)
            local_max=max(local_max,p75.max())
        if add_total:
            tot=np.sum(list(result.values()),axis=0)
            p25,p75=np.percentile(tot,[5,95],axis=0)
            mean = np.mean(v, axis=0)
            ax.fill_between(t,p25,p75,alpha=0.1)
            ax.plot(t,mean,label="Total",linewidth=2)
            local_max=max(local_max,p75.max())
        ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        ax.set_ylim([0, ax.get_ylim()[1]])
        return local_max

    def plot_bars(self, ax, tco, width=0.5, gap=0.1):
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        cost_keys = [k for k in next(iter(tco.values())).keys() if k != 'Total']
        colour_map = {k: colours[i % len(colours)] for i,k in enumerate(cost_keys)}

        for p, dic in tco.items():
            offset = list(tco).index(p)*(width+gap) - (width+gap)*(len(tco)-1)/2
            n_years = len(self.sample_years)
            total_pos = np.zeros(n_years)
            total_neg = np.zeros(n_years)

            for k in cost_keys:
                cost = dic[k][:, self.sample_years-START_YEAR].mean(axis=0)
                pos_mask = cost >= 0
                neg_mask = cost < 0

                if pos_mask.any():
                    ax.bar(self.sample_years+offset, cost*pos_mask, bottom=total_pos, width=width, color=colour_map[k])
                    total_pos += cost*pos_mask
                if neg_mask.any():
                    ax.bar(self.sample_years+offset, cost*neg_mask, bottom=total_neg, width=width, color=colour_map[k])
                    total_neg += cost*neg_mask

            # Powertrain labels
            tops = np.maximum(total_pos, 0)
            for i, yval in enumerate(self.sample_years):
                ax.text(yval+offset, tops[i] + 0.02*abs(tops[i]), p, ha='center', va='bottom', fontsize=8, rotation=90)

            # NPV whiskers
            npv = sum(dic[k][:, self.sample_years-START_YEAR] for k in [k for k in dic.keys()])
            p05, p25, p50, p75, p95 = np.percentile(npv, [5,25,50,75,95], axis=0)
            bxp_data = [{'med': p50[i], 'q1': p25[i], 'q3': p75[i],
                        'whislo': p05[i], 'whishi': p95[i], 'fliers': []}
                        for i in range(len(self.sample_years))]
            ax.bxp(bxp_data, positions=self.sample_years+offset, widths=width,
                manage_ticks=False, medianprops=dict(color='black'))

    def plot_all_bars(self, tco_all, width=0.5, gap=0.1, xlabel=None, ylabel=None, title=None):
        Ks = list(tco_all.keys())
        n_k = len(Ks)
        fig, axes = plt.subplots(n_k, 1, figsize=(15,5*n_k), sharey=False)
        axes = np.atleast_1d(axes)

        for ax, k in zip(axes, Ks):
            self.plot_bars(ax, tco_all[k], width=width, gap=gap)
            ax.set_title(k)
            ax.set_xlabel(xlabel)
            if ax is axes[0]:
                ax.set_ylabel(ylabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]*1.3])

        first_tco = next(iter(tco_all.values()))
        first_powertrain = next(iter(first_tco.values()))
        cost_components = [c for c in first_powertrain.keys() if c != 'Total']

        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colour_map = {c: colours[i % len(colours)] for i, c in enumerate(cost_components)}
        handles = [plt.Rectangle((0, 0), 1, 1, color=colour_map[c], label=c) for c in cost_components]
        handles.append(plt.Line2D([0], [0], color='k', lw=1, label='NPV (box & whisker)'))

        axes[0].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.suptitle(title)
        plt.tight_layout()


SAVE = True
if __name__ == "__main__":
    n_runs = 100
    set_constant_params(params)
    drive_cycles = {k: DriveCycle(df['path'])
                    for k, df in params['Drive Cycles'].items()}
    inputs_distributions = dict(get_uncertainty_distributions(params))
    a = []
    for policy_key, policy_package in policy_packages.items():
        print(policy_key)
        np.random.seed(0)
        samples = np.random.rand(n_runs, len(inputs_distributions)).astype('float32')
        
        # Policy
        ctax = policy_package['Carbon Tax']
        lcfs=policy_package['LCFS']
        autonomous_permits=policy_package['Autonomous Permits']
        zev_mandate=policy_package['ZEV Mandate']
        pyrolysis=policy_package['Pyrolysis']
        pyrolysis_elec=policy_package['Electrified Pyrolysis']
        accelerated_retirement=policy_package['Accelerated Retirement']
        foresight=policy_package['Foresight']
        gvwl_increase=policy_package['ZEV GVWL Increase']
        break_mandate=policy_package['Break Mandate']
        zev_rebate = policy_package['ZEV Rebate']

        runs = list(range(n_runs))

        PARALLEL = False
        if n_runs >= 1_000 or PARALLEL:
            with Pool(processes=10, initializer=init_worker,
                    initargs=(
                        params,
                        drive_cycles, 
                        inputs_distributions, 
                        samples,
                        ctax,
                        lcfs,
                        autonomous_permits,
                        zev_mandate,
                        pyrolysis,
                        pyrolysis_elec,
                        accelerated_retirement,
                        foresight,
                        gvwl_increase,
                        break_mandate,
                        zev_rebate,
                    )
                ) as pool:
                out = pool.map(one_run, runs)
        else:
            out = []
            for r in runs:
                # print(r)
                out.append(one_run(r))

        outputs = merge_outputs(out)

        all_costs = copy.deepcopy(outputs['Cost'])
        for k in all_costs.keys():
            all_costs[k].update(outputs['Policy cost'][k])

        outputs['Activity cost (inc. policy)'] = {
            k: {comp: arr / outputs['Activity'][k]
                for comp, arr in all_costs[k].items()}
            for k in all_costs
        }
        outputs['Activity cost'] = {
            k: {comp: arr / outputs['Activity'][k]
                for comp, arr in outputs['Cost'][k].items()}
            for k in outputs['Cost']
        }
        plotting = Plotting()
        # TCO and NPV
        plotting.plot_by_inner(outputs['Emissions'], x_label='Years', y_label='Emissions (MtCO2)', add_total=True)
        # plotting.plot_by_both(outputs['Activity cost (inc. policy)'], x_label='Years', y_label='Cost ($)', add_total=True)
        # plotting.plot_by_inner(outputs['Fuel Energy'], x_label='Years', y_label='Useful Energy (J)')
        plotting.plot_by_both(outputs['Stock'], x_label='Years', y_label='Stock (thousands)', add_total=True)
        plotting.plot_by_both(outputs['Sales'], x_label='Years', y_label='Sales (thousands)')
        # plotting.plot_all_bars(outputs['TCO'], xlabel='Years', ylabel='Net Present Value (CAD$ million)', title=policy_key)
        # plotting.plot_all_bars(outputs['LCA'], xlabel='Years', ylabel='LCA Emissions (tCO2e)')
        # plotting.plot_all_bars(outputs['Capital'], xlabel='Years', ylabel='Capital Cost (CAD$ thousand)')
        # a.append(sum(outputs['Activity cost (inc. policy)']['Sleeper'].values()))
        plt.show()

        # Save results
        if SAVE:
            with open(f'../Results_19_3_2026/{policy_key}.pkl', 'wb') as f:
                pickle.dump(outputs, f)
