""" Plot stock trajectories. """
import numpy as np
import matplotlib.pyplot as plt
import pickle

from parallel_test import Plotting

fname = '../Results_19_3_2026/Base.pkl'

with open(fname, 'rb') as f:
    outputs = pickle.load(f)

plotting = Plotting()
plotting.plot_by_both(outputs['Stock'], x_label='Years', y_label='Stock (thousands)')
plotting.plot_by_both(outputs['Sales'], x_label='Years', y_label='Sales (thousands)')

fuel_energy = outputs['Fuel Energy']
for k, v in fuel_energy.items():
    v['Hydrogen'] += v['Hydrogen (pyrolysis)'] + v['Hydrogen (pyrolysis + elec.)']
    v['Electricity'] = v['Slow Charge'] + v['Fast Charge']
    v.pop('Slow Charge')
    v.pop('Fast Charge')
    v.pop('Hydrogen (pyrolysis)')
    v.pop('Hydrogen (pyrolysis + elec.)')
    for p, vv in v.items():
        vv /= 1e15
plotting.plot_by_inner(fuel_energy, x_label='Years', y_label='Useful Energy (PJ)')

plotting.plot_by_inner(outputs['Emissions'], x_label='Years', y_label='Emissions (MtCO$_2$e)', add_total=True)


fname = '../Results_19_3_2026/Policy Package.pkl'

with open(fname, 'rb') as f:
    outputs = pickle.load(f)

plotting = Plotting()
plotting.plot_by_both(outputs['Stock'], x_label='Years', y_label='Stock (thousands)')
plotting.plot_by_both(outputs['Sales'], x_label='Years', y_label='Sales (thousands)')

fuel_energy = outputs['Fuel Energy']
for k, v in fuel_energy.items():
    v['Hydrogen'] += v['Hydrogen (pyrolysis)'] + v['Hydrogen (pyrolysis + elec.)']
    v['Electricity'] = v['Slow Charge'] + v['Fast Charge']
    v.pop('Slow Charge')
    v.pop('Fast Charge')
    v.pop('Hydrogen (pyrolysis)')
    v.pop('Hydrogen (pyrolysis + elec.)')
    for p, vv in v.items():
        vv /= 1e15

plotting.plot_by_inner(fuel_energy, x_label='Years', y_label='Useful Energy (PJ)')

plotting.plot_by_inner(outputs['Emissions'], x_label='Years', y_label='Emissions (MtCO$_2$e)')

total_by_fuel = {f: sum(fuel_energy[k][f] for k in fuel_energy.keys()) for f in fuel_energy['Sleeper'].keys()}
total = sum({k: sum(fuel_energy[k].values()) for k in fuel_energy.keys()}.values())
for f, v in total_by_fuel.items():
    frac = v/total
    print(f, np.percentile(frac[:, -1], [5, 95]))
