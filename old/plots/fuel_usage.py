""" Plot stock trajectories. """
import numpy as np
import matplotlib.pyplot as plt
import pickle

from parallel_test import Plotting

fname = '../Results/Base Case.pkl'
fname = '../Results/Policy Package.pkl'

with open(fname, 'rb') as f:
    outputs = pickle.load(f)

fuel_energy = outputs['Fuel Energy']
for k, v in fuel_energy.items():
    v.pop('Hydrogen (pyrolysis)')
    v.pop('Hydrogen (pyrolysis + elec.)')
    v['Electricity'] = v['Fast Charge'] + v['Slow Charge']
    v.pop('Slow Charge')
    v.pop('Fast Charge')
    for kk, vv in v.items():
        vv /= 1e15

plotting = Plotting()
plotting.plot_by_inner(fuel_energy, x_label='Years', y_label='Useful Energy (PJ)')



fuel_energy = outputs['Emissions']
for k, v in fuel_energy.items():
    v.pop('Embodied')
    # for kk, vv in v.items():
    #     vv /= 1e15

plotting.plot_by_inner(outputs['Emissions'], x_label='Years', y_label='Emissions (MtCO2)', add_total=True)
