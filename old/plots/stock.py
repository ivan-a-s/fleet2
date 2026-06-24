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

fname = '../Results_19_3_2026/Policy Package.pkl'

with open(fname, 'rb') as f:
    outputs = pickle.load(f)

plotting = Plotting()
plotting.plot_by_both(outputs['Stock'], x_label='Years', y_label='Stock (thousands)')
plotting.plot_by_both(outputs['Sales'], x_label='Years', y_label='Sales (thousands)')

