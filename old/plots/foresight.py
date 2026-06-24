""" Plots to deomstrate the power of foresight on vehicle adoption. """
import numpy as np
import matplotlib.pyplot as plt
import pickle

from parallel_test import Plotting

fname = '../Results_19_3_2026/Base.pkl'

with open(fname, 'rb') as f:
    outputs = pickle.load(f)

# Fuel emissions
emissions = outputs['Emissions']
for k, v in emissions.items():
    emissions[k] = v['Fuel Combustion'] + v['Fuel Supply']
base_emissions = emissions['Sleeper'] + emissions['Day Cab'] + emissions['Class-8 Straight']

plt.figure()
years = np.arange(2025, 2051)
p5, p50, p95 = np.percentile(base_emissions, [5, 50, 95], axis=0)
plt.plot(years, p50)
plt.fill_between(years, p5, p95, alpha=0.1)


fname = '../Results_19_3_2026/Foresight.pkl'
with open(fname, 'rb') as f:
    outputs = pickle.load(f)

# Fuel emissions
emissions = outputs['Emissions']
for k, v in emissions.items():
    emissions[k] = v['Fuel Combustion'] + v['Fuel Supply']
foresight_emissions = emissions['Sleeper'] + emissions['Day Cab'] + emissions['Class-8 Straight']

years = np.arange(2025, 2051)
p5, p50, p95 = np.percentile(foresight_emissions, [5, 50, 95], axis=0)
plt.plot(years, p50)
plt.fill_between(years, p5, p95, alpha=0.1)


ratio = 1-foresight_emissions/base_emissions
print(np.percentile(ratio[:, -1], [5, 95]))

fname = '../Results_19_3_2026/Policy Package.pkl'

with open(fname, 'rb') as f:
    outputs = pickle.load(f)

# Fuel emissions
emissions = outputs['Emissions']
for k, v in emissions.items():
    emissions[k] = v['Fuel Combustion'] + v['Fuel Supply']
base_emissions = emissions['Sleeper'] + emissions['Day Cab'] + emissions['Class-8 Straight']

plt.figure()
years = np.arange(2025, 2051)
p5, p50, p95 = np.percentile(base_emissions, [5, 50, 95], axis=0)
plt.plot(years, p50)
plt.fill_between(years, p5, p95, alpha=0.1)


fname = '../Results_19_3_2026/Foresight (PP).pkl'
with open(fname, 'rb') as f:
    outputs = pickle.load(f)

# Fuel emissions
emissions = outputs['Emissions']
for k, v in emissions.items():
    emissions[k] = v['Fuel Combustion'] + v['Fuel Supply']
foresight_emissions = emissions['Sleeper'] + emissions['Day Cab'] + emissions['Class-8 Straight']

years = np.arange(2025, 2051)
p5, p50, p95 = np.percentile(foresight_emissions, [5, 50, 95], axis=0)
plt.plot(years, p50)
plt.fill_between(years, p5, p95, alpha=0.1)


ratio = 1-foresight_emissions/base_emissions
np.percentile(ratio[:, -1], [5, 95])

# fname = '../Results_19_3_2026/Policy Package.pkl'
# with open(fname, 'rb') as f:
#     outputs = pickle.load(f)

# # Fuel emissions
# emissions = outputs['Emissions']
# for k, v in emissions.items():
#     emissions[k] = v['Fuel Combustion'] + v['Fuel Supply']
# total_emissions = emissions['Sleeper'] + emissions['Day Cab'] + emissions['Class-8 Straight']

# years = np.arange(2025, 2051)
# p5, p50, p95 = np.percentile(total_emissions, [5, 50, 95], axis=0)
# plt.plot(years, p50)
# plt.fill_between(years, p5, p95, alpha=0.1)


