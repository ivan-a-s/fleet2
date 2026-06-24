""" Plots uncertainty contribution from each variable. """
import numpy as np
import matplotlib.pyplot as plt
import pickle

from data import PARAMS
from model import *
from sobol import *

# Get emissions
fname = '../Results/Base Case.pkl'
with open(fname, 'rb') as f:
    outputs = pickle.load(f)
emissions = outputs['Emissions']
for k, v in emissions.items():
    emissions[k].pop('Embodied')
    emissions[k] = sum(vv for kk, vv in v.items())
emissions = sum(v for k, v in emissions.items())[:, 0]

# Get the sample distributions
params = PARAMS
n_runs = len(emissions)
np.random.seed(0)
inputs_distributions = dict(get_uncertainty_distributions(params))
samples = np.random.rand(n_runs, len(inputs_distributions)).astype('float32')





import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# X = samples, y = emissions (from your previous code)
X = samples
y = np.asarray(emissions).ravel()

# Split for testing fit quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit Random Forest surrogate
rf = RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=5,
    random_state=0
)
rf.fit(X_train, y_train)

# Evaluate surrogate
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Surrogate fit R^2: {r2:.3f}, RMSE: {rmse:.3f}")


from SALib.sample import saltelli
from SALib.analyze import sobol

# Build the SALib problem
problem = {
    'num_vars': X.shape[1],
    'names': [str(k) for k in inputs_distributions.keys()],
    'bounds': [[0, 1]] * X.shape[1],  # matches your normalized samples
}

# Generate samples
N_sobol = 1000  # base sample size; increase for stability
param_values = saltelli.sample(problem, N_sobol, calc_second_order=True)

# Evaluate surrogate at Sobol samples
Y_sobol = rf.predict(param_values)


Si = sobol.analyze(
    problem,
    Y_sobol,
    calc_second_order=True,
    print_to_console=False
)

# First-order
S1 = dict(zip(problem['names'], Si['S1']))
# Total-order
ST = dict(zip(problem['names'], Si['ST']))

import matplotlib.pyplot as plt

threshold = 0.01
names = np.array(list(S1.keys()))
values = np.array(list(S1.values()))

mask = values > threshold
names_f = names[mask]
values_f = values[mask]

# sort descending
order = np.argsort(values_f)[::-1]
names_f = names_f[order]
values_f = values_f[order]

plt.figure(figsize=(10, 4))
plt.bar(names_f, values_f)
plt.ylabel("First-order Sobol index")
plt.title("First-order Sobol sensitivity (>1%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

