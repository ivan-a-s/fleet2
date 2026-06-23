""" Generate a FASTSim vehicle dict from a YAML file. """
import os
import fastsim as fsim
import pprint

_HERE = os.path.dirname(os.path.abspath(__file__))

# Uncomment the vehicle to inspect
# fname = os.path.join(_HERE, 'vehicles', 'Toyota Mirai.yaml')
# fname = os.path.join(_HERE, 'vehicles', '2022 Ford F-150 Lightning 4WD.yaml')
# fname = os.path.join(_HERE, 'vehicles', '2016 BMW i3 REx PHEV.yaml')
fname   = os.path.join(_HERE, 'vehicles', '2016 KIA Optima Hybrid.yaml')

with open(fname, "r") as f:
    data = f.read()

veh = fsim.Vehicle.from_yaml(data)
pprint.pprint(veh.to_pydict())
