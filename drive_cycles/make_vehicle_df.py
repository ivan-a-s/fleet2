""" Generate a FASTSIM vehicle dataframe. """
import fastsim as fsim
import pprint

fname = 'drive_cycles/Toyota Mirai.yaml'
fname = 'drive_cycles/2022 Ford F-150 Lightning 4WD.yaml'
fname = 'drive_cycles/2016 BMW i3 REx PHEV.yaml'
fname = 'drive_cycles/2016 KIA Optima Hybrid.yaml'

with open(fname, "r") as f:
    data = f.read()

veh = fsim.Vehicle.from_yaml(data)
veh_dict = veh.to_pydict()
pprint.pprint(veh_dict)
