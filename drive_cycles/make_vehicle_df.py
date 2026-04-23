""" Generate a FASTSIM vehicle dataframe. """
import fastsim as fsim

with open('drive_cycles/2022 Ford F-150 Lightning 4WD.yaml', "r") as f:
    data = f.read()

veh = fsim.Vehicle.from_yaml(data)
veh_dict = veh.to_pydict()
print(veh_dict)
