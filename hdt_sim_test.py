import fastsim as fsim

# Load vehicle
veh = fsim.Vehicle.from_resource("2012_Ford_Fusion.yaml")
# Re-configure
veh_dict = veh.to_pydict()
veh_dict['mass_kilograms'] = 30000
veh_dict['chassis']['drag_coef'] = 0.6
veh_dict['chassis']['frontal_area_square_meters'] = 10
veh_dict['chassis']['wheel_rr_coef'] = 0.006
veh_dict['pt_type']['Conv']['fc']['pwr_out_max_watts'] = 400_000
veh_dict['pt_type']['Conv']['fc']['pwr_ramp_lag_seconds'] = 2.0
veh_dict['pt_type']['Conv']['transmission']['eff_interp'] = 0.92
veh_dict['chassis']['wheel_radius_meters'] = 0.5
veh_dict['chassis']['num_wheels'] = 10
# Re-load vehicle
veh = fsim.Vehicle.from_pydict(veh_dict)

# Load drive-cycle
cyc = fsim.Cycle.from_resource("hwfet.csv")

sim = fsim.SimDrive(veh, cyc)

