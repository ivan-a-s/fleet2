vehicle_modelling/
==================

FASTSim-based surrogate training for the fleet2 HDT adoption model.
These files are NOT imported by model.py at runtime; they are used offline
to generate surrogates.json and the drive cycle JSONs.

Files
-----
fuel_consumption.py
    Trains polynomial surrogates (degree-2 interaction) mapping vehicle
    physical parameters (mass, drag_coef, accessory_load, inv_eff) to
    fuel consumption for each (powertrain, drive_cycle) pair.
    Writes results to surrogates.json via save_surrogate().
    Run directly:  python fuel_consumption.py

make_vehicle_df.py
    Loads a FASTSim vehicle YAML and prints its parameter dict.
    Useful for inspecting reference vehicle parameters when calibrating
    the VEHICLES dict in fuel_consumption.py.
    Uncomment the desired vehicle path and run directly.

create_drive_cycle_json.py
    Reads a raw Fleet DNA or UDDS CSV, applies Savitzky-Golay smoothing,
    and writes a FASTSim-compatible JSON to drive_cycles/.
    Uncomment the desired input file and run directly.

surrogates.json
    Merged surrogate coefficients for all trained (powertrain, drive_cycle)
    pairs. Loaded once at module level in model.py as _SURROGATES.
    Structure: {powertrain: {drive_cycle: {features, intercept, r2, mape}}}

Subdirectories
--------------
drive_cycles/
    *.csv   Raw representative drive cycles from NREL Fleet DNA and UDDS.
    *.json  Smoothed (Savitzky-Golay, window=61) FASTSim cycle files.
            cruise_hdt = CARB HHDDT cruise segment
            udds_hdt   = Urban Dynamometer Driving Schedule (HDV)
            short_haul / regional_haul / long_haul = Fleet DNA cycles

vehicles/
    FASTSim YAML definitions for reference vehicles used to calibrate
    FASTSim model parameters (powertrain efficiencies, drag, etc.).
    Not used at runtime.
