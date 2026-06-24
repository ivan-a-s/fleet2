# vehicle_modelling/

FASTSim-based surrogate training for the fleet2 HDT adoption model.
These files are **not imported by `model.py` at runtime** — they are used offline
to generate `surrogates.json` and the drive cycle JSONs.

---

## Files

### `fuel_consumption.py`
Trains polynomial surrogates (degree-2 with interactions) mapping vehicle physical
parameters (mass, drag coefficient, accessory load, inverter efficiency) to fuel
consumption for each (powertrain, drive cycle) pair. Writes results to `surrogates.json`
via `save_surrogate()`.

```
python vehicle_modelling/fuel_consumption.py
```

### `make_vehicle_df.py`
Loads a FASTSim vehicle YAML and prints its parameter dict. Useful for inspecting
reference vehicle parameters when calibrating the `VEHICLES` dict in `fuel_consumption.py`.
Uncomment the desired vehicle path and run directly.

### `create_drive_cycle_json.py`
Reads a raw Fleet DNA or UDDS CSV, applies Savitzky-Golay smoothing, and writes a
FASTSim-compatible JSON to `drive_cycles/`. Uncomment the desired input file and run directly.

### `surrogates.json`
Merged surrogate coefficients for all trained (powertrain, drive cycle) pairs.
Loaded once at module level in `model.py` as `_SURROGATES`.

Structure: `{powertrain: {drive_cycle: {features, intercept, r2, mape}}}`

Surrogate notes:
- `hice` and `dhice` reuse the `he` surrogate (all 5 drive cycles)
- `phe` only has `udds_hdt` and `cruise_hdt` (no haul-specific files)

---

## Subdirectories

### `drive_cycles/`

| File | Description |
|------|-------------|
| `*.csv` | Raw representative drive cycles from NREL Fleet DNA and UDDS |
| `cruise_hdt.json` | CARB HHDDT cruise segment (smoothed) |
| `udds_hdt.json` | Urban Dynamometer Driving Schedule (heavy-duty variant, smoothed) |
| `short_haul.json` | Fleet DNA local delivery cycle |
| `regional_haul.json` | Fleet DNA regional haul cycle |
| `long_haul.json` | Fleet DNA long-haul cycle |

Smoothing: Savitzky-Golay filter, window=61.

### `vehicles/`
FASTSim YAML definitions for reference vehicles used to calibrate powertrain
efficiencies, drag coefficients, etc. Not used at runtime.
