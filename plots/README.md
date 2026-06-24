# plots/

Diagnostic and results plots for the fleet2 HDT adoption model.
Both scripts import `model.py` and `data.py` from the root directory and
run a deterministic Fleet at median params. Run them from the fleet2 root.

---

## Files

### `vehicle_plots.py`
Per-cohort sanity-check plots for a single vehicle type and cohort year.
Useful for inspecting whether individual `Vehicles` outputs look right before
looking at fleet-level results.

Plots produced (as subplots or separate figures):
- Mass breakdown by component (stacked bar)
- Fuel consumption by age and drive cycle
- Range by age
- Annual distance by age
- Capital cost breakdown (stacked bar)
- TCO decomposition (stacked bar)
- Emissions by stream (embodied, supply, use) by age

```
python plots/vehicle_plots.py
```

### `fleet_plots.py`
Fleet-level time-series plots across 2025–2050.

Plots produced:
- Total stock by powertrain (stacked area)
- Annual sales by powertrain
- Market share by powertrain
- Fuel usage by fuel type
- Emissions by stream (embodied, supply, use)
- System costs by component

Supports single deterministic runs and Monte Carlo output (mean + p5/p95 bands)
once `run.py` is written.

```
python plots/fleet_plots.py
```
