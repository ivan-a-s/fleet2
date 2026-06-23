fleet2 — HDT Fleet Adoption Model (Paper 2)
===========================================

Multinomial logit model of heavy-duty truck (HDT) fleet adoption in BC, Canada,
2025–2050. Rewrite of the Paper 1 model. Key architectural fix: shared component
parameters (e.g. ICE efficiency) are drawn once per Monte Carlo run and shared
across all powertrains that contain that component, rather than sampled independently.

Vehicle types:  sleeper, day_cab, straight (Class 8)
Powertrains:    dice, he, phe, be, fc, hice, dhice
Fuels:          diesel, h2, h2_p, h2_pe, fast_charge, slow_charge
Horizon:        2025–2050 (annual steps)

Run
---
    python model.py

Files
-----
data.json
    All model parameters. No Python expressions — arrays use
    {"array": "logistic"|"linspace"|"step"|"constant", ...} specs.

data.py
    Thin loader: reads data.json, expands array specs to numpy arrays,
    exports module-level constants (PARAMS, etc.).

model.py
    Vehicles and Fleet classes. Main entry point.
    - Vehicles: mass, fuel consumption (FASTSim surrogate), range,
      annual_distance, FC replacements, emissions, capital_cost,
      annual_cost, TCO, NPV
    - Fleet: initial stock build, year-by-year rollover, multinomial
      logit market share with iterative production cap, aggregation

vehicle_plots.py
    Vehicle-level sanity-check plots (age profiles, stacked bars for
    mass / capital cost / TCO / LCA emissions). Run directly.

fleet_plots.py
    Fleet-level results plots (stock, sales, market share, fuel usage,
    emissions, costs). Supports single-run and Monte Carlo (mean + p5/p95).
    Run directly for a deterministic single run.

Subdirectories
--------------
vehicle_modelling/
    Offline FASTSim surrogate training. Not imported at runtime.
    See vehicle_modelling/README.txt for details.
