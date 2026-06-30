# fleet2 — HDT Fleet Adoption Model (Paper 2)

Multinomial logit model of heavy-duty truck (HDT) fleet adoption in BC, Canada, 2025–2050.
Rewrite of the Paper 1 model (`old/`). Key architectural fix: shared component parameters
(e.g. ICE efficiency) are drawn once per Monte Carlo run and shared across all powertrains
that contain that component, rather than sampled independently.

- **Vehicle types:** `sleeper`, `day_cab`, `straight` (Class 8)
- **Powertrains:** `dice`, `he`, `phe`, `be`, `fc`, `hice`, `dhice`
- **Fuels:** `diesel`, `h2`, `h2_p`, `h2_pe`, `fast_charge`, `slow_charge`
- **Horizon:** 2025–2050, annual steps

---

## Running

```
python model.py
```

---

## Files

### `params.py`
All model parameters. Every numeric leaf is wrapped in
`Param(value, src, units, notes)` for citation tracking. Array specs
(`{"array": "logistic"|...}`) and distribution specs (`{"dist": "uniform"|...}`)
are stored as the `Param.value` and expanded/sampled downstream.

### `data.py`
Thin loader: strips `Param` wrappers via `_strip_params()`, expands array specs
to numpy arrays, exports module-level constants (`PARAMS`, `START_YEAR`, `MAX_AGE`, etc.).

### `documentation/build_appendix.py`
Reads `params.py` and generates `documentation/appendix.md` — supplementary
material tables (A1–A13) with values and citations. Run after any parameter change:
```
python documentation/build_appendix.py
```

### `model.py`
`Vehicles` and `Fleet` classes. Main entry point.

- **`Vehicles`** — per-cohort calculations: mass, fuel consumption (FASTSim surrogate),
  range, annual distance, FC stack replacements, emissions, capital cost, annual cost, TCO, NPV.
- **`Fleet`** — initial stock build, year-by-year rollover, multinomial logit market share
  with iterative production cap, aggregation of stock/fuel/emissions/costs.

---

## Subdirectories

### `plots/`
Vehicle-level sanity-check plots and fleet-level results plots.
See [`plots/README.md`](plots/README.md).

### `vehicle_modelling/`
Offline FASTSim surrogate training. Not imported at runtime.
See [`vehicle_modelling/README.md`](vehicle_modelling/README.md).

### `verification/`
Test suite and diagnostics (pytest unit tests, sensitivity analysis, benchmarks,
regression snapshot). See [`verification/README.md`](verification/README.md).

### `old/`
Archived Paper 1 model files. Not used by fleet2.
