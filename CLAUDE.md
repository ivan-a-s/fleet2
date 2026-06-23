# fleet2 — HDT Fleet Adoption Model (Paper 2)

Multinomial logit model of heavy-duty truck (HDT) fleet adoption in BC, Canada, 2025–2050.
Rewrite of the Paper 1 model (`fleet/`). The key architectural fix over Paper 1: shared component
parameters (e.g. ICE efficiency) are drawn once per Monte Carlo run and shared across all
powertrains that contain that component, rather than sampled independently.

## Scope

- **Vehicle types:** `sleeper`, `day_cab`, `straight` (Class 8)
- **Powertrains:** `dice`, `he` (mild hybrid), `phe` (plug-in hybrid), `be`, `fc`, `hice`, `dhice`
- **Fuels:** `diesel`, `h2` (electrolysis), `h2_p` (pyrolysis), `h2_pe` (electrified pyrolysis), `fast_charge`, `slow_charge`
- **Policies:** not yet implemented — base model first

## Key files

| File | Purpose |
|------|---------|
| `data.json` | All parameters. No Python expressions — arrays use `{"array": ...}` specs |
| `data.py` | Thin loader: reads JSON, expands array specs to numpy, exports module-level constants |
| `model.py` | `Vehicles` and `Fleet` classes; main entry point |
| `fuel_consumption.py` | FASTSim surrogate training code — **not imported by model.py** |
| `drive_cycles/*.json` | Surrogate coefficients per (powertrain, drive_cycle) used for inference |

## Running

```
C:\Users\ivana\anaconda3\python.exe model.py
```

## Conventions

- **Never silently change parameter values.** All values come from calibrated sources. If something looks wrong, flag it and ask.
- Dict/array attributes throughout — no dataclasses for computed outputs.
- `set_param()` / `set_year()` / `realise_uncertainties()` in `model.py` handle all uncertainty sampling.
- Array specs in `data.json` use `{"array": "logistic"|"linspace"|"step"|"constant", ...}` — expanded by `data.py` loader.
- `load_model_params()` and `estimate_fuel_consumption()` are copied directly into `model.py` (not imported from `fuel_consumption.py`) to avoid the fastsim dependency at runtime.

## Build status

- [x] `data.json` + `data.py` loader — complete, all 3 vehicle types × 7 powertrains
- [x] `Vehicles` class — complete: mass, fuel consumption (FASTSim surrogate), range, annual_distance, FC replacements, emissions, capital_cost, annual_cost, TCO, NPV
- [x] `Fleet._build_initial_stock()` — complete: pre-2025 diesel cohorts sized to match activity requirement
- [x] `Fleet._run()` — complete: year-by-year roll-over, vehicle creation, market share, new purchases
- [x] `Fleet._calculate_market_share()` — complete: multinomial logit + iterative production cap
- [x] `Fleet._aggregate()` — complete: total_stock, sales, fuel_usage, emissions, system_costs
- [ ] Physical plausibility check — total mass 28.9 t for sleeper diesel seems low (expect ~36-40 t loaded); payload calculation may need review
- [ ] Monte Carlo runner (`run.py`) — not yet written
- [ ] Policies (carbon tax, LCFS, ZEV mandate) — not yet implemented

## Current model.py structure

```
Helper functions:   load_model_params, estimate_fuel_consumption,
                    get_uncertainty_distributions, set_param_, set_param,
                    convert_to_float32, set_year

Module constants:   SURROGATE_NAME, PHE_DC_MAP, EFF_COMPONENT,
                    ZEV_POWERTRAINS, HICE_POWERTRAINS,
                    AFTERTREATMENT_POWERTRAINS, CHARGER_POWERTRAINS

Vehicles class:     _calculate_mass, _calculate_fuel_consumption,
                    _split_surrogate_output, _calculate_range,
                    _calculate_annual_distance, _track_fc_replacements,
                    _calculate_emissions, _calculate_capital_cost,
                    _calculate_annual_cost, _discount, _calculate_tco_npv

Fleet class:        _build_initial_stock, _run, _calculate_market_share,
                    _aggregate, select_vehicle_params, realise_uncertainties

Module function:    _market_share_limit (production cap helper)
```

## Key data.json fields to know

- `settings`: max_age=25, start_year=2025, end_year=2050, discount_rate=0.08, growth_rate=0.02
- `fleet`: initial_activity, activity_growth=0.02, price_lambda=0.00003, autonomous_t50=2040
- `vehicles.components`: shared component defs (`converter`, `ess`, `transmission`) — each powertrain references these by type to avoid independent MC draws
- Per-powertrain: `init_market_limit` (1.0 for dice, 0.02 for others), `cagr_nacent`, `cagr_mature`
- Surrogate mapping: `hice`/`dhice` reuse `dice`/`he_parallel` surrogates; `phe_parallel` only has `udds_hdt`/`cruise_hdt` (no haul-specific files)

## Next steps

1. Check total mass plausibility — sleeper diesel showing 28.9 t loaded (expect ~36-40 t)
2. Write `run.py` Monte Carlo runner (multiprocessing, per-scenario output aggregation)
3. Add policy layers (carbon tax, LCFS, ZEV mandate) to `Vehicles.annual_cost`
