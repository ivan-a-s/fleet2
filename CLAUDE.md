# fleet2 — HDT Fleet Adoption Model (Paper 2)

Multinomial logit model of heavy-duty truck (HDT) fleet adoption in BC, Canada, 2025–2050.
Rewrite of the Paper 1 model (`fleet/`). The key architectural fix over Paper 1: shared component
parameters (e.g. ICE efficiency) are drawn once per Monte Carlo run and shared across all
powertrains that contain that component, rather than sampled independently.

## Scope

- **Vehicle types:** `sleeper`, `day_cab`, `straight` (Class 8)
- **Powertrains:** `dice`, `he` (mild hybrid), `phe` (plug-in hybrid), `be`, `fc`, `hice`, `dhice`
- **Fuels:** `diesel`, `h2` (electrolysis), `h2_p` (pyrolysis), `h2_pe` (electrified pyrolysis), `fast_charge`, `slow_charge`
- **Policies:** `CarbonTax`, `GVWLExemption`, `LCFS`, `ZEVMandate` — all implemented in `policies.py`

## Key files

| File | Purpose |
|------|---------|
| `data.json` | All parameters. No Python expressions — arrays use `{"array": ...}` specs |
| `data.py` | Thin loader: reads JSON, expands array specs to numpy, exports module-level constants |
| `model.py` | `Vehicles` and `Fleet` classes; main entry point |
| `policies.py` | All policy classes: `CarbonTax`, `GVWLExemption`, `LCFS`, `ZEVMandate`, `Policies` container |
| `vehicle_modelling/fuel_consumption.py` | FASTSim surrogate training code — **not imported by model.py** |
| `vehicle_modelling/surrogates.json` | Surrogate coefficients per (powertrain, drive_cycle) used for inference |
| `plots/vehicle_plots.py` | Per-cohort sanity-check plots (mass, FC, TCO, emissions, etc.) |
| `plots/fleet_plots.py` | Fleet-level line plots (stock, sales, fuel use, emissions, costs) |
| `plots/policy_plots/carbon_tax.py` | Carbon tax comparison plots (NPV, sales) |
| `plots/policy_plots/zev_gvwl.py` | GVWL exemption comparison plots (NPV, FC, sales) |
| `plots/policy_plots/lcfs.py` | LCFS comparison plots (NPV, sales) |
| `plots/policy_plots/zev_mandate.py` | ZEV mandate comparison plots (NPV, sales) |
| `verification/profile_fleet.py` | cProfile script for `Fleet()`; phase table + top-25 by self/cumtime; saves `profile.prof` |

## Running

```
C:\Users\ivana\anaconda3\python.exe model.py
```

## Regression snapshot

Before making any code changes, run:
```
C:\Users\ivana\anaconda3\python.exe verification/snapshot.py save
```
After changes, run:
```
C:\Users\ivana\anaconda3\python.exe verification/snapshot.py check
```
This compares 2054 fleet output values and reports any that shifted by more than 0.01%. If outputs change unexpectedly, investigate before proceeding. `verification/snapshot.npz` is the saved baseline — overwrite it intentionally when a change is deliberate.

## Conventions

- **Never silently change parameter values.** All values come from calibrated sources. If something looks wrong, flag it and ask.
- Dict/array attributes throughout — no dataclasses for computed outputs.
- `set_param()` / `set_year()` / `realise_uncertainties()` in `model.py` handle all uncertainty sampling.
- Array specs in `data.json` use `{"array": "logistic"|"linspace"|"step"|"constant", ...}` — expanded by `data.py` loader.
- `load_model_params()` and `estimate_fuel_consumption()` are copied directly into `model.py` (not imported from `fuel_consumption.py`) to avoid the fastsim dependency at runtime.
- **`set_year()` is non-mutating** — it returns new dicts/scalars and never modifies its input. This means `select_vehicle_params()` only needs shallow `dict()` copies, not `deepcopy`. Do not change `set_year()` to mutate in-place.
- **`_discount_factor`** is precomputed once in `Vehicles.__init__` as `survival_rate / (1+r)^age`. Use `v._discount_factor[0]` for NPV adjustments at age 0 (e.g. ZEV mandate penalty) rather than recomputing the full discount sum.

## Build status

- [x] `data.json` + `data.py` loader — complete, all 3 vehicle types × 7 powertrains
- [x] `Vehicles` class — complete: mass, fuel consumption (FASTSim surrogate), range, annual_distance, FC replacements, emissions, capital_cost, annual_cost, TCO, NPV
- [x] `Fleet._build_initial_stock()` — complete: pre-2025 diesel cohorts sized to match activity requirement
- [x] `Fleet._run()` — complete: year-by-year roll-over, vehicle creation, market share, new purchases
- [x] `Fleet._calculate_market_share()` — complete: multinomial logit + iterative production cap
- [x] `Fleet._aggregate()` — complete: total_stock, sales, fuel_usage, emissions, system_costs
- [x] `plots/vehicle_plots.py` — per-cohort sanity checks; `plots/fleet_plots.py` — fleet-level line plots
- [x] `policies.py` — `CarbonTax`, `GVWLExemption`, `LCFS`, `ZEVMandate`, `Policies` container; comparison plots in `plots/policy_plots/`
- [ ] Monte Carlo runner (`run.py`) — not yet written

## Current model.py structure

```
Helper functions:   load_model_params, estimate_fuel_consumption,
                    get_uncertainty_distributions, set_param_, set_param,
                    convert_to_float32, set_year

Module constants:   _SURROGATES (loaded from vehicle_modelling/surrogates.json),
                    SURROGATE_NAME, EFF_COMPONENT,
                    ZEV_POWERTRAINS, HICE_POWERTRAINS,
                    CHARGER_POWERTRAINS, ALL_POWERTRAINS,
                    _YEAR0 (= START_YEAR - MAX_AGE, first year in all realised arrays)

Vehicles class:     _calculate_mass, _calculate_fuel_consumption,
                    _split_surrogate_output, _calculate_range,
                    _calculate_annual_distance, _track_fc_replacements,
                    _calculate_emissions, _cap_cost, _op_cost_array,
                    _calculate_capital_cost, _calculate_annual_cost,
                    _discount, _calculate_tco_npv

Fleet class:        _make_vehicle, _apply_mandate_penalty, _build_initial_stock, _run,
                    _calculate_market_share, _aggregate,
                    select_vehicle_params, realise_uncertainties

Module function:    _market_share_limit (production cap helper)
```

## Policy architecture (`policies.py`)

Two-phase hook interface called from `Fleet._make_vehicle()`:
- **`pre_apply(params, k, p, t)`** — modifies the params dict *before* `Vehicles()` is constructed. For physics policies that propagate through mass → FC → cost (e.g. GVWL exemption).
- **`apply(v)`** — writes cost terms into `v.annual_cost` *after* construction, then calls `v._calculate_tco_npv()` once. For cost-only policies (carbon tax, LCFS).

Endogenous policies (ZEV mandate) run in an outer convergence loop inside `Fleet._run()`, not via the hooks. The loop warm-starts from the previous year's penalty, uses 30/70 damped updates, bisects on oscillation, and exits cleanly when the penalty converges (production cap binding) — only warns on true numerical non-convergence.

`COST_CATEGORIES` in `model.py` drives `_aggregate()`:
```python
COST_CATEGORIES = {
    'system': ('capital', 'operational', 'fuel', 'driver', 'fc_replacements'),
    'policy': ('carbon_tax', 'lcfs', 'zev_mandate'),
}
```

**LCFS calibration:** `baseline_fc[k]` is extracted from a throwaway START_YEAR diesel vehicle built in `Fleet.__init__` between `_build_initial_stock()` and `_run()`. This means LCFS has non-zero costs for diesel from 2025 onwards (target starts at 18.3%).

**ZEV mandate scopes:**
- `scope='fleet'` — ZEV share target applies across all k combined; `targets = {year_str: fraction}`
- `scope='per_k'` — independent target per vehicle type; `targets = {k: {year_str: fraction}}`

## Key data.json fields to know

- `settings`: max_age=25, start_year=2025, end_year=2050, discount_rate=0.08, growth_rate=0.02
- `fleet`: initial_activity, activity_growth=0.02, price_lambda=0.00003, autonomous_t50=2040
- `vehicles.components`: shared component defs (`converter`, `ess`, `transmission`) — each powertrain references these by type to avoid independent MC draws
- Per-powertrain: `init_market_limit` (1.0 for dice, 0.02 for others), `cagr_nacent`, `cagr_mature`
- Surrogate mapping: both `hice` and `dhice` reuse the `he` surrogate (all 5 drive cycles); `phe` only has `udds_hdt`/`cruise_hdt` (no haul-specific files)
- `hice` is modelled as a hybridised H2 ICE: motor (220 kW), battery (10/5/5 kWh), regen_efficiency=0.71, accessory_load=3400 — mirrors `he` component set with H2 tank instead of diesel tank

## Next steps

1. Check total mass plausibility — sleeper diesel showing 28.9 t loaded (expect ~36-40 t)
2. Write `run.py` Monte Carlo runner (multiprocessing, per-scenario output aggregation)
3. Verify `model.py` checked up to `_calculate_fuel_consumption` — remaining methods still need review
4. Stage 4 policy improvements: dynamic LCFS credit pricing, ZEV purchase rebate, joint mandate+LCFS convergence (see `POLICY_PLAN.md`)

## Performance notes

Single `Fleet()` run takes ~0.53 s at median params. The dominant cost is constructing 543 `Vehicles`
objects in `_run()` — inherent physics, not easily vectorised. Further optimisation opportunities
(ZEV mandate loop, Monte Carlo parallelism, `activity_met` vectorisation) are documented in the
`verification/profile_fleet.py` docstring under "Remaining optimisation opportunities".
