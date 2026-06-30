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
| `params.py` | All parameters. Each leaf wrapped in `Param(value, src, units, notes)` for citation tracking |
| `data.py` | Thin loader: strips `Param` wrappers, expands array specs to numpy, exports module-level constants |
| `model.py` | `Vehicles` and `Fleet` classes; main entry point for single deterministic runs |
| `policies.py` | All policy classes: `CarbonTax`, `GVWLExemption`, `LCFS`, `ZEVMandate`, `Policies` container |
| `scenarios.py` | Policy scenario definitions — one `Policies` object per named scenario; edit values here |
| `run.py` | Monte Carlo runner: parallel execution, KS convergence, per-scenario `.npz` output |
| `vehicle_modelling/fuel_consumption.py` | FASTSim surrogate training code — **not imported by model.py** |
| `vehicle_modelling/surrogates.json` | Surrogate coefficients per (powertrain, drive_cycle) used for inference |
| `plots/vehicle_plots.py` | Per-cohort sanity-check plots (mass, FC, TCO, emissions, etc.) |
| `plots/fleet_plots.py` | Fleet-level line plots (stock, sales, fuel use, emissions, costs) |
| `plots/policy_plots/carbon_tax.py` | Carbon tax comparison plots (NPV, sales) |
| `plots/policy_plots/zev_gvwl.py` | GVWL exemption comparison plots (NPV, FC, sales) |
| `plots/policy_plots/lcfs.py` | LCFS comparison plots (NPV, sales) |
| `plots/policy_plots/zev_mandate.py` | ZEV mandate comparison plots (NPV, sales) |
| `verification/profile_fleet.py` | cProfile script for `Fleet()`; phase table + top-25 by self/cumtime; saves `profile.prof` |
| `documentation/build_appendix.py` | Generates `documentation/appendix.md` from `params.py`; run after any parameter change |
| `documentation/appendix.md` | Auto-generated supplementary material tables (A1–A13) with citations; do not edit by hand |

## Running

Single deterministic run (median parameters, no policies):
```
C:\Users\ivana\anaconda3\python.exe model.py
```

Monte Carlo runner (all scenarios, default settings):
```
C:\Users\ivana\anaconda3\python.exe run.py
```

Common MC options:
```
# Quick smoke test — 10 runs, baseline only
C:\Users\ivana\anaconda3\python.exe run.py --max-runs 10 --scenarios baseline

# Specific scenarios with tighter convergence
C:\Users\ivana\anaconda3\python.exe run.py --scenarios baseline carbon_tax lcfs --tol 0.04

# Adjust worker count
C:\Users\ivana\anaconda3\python.exe run.py --workers 4
```

Output lands in `results/<scenario>.npz` (shape `(n_runs, 26)` float32 arrays) and
`results/<scenario>_meta.json` (n_runs, seed, tol, wall_time_s).  Load with:
```python
import numpy as np
d = np.load('results/baseline.npz')
zev_sleeper = d['zev_stock_sleeper']   # shape (n_runs, 26)
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
This compares 2210 fleet output values and reports any that shifted by more than 0.01%. If outputs change unexpectedly, investigate before proceeding. `verification/snapshot.npz` is the saved baseline — overwrite it intentionally when a change is deliberate.

## Conventions

- **Never silently change parameter values.** All values come from calibrated sources. If something looks wrong, flag it and ask.
- Dict/array attributes throughout — no dataclasses for computed outputs.
- `set_param()` / `set_year()` / `realise_uncertainties()` in `model.py` handle all uncertainty sampling.
- Array specs in `params.py` use `{"array": "logistic"|"linspace"|"step"|"constant", ...}` as the `Param.value` — expanded to numpy by `data.py`'s `_expand_arrays` after stripping wrappers.
- `load_model_params()` and `estimate_fuel_consumption()` are copied directly into `model.py` (not imported from `fuel_consumption.py`) to avoid the fastsim dependency at runtime.
- **`set_year()` is non-mutating** — it returns new dicts/scalars and never modifies its input. This means `select_vehicle_params()` only needs shallow `dict()` copies, not `deepcopy`. Do not change `set_year()` to mutate in-place.
- **`_discount_factor`** is precomputed once in `Vehicles.__init__` as `survival_rate / (1+r)^age`. Use `v._discount_factor[0]` for NPV adjustments at age 0 (e.g. ZEV mandate penalty) rather than recomputing the full discount sum.

## Build status

- [x] `params.py` + `data.py` loader — complete, all 3 vehicle types × 7 powertrains; citations in `Param.src` fields
- [x] `Vehicles` class — complete: mass, fuel consumption (FASTSim surrogate), range, annual_distance, FC replacements, emissions, capital_cost, annual_cost, TCO, NPV
- [x] `Fleet._build_initial_stock()` — complete: pre-2025 diesel cohorts sized to match activity requirement
- [x] `Fleet._run()` — complete: year-by-year roll-over, vehicle creation, market share, new purchases
- [x] `Fleet._calculate_market_share()` — complete: multinomial logit + iterative production cap
- [x] `Fleet._aggregate()` — complete: total_stock, sales, fuel_usage, emissions, system_costs
- [x] `plots/vehicle_plots.py` — per-cohort sanity checks; `plots/fleet_plots.py` — fleet-level line plots
- [x] `policies.py` — `CarbonTax`, `GVWLExemption`, `LCFS`, `ZEVMandate`, `Policies` container; comparison plots in `plots/policy_plots/`
- [x] `scenarios.py` — 6 named policy scenarios (`baseline`, `carbon_tax`, `lcfs`, `zev_mandate`, `gvwl`, `full_policy`)
- [x] `run.py` — Monte Carlo runner: grouped sampling, parallel execution, KS convergence, `.npz` output

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

## Key params.py fields to know

- `settings`: max_age=25, start_year=2025, end_year=2050, discount_rate=0.08, growth_rate=0.02
- `fleet`: initial_activity, activity_growth=0.02, price_lambda=0.00003, autonomous_t50=2040
- `vehicles.components`: shared component defs (`converter`, `ess`, `transmission`) — each powertrain references these by type to avoid independent MC draws
- Per-powertrain: `init_market_limit` (1.0 for dice, 0.02 for others), `cagr_nacent`, `cagr_mature`
- Surrogate mapping: both `hice` and `dhice` reuse the `he` surrogate (all 5 drive cycles); `phe` only has `udds_hdt`/`cruise_hdt` (no haul-specific files)
- `hice` is modelled as a hybridised H2 ICE: motor (220 kW), battery (10/5/5 kWh), regen_efficiency=0.71, accessory_load=3400 — mirrors `he` component set with H2 tank instead of diesel tank

## params.py conventions

- Every numeric leaf is wrapped: `P(value, src="Author, Year", units="unit", notes="optional")`
- `src=None` or a string starting with `"UNREF --"` flags values without a paper citation
- Container dicts (those without a `"dist"` or `"array"` key) are left as plain Python dicts
- `data.py` calls `_strip_params()` to remove all `Param` wrappers before passing to `_expand_arrays()`
- `documentation/build_appendix.py` reads `PARAM_DICT` directly to generate supplementary tables
- To update a parameter: edit the value and/or `src` in `params.py`, then run `python documentation/build_appendix.py`

## Next steps

1. Check total mass plausibility — sleeper diesel showing 28.9 t loaded (expect ~36-40 t)
2. Verify `model.py` checked up to `_calculate_fuel_consumption` — remaining methods still need review
3. Stage 4 policy improvements: dynamic LCFS credit pricing, ZEV purchase rebate, joint mandate+LCFS convergence (see `POLICY_PLAN.md`)

## Performance notes

Single `Fleet()` run takes ~0.5–0.9 s depending on machine and policies active. The dominant cost
is constructing 543 `Vehicles` objects in `_run()` — inherent physics, not easily vectorised.
Further optimisation opportunities (ZEV mandate loop, `activity_met` vectorisation) are documented
in the `verification/profile_fleet.py` docstring under "Remaining optimisation opportunities".

Monte Carlo throughput with `run.py` at 8 workers: ~4–5× wall-clock speedup over serial.
Baseline scenario (no policies, high distributional uncertainty) converges at ~3 000–4 000 runs
with `--tol 0.05` (~8 min). Policy scenarios converge faster (~1 000–2 000 runs) because the
mandate/tax constrains the ZEV adoption distribution.

## Monte Carlo architecture (`run.py` + `scenarios.py`)

**Sampling:** `get_uncertainty_distributions(PARAMS)` walks the params tree and returns every
leaf with a `'dist'` key. Parameters sharing a `'group'` field receive the same cumulative
probability draw (correlated); others are independent. `_build_col_map()` maps each parameter
path to a column index in the `(max_runs, n_independent)` sample matrix, which is pre-generated
once with `np.random.default_rng(seed)` so all scenarios use identical draws for counterfactual
comparison.

**Parallel execution:** All `max_runs` futures are submitted to a single `ProcessPoolExecutor`
at once (no per-batch pool overhead). `as_completed()` yields results as workers finish. Below
`PARALLEL_THRESHOLD = 50` runs a flat serial loop is used instead (pool setup cost exceeds
any speedup for small runs).

**Convergence:** After every `--check-every` completions (minimum 200), a half-sample
two-sample KS test is applied. The accumulated results are split into two equal halves and
`scipy.stats.ks_2samp` is called for each monitored series at each calendar year:

    D = max_x |F_A(x) - F_B(x)|

D ∈ [0, 1] is scale-free — no normalisation denominator needed. For i.i.d. draws,
E[D] ≈ 0.8/√n per half, so the convergence rate is predictable. Converged when
`max(D) < tol` across all monitored series and years. Remaining queued futures are
cancelled (already in-flight workers finish naturally).

**Monitored series** (15 total = 5 metrics × 3 vehicle types):
- `zev_stock_{k}` — total ZEV stock
- `emissions_{k}_use` / `emissions_{k}_supply` — fleet emissions
- `system_costs_{k}_capital` / `system_costs_{k}_fuel` — cost spread

**Output arrays** (82 series per scenario):

| Key pattern | Shape | Content |
|-------------|-------|---------|
| `total_stock_{k}_{p}` | (n, 26) | Total on-road stock by powertrain |
| `sales_{k}_{p}` | (n, 26) | New sales per year |
| `zev_stock_{k}` | (n, 26) | Aggregate ZEV stock |
| `emissions_{k}_{embodied\|supply\|use}` | (n, 26) | Fleet emissions (kgCO2e/yr) |
| `system_costs_{k}_{category}` | (n, 26) | Annual costs ($/yr) |
| `fuel_usage_{k}_{f}` | (n, 26) | Fuel consumed (L, kg, or kWh/yr) |

**Scenarios** (`scenarios.py`): `baseline`, `carbon_tax`, `lcfs`, `zev_mandate`, `gvwl`,
`full_policy`. Policy numeric values (tax schedules, LCFS targets, ZEV mandate fractions) all
live in `scenarios.py` — edit there without touching `run.py` or `model.py`.
