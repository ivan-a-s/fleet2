# Verification — fleet2

Verification and reliability suite for the fleet2 HDT fleet adoption model.
All scripts use a single deterministic Fleet run at median parameter values (cp=0.5 for every
uncertain parameter), so there is no Monte Carlo noise in the results.

---

## Running everything

```
pytest verification/ -v
python verification/sensitivity.py
python verification/benchmarks.py
python verification/snapshot.py check
```

---

## test_fleet_consistency.py — Stock-flow accounting identities

Checks that the fleet's internal accounting is self-consistent at every simulated year
(2025–2050) across all vehicle types and powertrains. These are mathematical identities
that must hold regardless of parameter values — a failure here indicates a bug in the
aggregation code, not a calibration issue.

**How it runs:** one shared Fleet fixture (median params) used by all 7 tests.

| Test | What it checks |
|------|---------------|
| `test_total_stock_equals_cohort_sum` | `total_stock[k,p,t]` equals the direct sum of `stock[k,p,y,t]` over all cohort years. Verifies `_aggregate()` is doing the sum correctly. |
| `test_sales_equal_age_zero_stock` | `sales[k,p,t]` equals `stock[k,p,t,t]` — the age-0 cohort in year t. Both come from the same dict entry so this is a consistency check between two output arrays. |
| `test_fleet_meets_activity_requirement` | Total tonne-km delivered by all surviving cohorts equals `activity_req[k,t]`. Tolerance is 1% because new purchases are sized to close the activity gap and floating-point accumulation is non-trivial. |
| `test_fuel_usage_matches_cohort_aggregation` | `fuel_usage[k,fuel,t]` equals the sum of `stock × annual_fuel[fuel][age]` over all cohorts. Checks every fuel type that appears in any vehicle. |
| `test_emissions_match_cohort_aggregation` | Same check for `emissions[k]` — three streams: embodied (manufacturing), supply (upstream), and use (tailpipe). |
| `test_market_shares_sum_to_one` | Sum of market shares over all powertrains equals 1.0 every year. The iterative production-cap loop must not leak or double-count share. |
| `test_rollover_applies_conditional_survival` | `stock[k,p,y,t]` equals `stock[k,p,y,t-1] × survival[age] / survival[age-1]`. Checks the marginal-decay rollover formula used in `Fleet._run()`. |

```
pytest verification/test_fleet_consistency.py -v
```

---

## test_limits.py — Logit boundary cases

Tests `_calculate_market_share()` against outcomes that are analytically known
without needing calibrated parameter values. Uses a minimal duck-typed mock object
so the real method is called in isolation, without constructing a full Fleet.

**How it runs:** most tests call `_calculate_market_share` directly on a mock;
two tests use a full `Fleet` with `exclude_powertrains` to isolate diesel.

| Test | What it checks |
|------|---------------|
| `test_equal_npv_gives_uniform_shares` | When all N powertrains have identical NPV, every share must be exactly 1/N. Tested for N=2, 3, 7. |
| `test_zero_lambda_gives_uniform_shares` | As price_lambda→0, exp(λ·NPV)→1 for all powertrains, collapsing to uniform shares regardless of NPV spread. |
| `test_higher_npv_gets_higher_share` | The powertrain with the highest NPV gets the largest share — verifies the sign of price_lambda in the logit. |
| `test_single_powertrain_gets_full_market` | With only one powertrain, its share must be exactly 1.0. |
| `test_non_binding_cap_leaves_shares_unchanged` | When the production cap is above the unconstrained logit share for every powertrain, the iterative loop must converge on the first pass and return plain logit values. |
| `test_zero_init_limit_caps_powertrain_to_zero` | A powertrain with `init_market_limit=0` and no prior market presence must get 0% share even with a large NPV advantage. The sole uncapped powertrain inherits 100%. |
| `test_diesel_gets_full_market_when_sole_powertrain` | Full Fleet run with all non-diesel powertrains excluded — diesel must be 100% every year. |
| `test_no_zev_stock_when_powertrains_excluded` | Following from above, total_stock for every excluded powertrain must be 0 in every year. |

```
pytest verification/test_limits.py -v
```

---

## test_vehicles.py — Per-vehicle calculated quantities

Unit tests on the `Vehicles` class methods. Uses a real Fleet run at median params
rather than constructing minimal param dicts from scratch — this tests against the
actual calibrated model and avoids fragile fixture plumbing.

**How it runs:** one Fleet fixture shared across all tests; a second `vehicles` fixture
returns all (k, p) Vehicles objects at START_YEAR.

### Mass (4 tests)

| Test | What it checks |
|------|---------------|
| `test_unloaded_mass_positive` | Unloaded mass > 0 for all (k, p). |
| `test_total_mass_exceeds_unloaded` | total_mass >= unloaded_mass at every age (payload adds, never subtracts). |
| `test_sleeper_diesel_mass_in_plausible_range` | Sleeper diesel loaded mass between 20–55 t. Wide bounds accommodate the known mass underestimate (target 36–40 t — see benchmarks). |
| `test_zev_payload_positive` | BEV and FC payload > 0. Does not compare to diesel — GVWL exemptions may allow ZEVs higher payload in future. |

### Fuel consumption (5 tests)

| Test | What it checks |
|------|---------------|
| `test_fuel_consumption_positive` | All fuel consumption values > 0 for all (k, p, fuel). |
| `test_dice_has_only_diesel` | DICE vehicles have exactly one fuel key: `diesel`. |
| `test_hice_has_only_h2` | HICE vehicles have exactly one fuel key: `h2`. |
| `test_dhice_has_diesel_and_h2` | DHICE vehicles have exactly two fuel keys: `diesel` and `h2`. |
| `test_be_has_a_charge_fuel` | BEV vehicles have at least one fuel key containing `charge`. |
| `test_dhice_fuel_proportions_match_params` | The energy split between diesel and H2 for DHICE matches the proportions declared in data.json, within LHV conversion rounding. |

### Range (2 tests)

| Test | What it checks |
|------|---------------|
| `test_range_positive` | Range > 0 for all (k, p) at every age. |
| `test_be_range_less_than_diesel_range` | BEV range at age 0 is less than diesel range — BEV has a smaller effective tank. |

### Annual distance (2 tests)

| Test | What it checks |
|------|---------------|
| `test_annual_distance_positive` | Annual distance > 0 for all (k, p, age). |
| `test_annual_distance_does_not_exceed_target` | Annual km never exceeds the target distance (with 1% floating-point margin). The time budget is fixed — en-route charging/refuelling stops cannot add unbounded km. |

### FC replacements (1 test)

| Test | What it checks |
|------|---------------|
| `test_non_fc_powertrains_have_no_replacements` | DICE, HE, BE, HICE vehicles have fc_replacements == 0 at every age. Only PHE and FC can have stack replacements. |

### Emissions (4 tests)

| Test | What it checks |
|------|---------------|
| `test_embodied_nonnegative_and_positive_at_purchase` | Embodied emissions > 0 at age 0 (manufacturing); >= 0 at all ages. Ages > 0 may be nonzero once FC stack replacement embodied emissions are added. |
| `test_zev_has_zero_tailpipe_emissions` | All ZEV powertrains have emissions_use == 0 at every age. |
| `test_diesel_has_positive_tailpipe_emissions` | DICE has emissions_use > 0 at every age. |
| `test_supply_emissions_nonnegative` | Upstream (well-to-tank) emissions >= 0 for all (k, p). |

### Discount formula and TCO (3 tests)

| Test | What it checks |
|------|---------------|
| `test_discount_matches_survival_weighted_formula` | `_discount(C)` equals `C × Σ survival[a] / (1+r)^a` — the exact closed-form annuity. Tests against sleeper diesel. |
| `test_tco_equals_sum_of_discounted_cost_components` | TCO equals `Σ _discount(arr)` over all entries in `annual_cost`. Checks that the TCO is a complete decomposition. |
| `test_npv_equals_discounted_revenue_minus_tco` | NPV equals `_discount(revenue) - tco`. |
| `test_diesel_tco_in_plausible_range` | Sleeper diesel TCO between $200k–$2M. Wide bounds; see benchmarks for the tighter literature comparison. |

```
pytest verification/test_vehicles.py -v
```

---

## sensitivity.py — Elasticity diagnostics

Perturbs three parameters by +/-20% and reports the effect on BEV market share in 2040.
The key trick: after a baseline Fleet run, `fleet.params` contains already-realised
scalar/array values. Deep-copying and modifying these (with `param_cps={}`) creates a
perturbed fleet without touching the distribution specs in data.json.

| Parameter perturbed | Expected direction | Why |
|---------------------|-------------------|-----|
| Battery cost ($/kWh) | Lower cost → higher BEV share | Cheaper battery improves BEV NPV |
| H2 fuel price ($/kg) | Higher price → higher BEV share | Pricier H2 hurts FC/HICE relative to BEV |
| price_lambda | Sign depends on BEV NPV vs field average | Higher lambda sharpens the logit; dominant powertrain gains |

Reports elasticity via central difference: `(share_hi - share_lo) / (share_base × 2 × 0.20)`.
Shares are printed for all three vehicle types; elasticity is computed on sleeper.

```
python verification/sensitivity.py
```

---

## benchmarks.py — Literature range checks

Compares 8 model outputs to literature ranges and prints PASS/FAIL. All quantities come
from the sleeper diesel (or BEV/FC) Vehicles object at START_YEAR, median params.
FAIL means the quantity is outside the range and warrants investigation — the script
always completes successfully.

| Quantity | Literature range | Source |
|----------|----------------|--------|
| Loaded mass — sleeper diesel | 36–40 t | GVW regs, NRCan |
| Fuel consumption — sleeper diesel | 35–45 L/100km | NRCan, DOE |
| Annual distance — sleeper diesel | 150,000–200,000 km/yr | StatCan, NRCan |
| Capital cost — sleeper diesel | $180k–$250k CAD | Industry reports |
| TCO — sleeper diesel | $350k–$600k CAD | Literature TCO studies |
| BEV range — sleeper | 300–500 km | Tesla Semi, eCascadia OEM specs |
| BEV / diesel capital ratio | 2–3× | BloombergNEF, ICCT |
| FC / diesel capital ratio | 3–5× | ICCT, H2 Council |

Note: the TCO literature range ($350k–$600k) typically covers a 5–10 year horizon and
may exclude driver costs. The model runs 25 years including driver wages, so a direct
comparison requires care.

```
python verification/benchmarks.py
```

---

## snapshot.py — Regression snapshot

Saves 1820 fleet output values (market shares, total stock, fuel usage, emissions, and
system costs across all vehicle types, powertrains, and years) to `snapshot.npz`.
Run `save` before making code changes, `check` after, to catch accidental numerical shifts.
Tolerance is 0.01%. A mismatch on a functional change (parameter correction, calculation
fix) is expected — re-run `save` to update the baseline in that case.

```
python verification/snapshot.py save    # before changes
python verification/snapshot.py check   # after changes
```

---

## To do

<!-- Add implementation items here -->
