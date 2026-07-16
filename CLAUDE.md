# fleet2 — HDT Fleet Adoption Model (Paper 2)

Multinomial logit model of heavy-duty truck (HDT) fleet adoption in BC, Canada, 2025–2050.
Rewrite of the Paper 1 model (`fleet/`). The key architectural fix over Paper 1: shared component
parameters (e.g. ICE efficiency) are drawn once per Monte Carlo run and shared across all
powertrains that contain that component, rather than sampled independently.

## Scope

- **Vehicle types:** `sleeper`, `day_cab`, `straight` (Class 8)
- **Powertrains:** `dice`, `he` (mild hybrid), `phe` (plug-in hybrid), `be`, `fc`, `hice`, `dhice`
- **Fuels:** `diesel`, `h2` (electrolysis), `h2_p` (pyrolysis), `h2_pe` (electrified pyrolysis), `fast_charge`, `slow_charge`, `public_slow_charge` (non-depot slow charging, e.g. sleeper PHE charging during a rest break; BC Hydro Fleet Demand Transition Rate)
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
| `verification/global_sensitivity.py` | SRRC global sensitivity diagnostic from saved `run.py` MC output; rank-regression bar chart per metric/scenario/year |
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

Output lands in `results/<scenario>.npz` (shape `(n_runs, 26)` float32 arrays, plus
`_mc_cp_samples` — the `(n_runs, n_cols)` array of per-run `cp` draws actually used,
row-aligned with every other array) and `results/<scenario>_meta.json` (n_runs, seed, tol,
wall_time_s, `col_labels`, `zero_variance_cols`, `n_cols`).  Load with:
```python
import numpy as np
d = np.load('results/baseline.npz')
zev_sleeper = d['zev_stock_sleeper']   # shape (n_runs, 26)
```

`verification/global_sensitivity.py` reads these files directly for an SRRC (rank-regression)
sensitivity diagnostic — see the Monte Carlo architecture section below.

## Regression snapshot

Before making any code changes, run:
```
C:\Users\ivana\anaconda3\python.exe verification/snapshot.py save
```
After changes, run:
```
C:\Users\ivana\anaconda3\python.exe verification/snapshot.py check
```
This compares 2236 fleet output values and reports any that shifted by more than 0.01%. If outputs change unexpectedly, investigate before proceeding. `verification/snapshot.npz` is the saved baseline — overwrite it intentionally when a change is deliberate.

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
- [x] `Fleet._calculate_market_share()` — complete: nested logit (`NEST_TREE`) + production caps enforced via shadow pricing (see "Market-share allocation architecture" below)
- [x] `Fleet._aggregate()` — complete: total_stock, sales, fuel_usage, emissions, system_costs
- [x] `plots/vehicle_plots.py` — per-cohort sanity checks; `plots/fleet_plots.py` — fleet-level line plots
- [x] `policies.py` — `CarbonTax`, `GVWLExemption`, `LCFS`, `ZEVMandate`, `Policies` container; comparison plots in `plots/policy_plots/`
- [x] `scenarios.py` — 7 named policy scenarios (`baseline`, `carbon_tax`, `lcfs`, `lcfs_bc_eer`, `zev_mandate`, `gvwl`, `full_policy`)
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
                    NEST_TREE (nested-logit tree, see "Market-share allocation
                    architecture" below), _NEST_TREE_LEAVES,
                    _YEAR0 (= START_YEAR - MAX_AGE, first year in all realised arrays)

Vehicles class:     _calculate_mass, _calculate_fuel_consumption,
                    _split_surrogate_output, _calculate_range,
                    _calculate_annual_distance, _track_fc_replacements,
                    _calculate_emissions, _cap_cost, _op_cost_array,
                    _calculate_capital_cost, _calculate_annual_cost,
                    _discount, _calculate_tco_npv

Fleet class:        _make_vehicle, _apply_mandate_credit, _build_initial_stock,
                    _system_cost_per_tkm, _new_vehicle_cost_per_tkm, _run,
                    _calculate_market_share, _aggregate,
                    select_vehicle_params, realise_uncertainties
                    (self._mu_warm_start: (k, p) -> float, persists shadow costs across
                    _calculate_market_share calls -- see "Market-share allocation
                    architecture" below)

Module functions:   _market_share_limit (production cap helper, unchanged),
                    _prune_tree, _bottom_up_utility, _top_down_shares (nested-logit
                    tree evaluator, called from inside _shadow_price_shares),
                    _leaf_ancestor_path, _recompute_path_utilities, _path_share
                    (single-leaf incremental tree evaluator, used by
                    _shadow_price_shares's bisection instead of _bottom_up_utility/
                    _top_down_shares to avoid re-evaluating unaffected sibling nests
                    -- see "Performance" below),
                    _shadow_price_shares (shadow-price solver -- see below)
```

## Market-share allocation architecture (`Fleet._calculate_market_share`, `_shadow_price_shares`)

**Status: nested logit + shadow pricing is done and verified (snapshot re-saved with the real
tree lambdas; see below for the verification sequence and what it found).**

Utility for powertrain `p` is `price_lambda * NPV(p)` (higher NPV = more desirable; `price_lambda`
is positive, `params.py: fleet.price_lambda` -- now MC-varied, see "Key params.py fields to know").
Production caps (`_market_share_limit()`, unchanged from before — ratchets a share cap off last
year's own share via `init_market_limit` / `cagr_nacent` / `cagr_mature`) used to be enforced by
**clipping**: run the flat logit, pin any powertrain that exceeds its cap at the cap value, remove
it, and re-run the logit over the survivors, repeating until nothing new binds (≤10 passes). This
was replaced with **shadow pricing**: a shadow cost `mu_p >= 0` is solved for every powertrain so
that `V_p = price_lambda * (NPV(p) - mu_p)` enters the choice set as a leaf utility (nobody is ever
removed from the tree), with complementary slackness — `mu_p > 0` only where the cap binds exactly.
Under a single flat logit, clipping and shadow pricing are provably equivalent (IIA: a third
powertrain's relative odds against the others don't depend on *how* a capped powertrain's excess
demand is discouraged, only on the share it ends up with) — this was confirmed by an exact match
(2236/2236 values within 0.01%) against the pre-shadow-pricing snapshot, back when the choice set
was still flat. They do **not** stay equivalent once nesting is live — that's the actual reason
shadow pricing was built before nesting: a capped leaf that's *removed* rather than *discounted*
leaks a wrong (inflated) inclusive value up through its nest, distorting every sibling nest's share.
That distortion is largest exactly when several members of the same nest are simultaneously and
severely capped — i.e. nascent ZEV/H2 technology in the early simulation years, the part of the
trajectory the whole model exists to get right.

**The nested tree** (module-level `NEST_TREE` in `model.py`, a plain nested-tuple structure —
`(name, lambda_key, children)` for non-leaf nodes, bare powertrain strings for leaves; the tree
shape is code, not a `params.py` value, since it encodes a structural/modeling decision with no
number to cite, same status as `ZEV_POWERTRAINS`):

```
Liquid (lambda=0.7)
  Conventional (lambda=0.4): dice, he
  phe
Hydrogen (lambda=0.6): fc, hice, dhice
Electric (lambda=1.0): be
```

`dhice` sits in Hydrogen (not Liquid) despite being a 75%-diesel/25%-H2 dual-fuel ICE — grouped
with fc/hice by ZEV-adjacent substitution pattern, not by fuel share. The four lambda *values*
(0.7/0.4/0.6/1.0) live in `params.py: fleet.nest_lambdas` as plain `P(value, src="UNREF --
assumption")` scalars — not yet calibrated from data.

**Math convention:** McFadden nested logit, `U_n = lambda_n * ln(sum_c exp(U_c / lambda_n))`
feeding each node's utility up to its parent (leaf utility `V_p` at the bottom), with
`P(c|n) = exp(U_c/lambda_n) / sum exp(U_c'/lambda_n)` distributing probability back down; the
root (parent of Liquid/Hydrogen/Electric) has its own scale fixed at 1.0 (nothing sits above it to
rescale against, so it isn't a `params.py` value). This specific convention — scaling the
log-sum-exp by `lambda_n` before handing it to the parent, rather than passing a raw un-scaled
inclusive value up — is what makes an all-`lambda=1` tree collapse to exactly the flat softmax at
any depth (verified both by hand and empirically: `verification/test_limits.py`'s
`test_all_nest_lambdas_one_matches_flat_mnl`, and a full-`Fleet()` run with `nest_lambdas` forced
to all-1.0 matched the pre-nesting snapshot exactly before the real lambdas were ever turned on).
A singleton nest (e.g. Electric = `{be}` alone) needs no special-case code — `U_n` collapses to
`V_be` algebraically for any `lambda_n`, so a nest that *becomes* a singleton via
`exclude_powertrains` or zero-cap pruning (e.g. `he` excluded, leaving `dice` alone in
Conventional) also collapses correctly for free. Zero-cap leaves are pruned out of the tree
entirely before evaluation (not just zeroed in final probability) so they can't pollute their
former nest's inclusive value — see `_prune_tree`/`_shadow_price_shares`.

**How to think about the lambda values:** each is a knob on how much a nest's members are seen as
variants of the same underlying choice vs. genuinely independent alternatives.  `lambda=1` = no
family effect (behaves like flat MNL locally). `lambda -> 0` = near-perfect substitutes — a shift
between two nest-mates mostly reallocates share *between them*, and the nest avoids the classic
red-bus/blue-bus over-counting (two near-identical options don't out-compete a dissimilar one just
by being two draws instead of one). Concretely: dice/he (lambda=0.4, tightest) are "the same truck,
optionally hybridized" — a trim choice, not a technology bet. phe joins them under Liquid
(lambda=0.7, looser) since it's still liquid-fuelled but a more distinct drivetrain. fc/hice/dhice
(lambda=0.6) share H2 supply-chain/infrastructure risk. Electric's lambda is a placeholder (no
effect while it's a singleton). One clean way to see the effect empirically: within a nest, the
effective sensitivity comparing two members is `price_lambda/lambda_n` (2.5x for Conventional) —
this is exactly why nesting pushed already-losing same-nest powertrains (e.g. residual `dice` share
against a dominant `he`) further toward zero relative to the pre-nesting numbers; across nests, the
root's own scale stays fixed at 1.0, so top-level comparisons aren't sharpened, only each nest's
internal "diversity bonus" is corrected.

**Solver (`_shadow_price_shares`, `model.py`):** Gauss-Seidel sweeps — one powertrain's `mu` at a
time, by **bisection**, holding every other powertrain's `mu` fixed at its current value, cycling
through all powertrains repeatedly. Bisection (not a joint Newton/multiplicative fixed-point step)
was chosen because `share_p(mu_p)`, holding everything else fixed, is monotonically
non-increasing — bisection can't overshoot regardless of how close a share is to saturating at 0
or 1. This still holds under nesting: raising `mu_p` strictly lowers leaf `p`'s own utility, which
strictly lowers its within-nest conditional share and non-decreasingly lowers its nest's inclusive
value relative to sibling nests — a product of factors each non-decreasing (one strictly
increasing) in that utility can't increase as the utility falls, at any tree depth. An earlier
joint-Newton version overshot badly and oscillated once several powertrains were
simultaneously and severely capped (residual pinned at a large constant, never shrinking) — the
same lesson this codebase already learned once for the ZEV-mandate credit-price search (bisection
over a damped blend, `policies.py`/`Fleet._run`, because a damped blend oscillated on a steep
transition there too).

**A real bug worth remembering if this solver is touched again:** the convergence check must
verify full complementary slackness, not just feasibility. Checking only `share_p <= cap_p` is not
enough — with a warm-started `mu` (see below), an early powertrain in the sweep order can be
judged against *other* powertrains' stale, not-yet-updated values and pick up an unwarranted
`mu > 0`; once those others are corrected later in the same sweep, the wrongly-capped one's share
settles comfortably *under* its own cap, which passes a feasibility-only check even though it
should have relaxed back to `mu == 0`. This produced answers up to ~99% wrong in one real (k, t)
case before being caught by the snapshot check. The fix: for every powertrain with `mu > 0`,
require `share_p == cap_p` (not just `<=`), which forces another sweep instead of accepting that
self-consistent-but-wrong state. Trust the snapshot check over "no warnings raised" when validating
any future change here — the buggy version also produced *zero* convergence warnings.

**Warm-starting:** `Fleet._mu_warm_start` (keyed `(k, p)`) persists each powertrain's converged
shadow cost across calls — both across the ZEV-mandate bisection's ~30 calls/year (which only
shift NPV slightly as the credit price changes) and across years. This changes nothing about the
converged answer (still verified against the snapshot), only how many sweeps it takes to reach it.

**Performance** (`Fleet(PARAMS, param_cps, policies=...)`, cp=0.5, directly measured not
extrapolated): baseline 0.96 s, `zev_mandate` 3.86 s, `full_policy` 6.57 s (2026-07, current
codebase). The `zev_mandate`/`full_policy` figures were previously found to be ~33 s /
correspondingly higher (see git history) — an **8.4x** total speedup on `zev_mandate`
(32.6 s -> 3.86 s), from three sequential fixes, all verified bit-compatible with the pre-change
snapshot (2236/2236 within 0.01%):

1. **Plain-Python log-sum-exp/softmax.** cProfile showed `_bottom_up_utility`/`_top_down_shares`
   (the NEST_TREE evaluator) spending ~80% of runtime in numpy array-reduction overhead
   (`np.array`/`.max()`/`np.sum`/`np.exp`) on per-node child arrays of only 1-7 elements, called
   ~3.6M times across the ~300k `shares_at` evaluations one `zev_mandate` build makes. Numpy's
   per-call dispatch overhead dwarfs the actual arithmetic at that array size. Rewrote both
   functions with plain-Python `math.exp`/`math.log`/`max`/`sum` instead — 3.3x alone
   (32.6 s -> 9.8 s).
2. **Incremental bisection (avoid re-evaluating unaffected sibling nests).** `share_of`'s
   bisection of one leaf's `mu` used to call `shares_at`, which re-evaluated the **entire** tree
   on every probe (up to `max_bisect`=40 times per leaf), including sibling nests wholly
   unaffected by that leaf's `mu`. Replaced with `node_utils_cache` (module-level
   `_leaf_ancestor_path`/`_recompute_path_utilities`/`_path_share`): a persistent per-node
   utility cache kept consistent with the current `mu` at all times (every leaf commits its
   final value into it after its own bisection), so a probe only recomputes that leaf's own
   ancestor chain (root -> ... -> leaf, typically 2-3 nodes) and reuses cached values for every
   sibling — a further 2.2x (9.8 s -> 4.5 s). Profiling a **baseline** build (no mandate at all)
   after fixes 1-2 showed `_shadow_price_shares` was still ~70% of total cost -- most non-`dice`
   powertrains start nascent (`init_market_limit=0.02`), so multi-cap bisection is routine even
   without an active mandate, not just under one.
3. **`max_bisect` 40 -> 30.** 40 bisection iterations narrows the mu search interval by 2^40 --
   far more precision than `tol=1e-5` needs. Tested 40/30/25/20: 20 broke (`RuntimeError`,
   genuine non-convergence, confirming 40 wasn't pure padding); 25 and 30 both passed a 50-draw
   stress test (40 baseline + 10 `zev_mandate` random draws, 0 failures each). Chose 30 over 25
   for margin from the observed failure point at 20, trading a few points of speedup for
   distance from a known failure mode -- ~10% further (4.5 s -> ~4.0 s on `zev_mandate`).

Confirms this was **not** fundamentally a solver-convergence problem: `max_sweeps` (300 vs 2000)
made zero measurable difference at median parameters before fix 1, and a 3x tolerance loosening
only bought ~10% -- the cost was call-count x per-call overhead, not sweep count. `max_bisect`
is a genuine (if narrow) precision/speed tradeoff, unlike fixes 1-2 which were pure "same math,
faster" wins.

A structurally different, much larger further win remains **unimplemented and undecided**:
replacing the hard production-cap + shadow-pricing mechanism with a smooth acquisition-friction
penalty computed from **prior-year** state only (`prev_share`, `cagr_nacent`, etc. -- all known
before the year's logit runs, same pattern already used for `revenue_per_tkm`), which would
eliminate the iterative bisection entirely rather than just speeding it up. This is a genuine
re-specification of production-constraint economics (soft frictions vs. hard caps), not a
performance refactor -- would need a defensible functional form/calibration, a deliberately
re-saved snapshot (results should shift), and a validation pass against current trajectories
before being implemented. Discussed 2026-07; not started.

**Convergence tuning (`max_sweeps=2000`, `tol=1e-5`):** originally `max_sweeps=200`, `tol=1e-7`.
Once `fleet.price_lambda` gained a `dist` (see "Key params.py fields to know"), the median run
(cp=0.5) started drawing `price_lambda=0.00002` instead of the old fixed `0.00003`, and one
early-year sleeper case (four powertrains simultaneously capped across three different nests —
`he`/Conventional, `phe`/Liquid, `be`/Electric, `fc`/Hydrogen) needed 262 sweeps to clear the old
`1e-7` tolerance. Traced sweep-by-sweep: the gap shrank monotonically and geometrically the whole
way (not the oscillation failure mode described above) — nesting adds an extra coupling channel
between simultaneously-capped powertrains in *different* nests via their shared parent's inclusive
value, on top of a lower `price_lambda` flattening the logit and requiring larger `mu` excursions
to hit the same cap. `tol=1e-5` is still 10x tighter than `verification/snapshot.py`'s own
`RTOL=1e-4` materiality threshold, so any leftover imprecision stays invisible to what the project
already treats as "a real change" -- **kept unchanged rather than loosened** (see below).
`max_sweeps` is pure headroom (can't change the converged answer, only whether the loop reaches it
in time) and costs nothing on the vast majority of calls that already converge in single digits
via warm-starting.

`max_sweeps=300` (the value chosen alongside the `1e-7`->`1e-5` change above) turned out to still
be too tight once non-convergence was changed from a swallowed `warnings.warn` to a hard `raise`
(2026-07 audit -- see "A real bug worth remembering" below for why silently returning a
cap-violating result on non-convergence was itself the bug being fixed): a 30-random-draw
reproduction across the full uncertainty space (not just `price_lambda`'s extremes) hit genuine
non-convergence at `max_sweeps=300` in 1/30 trials (~3%), with gaps up to `8.17e-04` -- 8x
*larger* than the snapshot's own `1e-4` materiality bar, i.e. large enough that silently accepting
it (via a loosened `tol`) could have changed a downstream number the project would otherwise flag
as real. Verified the gap is genuine slowness, not a harder failure: raising `max_sweeps` alone
(no `tol` change) to 2000 cleared all 30 draws with zero non-convergence, confirming (as the
solver's own monotonic-non-increasing proof predicts) these are legitimately slow-converging
points, not oscillation or infeasibility -- more sweeps is strictly the correct lever here, since
it preserves the existing 10x precision margin everywhere rather than trading it away just to
paper over the rare slow case.

**Verification sequence actually used** (worth repeating if this solver or the tree is touched
again): (1) confirm the pre-change snapshot passes; (2) implement; (3) force all four
`nest_lambdas` to 1.0 *in memory only* (never edit `params.py` for this step) and check against
the **old, not-yet-resaved** snapshot — this is the strongest end-to-end proof the tree evaluator
is wired correctly, since it exercises the full `Fleet()` simulation, not just hand-picked unit
tests; (4) revert to real lambdas, run `verification/test_limits.py`; (5) check the snapshot with
real lambdas — expect genuine, broad mismatches now (not a bug) and inspect only for plausibility
(shifts concentrated in same-nest ZEV/H2 competition and the Liquid-branch restructuring, no sign
flips or blow-ups); (6) re-save intentionally once satisfied.

## Policy architecture (`policies.py`)

Two-phase hook interface called from `Fleet._make_vehicle()`:
- **`pre_apply(params, k, p, t)`** — modifies the params dict *before* `Vehicles()` is constructed. For physics policies that propagate through mass → FC → cost (e.g. GVWL exemption).
- **`apply(v)`** — writes cost terms into `v.annual_cost` *after* construction, then calls `v._calculate_tco_npv()` once. For cost-only policies (carbon tax, LCFS).

Endogenous policies (ZEV mandate) run in an outer convergence loop inside `Fleet._run()`, not via the hooks. Each vehicle sold is worth `credits_per_vehicle[k]` ZEV credits (default 2.5). `ZEVMandate.credit_price(target, p_zev)` is a smooth logistic function of the compliance gap (~`penalty_max` deep below target, collapsing toward 0 at/above target — no hard cliff). Since `credit_price(target, p_zev)` is monotonically decreasing in `p_zev` and the market's share response to price is monotonically non-decreasing, their composition minus `p_zev` is monotonic with at most one root — so the loop solves for the fixed point via **bisection on `p_zev` in `[0, 1]`** (probe the bracket midpoint, apply the implied price, measure the resulting ZEV share, narrow the bracket by comparison, repeat) rather than a damped linear blend. This is robust to how steep the credit-price transition is (a narrow transition band made a damped-blend version oscillate near the target; bisection doesn't, since it only ever uses the sign of the gap, not its magnitude). The mandate applies economic pressure and the market settles wherever the bracket converges; the loop is **not** a search that stops the instant the target is hit. Production-cap-bound years converge the same way (the bracket still narrows, just to a `p_zev` below target) — this is not a special case. Only warns on true numerical non-convergence (30-iteration limit without the bracket narrowing below `1e-4`). No cross-year warm-start (each year bisects the full `[0, 1]` range from scratch, ~14 iterations to converge) — this matches Paper 1 (`old/model_old.py:885-915`, which also resets its penalty search fresh every year) more closely than fleet2's previous warm-started version did. Paper 1's mechanism is a hinge-clamped linear formula with the same "only exit is the search variable stabilizing" property; this replaces the hinge with a smooth logistic and per-vehicle credits in place of a flat $/vehicle penalty.

`Fleet._apply_mandate_credit(t, credit_price, target, p_zev, k=None)` writes the actual dollar amounts and is bounded so the government never pays out more than it collects. Non-ZEVs always owe a flat `credits_per_vehicle[k] * credit_price * target` (their own share of the obligation, independent of population split). The total collected from non-ZEVs is a pool; ZEVs are paid the flat market rate for their own credits (`credits_per_vehicle[k] * credit_price`) if the pool covers it, otherwise payouts ration down proportionally (`min(1, target * p_nonzev / p_zev)`) so total payout never exceeds the pool. Net revenue is `>= 0` **at the bisection's converged fixed point** (within its `1e-4` bracket-width tolerance — `ration` is computed from the probe `p_zev`, not the realized stock split, so it can drift marginally negative mid-search) — positive when ZEV supply is undersized relative to target, `~0` once ZEV supply is abundant enough to exhaust the pool. This also assumes `credits_per_vehicle` is uniform across `k` under `scope='fleet'`, since `ration` is computed once from the fleet-wide `p_zev` but applied with each `k`'s own `credits_per_vehicle`.

`COST_CATEGORIES` in `model.py` drives `_aggregate()`:
```python
COST_CATEGORIES = {
    'system': ('capital', 'operational', 'fuel', 'driver', 'fc_replacements'),
    'policy': ('carbon_tax', 'lcfs', 'zev_mandate'),
}
```

**LCFS calibration:** `LCFS.apply()` computes cost per fuel directly from CI, EER, and LHV terms (see `policies.py`) — no baseline-diesel calibration step is needed. An earlier `set_baseline_fc` throwaway-vehicle mechanism (built once per `k` in `Fleet.__init__` solely to feed it) was removed as dead code in the 2026-07 audit once the LCFS formula moved to this energy-basis calculation; `set_baseline_fc` had already been a no-op for some time before that. LCFS has non-zero costs for diesel from 2025 onwards (target starts at 18.3%).

**ZEV mandate scopes:**
- `scope='fleet'` — ZEV share target applies across all k combined; `targets = {year_str: fraction}`
- `scope='per_k'` — independent target per vehicle type; `targets = {k: {year_str: fraction}}`

`credits_per_vehicle` (default `{'sleeper': 2.5, 'day_cab': 2.5, 'straight': 2.5}`) and
`transition_width` (default `0.02`, i.e. credit price is ~95%/~5% of `penalty_max` at a
2-percentage-point deficit/surplus) are constructor kwargs on `ZEVMandate`, set per scenario
in `scenarios.py` like `targets`/`penalty`/`scope` — not calibrated params.py values.

## Key params.py fields to know

- `settings`: max_age=25, start_year=2025, end_year=2050, discount_rate=0.08, growth_rate=0.02
- `fleet`: initial_activity, activity_growth=0.02, autonomous_t50=2040
- `fleet.price_lambda`: `dist: uniform, min=0.00001, max=0.00003` (MC-varied; median/cp=0.5 draw
  is 0.00002, not the old fixed 0.00003 — the upper bound is the original Table 7 point estimate,
  the lower bound is `UNREF -- assumption`, "a finger-in-the-air number")
- `fleet.nest_lambdas`: `{liquid: 0.7, conventional: 0.4, hydrogen: 0.6, electric: 1.0}` — fixed
  (no `dist`), see "Market-share allocation architecture" above
- `vehicles.components`: shared component defs (`converter`, `ess`, `transmission`) — each powertrain references these by type to avoid independent MC draws
- Per-powertrain: `init_market_limit` (1.0 for dice, 0.02 for others), `cagr_nacent`, `cagr_mature`
- Surrogate mapping: `hice` reuses the `he` surrogate (all 5 drive cycles); `dhice` reuses the `dice` surrogate, not `he` (`dhice` has no motor/battery, so it's modelled as a non-hybrid H2 ICE — `model.py: SURROGATE_NAME`); `phe` only has `udds_hdt`/`cruise_hdt` (no haul-specific files)
- `hice` is modelled as a hybridised H2 ICE: motor (220 kW), battery (10/5/5 kWh), regen_efficiency=0.71, accessory_load=3400 — mirrors `he` component set with H2 tank instead of diesel tank

## params.py conventions

- Every numeric leaf is wrapped: `P(value, src="Author, Year", units="unit", notes="optional")`
- `src=None` or a string starting with `"UNREF --"` flags values without a paper citation
- Container dicts (those without a `"dist"` or `"array"` key) are left as plain Python dicts
- `data.py` calls `_strip_params()` to remove all `Param` wrappers before passing to `_expand_arrays()`
- `documentation/build_appendix.py` reads `PARAM_DICT` directly to generate supplementary tables
- To update a parameter: edit the value and/or `src` in `params.py`, then run `python documentation/build_appendix.py`

## Next steps

1. ~~Nested logit~~ — **done.** `NEST_TREE` + `fleet.nest_lambdas` implemented, verified (degenerate
   all-lambda=1.0 case matched the pre-nesting snapshot exactly; real lambdas re-saved
   intentionally after inspecting for plausibility), and `fleet.price_lambda` now varies in MC
   (`dist: uniform, min=0.00001, max=0.00003`) — see "Market-share allocation architecture" and
   "Key params.py fields to know" above. Contrary to the original prediction here, none of the 7
   pre-existing `verification/test_limits.py` cases actually needed changes — every one uses at
   most one real powertrain per top-level branch, which collapses to flat MNL regardless of
   lambda (traced explicitly). 6 new nested-logit-specific tests were added instead (degenerate
   reduction, within-nest ratio invariance, cross-nest IIA-violation demonstration, singleton-nest
   invariance, zero-cap exclusion from inclusive value, multi-same-nest-binding-cap convergence).
   Discovered along the way: widening `price_lambda`'s range exposed a slow (not oscillating)
   shadow-pricing convergence case at the low end, fixed by loosening `max_sweeps`/`tol` (see
   "Convergence tuning" above) rather than narrowing the range.
2. ~~Check total mass plausibility~~ — **resolved, not a bug.** Sleeper diesel's 28.9 t is the
   correct average *operating* mass at a 16 t payload (12,976 kg unloaded + 15,995 kg payload,
   `model.py:263-291`); the truck's legal max is `gvwl=53,500 kg` (`params.py:563`). The "~36-40 t"
   expectation in this item conflated legal GVW max with typical loaded mass — 36-40 t is a generic
   fully-loaded US Class-8 figure (80,000 lb), not what this BC-specific spec should show at
   average payload. Matches Paper 1 exactly (`old/model_old.py:42`, `old/data_old.py:275` ->
   28,938 kg). Payload is age/drive-cycle-varying (16 t long-haul -> 10 t regional-haul past age 9,
   `data.py:47-51`), which Paper 1 held constant across vehicle life — **confirmed intentional**
   (user sign-off, 2026-07); `fleet.initial_activity` (`params.py:30`) was deliberately recalibrated
   ~3% below its Table 8 point estimate as a consequence, same underlying sources.
3. ~~Verify `model.py` checked up to `_calculate_fuel_consumption`~~ — **done.** `_calculate_range`
   through `_calculate_tco_npv` (`model.py:400-818`) reviewed in full (exhaustive audit, 2026-07):
   no critical/high correctness bugs, all unit-handling checks pass (mass kg-consistent, fuel-consumption
   units match the surrogate, cost math doesn't mix $/vehicle/$/km/$/year, discounting applied
   exactly once via `_discount_factor`). The low-severity items found (stale comment at
   `model.py:525`; `diesel_tank.refuel_rate` unit label at `params.py:392`; `default_unloaded_mass`
   at `params.py:562` 38 kg off from the summed diesel component mass) have since been fixed/flagged.
4. Stage 4 policy improvements: dynamic LCFS credit pricing, ZEV purchase rebate, joint mandate+LCFS convergence (see `POLICY_PLAN.md`)
5. Re-introduce autonomous vehicles (dropped from Paper 1's `AutonomousPermits` policy and the
   `(1-autonomous_penetration)` driver-cost discount, `old/model_old.py:241-251,685`). Confirmed
   future work (2026-07), not a regression — `fleet.autonomous_t50` (`params.py`) is currently
   unused dead config until this lands.

## Performance notes

Single `Fleet()` run at cp=0.5 (2026-07 measurement, current codebase): ~1.1 s baseline, ~4.5 s
`zev_mandate`, ~8.0 s `full_policy` — see "Market-share allocation architecture" above for the
full root-cause (mandate-bisection call volume into the NEST_TREE evaluator) and the two fixes
that brought `zev_mandate` down 7.3x from ~33 s. The dominant cost for scenarios without an active
mandate is still constructing 543 `Vehicles` objects in `_run()` — inherent physics, not easily
vectorised. Further optimisation opportunities (`activity_met` vectorisation; profiling ruled this
out as the mandate-scenario bottleneck, but it's still a real, uninvestigated inefficiency — see
`_run`'s inner bisection loop) are documented in the `verification/profile_fleet.py` docstring
under "Remaining optimisation opportunities".

Monte Carlo throughput with `run.py` at 8 workers: ~4–5× wall-clock speedup over serial.
Baseline scenario (no policies, high distributional uncertainty) converges at ~3 000–4 000 runs
with `--tol 0.05` (~8 min). Policy scenarios converge faster in *run count* (~1 000–2 000 runs)
because the mandate/tax constrains the ZEV adoption distribution; `zev_mandate`/`full_policy` runs
are still individually more expensive than baseline (see per-run costs above) — factor that into
wall-clock estimates for a large MC pass on either scenario rather than assuming baseline's cost.

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

**Output arrays** (82 per-year `(n, 26)` series per scenario, plus the scalar/mandate series below):

| Key pattern | Shape | Content |
|-------------|-------|---------|
| `total_stock_{k}_{p}` | (n, 26) | Total on-road stock by powertrain |
| `sales_{k}_{p}` | (n, 26) | New sales per year |
| `zev_stock_{k}` | (n, 26) | Aggregate ZEV stock |
| `emissions_{k}_{embodied\|supply\|use}` | (n, 26) | Fleet emissions (kgCO2e/yr) |
| `system_costs_{k}_{category}` | (n, 26) | Annual costs ($/yr) |
| `fuel_usage_{k}_{f}` | (n, 26) | Fuel consumed (L, kg, or kWh/yr) |
| `npv_{k}_{p}_{y}` | (n,) | Per-vehicle NPV at cohort year `y` in {2030, 2040, 2050} — scalar per run, **not** a (n, 26) time series |
| `mandate_penalty_frac` | (n, 26) | ZEV-mandate credit price / `penalty_max` per year — only present when a ZEV mandate policy is active |

Plus one extra array, `_mc_cp_samples` (n, n_cols) — the raw `cp` draw used for each surviving
run in each sample-matrix column, row-aligned with all series above via the same `iRun` order
`run_scenario()` accumulates results in. `_meta.json` carries the matching `col_labels` (one
label per column — the `group` name if grouped, else the dotted `params.py` leaf path) and
`zero_variance_cols` (column indices where every mapped leaf is invariant to `cp` — e.g.
`interp`/`const` anchors that are bare values rather than distributions — so they can be
excluded from a sensitivity regression rather than plotted as meaningless near-zero bars).

**Global sensitivity (`verification/global_sensitivity.py`):** a pure post-hoc reader of the
above — no `Fleet`/`ProcessPoolExecutor` dependency. Computes Standardized Rank Regression
Coefficients (SRRC): rank-transform `_mc_cp_samples` and a chosen scalar output metric
(`emissions_total`, `zev_share`, or `system_cost_total`, evaluated at `--year`), standardize,
fit OLS; each input's squared coefficient approximates its share of output variance, and the
regression's own rank-R^2 flags whether that decomposition is trustworthy (SRRC only sees
monotonic marginal effects, not interactions — the ZEV-mandate bisection loop and
production-cap kinks are the likely source of any low R^2). This is an approximate,
cheap-to-run diagnostic for day-to-day development; a full Sobol'/Saltelli variance
decomposition (needs its own A/B/AB sampling design, not the plain random draws here) is
planned separately for pre-publication use and is not implemented.

**Scenarios** (`scenarios.py`): `baseline`, `carbon_tax`, `lcfs`, `lcfs_bc_eer`, `zev_mandate`,
`gvwl`, `full_policy`. `lcfs_bc_eer` is `lcfs` re-run with the BC LCFS technical regulation's
legislated Energy Effectiveness Ratios (`_EER_BC`) instead of the model-derived set (`_EER_MODEL`)
used by `lcfs`/`full_policy` — a direct comparison of model-derived vs. regulatory EER assumptions.
Policy numeric values (tax schedules, LCFS targets, ZEV mandate fractions) all
live in `scenarios.py` — edit there without touching `run.py` or `model.py`.
