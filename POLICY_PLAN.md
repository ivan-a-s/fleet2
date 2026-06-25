# Policy Implementation Strategic Plan — fleet2

## Context

The fleet2 model needs policy layers to simulate BC's HDT adoption trajectory under regulatory scenarios.
Carbon tax is already implemented. Three policies remain: GVWL weight exemption for ZEVs, LCFS (Low Carbon
Fuel Standard), and ZEV Mandate. The goal of Stages 0–3 is functional equivalence with `old/model_old.py`.
Stage 4 improves them (dynamic LCFS credit pricing, joint convergence).

**Key architectural decisions from design session:**
- **Exogenous policies** (carbon tax, LCFS static, GVWL) → `pre_apply()`/`apply()` hooks in `_make_vehicle()`
- **Endogenous policies** (ZEV mandate, future dynamic LCFS) → outer convergence loop inside `Fleet._run()`
- ZEV mandate lives in `_run()`, not in `_calculate_market_share()` (mandate is cross-k; that method is per-k)
- `_calculate_tco_npv()` called once by `Policies.apply()` after all cost terms written — not by individual policies
- New model's `ZEV_POWERTRAINS = {'be', 'fc', 'hice'}`; non-ZEV = `{'dice', 'he', 'phe', 'dhice'}`

---

## Always first: snapshot

```
C:\Users\ivana\anaconda3\python.exe verification/snapshot.py save
```
Run this before touching any code. After each stage:
```
C:\Users\ivana\anaconda3\python.exe verification/snapshot.py check
```
With `policies=None`, outputs must be numerically identical to the pre-change baseline.

---

## Stage 0: Structural prerequisites ✓ COMPLETE

These changes are load-bearing for all subsequent stages. With `policies=None` they must be no-ops.

### 0a. Pre-initialize policy cost slots — `model.py: Vehicles._calculate_annual_cost()` (~line 660)

Add alongside `'carbon_tax'`:
```python
'lcfs':        np.zeros(len(self.age), dtype=np.float32),
'zev_mandate': np.zeros(len(self.age), dtype=np.float32),
```
`_calculate_tco_npv()` already sums `annual_cost.values()`, so these zero slots are automatically included
without changing that method.

### 0b. Update `COST_CATEGORIES` — `model.py` module constants (~line 86)

```python
COST_CATEGORIES = {
    'system': ('capital', 'operational', 'fuel', 'driver', 'fc_replacements'),
    'policy': ('carbon_tax', 'lcfs', 'zev_mandate'),
}
```

### 0c. Make `_aggregate()` data-driven — `model.py: Fleet._aggregate()` (~line 894, 913)

Replace hardcoded cost key tuples with `COST_CATEGORIES`.

`system_costs` initialization:
```python
all_costs = COST_CATEGORIES['system'] + COST_CATEGORIES['policy']
self.system_costs = {k: {c: np.zeros(len(T)) for c in all_costs} for k in self.K}
```

Accumulation loop (replaces the hardcoded tuple at ~line 913):
```python
flow_costs = tuple(c for c in all_costs if c != 'capital')
for c in flow_costs:
    self.system_costs[k][c][i] += n * v.annual_cost[c][a]
```
(`'capital'` is still handled separately at point-of-sale, unchanged.)

### 0d. Fix `CarbonTax.apply()` — `policies.py` (~line 44)

Remove `v._calculate_tco_npv()` from `CarbonTax.apply()`. Move it to `Policies.apply()`, called once
after all cost-writing policies have run:
```python
def apply(self, v):
    if self.carbon_tax:
        self.carbon_tax.apply(v)
    if self.lcfs:
        self.lcfs.apply(v)
    v._calculate_tco_npv()   # single call after all policies
```

---

## Stage 1: GVWL Exemption ✓ COMPLETE

**Mechanism:** For ZEV powertrains only, BC allows additional gross weight. More GVWL headroom
→ higher payload fraction → better revenue/TCO → slightly higher fuel consumption (heavier).

**GVWL increases are policy parameters, not physical constants.** They live in `GVWLExemption`,
not in `data.json`. Remove `gvwl_increase` from each vehicle type's shared params in `data.json`.

**Hook already wired:** `Vehicles._calculate_mass()` reads `self.params.get('gvwl_exemption_kg', 0.0)`.

### `policies.py`: Add `GVWLExemption` class

```python
class GVWLExemption:
    # Default values match BC regulation (kg)
    _DEFAULT_INCREASES = {'sleeper': 5000, 'day_cab': 3000, 'straight': 2000}

    def __init__(self, increases: dict = None):
        self._increases = increases if increases is not None else self._DEFAULT_INCREASES

    def pre_apply(self, params, k, p, t):
        if p in ZEV_POWERTRAINS:
            params['gvwl_exemption_kg'] = float(self._increases.get(k, 0.0))
```

### `policies.py`: Update `Policies.__init__`

Add `gvwl_exemption=None` parameter; call `self.gvwl_exemption.pre_apply(params, k, p, t)` from
`Policies.pre_apply()`.

### `data.json`: Remove `gvwl_increase`

Delete the `gvwl_increase` field from `vehicles.types.sleeper.shared`, `day_cab.shared`, and
`straight.shared`. Verify no other code reads this key (grep for `gvwl_increase` before deleting).

**Verification:** For a ZEV vehicle (e.g., `be`, sleeper), check `v.mass['payload']` increases
vs. no-exemption case. Diesel vehicles should be unaffected.

---

## Stage 2: LCFS (static credit price) ✓ COMPLETE

**Mechanism:** Annual LCFS cost per vehicle =
`annual_distance × (actual_CI_per_km − baseline_CI_per_km × (1 − target[year])) × credit_price / 1000`

- `actual_CI_per_km` = `(v.emissions_supply + v.emissions_use) / v.annual_distance` [kgCO2e/km]
- `baseline_CI_per_km` = diesel CI × baseline diesel FC per km (calibrated from 2025 diesel vehicle)
- `target` = CI reduction schedule, e.g., linearly from 18.3% (2025) to 76% (2050), zero before 2025
- `credit_price` = $/tCO2e (exogenous, fixed)
- Negative cost = credits (revenue for low-CI vehicles); positive = deficits (cost for high-CI)

Old model reference: `old/model_old.py` lines 215–227 (LCFS class), line 852 (baseline calibration),
line 687 (annual cost formula).

### Baseline calibration

`baseline_fc[k]` = diesel fuel consumption per km of the 2025 diesel vehicle, extracted after
`_build_initial_stock()` runs. In `Fleet.__init__`, between `_build_initial_stock()` and `_run()`:

```python
if self.policies and self.policies.lcfs:
    for k in self.K:
        self.policies.lcfs.set_baseline_fc(k, self.vehicles[k, 'dice', START_YEAR])
```

`LCFS.set_baseline_fc(k, v)` stores `v.annual_fuel['diesel'][0] / v.annual_distance[0]` (L/km).

### `policies.py`: `LCFS` class

```python
class LCFS:
    def __init__(self, credit_price: float, start_target=0.183, end_target=0.76):
        # Interpolate CI reduction target schedule over _YEAR0..END_YEAR; zero before START_YEAR
        self._target_arr = ...
        self._credit_price = float(credit_price)
        self._baseline_fc = {}        # populated by set_baseline_fc()
        self._baseline_ci = ...       # diesel combustion + supply CI (kgCO2e/L) from data constants

    def set_baseline_fc(self, k, v):
        self._baseline_fc[k] = v.annual_fuel['diesel'][0] / v.annual_distance[0]

    def apply(self, v):
        target      = self._target_arr[v.operation_years - _YEAR0]
        actual_ci   = (v.emissions_supply + v.emissions_use) / np.maximum(v.annual_distance, 1.0)
        baseline_ci = self._baseline_ci * self._baseline_fc[v.k] * (1.0 - target)
        cost = v.annual_distance * (actual_ci - baseline_ci) * self._credit_price / 1000.0
        v.annual_cost['lcfs'] = cost.astype(np.float32)
        # No _calculate_tco_npv() here — Policies.apply() handles it
```

Diesel CI constants come from `data.json` fuels.diesel: supply 0.88 + use 2.52 = 3.40 kgCO2e/L.
Confirm these are per-litre and consistent with `v.annual_fuel` units.

**Verification:** 2025 diesel vehicle → near-zero LCFS cost. 2025 BE sleeper → negative cost (credits).
Magnitude matches old model at equivalent `credit_price`.

---

## Stage 3: ZEV Mandate ✓ COMPLETE

**Mechanism:** Iterative convergence loop each year t. If ZEV share of new sales falls below `target[t]`,
a penalty is applied to non-ZEV vehicles and a rebate to ZEV vehicles, shifting market shares until
convergence. Supports fleet-wide scope (aggregate across all k) or per-k scope (independent per type).

Old model reference: `old/model_old.py` lines 229–237 (ZEVMandate class), lines 855–913 (loop).

### `policies.py`: `ZEVMandate` class (implemented)

```python
ZEVMandate(targets={'2030': 0.30, '2050': 1.00}, penalty=200_000, scope='fleet')
ZEVMandate(targets={'sleeper': {'2035': 0.20, '2050': 1.00}, ...}, penalty=300_000, scope='per_k')
```

- `target_at(t, k=None)` — returns mandate fraction for year t (and vehicle type k if per_k scope)
- Targets are linearly interpolated; zero before `START_YEAR`

### `model.py: Fleet._run()` — convergence loop (implemented)

- Warm-start: carries converged penalty from previous year into the next
- 30-iteration limit per year; 30/70 damped update: `new_pen = 0.3*penalty + 0.7*raw`
- Oscillation guard (n ≥ 5): bisect the step if direction flips
- Exits cleanly when penalty stabilizes (< $1 change) — production cap binding, not an error
- `warnings.warn` only on true non-convergence (penalty still oscillating after 30 iters)

### `model.py`: `Fleet._apply_mandate_penalty(t, penalty, p_zev, k=None)` (implemented)

Sets `v.annual_cost['zev_mandate'][0]` for all year-t vehicles, then calls `v._calculate_tco_npv()`.
Rebate = `min(penalty, penalty * (1−p_zev) / max(p_zev, ε))` so total payments balance across the fleet.

**Verified:** snapshot identical with `policies=None`. With 30% target from 2030, production cap
limits early years (expected); mandate enforces naturally from ~2038 onwards.

---

## Stage 4 (future): Improvements

### 4a. Dynamic LCFS credit pricing
The LCFS credit price becomes endogenous — determined each year by the fleet's aggregate net CI
credits vs. the declining standard. Requires:
- Accumulating CI credits/deficits across all surviving cohorts for year t
- Computing equilibrium credit price from supply/demand (or BC LCFS price trajectory)
- Adding LCFS to the outer convergence loop alongside ZEV mandate for joint convergence

### 4b. Additional policies
- ZEV purchase rebate (capital cost reduction at age 0, exogenous → `apply()`)
- LCFS + ZEV mandate joint convergence when both are active
- Accelerated retirement (survival_rate modification → `pre_apply()`)

---

## Files changed per stage

| Stage | Files |
|-------|-------|
| 0 | `model.py` (`Vehicles._calculate_annual_cost`, `COST_CATEGORIES`, `Fleet._aggregate`), `policies.py` (`CarbonTax.apply`, `Policies.apply`) |
| 1 | `policies.py` (`GVWLExemption` class, `Policies` update), `data.json` (remove `gvwl_increase`) |
| 2 | `policies.py` (`LCFS` class), `model.py` (`Fleet.__init__` baseline calibration) |
| 3 | `policies.py` (`ZEVMandate` class), `model.py` (`Fleet._run` outer loop, `Fleet._apply_mandate_penalty`) |
