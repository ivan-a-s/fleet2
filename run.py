"""
Monte Carlo runner for the fleet2 HDT adoption model.

Usage
-----
    python run.py                              # all scenarios, defaults
    python run.py --scenarios baseline lcfs    # specific scenarios
    python run.py --max-runs 200 --workers 4   # quick test
    python run.py --tol 0.04                   # tighter convergence (~3000 runs)

Convergence criterion
---------------------
Runs are submitted all at once to a single process pool (no per-batch pool overhead).
After every --check-every completions (minimum 200 runs), a half-sample Kolmogorov-
Smirnov (KS) test is applied: the accumulated results are split into two equal halves
and the two-sample KS statistic D is computed for each monitored output series at every
calendar year.

    D = max_x |F_A(x) - F_B(x)|

D is the maximum vertical gap between the two empirical CDFs.  It lies in [0, 1] and is
scale-free -- no denominator or normalization is needed, so it is immune to near-zero
lower-tail values and narrow CI bands.  For n i.i.d. samples per half from the same
distribution, E[D] ~ 0.8/sqrt(n); so the convergence rate is predictable and the
threshold is directly interpretable: D < 0.05 means the two halves' CDFs agree to within
5 percentage points everywhere.

Convergence is declared when max(D) < tol across all monitored output series and years.

If --max-runs is reached before convergence the run completes anyway; a warning is printed.

Output
------
results/<scenario>.npz      -- dict of (n_runs, 26) float32 arrays, one per output series
results/<scenario>_meta.json -- n_runs, seed, tol, wall_time_s
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from data import PARAMS, START_YEAR, END_YEAR
from model import Fleet, get_uncertainty_distributions, ZEV_POWERTRAINS
from scenarios import SCENARIOS

# --- constants ---------------------------------------------------------------

PARALLEL_THRESHOLD = 50   # below this, pool setup cost exceeds any speedup

_YEARS = np.arange(START_YEAR, END_YEAR + 1)
_T     = len(_YEARS)      # 26

# Output series monitored for KS convergence -- built per vehicle type k by
# _make_convergence_keys().  Covers ZEV adoption, emissions, and cost spread.
_CONVERGENCE_SUFFIXES = (
    'zev_stock',
    'emissions_use', 'emissions_supply',
    'system_costs_capital', 'system_costs_fuel',
)


# --- sampling ----------------------------------------------------------------

def _build_col_map(params):
    """
    Walk the params tree for uncertain parameters and assign each independent
    draw a column index in the samples matrix.

    Parameters in the same 'group' share one column (correlated draw); all
    others get their own.  Returns (col_map, n_cols) where col_map maps each
    parameter path tuple to its column index.
    """
    distributions = get_uncertainty_distributions(params)
    group_col: dict[str, int] = {}
    col_map: dict[tuple, int] = {}
    n_cols = 0
    for path, group in distributions:
        if group is not None:
            if group not in group_col:
                group_col[group] = n_cols
                n_cols += 1
            col_map[path] = group_col[group]
        else:
            col_map[path] = n_cols
            n_cols += 1
    return col_map, n_cols


# --- worker ------------------------------------------------------------------

def _one_run(args):
    """Worker entry point.  Runs in a subprocess; PARAMS is re-imported there."""
    iRun, samples_row, col_map, policies = args
    param_cps = {path: np.float32(samples_row[col]) for path, col in col_map.items()}
    fleet = Fleet(PARAMS, param_cps, policies=policies)
    return _extract(fleet)


def _extract(fleet):
    """Convert fleet outputs to a flat dict of (T,) float32 arrays."""
    out: dict[str, np.ndarray] = {}

    for k in fleet.K:
        # Per-powertrain stock and sales
        for p in fleet.P[k]:
            out[f'total_stock_{k}_{p}'] = np.array(
                [fleet.total_stock.get((k, p, t), 0.0) for t in _YEARS], dtype=np.float32)
            out[f'sales_{k}_{p}'] = np.array(
                [fleet.sales.get((k, p, t), 0.0) for t in _YEARS], dtype=np.float32)

        # ZEV aggregate per k (sum over ZEV powertrains present for this k)
        zev_ps = [p for p in fleet.P[k] if p in ZEV_POWERTRAINS]
        if zev_ps:
            zev_arr = np.zeros(_T, dtype=np.float32)
            for p in zev_ps:
                zev_arr += np.array(
                    [fleet.total_stock.get((k, p, t), 0.0) for t in _YEARS], dtype=np.float32)
            out[f'zev_stock_{k}'] = zev_arr

        # Emissions
        for etype in ('embodied', 'supply', 'use'):
            out[f'emissions_{k}_{etype}'] = fleet.emissions[k][etype].astype(np.float32)

        # System costs
        for c, arr in fleet.system_costs[k].items():
            out[f'system_costs_{k}_{c}'] = arr.astype(np.float32)

    # Fuel usage -- collect unique (k, f) pairs then build dense T-length arrays
    fuel_pairs = set()
    for k, f, _ in fleet.fuel_usage:
        fuel_pairs.add((k, f))
    for k, f in sorted(fuel_pairs):
        out[f'fuel_usage_{k}_{f}'] = np.array(
            [fleet.fuel_usage.get((k, f, t), 0.0) for t in _YEARS], dtype=np.float32)

    # Per-vehicle NPV at key cohort years (scalar per run, shape () not (T,))
    for k in fleet.K:
        for p in fleet.P[k]:
            for y in (2030, 2040, 2050):
                if (k, p, y) in fleet.vehicles:
                    out[f'npv_{k}_{p}_{y}'] = np.float32(fleet.vehicles[k, p, y].npv)

    # Mandate penalty fraction time series (penalty / penalty_max per year)
    if fleet.penalty_history:
        arr = np.zeros(_T, dtype=np.float32)
        for t, frac in fleet.penalty_history.items():
            if START_YEAR <= t <= END_YEAR:
                arr[int(t) - START_YEAR] = float(frac)
        out['mandate_penalty_frac'] = arr

    return out


# --- merge & convergence -----------------------------------------------------

def _merge(results: list[dict]) -> dict[str, np.ndarray]:
    """Stack per-run dicts into (n_runs, T) arrays."""
    return {key: np.stack([r[key] for r in results]) for key in results[0]}


def _converged(results: list[dict], tol: float, convergence_keys: list[str]) -> bool:
    """
    Half-sample two-sample Kolmogorov-Smirnov convergence test.

    Split the accumulated results into two equal halves (by order of completion
    under as_completed -- effectively two random subsets).  For each monitored
    output series and each calendar year t, compute the KS statistic:

        D_t = max_x |F_A(x) - F_B(x)|

    D is the maximum vertical gap between the two empirical CDFs, always in [0,1].
    It is scale-free -- no normalization needed -- and immune to near-zero lower
    tails and narrow CI bands that caused instability with percentile-based tests.

    For n i.i.d. draws per half from the same distribution: E[D] ~ 0.8/sqrt(n).
    At n=850 per half (1700 total), E[D] ~ 0.027; tol=0.05 means convergence is
    expected around n=512 per half (1024 total) for Gaussian-like outputs, and
    somewhat more for the skewed ZEV adoption distribution (~1500 per half).

    Converged when max_{key,t}(D) < tol.
    """
    n    = len(results)
    half = n // 2
    for key in convergence_keys:
        if key not in results[0]:
            continue
        arr_a = np.stack([r[key] for r in results[:half]])        # (half, T)
        arr_b = np.stack([r[key] for r in results[half:2*half]])  # (half, T)
        T = arr_a.shape[1]
        for t in range(T):
            D, _ = ks_2samp(arr_a[:, t], arr_b[:, t])
            if D > tol:
                return False
    return True


def _make_convergence_keys(K: list[str]) -> list[str]:
    keys = []
    for k in K:
        keys += [
            f'zev_stock_{k}',
            f'emissions_{k}_use',
            f'emissions_{k}_supply',
            f'system_costs_{k}_capital',
            f'system_costs_{k}_fuel',
        ]
    return keys


def _time_one_run(col_map: dict, policies) -> float:
    """Time a single Fleet() call in the main process for the speedup report."""
    n_cols = max(col_map.values()) + 1
    row    = np.random.default_rng(99).random(n_cols).astype(np.float32)
    t0     = time.perf_counter()
    _one_run((0, row, col_map, policies))
    return time.perf_counter() - t0


# --- scenario runner ---------------------------------------------------------

def run_scenario(scenario_name: str, policies, all_samples: np.ndarray,
                 col_map: dict, n_workers: int, tol: float,
                 check_every: int, per_run_s: float) -> tuple[dict, int, float]:
    K = list(PARAMS['vehicles']['types'].keys())
    convergence_keys = _make_convergence_keys(K)

    n_max = len(all_samples)
    args  = [(i, all_samples[i], col_map, policies) for i in range(n_max)]
    accumulated: list[dict] = []
    converged   = False
    t0          = time.perf_counter()

    if n_max < PARALLEL_THRESHOLD:
        for a in args:
            accumulated.append(_one_run(a))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_one_run, a) for a in args]
            done_count = 0
            for fut in as_completed(futures):
                accumulated.append(fut.result())
                done_count += 1
                if done_count >= 200 and done_count % check_every == 0:
                    # need even count so both halves are equal size
                    if done_count % 2 == 0 and _converged(accumulated, tol, convergence_keys):
                        for f in futures:
                            f.cancel()
                        converged = True
                        break

    wall      = time.perf_counter() - t0
    n_conv    = len(accumulated)
    serial_est = n_conv * per_run_s
    speedup   = serial_est / max(wall, 1e-6)

    status = 'converged' if converged else 'max_runs reached'
    print(f'{scenario_name}: {n_conv} runs ({status}) | '
          f'wall: {wall:.1f} s | serial est: {serial_est:.0f} s | speedup: {speedup:.1f}x')
    if not converged and n_max >= PARALLEL_THRESHOLD:
        print(f'  Warning: convergence not reached at tol={tol}; increase --max-runs or relax --tol')

    return _merge(accumulated), n_conv, wall


# --- entry point -------------------------------------------------------------

def main(args):
    Path('results').mkdir(exist_ok=True)

    col_map, n_cols = _build_col_map(PARAMS)
    print(f'Uncertainty parameters: {len(col_map)} paths, {n_cols} independent draws\n')

    # Time a single run so the speedup report uses the actual machine speed.
    first_scenario = args.scenarios[0]
    print(f'Timing one run (scenario: {first_scenario})...', end=' ', flush=True)
    per_run_s = _time_one_run(col_map, SCENARIOS[first_scenario])
    print(f'{per_run_s:.3f} s\n')

    # Pre-generate the full sample matrix once so all scenarios use identical draws.
    rng         = np.random.default_rng(args.seed)
    all_samples = rng.random((args.max_runs, n_cols)).astype(np.float32)

    for scenario_name in args.scenarios:
        if scenario_name not in SCENARIOS:
            print(f'Warning: unknown scenario {scenario_name!r} -- skipping')
            continue
        policies = SCENARIOS[scenario_name]

        merged, n_conv, wall = run_scenario(
            scenario_name, policies, all_samples, col_map,
            n_workers   = args.workers,
            tol         = args.tol,
            check_every = args.check_every,
            per_run_s   = per_run_s,
        )

        out_path = Path('results') / f'{scenario_name}.npz'
        np.savez_compressed(out_path, **merged)

        meta = {
            'scenario':    scenario_name,
            'n_runs':      n_conv,
            'seed':        args.seed,
            'tol':         args.tol,
            'wall_time_s': round(wall, 2),
        }
        with open(Path('results') / f'{scenario_name}_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        print(f'  Saved -> results/{scenario_name}.npz  '
              f'({n_conv} runs x {_T} years, {len(merged)} series)\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Monte Carlo runner for fleet2 HDT adoption model',
    )
    parser.add_argument(
        '--max-runs', type=int, default=5000,
        help='Safety cap on total runs per scenario (default: 5000)',
    )
    parser.add_argument(
        '--tol', type=float, default=0.05,
        help='Convergence tolerance: max two-sample KS statistic D across all '
             'monitored outputs and years (default: 0.05).  D is the maximum vertical '
             'gap between the empirical CDFs of two equal halves of accumulated runs; '
             'D < 0.05 means the CDFs agree within 5 percentage points everywhere.  '
             'For the baseline scenario expect ~2000-3000 runs; policy scenarios '
             'converge faster.  Use 0.04 for tighter convergence.',
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Parallel worker processes (default: 8)',
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='RNG seed; same seed gives identical draws across scenarios (default: 0)',
    )
    parser.add_argument(
        '--scenarios', nargs='+', default=list(SCENARIOS),
        help='Scenario names to run (default: all)',
    )
    parser.add_argument(
        '--check-every', type=int, default=100,
        help='Check convergence every N completed runs (default: 100)',
    )
    args, _ = parser.parse_known_args()  # parse_known_args silently ignores Jupyter kernel args
    main(args)
