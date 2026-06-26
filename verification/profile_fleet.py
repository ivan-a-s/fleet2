"""
Performance profile of Fleet() instantiation.

Run from the fleet2 root:
    python verification/profile_fleet.py

Output:
  1. Phase timing table (_build_initial_stock / _run / _aggregate)
  2. Top 25 functions by self-time (tottime)   -where CPU cycles land
  3. Top 25 functions by cumulative time (cumtime) -call-tree weight
  4. Saves verification/profile.prof for interactive exploration

Optional flame graph:
    pip install snakeviz
    snakeviz verification/profile.prof

Optimisation history
--------------------
  - deepcopy elimination   (select_vehicle_params): set_year() made non-mutating;
                            3x deepcopy -> 3x dict().  ~39% faster (0.93s -> 0.57s).
  - discount factor precompute (_discount):         _discount_factor cached on Vehicles;
                            _discount() becomes a single np.sum.  _discount tottime:
                            0.020s -> 0.013s (~35% for that function).
  - _aggregate vectorised:  inner calendar-year loop replaced with numpy slice ops;
                            valid-year window per cohort, no Python inner loop.
                            _aggregate tottime: 0.088s -> ~0.05s (~45% for that function).
  - _calculate_annual_distance (non-BEV):           numpy path for dice/he/fc/hice/dhice;
                            BEV Python loop preserved for battery degradation.  Marginal
                            improvement in practice due to intermediate array allocation.
  - set_year np.clip -> scalar max/min:             avoids numpy overhead on scalar index.

Remaining optimisation opportunities
-------------------------------------
  (a) activity_met vectorisation (_run):
      Currently a generator sum over (powertrain x vintage) pairs.  Deferred because
      when the ZEV mandate is added, activity_met should move outside the ZEV iteration
      loop (it sums surviving cohorts y < t, which are constant during ZEV convergence).
      At that point, cache per-cohort activity contributions as a pre-built dict and
      accumulate incrementally as cohorts are added.  Estimated saving: ~2%.

  (b) ZEV mandate inner loop (add when implementing ZEV mandate):
      The mandate runs _calculate_market_share() up to ~20x per year until the ZEV-share
      penalty converges.  With _discount_factor precomputed, updating NPV on each iteration
      is: v.npv += v._discount_factor[0] * delta_cost (one multiply, no loop).  The
      remaining cost is the logit denominator sum over ~7 powertrains, which is already
      fast.  Profile the ZEV loop explicitly once the mandate is implemented.

  (c) Monte Carlo parallelism (run.py):
      Each MC draw is fully independent.  Use ProcessPoolExecutor with chunk size ~4-8
      draws per worker to amortise inter-process overhead.  Expected near-linear scaling
      up to physical core count.

  (d) More vehicle types / powertrains (to-do items):
      Adding resource-haul or PHE increases cohort count linearly.  The vectorised
      _aggregate already scales O(cohorts) rather than O(cohorts x years), so growth
      is cheaper than before.  Re-profile after each new type is added.

  (e) BEV annual-distance loop:
      The per-age Python loop in _calculate_annual_distance has a serial battery
      degradation dependency (cycles accumulates age-by-age).  If a simpler closed-form
      degradation model is acceptable (e.g. linear fade without cycle feedback), the BEV
      path could also be vectorised.  Only worth pursuing if BEV share grows large enough
      to make BEV cohorts a significant fraction of total runtime.
"""
import sys
import os
import cProfile
import pstats
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data import PARAMS
from model import Fleet, get_uncertainty_distributions

SEP = "-" * 80

# Deterministic median params -same convention as benchmarks.py
inputs    = dict(get_uncertainty_distributions(PARAMS))
param_cps = {k: 0.5 for k in inputs}

print(SEP)
print("Profiling Fleet(PARAMS, param_cps)  [median params, single run]")
print(SEP)

pr = cProfile.Profile()
pr.enable()
fleet = Fleet(PARAMS, param_cps)
pr.disable()

# Save .prof for snakeviz
prof_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile.prof")
pr.dump_stats(prof_path)

# -- Phase summary -------------------------------------------------------------
# pstats.Stats.stats keys: (filename, lineno, funcname)
# values: (prim_calls, calls, tottime, cumtime, callers)
PHASES = ("_build_initial_stock", "_run", "_aggregate")

stats_obj = pstats.Stats(pr)
raw_stats = stats_obj.stats

phase_data = {}
for (fname, lineno, funcname), (prim, calls, tt, ct, _callers) in raw_stats.items():
    if funcname in PHASES:
        phase_data[funcname] = (calls, tt, ct)

print()
print(f"  {'Phase':<32}  {'calls':>6}  {'self (s)':>10}  {'cumul (s)':>10}  {'%':>5}")
print("  " + "-" * 70)
total_ct = sum(v[2] for v in phase_data.values()) or 1.0
for phase in PHASES:
    if phase in phase_data:
        calls, tt, ct = phase_data[phase]
        print(f"  {phase:<32}  {calls:>6}  {tt:>10.3f}  {ct:>10.3f}  {100*ct/total_ct:>4.0f}%")

print("  " + "-" * 70)
print(f"  {'Total (3 phases)':32}  {'':>6}  {'':>10}  {total_ct:>10.3f}  {'100':>4}%")
print()
print(f"  Vehicles objects in fleet.vehicles : {len(fleet.vehicles)}")
print(f"  Vehicles objects in fleet.stock    : {len(fleet.stock)}")
print()

# -- Top 25 by self-time -------------------------------------------------------
print(SEP)
print("Top 25 functions by self-time (tottime) -where CPU cycles land")
print(SEP)
s = io.StringIO()
pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime").print_stats(25)
print(s.getvalue())

# -- Top 25 by cumulative time -------------------------------------------------
print(SEP)
print("Top 25 functions by cumulative time (cumtime) -call-tree weight")
print(SEP)
s2 = io.StringIO()
pstats.Stats(pr, stream=s2).strip_dirs().sort_stats("cumtime").print_stats(25)
print(s2.getvalue())

print(SEP)
print(f"Profile saved: {prof_path}")
print(f"Flame graph  : pip install snakeviz && snakeviz {prof_path}")
print(SEP)
