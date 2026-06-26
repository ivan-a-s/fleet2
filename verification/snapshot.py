"""
Regression snapshot tool for fleet2.

Save a snapshot before making changes, then check after to confirm
nothing shifted unintentionally.

Usage:
    python verification/snapshot.py save   # save current outputs to snapshot.npz
    python verification/snapshot.py check  # compare current outputs to saved snapshot

Or from the interactive window:
    import os
    os.chdir(r"c:/Users/ivana/OneDrive - UBC/PhD/Paper 2/Code/Laptop/fleet2")
    exec(open("verification/snapshot.py").read())   # runs 'check' if snapshot exists, else 'save'
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data import PARAMS
from model import Fleet, get_uncertainty_distributions

SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshot.npz')
RTOL = 1e-4   # 0.01% tolerance -- enough to catch real changes, ignores float32 noise


def _run():
    inputs    = dict(get_uncertainty_distributions(PARAMS))
    param_cps = {keys: 0.5 for keys in inputs}
    return Fleet(PARAMS, param_cps)


def _extract(fleet):
    """Flatten key fleet outputs into a {name: scalar_or_array} dict."""
    data = {}
    for (k, p, t), v in fleet.market_share.items():
        data[f"ms__{k}__{p}__{t}"] = float(v)
    for (k, p, t), v in fleet.total_stock.items():
        data[f"stock__{k}__{p}__{t}"] = float(v)
    for (k, fuel, t), v in fleet.fuel_usage.items():
        data[f"fuel__{k}__{fuel}__{t}"] = float(v)
    for k in fleet.K:
        for stream, arr in fleet.emissions[k].items():
            for i, t in enumerate(fleet.years):
                data[f"emis__{k}__{stream}__{t}"] = float(arr[i])
        for cost, arr in fleet.system_costs[k].items():
            for i, t in enumerate(fleet.years):
                data[f"cost__{k}__{cost}__{t}"] = float(arr[i])
    return data


def save():
    print("Running model...")
    fleet = _run()
    data  = _extract(fleet)
    np.savez(SNAPSHOT_PATH, **data)
    print(f"Snapshot saved: {len(data)} values -> {SNAPSHOT_PATH}")


def check():
    if not os.path.exists(SNAPSHOT_PATH):
        print("No snapshot found. Run with 'save' first.")
        return

    print("Running model...")
    fleet   = _run()
    current = _extract(fleet)
    saved   = dict(np.load(SNAPSHOT_PATH))

    missing  = [k for k in saved  if k not in current]
    new_keys = [k for k in current if k not in saved]
    mismatches = []

    for key in saved:
        if key not in current:
            continue
        a, b = float(saved[key]), current[key]
        if a == 0 and b == 0:
            continue
        denom = max(abs(a), abs(b), 1e-12)
        if abs(a - b) / denom > RTOL:
            mismatches.append((key, a, b, abs(a - b) / denom))

    if not mismatches and not missing and not new_keys:
        print(f"OK -- all {len(saved)} values match within {RTOL*100:.2f}%")
        return

    if mismatches:
        mismatches.sort(key=lambda x: -x[3])
        print(f"\nMISMATCHES ({len(mismatches)}):")
        for key, old, new, rel in mismatches[:20]:
            print(f"  {key}")
            print(f"    saved={old:.6g}  current={new:.6g}  rel_diff={rel:.2e}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches)-20} more")
    if missing:
        print(f"\nKEYS IN SNAPSHOT BUT NOT IN CURRENT ({len(missing)}):")
        for k in missing[:10]:
            print(f"  {k}")
    if new_keys:
        print(f"\nNEW KEYS NOT IN SNAPSHOT ({len(new_keys)}):")
        for k in new_keys[:10]:
            print(f"  {k}")

    print(f"\nFAIL -- {len(mismatches)} mismatches, {len(missing)} missing, {len(new_keys)} new")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__' or '__file__' not in dir():
    # When exec()'d from interactive window, default to check if snapshot exists
    args = sys.argv[1:] if '__file__' in dir() else []
    mode = args[0] if args else ('check' if os.path.exists(SNAPSHOT_PATH) else 'save')

    if mode == 'save':
        save()
    elif mode == 'check':
        check()
    else:
        print("Usage: python verification/snapshot.py [save|check]")
