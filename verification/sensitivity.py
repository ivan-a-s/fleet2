"""
Strand 4 — Sensitivity / elasticity diagnostics for fleet2.

Varies three parameters by ±20% and checks that BEV market share in 2040
responds in the correct direction.  Does not require literature values —
validates wiring, not calibration.

Run from the fleet2 root:
    python verification/sensitivity.py

Or paste this in the interactive window:
    import os
    os.chdir(r"c:/Users/ivana/OneDrive - UBC/PhD/Paper 2/Code/Laptop/fleet2")
    exec(open("verification/sensitivity.py").read())
"""
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data import PARAMS
from model import Fleet, get_uncertainty_distributions


TARGET_YEAR = 2040
DELTA       = 0.20   # ±20%


def _baseline():
    inputs    = dict(get_uncertainty_distributions(PARAMS))
    param_cps = {keys: 0.5 for keys in inputs}
    return Fleet(PARAMS, param_cps)


def _perturb(base_fleet, key_path, factor):
    """
    Deep-copy base_fleet's realized params, scale the value at key_path by
    factor, and return a new Fleet.  param_cps={} because all distribution
    specs were already consumed by the base run.
    """
    params = copy.deepcopy(base_fleet.params)
    d = params
    for k in key_path[:-1]:
        d = d[k]
    d[key_path[-1]] = d[key_path[-1]] * factor
    return Fleet(params, param_cps={})


def _be_shares(fleet):
    """BEV market share in TARGET_YEAR by vehicle type."""
    return {k: fleet.market_share.get((k, 'be', TARGET_YEAR), 0.0)
            for k in fleet.K}


def _elasticity(share_lo, share_hi, share_base):
    """Central-difference elasticity: (Δshare/share_base) / (Δparam/param_base)."""
    if share_base < 1e-9:
        return float('nan')
    return (share_hi - share_lo) / (share_base * 2 * DELTA)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("Building baseline fleet (cp=0.5)...")
base        = _baseline()
base_shares = _be_shares(base)
K           = base.K

experiments = [
    {
        'name':          'battery_cost',
        'key_path':      ('vehicles', 'costs', 'battery'),
        'expected_sign': '-',
        'note':          'cheaper battery -> higher BEV NPV -> higher share',
    },
    {
        'name':          'h2_price',
        'key_path':      ('fuels', 'h2', 'cost'),
        'expected_sign': '+',
        'note':          'pricier H2 -> lower FC/HICE NPV -> share shifts toward BEV',
    },
    {
        'name':          'price_lambda',
        'key_path':      ('fleet', 'price_lambda'),
        'expected_sign': '?',
        'note':          'higher lambda sharpens logit; sign depends on BEV NPV vs field average',
    },
]

rows = []
for exp in experiments:
    print(f"Perturbing {exp['name']} +/-{DELTA*100:.0f}%...")
    lo_fleet = _perturb(base, exp['key_path'], 1.0 - DELTA)
    hi_fleet = _perturb(base, exp['key_path'], 1.0 + DELTA)
    lo_shares = _be_shares(lo_fleet)
    hi_shares = _be_shares(hi_fleet)
    elast = _elasticity(
        lo_shares.get('sleeper', 0.0),
        hi_shares.get('sleeper', 0.0),
        base_shares.get('sleeper', 0.0),
    )
    rows.append({**exp, 'lo': lo_shares, 'hi': hi_shares, 'elasticity': elast})

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

col_w = 10
header_types = "  ".join(f"{k:>{col_w}}" for k in K)
print(f"\n\nBEV market share in {TARGET_YEAR} -- sensitivity to +/-{DELTA*100:.0f}% perturbations\n")
print(f"{'Parameter':<16} {'chg':>5}  {header_types}  {'Elasticity':>12}  {'Sign OK?':>8}")
print("-" * (16 + 7 + len(K) * (col_w + 2) + 14 + 10))

# Baseline row
shares_str = "  ".join(f"{base_shares.get(k, 0):{col_w}.4f}" for k in K)
print(f"{'baseline':<16} {'--':>5}  {shares_str}  {'--':>12}  {'--':>8}")

for row in rows:
    for label, shares, delta_str in [('-20%', row['lo'], f"-{DELTA*100:.0f}%"),
                                      ('+20%', row['hi'], f"+{DELTA*100:.0f}%")]:
        shares_str = "  ".join(f"{shares.get(k, 0):{col_w}.4f}" for k in K)
        if label == '+20%':
            elast     = row['elasticity']
            elast_str = f"{elast:+.2f}" if not np.isnan(elast) else "nan"
            exp_sign  = row['expected_sign']
            if exp_sign == '?':
                ok_str = '?'
            elif exp_sign == '-':
                ok_str = 'PASS' if elast < 0 else 'FAIL'
            else:
                ok_str = 'PASS' if elast > 0 else 'FAIL'
            print(f"{row['name']:<16} {delta_str:>5}  {shares_str}  {elast_str:>12}  {ok_str:>8}")
        else:
            print(f"{'':<16} {delta_str:>5}  {shares_str}")

print()
print("Notes:")
for row in rows:
    print(f"  {row['name']:<16}  expected sign: {row['expected_sign']}  ({row['note']})")
