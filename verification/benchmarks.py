"""
Strand 5 -- Literature benchmarks for fleet2.

Runs the model at median params and compares 8 key quantities to
literature ranges. FAIL means the value is outside the range and
warrants investigation -- it is not a hard error.

Run from the fleet2 root:
    python verification/benchmarks.py

Or paste this in the interactive window:
    import os
    os.chdir(r"c:/Users/ivana/OneDrive - UBC/PhD/Paper 2/Code/Laptop/fleet2")
    exec(open("verification/benchmarks.py").read())
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import PARAMS, START_YEAR
from model import Fleet, get_uncertainty_distributions


# ---------------------------------------------------------------------------
# Run model at median params
# ---------------------------------------------------------------------------

print("Running model at median params...")
inputs    = dict(get_uncertainty_distributions(PARAMS))
param_cps = {keys: 0.5 for keys in inputs}
fleet     = Fleet(PARAMS, param_cps)

dice = fleet.vehicles['sleeper', 'dice', START_YEAR]
be   = fleet.vehicles['sleeper', 'be',   START_YEAR]
fc   = fleet.vehicles['sleeper', 'fc',   START_YEAR]


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------

benchmarks = [
    {
        'label':  'Sleeper diesel mass (t)',
        'value':  dice.total_mass[0] / 1000.0,
        'lo':     36.0,
        'hi':     40.0,
        'fmt':    '.1f',
        'source': 'GVW regs, NRCan',
    },
    {
        'label':  'Diesel FC (L/100km)',
        'value':  dice.fuel_consumption['diesel'][0] * 100.0,
        'lo':     35.0,
        'hi':     45.0,
        'fmt':    '.1f',
        'source': 'NRCan, DOE',
    },
    {
        'label':  'Annual distance diesel (km/yr)',
        'value':  dice.annual_distance[0],
        'lo':     150_000.0,
        'hi':     200_000.0,
        'fmt':    '.0f',
        'source': 'StatCan, NRCan',
    },
    {
        'label':  'Diesel capital cost ($k CAD)',
        'value':  dice.capital_total / 1000.0,
        'lo':     180.0,
        'hi':     250.0,
        'fmt':    '.1f',
        'source': 'Industry reports',
    },
    {
        'label':  'Diesel TCO ($k CAD)',
        'value':  dice.tco / 1000.0,
        'lo':     350.0,
        'hi':     600.0,
        'fmt':    '.1f',
        'source': 'Literature TCO studies',
    },
    {
        'label':  'BEV range (km)',
        'value':  be.range[0],
        'lo':     300.0,
        'hi':     500.0,
        'fmt':    '.0f',
        'source': 'Tesla Semi, eCascadia OEM specs',
    },
    {
        'label':  'BEV/diesel capital ratio',
        'value':  be.capital_total / dice.capital_total,
        'lo':     2.0,
        'hi':     3.0,
        'fmt':    '.2f',
        'source': 'BloombergNEF, ICCT',
    },
    {
        'label':  'FC/diesel capital ratio',
        'value':  fc.capital_total / dice.capital_total,
        'lo':     3.0,
        'hi':     5.0,
        'fmt':    '.2f',
        'source': 'ICCT, H2 Council',
    },
]


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

W_LABEL = 34
W_MODEL = 12
W_RANGE = 22
W_PASS  = 10

header = (
    f"{'Quantity':<{W_LABEL}}"
    f"{'Model':>{W_MODEL}}"
    f"  {'Literature range':<{W_RANGE}}"
    f"{'Pass/Fail':>{W_PASS}}"
)
sep = '-' * (W_LABEL + W_MODEL + 2 + W_RANGE + W_PASS)

print()
print(f"Sleeper benchmarks vs literature (median params, {START_YEAR})")
print()
print(header)
print(sep)

n_pass = 0
fails  = []

for b in benchmarks:
    val   = b['value']
    lo    = b['lo']
    hi    = b['hi']
    ok    = lo <= val <= hi
    flag  = 'PASS' if ok else 'FAIL'
    if ok:
        n_pass += 1
    else:
        fails.append(b['label'])

    val_str   = format(val, b['fmt'])
    range_str = f"{format(lo, b['fmt'])} - {format(hi, b['fmt'])}"

    print(
        f"{b['label']:<{W_LABEL}}"
        f"{val_str:>{W_MODEL}}"
        f"  {range_str:<{W_RANGE}}"
        f"{flag:>{W_PASS}}"
    )

print(sep)
print(f"\n{n_pass}/{len(benchmarks)} benchmarks within literature range.")

if fails:
    print("\nQuantities outside range (investigate separately):")
    for label in fails:
        print(f"  - {label}")
