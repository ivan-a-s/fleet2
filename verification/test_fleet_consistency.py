"""
Strand 3 -- Stock-flow consistency checks for fleet2.

Asserts six accounting identities at every simulated year, using a single
deterministic Fleet run (all uncertain parameters at median, cp=0.5).

Run from the fleet2 root:
    pytest verification/test_fleet_consistency.py -v

    
Or paste this in the interactive window:

import os
os.chdir(r"c:\Users\ivana\OneDrive - UBC\PhD\Paper 2\Code\Laptop\fleet2")
import pytest
pytest.main(["verification/test_fleet_consistency.py", "-v"])

"""
import numpy as np
import numpy.testing as npt
import pytest

from data import PARAMS, START_YEAR, END_YEAR, MAX_AGE
from model import Fleet, get_uncertainty_distributions


@pytest.fixture(scope="module")
def fleet():
    """Deterministic Fleet at median parameter values (cp=0.5 for every uncertain param)."""
    inputs = dict(get_uncertainty_distributions(PARAMS))
    param_cps = {keys: 0.5 for keys in inputs}
    return Fleet(PARAMS, param_cps)


# ---------------------------------------------------------------------------
# Identity 1 -- total_stock matches the sum of individual cohort stocks
# ---------------------------------------------------------------------------

def test_total_stock_equals_cohort_sum(fleet):
    """
    total_stock[k, p, t] must equal sum of stock[k, p, y, t] for all cohort years y.
    The aggregate is built in _aggregate(); this checks it against a direct cohort sum.
    """
    for k in fleet.K:
        for p in fleet.P[k]:
            for t in fleet.years:
                expected = sum(
                    fleet.stock.get((k, p, y, t), 0.0)
                    for y in range(t - MAX_AGE + 1, t + 1)
                )
                actual = float(fleet.total_stock[k, p, t])
                npt.assert_allclose(
                    actual, expected, rtol=1e-5,
                    err_msg=f"total_stock mismatch: k={k}, p={p}, t={t}",
                )


# ---------------------------------------------------------------------------
# Identity 2 -- sales equal the age-0 cohort stock in the same year
# ---------------------------------------------------------------------------

def test_sales_equal_age_zero_stock(fleet):
    """
    sales[k, p, t] must equal stock[k, p, t, t] -- i.e. the new cohort
    purchased in year t.  Both are derived from the same dict entry.
    """
    for k in fleet.K:
        for p in fleet.P[k]:
            for t in fleet.years:
                expected = float(fleet.stock.get((k, p, t, t), 0.0))
                actual   = float(fleet.sales[k, p, t])
                npt.assert_allclose(
                    actual, expected, rtol=1e-5,
                    err_msg=f"sales mismatch: k={k}, p={p}, t={t}",
                )


# ---------------------------------------------------------------------------
# Identity 3 -- fleet delivers the required activity every year
# ---------------------------------------------------------------------------

def test_fleet_meets_activity_requirement(fleet):
    """
    Total t-km delivered by all surviving cohorts (including new sales) must
    equal activity_req[k, t].  New purchases are sized to close the gap, so
    the check should hold to within floating-point tolerance.
    """
    for k in fleet.K:
        for t in fleet.years:
            activity_total = sum(
                fleet.stock.get((k, p, y, t), 0.0)
                * fleet.vehicles[k, p, y].annual_distance[t - y]
                * float(np.asarray(fleet.vehicles[k, p, y].mass['payload'])[t - y]) / 1000.0
                for p in fleet.P[k]
                for y in range(t - MAX_AGE + 1, t + 1)
                if (k, p, y) in fleet.vehicles
            )
            required = fleet.activity_req[k, t]
            npt.assert_allclose(
                activity_total, required, rtol=1e-2,
                err_msg=(
                    f"activity gap: k={k}, t={t} -- "
                    f"delivered={activity_total:.1f}, required={required:.1f}"
                ),
            )


# ---------------------------------------------------------------------------
# Identity 4 -- fuel_usage matches the cohort-level aggregation
# ---------------------------------------------------------------------------

def test_fuel_usage_matches_cohort_aggregation(fleet):
    """
    fleet.fuel_usage[k, fuel, t] must equal the sum of
    n * annual_fuel[fuel][age] over all cohorts that operate in year t.
    """
    # Collect every fuel key that appears in any vehicle's annual_fuel
    all_fuels: dict[str, set] = {k: set() for k in fleet.K}
    for k in fleet.K:
        for p in fleet.P[k]:
            for y in range(START_YEAR - MAX_AGE + 1, END_YEAR + 1):
                if (k, p, y) in fleet.vehicles:
                    all_fuels[k].update(fleet.vehicles[k, p, y].annual_fuel)

    for k in fleet.K:
        for fuel in all_fuels[k]:
            for t in fleet.years:
                expected = sum(
                    fleet.stock.get((k, p, y, t), 0.0)
                    * fleet.vehicles[k, p, y].annual_fuel[fuel][t - y]
                    for p in fleet.P[k]
                    for y in range(t - MAX_AGE + 1, t + 1)
                    if (k, p, y) in fleet.vehicles
                    and 0 <= (t - y) < MAX_AGE
                    and fuel in fleet.vehicles[k, p, y].annual_fuel
                )
                actual = float(fleet.fuel_usage.get((k, fuel, t), 0.0))
                npt.assert_allclose(
                    actual, expected, rtol=1e-4,
                    err_msg=f"fuel_usage mismatch: k={k}, fuel={fuel}, t={t}",
                )


# ---------------------------------------------------------------------------
# Identity 5 -- emissions totals match the cohort aggregation
# ---------------------------------------------------------------------------

def test_emissions_match_cohort_aggregation(fleet):
    """
    fleet.emissions[k][stream][i] must equal the sum of n * per-vehicle
    emission[age] over all cohorts active in year t, for each stream
    (embodied, supply, use).
    """
    getters = {
        'embodied': lambda v: v.embodied,
        'supply':   lambda v: v.emissions_supply,
        'use':      lambda v: v.emissions_use,
    }
    for k in fleet.K:
        for stream, getter in getters.items():
            for i, t in enumerate(fleet.years):
                expected = sum(
                    fleet.stock.get((k, p, y, t), 0.0)
                    * getter(fleet.vehicles[k, p, y])[t - y]
                    for p in fleet.P[k]
                    for y in range(t - MAX_AGE + 1, t + 1)
                    if (k, p, y) in fleet.vehicles and 0 <= (t - y) < MAX_AGE
                )
                actual = float(fleet.emissions[k][stream][i])
                npt.assert_allclose(
                    actual, expected, rtol=1e-4,
                    err_msg=f"emissions[{stream}] mismatch: k={k}, t={t}",
                )


# ---------------------------------------------------------------------------
# Identity 6 -- market shares sum to 1
# ---------------------------------------------------------------------------

def test_market_shares_sum_to_one(fleet):
    """
    The multinomial logit with iterative production caps must allocate exactly
    100% of the market each year.
    """
    for k in fleet.K:
        for t in fleet.years:
            total = sum(fleet.market_share.get((k, p, t), 0.0) for p in fleet.P[k])
            npt.assert_allclose(
                total, 1.0, atol=1e-5,
                err_msg=f"market shares don't sum to 1: k={k}, t={t} (sum={total:.8f})",
            )


# ---------------------------------------------------------------------------
# Identity 7 -- rollover applies the correct conditional survival ratio
# ---------------------------------------------------------------------------

def test_rollover_applies_conditional_survival(fleet):
    """
    For surviving cohorts, stock[k, p, y, t] must equal
    stock[k, p, y, t-1] * survival_rate[age] / survival_rate[age-1].
    This is the marginal-decay formula used in Fleet._run().
    """
    for k in fleet.K:
        for p in fleet.P[k]:
            for y in range(START_YEAR - MAX_AGE + 1, END_YEAR):
                if (k, p, y) not in fleet.vehicles:
                    continue
                surv = fleet.vehicles[k, p, y].params['survival_rate']
                for t in fleet.years[1:]:
                    a = t - y
                    if not (1 <= a < MAX_AGE):
                        continue
                    prev = fleet.stock.get((k, p, y, t - 1), 0.0)
                    if prev == 0.0:
                        continue
                    expected = float(prev) * float(surv[a]) / max(float(surv[a - 1]), 1e-9)
                    actual   = float(fleet.stock.get((k, p, y, t), 0.0))
                    npt.assert_allclose(
                        actual, expected, rtol=1e-4,
                        err_msg=f"rollover error: k={k}, p={p}, y={y}, t={t}",
                    )
