"""
Strand 2 — Limiting / degenerate cases for fleet2.

Tests boundary conditions in _calculate_market_share() where the expected
outcome is analytically known, without requiring calibrated parameter values.
The real Fleet._calculate_market_share method is called on minimal duck-typed
mock objects so no parameter files are needed for most tests.

Run from the fleet2 root:
    pytest verification/test_limits.py -v

    
Or paste this in the interactive window:

import os
os.chdir(r"c:/Users/ivana/OneDrive - UBC/PhD/Paper 2/Code/Laptop/fleet2")
import pytest
pytest.main(["verification/test_limits.py", "-v"])

"""
import numpy as np
import numpy.testing as npt
import pytest

from data import PARAMS, START_YEAR
from model import Fleet, get_uncertainty_distributions


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockVehicle:
    """Minimal stub with the attributes _calculate_market_share reads."""
    def __init__(self, npv, init_limit=1.0, cagr_nacent=1.0, cagr_mature=1.0):
        self.npv = npv
        self.params = {
            'init_market_limit': init_limit,
            'cagr_nacent':       cagr_nacent,
            'cagr_mature':       cagr_mature,
        }


def _run_market_share(powertrains, npvs, price_lambda,
                      prev_shares=None, init_limits=None,
                      cagr_nacent=1.0, cagr_mature=1.0):
    """
    Call the real Fleet._calculate_market_share on a minimal mock and return
    the resulting {powertrain: share} dict for year START_YEAR.

    prev_shares: previous year's shares (list, same order as powertrains).
        Defaults to uniform 1/N so production caps are non-binding.
    init_limits: init_market_limit per powertrain (list).
        Defaults to 1.0 (unconstrained).
    """
    t = START_YEAR
    N = len(powertrains)

    class _Mock:
        pass

    mock              = _Mock()
    mock.K            = ['k']
    mock.P            = {'k': list(powertrains)}
    mock.price_lambda = price_lambda
    mock.market_share = {}
    mock.vehicles     = {}

    for i, p in enumerate(powertrains):
        init = init_limits[i] if init_limits is not None else 1.0
        prev = prev_shares[i] if prev_shares is not None else 1.0 / N
        mock.vehicles['k', p, t] = _MockVehicle(npvs[i], init, cagr_nacent, cagr_mature)
        mock.market_share['k', p, t - 1] = prev

    Fleet._calculate_market_share(mock, 'k', t)
    return {p: mock.market_share['k', p, t] for p in powertrains}


# ---------------------------------------------------------------------------
# Case 1 — equal NPV → uniform shares (1/N)
# ---------------------------------------------------------------------------

def test_equal_npv_gives_uniform_shares():
    """
    When all N powertrains have identical NPV, the logit numerator equals
    1/N of the denominator, so every share must be exactly 1/N.
    Tested for N = 2, 3, and 7 (the full powertrain count).
    """
    for N in [2, 3, 7]:
        powertrains = [f'p{i}' for i in range(N)]
        shares = _run_market_share(powertrains, npvs=[50_000.0] * N, price_lambda=3e-5)
        for p in powertrains:
            npt.assert_allclose(
                shares[p], 1.0 / N, rtol=1e-5,
                err_msg=f"equal-NPV share wrong for N={N}, p={p}",
            )


# ---------------------------------------------------------------------------
# Case 2 — price_lambda → 0 → uniform shares regardless of NPV spread
# ---------------------------------------------------------------------------

def test_zero_lambda_gives_uniform_shares():
    """
    As price_lambda → 0, exp(λ × NPV) → 1 for all powertrains, collapsing
    the logit to uniform shares regardless of NPV differences.
    """
    powertrains = ['dice', 'be', 'fc']
    npvs        = [-500_000.0, 0.0, 200_000.0]   # large spread, but lambda is tiny
    shares      = _run_market_share(powertrains, npvs, price_lambda=1e-12)
    for p in powertrains:
        npt.assert_allclose(
            shares[p], 1.0 / 3, rtol=1e-5,
            err_msg=f"near-zero lambda: share wrong for {p}",
        )


# ---------------------------------------------------------------------------
# Case 3 — higher NPV → higher share (correct sign)
# ---------------------------------------------------------------------------

def test_higher_npv_gets_higher_share():
    """
    The vehicle with the best NPV must receive the largest market share.
    This checks that price_lambda enters with the correct sign.
    """
    powertrains = ['dice', 'be', 'fc']
    npvs        = [100_000.0, 200_000.0, 50_000.0]
    shares      = _run_market_share(powertrains, npvs, price_lambda=3e-5)
    assert shares['be'] > shares['dice'] > shares['fc'], (
        f"shares not ordered by NPV: {shares}"
    )


# ---------------------------------------------------------------------------
# Case 4 — single powertrain gets 100 % of the market
# ---------------------------------------------------------------------------

def test_single_powertrain_gets_full_market():
    """With only one powertrain available, its share must be exactly 1.0."""
    shares = _run_market_share(['dice'], npvs=[100_000.0], price_lambda=3e-5)
    npt.assert_allclose(shares['dice'], 1.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Case 5 — non-binding cap leaves shares at their unconstrained logit values
# ---------------------------------------------------------------------------

def test_non_binding_cap_leaves_shares_unchanged():
    """
    When every production cap exceeds the unconstrained logit share, the
    iterative loop must converge on the first pass and return the plain
    logit values.
    """
    powertrains = ['dice', 'be', 'fc']
    npvs        = [100_000.0, 80_000.0, 60_000.0]
    lam         = 3e-5

    # Expected unconstrained logit shares
    exps   = np.exp(lam * np.array(npvs))
    logits = exps / exps.sum()

    # High prev_shares (0.9) and high CAGR (2.0) → cap ≫ logit for all p
    shares = _run_market_share(
        powertrains, npvs, price_lambda=lam,
        prev_shares=[0.9, 0.9, 0.9],
        cagr_nacent=2.0, cagr_mature=2.0,
    )
    for p, logit in zip(powertrains, logits):
        npt.assert_allclose(
            shares[p], logit, rtol=1e-5,
            err_msg=f"non-binding cap altered share for {p}",
        )


# ---------------------------------------------------------------------------
# Case 6 — init_market_limit = 0 caps a powertrain to zero share
# ---------------------------------------------------------------------------

def test_zero_init_limit_caps_powertrain_to_zero():
    """
    A powertrain with init_market_limit=0 and no prior market presence must
    be capped at 0% regardless of NPV advantage, and the sole uncapped
    powertrain inherits the full market.
    """
    shares = _run_market_share(
        powertrains=['dice', 'be'],
        npvs=[50_000.0, 500_000.0],   # 'be' has a huge NPV edge
        price_lambda=3e-5,
        prev_shares=[1.0, 0.0],
        init_limits=[1.0, 0.0],
    )
    npt.assert_allclose(shares['be'],   0.0, atol=1e-10,
                        err_msg="capped powertrain got non-zero share")
    npt.assert_allclose(shares['dice'], 1.0, rtol=1e-5,
                        err_msg="sole uncapped powertrain did not get 100%")


# ---------------------------------------------------------------------------
# Case 7 — diesel-only fleet when all other powertrains are excluded
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def diesel_only_fleet():
    inputs    = dict(get_uncertainty_distributions(PARAMS))
    param_cps = {keys: 0.5 for keys in inputs}
    return Fleet(PARAMS, param_cps,
                 exclude_powertrains=['he', 'phe', 'be', 'fc', 'hice', 'dhice'])


def test_diesel_gets_full_market_when_sole_powertrain(diesel_only_fleet):
    """Diesel must receive 100% market share in every year when it is the only option."""
    fleet = diesel_only_fleet
    for k in fleet.K:
        for t in fleet.years:
            npt.assert_allclose(
                fleet.market_share.get((k, 'dice', t), 0.0), 1.0, rtol=1e-5,
                err_msg=f"diesel share < 1: k={k}, t={t}",
            )


def test_no_zev_stock_when_powertrains_excluded(diesel_only_fleet):
    """No ZEV stock should exist when all non-diesel powertrains are excluded."""
    fleet = diesel_only_fleet
    for k in fleet.K:
        for t in fleet.years:
            for p in ['he', 'phe', 'be', 'fc', 'hice', 'dhice']:
                stock = fleet.total_stock.get((k, p, t), 0.0)
                assert stock == 0.0, (
                    f"excluded powertrain has stock: k={k}, p={p}, t={t}, stock={stock}"
                )
