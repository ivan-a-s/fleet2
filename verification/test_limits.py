"""
Strand 2 -- Limiting / degenerate cases for fleet2.

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
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from data import PARAMS, START_YEAR
from model import Fleet, get_uncertainty_distributions, _market_share_limit, _shadow_price_shares
from policies import Policies, ZEVMandate


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


_REAL_NEST_LAMBDAS = {'liquid': 0.7, 'conventional': 0.4, 'hydrogen': 0.6, 'electric': 1.0}


def _run_market_share(powertrains, npvs, price_lambda,
                      prev_shares=None, init_limits=None,
                      cagr_nacent=1.0, cagr_mature=1.0,
                      nest_lambdas=None):
    """
    Call the real Fleet._calculate_market_share on a minimal mock and return
    the resulting {powertrain: share} dict for year START_YEAR.

    prev_shares: previous year's shares (list, same order as powertrains).
        Defaults to uniform 1/N so production caps are non-binding.
    init_limits: init_market_limit per powertrain (list).
        Defaults to 1.0 (unconstrained).
    nest_lambdas: dict nest-name -> scale. Defaults to the real params.py values.
    """
    t = START_YEAR
    N = len(powertrains)

    class _Mock:
        pass

    mock                = _Mock()
    mock.K              = ['k']
    mock.P              = {'k': list(powertrains)}
    mock.price_lambda   = price_lambda
    mock.nest_lambdas   = dict(_REAL_NEST_LAMBDAS if nest_lambdas is None else nest_lambdas)
    mock.market_share   = {}
    mock.vehicles       = {}
    mock._mu_warm_start = {}

    for i, p in enumerate(powertrains):
        init = init_limits[i] if init_limits is not None else 1.0
        prev = prev_shares[i] if prev_shares is not None else 1.0 / N
        mock.vehicles['k', p, t] = _MockVehicle(npvs[i], init, cagr_nacent, cagr_mature)
        mock.market_share['k', p, t - 1] = prev

    Fleet._calculate_market_share(mock, 'k', t)
    return {p: mock.market_share['k', p, t] for p in powertrains}


# ---------------------------------------------------------------------------
# Case 1 -- equal NPV -> uniform shares (1/N)
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
# Case 2 -- price_lambda -> 0 -> uniform shares regardless of NPV spread
# ---------------------------------------------------------------------------

def test_zero_lambda_gives_uniform_shares():
    """
    As price_lambda -> 0, exp(lam x NPV) -> 1 for all powertrains, collapsing
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
# Case 3 -- higher NPV -> higher share (correct sign)
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
# Case 4 -- single powertrain gets 100 % of the market
# ---------------------------------------------------------------------------

def test_single_powertrain_gets_full_market():
    """With only one powertrain available, its share must be exactly 1.0."""
    shares = _run_market_share(['dice'], npvs=[100_000.0], price_lambda=3e-5)
    npt.assert_allclose(shares['dice'], 1.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Case 5 -- non-binding cap leaves shares at their unconstrained logit values
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

    # High prev_shares (0.9) and high CAGR (2.0) -> cap >> logit for all p
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
# Case 6 -- init_market_limit = 0 caps a powertrain to zero share
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
# Case 7 -- diesel-only fleet when all other powertrains are excluded
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


# ---------------------------------------------------------------------------
# Case 8 -- nested logit (NEST_TREE): degenerate reduction, IIA-violation
# demonstration, singleton-nest invariance, zero-cap exclusion, same-nest caps
# ---------------------------------------------------------------------------

def test_all_nest_lambdas_one_matches_flat_mnl():
    """
    The explicit degenerate case: with every nest_lambdas value at 1.0, the nested tree must
    reduce to exactly flat MNL over all 7 powertrains, regardless of tree depth (proven by hand
    in _shadow_price_shares's docstring; this is the strongest single-call exercise of that
    proof, hitting the 2-level Liquid->Conventional branch, the 3-way Hydrogen nest, and the
    Electric singleton all at once).
    """
    powertrains = ['dice', 'he', 'phe', 'be', 'fc', 'hice', 'dhice']
    npvs        = [100_000.0, 90_000.0, 120_000.0, 200_000.0, 60_000.0, 70_000.0, 50_000.0]
    lam         = 3e-5

    shares = _run_market_share(
        powertrains, npvs, price_lambda=lam,
        prev_shares=[0.9] * 7, cagr_nacent=2.0, cagr_mature=2.0,
        nest_lambdas={'liquid': 1.0, 'conventional': 1.0, 'hydrogen': 1.0, 'electric': 1.0},
    )
    exps   = np.exp(lam * np.array(npvs))
    logits = exps / exps.sum()
    for p, logit in zip(powertrains, logits):
        npt.assert_allclose(
            shares[p], logit, rtol=1e-6,
            err_msg=f"all-lambda=1 nested share != flat MNL for {p}",
        )


def test_within_nest_ratio_invariant_to_other_nests():
    """
    dice and he's relative odds (both in Conventional) must depend only on their own NPVs and
    lambda_conventional -- never on what else is present, whether a fully separate nest (be) or
    a Liquid sibling one level up but outside Conventional (phe).
    """
    npv_dice, npv_he = 100_000.0, 90_000.0
    lam = 3e-5

    def ratio(extra_powertrains, extra_npvs):
        powertrains = ['dice', 'he'] + extra_powertrains
        npvs        = [npv_dice, npv_he] + extra_npvs
        n = len(powertrains)
        shares = _run_market_share(
            powertrains, npvs, price_lambda=lam,
            prev_shares=[0.9] * n, cagr_nacent=2.0, cagr_mature=2.0,
        )
        return shares['dice'] / shares['he']

    r_alone     = ratio([], [])
    r_plus_be   = ratio(['be'], [150_000.0])
    r_plus_phe  = ratio(['phe'], [110_000.0])
    r_plus_both = ratio(['be', 'phe'], [150_000.0, 110_000.0])

    for name, r in [('plus_be', r_plus_be), ('plus_phe', r_plus_phe), ('plus_both', r_plus_both)]:
        npt.assert_allclose(
            r, r_alone, rtol=1e-6,
            err_msg=f"dice/he ratio changed when adding {name}: {r} vs baseline {r_alone}",
        )


def test_cross_nest_ratio_shifts_with_within_nest_substitute():
    """
    dice (Conventional) vs fc (Hydrogen), both singleton branches, matches flat MNL exactly.
    Adding a competitive same-nest substitute (he, equal NPV to dice) changes dice/fc's relative
    odds by a factor of 2**(lambda_conventional - 1) != 1 -- an IIA violation flat MNL cannot
    produce, and the entire point of nesting.
    """
    npv_dice, npv_fc = 100_000.0, 80_000.0
    lam = 3e-5

    shares_a = _run_market_share(
        ['dice', 'fc'], [npv_dice, npv_fc], price_lambda=lam,
        prev_shares=[0.9, 0.9], cagr_nacent=2.0, cagr_mature=2.0,
    )
    ratio_a    = shares_a['dice'] / shares_a['fc']
    flat_ratio = np.exp(lam * (npv_dice - npv_fc))
    npt.assert_allclose(ratio_a, flat_ratio, rtol=1e-6,
                        err_msg="singleton-per-branch case should match flat MNL exactly")

    shares_b = _run_market_share(
        ['dice', 'he', 'fc'], [npv_dice, npv_dice, npv_fc], price_lambda=lam,
        prev_shares=[0.9, 0.9, 0.9], cagr_nacent=2.0, cagr_mature=2.0,
    )
    ratio_b = shares_b['dice'] / shares_b['fc']

    lambda_conv     = _REAL_NEST_LAMBDAS['conventional']
    expected_factor = 2.0 ** (lambda_conv - 1.0)
    npt.assert_allclose(
        ratio_b / ratio_a, expected_factor, rtol=1e-6,
        err_msg="adding a same-nest substitute should scale dice/fc odds by 2**(lambda_conv-1)",
    )
    assert not np.isclose(ratio_b, flat_ratio, rtol=1e-3), (
        "dice/fc ratio with a same-nest substitute should NOT match the flat-MNL-implied value"
    )


def test_singleton_nest_lambda_has_no_effect():
    """
    Electric contains only `be` -- its nest lambda must have zero effect on any share, since a
    singleton nest's utility collapses to its single leaf's utility regardless of lambda.
    """
    powertrains = ['dice', 'he', 'phe', 'be', 'fc', 'hice', 'dhice']
    npvs        = [100_000.0, 90_000.0, 120_000.0, 200_000.0, 60_000.0, 70_000.0, 50_000.0]
    lam         = 3e-5

    baseline = None
    for electric_lambda in [0.3, 1.0, 2.5]:
        nest_lambdas = dict(_REAL_NEST_LAMBDAS, electric=electric_lambda)
        shares = _run_market_share(
            powertrains, npvs, price_lambda=lam,
            prev_shares=[0.9] * 7, cagr_nacent=2.0, cagr_mature=2.0,
            nest_lambdas=nest_lambdas,
        )
        if baseline is None:
            baseline = shares
        else:
            for p in powertrains:
                npt.assert_allclose(
                    shares[p], baseline[p], rtol=1e-8,
                    err_msg=f"electric_lambda={electric_lambda} changed share of {p}",
                )


def test_zero_cap_leaf_excluded_from_nest_inclusive_value():
    """
    A zero-capped same-nest leaf must be excluded from its nest's inclusive value entirely, not
    merely zeroed in final probability -- otherwise the surviving nest-mate's share would be
    pulled toward the "extra option value" of a phantom alternative that supposedly has zero
    probability. dice's share with he zero-capped must exactly match dice's share in a model
    where he was never present at all.
    """
    npv_dice, npv_he, npv_fc = 100_000.0, 500_000.0, 80_000.0
    lam = 3e-5

    shares_with_zeroed_he = _run_market_share(
        ['dice', 'he', 'fc'], [npv_dice, npv_he, npv_fc], price_lambda=lam,
        prev_shares=[0.9, 0.0, 0.9], init_limits=[1.0, 0.0, 1.0],
        cagr_nacent=2.0, cagr_mature=2.0,
    )
    shares_without_he = _run_market_share(
        ['dice', 'fc'], [npv_dice, npv_fc], price_lambda=lam,
        prev_shares=[0.9, 0.9], cagr_nacent=2.0, cagr_mature=2.0,
    )

    npt.assert_allclose(shares_with_zeroed_he['he'], 0.0, atol=1e-10,
                        err_msg="zero-capped he got non-zero share")
    npt.assert_allclose(shares_with_zeroed_he['dice'], shares_without_he['dice'], rtol=1e-6,
                        err_msg="dice's share leaked he's inclusive-value contribution")
    npt.assert_allclose(shares_with_zeroed_he['fc'], shares_without_he['fc'], rtol=1e-6,
                        err_msg="fc's share leaked he's inclusive-value contribution")


def test_shadow_pricing_converges_with_multiple_same_nest_binding_caps():
    """
    Both dice and he (same Conventional nest) simultaneously and severely capped, alongside
    uncapped be/fc, must still converge with complementary slackness -- the scenario flagged in
    the architecture notes as highest risk (several same-nest members capped at once).
    """
    powertrains = ['dice', 'he', 'be', 'fc']
    npvs        = [500_000.0, 500_000.0, 50_000.0, 50_000.0]  # dice/he want most of the market
    lam         = 3e-5
    prev_shares = [0.01, 0.01, 0.49, 0.49]
    # dice/he given a nascent init_market_limit (0.02, like a real ZEV powertrain) so their tiny
    # prev_share isn't floored back up near 1 by _market_share_limit's init/(1+cagr) term --
    # that floor is what a mature (init=1.0) powertrain like dice normally relies on to make its
    # own cap non-binding in practice; this test deliberately defeats that to force both same-
    # nest members to bind simultaneously.
    cap_dice = _market_share_limit(0.01, 0.02, 0.05, 0.05)
    cap_he   = _market_share_limit(0.01, 0.02, 0.05, 0.05)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shares = _run_market_share(
            powertrains, npvs, price_lambda=lam,
            prev_shares=prev_shares, init_limits=[0.02, 0.02, 1.0, 1.0],
            cagr_nacent=0.05, cagr_mature=0.05,
        )
    assert not caught, f"unexpected warning(s): {[str(w.message) for w in caught]}"

    npt.assert_allclose(shares['dice'], cap_dice, rtol=1e-4,
                        err_msg="dice (same-nest, capped) share should equal its cap")
    npt.assert_allclose(shares['he'], cap_he, rtol=1e-4,
                        err_msg="he (same-nest, capped) share should equal its cap")


# ---------------------------------------------------------------------------
# Case 9 -- cross-nest simultaneous binding caps, sampled price_lambda extreme,
# warm-start robustness, nest_lambda > 1, negative-NPV leaves, and ZEV mandate x
# shadow-pricing interaction (2026-07 audit follow-ups)
# ---------------------------------------------------------------------------

def test_shadow_pricing_converges_with_cross_nest_binding_caps():
    """
    Four different nests each contribute one severely-capped member simultaneously -- he
    (Liquid/Conventional), phe (Liquid, direct child), be (Electric), fc (Hydrogen) -- with
    dice absorbing the residual uncapped demand. This is the exact real-world worst case
    documented in CLAUDE.md's "Convergence tuning" note (the 262-sweep case that motivated
    loosening max_sweeps/tol), previously untested: the only existing multi-cap test
    (test_shadow_pricing_converges_with_multiple_same_nest_binding_caps) binds two powertrains
    within a single nest, never exercising the extra coupling channel that runs through each
    capped nest's shared parent inclusive value.
    """
    powertrains = ['dice', 'he', 'phe', 'be', 'fc']
    npvs        = [50_000.0, 500_000.0, 500_000.0, 500_000.0, 500_000.0]
    lam         = 3e-5
    prev_shares = [0.6, 0.01, 0.01, 0.01, 0.01]
    init_limits = [1.0, 0.02, 0.02, 0.02, 0.02]

    caps = {
        p: _market_share_limit(prev, init, 0.05, 0.05)
        for p, prev, init in zip(powertrains, prev_shares, init_limits)
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shares = _run_market_share(
            powertrains, npvs, price_lambda=lam,
            prev_shares=prev_shares, init_limits=init_limits,
            cagr_nacent=0.05, cagr_mature=0.05,
        )
    assert not caught, f"unexpected warning(s): {[str(w.message) for w in caught]}"

    for p in ['he', 'phe', 'be', 'fc']:
        npt.assert_allclose(shares[p], caps[p], rtol=1e-4,
                            err_msg=f"{p} (cross-nest, capped) share should equal its cap")
    npt.assert_allclose(sum(shares.values()), 1.0, rtol=1e-6,
                        err_msg="shares must sum to 1")


def test_shadow_pricing_converges_at_sampled_price_lambda_lower_bound():
    """
    Every other nested/shadow-pricing test hardcodes price_lambda=3e-5 (the old fixed value /
    MC upper bound). fleet.price_lambda is now MC-sampled down to 1e-5 (params.py), and that
    lower extreme flattens the logit and needs larger mu excursions to hit the same cap --
    exactly where CLAUDE.md's "Convergence tuning" note says a slow (not oscillating)
    convergence case was found. Reruns the cross-nest binding-cap scenario at the sampled lower
    bound to pin the real worst-case sweep count against the max_sweeps=300 ceiling.
    """
    powertrains = ['dice', 'he', 'phe', 'be', 'fc']
    npvs        = [50_000.0, 500_000.0, 500_000.0, 500_000.0, 500_000.0]
    lam         = 1e-5
    prev_shares = [0.6, 0.01, 0.01, 0.01, 0.01]
    init_limits = [1.0, 0.02, 0.02, 0.02, 0.02]

    caps = {
        p: _market_share_limit(prev, init, 0.05, 0.05)
        for p, prev, init in zip(powertrains, prev_shares, init_limits)
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shares = _run_market_share(
            powertrains, npvs, price_lambda=lam,
            prev_shares=prev_shares, init_limits=init_limits,
            cagr_nacent=0.05, cagr_mature=0.05,
        )
    assert not caught, f"unexpected warning(s) at price_lambda=1e-5: {[str(w.message) for w in caught]}"

    for p in ['he', 'phe', 'be', 'fc']:
        npt.assert_allclose(shares[p], caps[p], rtol=1e-4,
                            err_msg=f"{p} (cross-nest, capped, low price_lambda) share should equal its cap")


def test_warm_start_relaxes_stale_mu_to_zero():
    """
    A stale, previously-converged mu for a leaf that should now be uncapped must relax back to
    mu == 0, not get stuck at a stale positive value that happens to look feasible under a
    feasibility-only check. Direct regression test for the exact bug documented in CLAUDE.md
    ("A real bug worth remembering..."): checking share <= cap alone accepted a
    self-consistent-but-wrong state after a bad warm start; only checking tightness (share ==
    cap whenever mu > 0) catches it. Builds the adversarial warm start by hand (calling
    _shadow_price_shares directly) rather than relying on incidentally hitting it via Fleet's
    own warm-start sequencing.
    """
    nest_lambdas = _REAL_NEST_LAMBDAS
    lam          = 3e-5

    # State A: dice and he both severely capped in the same nest (Conventional) -- converges
    # both mu_dice and mu_he to large positive values.
    npv_a  = {'dice': 500_000.0, 'he': 500_000.0, 'be': 50_000.0, 'fc': 50_000.0}
    caps_a = {'dice': 0.02, 'he': 0.02, 'be': 1.0, 'fc': 1.0}
    _, mu_a = _shadow_price_shares(npv_a, caps_a, lam, nest_lambdas)
    assert mu_a['dice'] > 0.0, "setup: state A should capped-bind dice"
    assert mu_a['he']   > 0.0, "setup: state A should capped-bind he"

    # State B: dice's cap is fully relaxed (1.0 -- can never bind) while he stays tight.
    # Warm-starting dice from state A's large stale mu is the adversarial case: it must relax
    # back to exactly 0, not get stuck self-consistently capped.
    npv_b  = {'dice': 500_000.0, 'he': 500_000.0, 'be': 50_000.0, 'fc': 50_000.0}
    caps_b = {'dice': 1.0, 'he': 0.02, 'be': 1.0, 'fc': 1.0}

    shares_fresh, mu_fresh = _shadow_price_shares(npv_b, caps_b, lam, nest_lambdas)
    shares_warm,  mu_warm  = _shadow_price_shares(npv_b, caps_b, lam, nest_lambdas, mu0=mu_a)

    for p in npv_b:
        npt.assert_allclose(shares_warm[p], shares_fresh[p], rtol=1e-4,
                            err_msg=f"warm-started result for {p} differs from a fresh solve")
    assert mu_warm['dice'] == pytest.approx(0.0, abs=1e-6), (
        f"dice's stale positive mu should have relaxed back to 0 once its cap was lifted, "
        f"got {mu_warm['dice']}"
    )


def test_nest_lambda_greater_than_one_produces_valid_shares():
    """
    All existing non-degenerate nest_lambda tests use lambda <= 1 (nest members MORE
    correlated than flat MNL). lambda > 1 (members treated as MORE independent than flat MNL)
    is valid McFadden as long as lambda <= the parent's own scale, and was previously entirely
    unexercised (the one lambda>1 case in the suite, test_singleton_nest_lambda_has_no_effect,
    is on a singleton nest, an explicit no-op). Checks shares stay finite and sum to 1, and that
    dice/fc's odds ratio scales by the expected factor 2**(lambda_conv - 1) in the >1 direction
    too (mirroring test_cross_nest_ratio_shifts_with_within_nest_substitute's <1 case).
    """
    npv_dice, npv_fc = 100_000.0, 80_000.0
    lam = 3e-5
    nest_lambdas = dict(_REAL_NEST_LAMBDAS, conventional=1.5)

    shares_a = _run_market_share(
        ['dice', 'fc'], [npv_dice, npv_fc], price_lambda=lam,
        prev_shares=[0.9, 0.9], cagr_nacent=2.0, cagr_mature=2.0,
        nest_lambdas=nest_lambdas,
    )
    ratio_a    = shares_a['dice'] / shares_a['fc']
    flat_ratio = np.exp(lam * (npv_dice - npv_fc))
    npt.assert_allclose(ratio_a, flat_ratio, rtol=1e-6,
                        err_msg="singleton-per-branch case should match flat MNL exactly regardless of lambda")

    shares_b = _run_market_share(
        ['dice', 'he', 'fc'], [npv_dice, npv_dice, npv_fc], price_lambda=lam,
        prev_shares=[0.9, 0.9, 0.9], cagr_nacent=2.0, cagr_mature=2.0,
        nest_lambdas=nest_lambdas,
    )
    ratio_b = shares_b['dice'] / shares_b['fc']
    npt.assert_allclose(sum(shares_b.values()), 1.0, rtol=1e-6,
                        err_msg="shares must sum to 1 with nest_lambda > 1")
    assert np.isfinite(ratio_b), "nest_lambda > 1 produced a non-finite share ratio"

    expected_factor = 2.0 ** (1.5 - 1.0)
    npt.assert_allclose(
        ratio_b / ratio_a, expected_factor, rtol=1e-6,
        err_msg="adding a same-nest substitute under lambda>1 should still scale dice/fc odds by 2**(lambda_conv-1)",
    )


def test_negative_npv_leaf_under_real_nesting_with_binding_cap():
    """
    Only test_zero_lambda_gives_uniform_shares uses a negative NPV, and it does so with
    price_lambda=1e-12 (making NPV irrelevant everywhere). Real early ZEVs can have negative
    NPV; this drives V_p = price_lambda * (npv - mu) strongly negative for one leaf (he) under
    REAL nest_lambdas while a same-nest sibling (dice) is simultaneously capped, confirming the
    tree evaluator and bisection bracket stay well-behaved (finite, in [0, 1], sums to 1) rather
    than assuming every leaf's NPV is positive.
    """
    powertrains = ['dice', 'he', 'be']
    npvs        = [500_000.0, -300_000.0, 50_000.0]   # dice desired+capped, he deeply negative
    lam         = 3e-5
    prev_shares = [0.01, 0.9, 0.9]
    init_limits = [0.02, 1.0, 1.0]

    cap_dice = _market_share_limit(0.01, 0.02, 0.05, 0.05)

    shares = _run_market_share(
        powertrains, npvs, price_lambda=lam,
        prev_shares=prev_shares, init_limits=init_limits,
        cagr_nacent=0.05, cagr_mature=0.05,
    )
    for p, s in shares.items():
        assert np.isfinite(s), f"non-finite share for {p}: {s}"
        assert -1e-9 <= s <= 1.0 + 1e-9, f"share out of [0, 1] for {p}: {s}"
    npt.assert_allclose(sum(shares.values()), 1.0, rtol=1e-6,
                        err_msg="shares must sum to 1 with a negative-NPV leaf present")
    npt.assert_allclose(shares['dice'], cap_dice, rtol=1e-4,
                        err_msg="dice should be capped-tight even alongside a deeply negative-NPV nest-mate")
    assert shares['he'] < shares['be'], (
        "deeply negative-NPV he should get less share than positive-NPV be"
    )


def test_zev_mandate_converges_with_active_shadow_price_cap():
    """
    The ZEV-mandate credit-price bisection in Fleet._run calls _calculate_market_share (and
    therefore _shadow_price_shares) up to ~30 times per year, warm-starting mu between probes
    as the credit price shifts NPVs slightly each time (CLAUDE.md's "Warm-starting" note). This
    interaction -- an active mandate driving repeated shadow-price solves within a single year,
    with ZEV powertrains also production-cap-bound at their default nascent limits -- was
    previously untested in isolation from either mechanism's own unit tests. Runs a real
    (not mocked) Fleet with a ZEV mandate at median (cp=0.5) parameters and checks the mandate
    bisection and the shadow-price solver both converge cleanly (no warnings) across every
    simulated year, with market shares summing to 1 throughout.
    """
    inputs    = dict(get_uncertainty_distributions(PARAMS))
    param_cps = {keys: 0.5 for keys in inputs}
    policies  = Policies(zev_mandate=ZEVMandate(
        targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
        penalty=30_000,
        scope='fleet',
    ))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fleet = Fleet(PARAMS, param_cps, policies=policies)
    assert not caught, f"unexpected warning(s): {[str(w.message) for w in caught]}"

    for k in fleet.K:
        for t in fleet.years:
            total = sum(fleet.market_share.get((k, p, t), 0.0) for p in fleet.P[k])
            npt.assert_allclose(
                total, 1.0, rtol=1e-4,
                err_msg=f"market shares don't sum to 1 for k={k}, t={t} under an active mandate",
            )
