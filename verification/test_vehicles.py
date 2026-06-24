"""
Strand 1 — Unit tests on Vehicles methods.

Uses a real Fleet run at median params rather than constructing minimal
params dicts from scratch — avoids fragile fixture plumbing and tests
against the actual calibrated model.

Run from the fleet2 root:
    pytest verification/test_vehicles.py -v

Or paste this in the interactive window:
    import os
    os.chdir(r"c:/Users/ivana/OneDrive - UBC/PhD/Paper 2/Code/Laptop/fleet2")
    import pytest
    pytest.main(["verification/test_vehicles.py", "-v"])
"""
import numpy as np
import numpy.testing as npt
import pytest

from data import PARAMS, START_YEAR, MAX_AGE, DISCOUNT_RATE
from model import Fleet, get_uncertainty_distributions, ZEV_POWERTRAINS


@pytest.fixture(scope="module")
def fleet():
    inputs    = dict(get_uncertainty_distributions(PARAMS))
    param_cps = {keys: 0.5 for keys in inputs}
    return Fleet(PARAMS, param_cps)


@pytest.fixture(scope="module")
def vehicles(fleet):
    """All (k, p) Vehicles objects at START_YEAR."""
    return {
        (k, p): fleet.vehicles[k, p, START_YEAR]
        for k in fleet.K
        for p in fleet.P[k]
    }


# ---------------------------------------------------------------------------
# Mass
# ---------------------------------------------------------------------------

def test_unloaded_mass_positive(vehicles):
    for (k, p), v in vehicles.items():
        assert v.unloaded_mass > 0, f"unloaded_mass <= 0: k={k}, p={p}"


def test_total_mass_exceeds_unloaded(vehicles):
    """Payload adds mass at every age — total_mass must never fall below unloaded."""
    for (k, p), v in vehicles.items():
        assert np.all(v.total_mass >= v.unloaded_mass), \
            f"total_mass < unloaded_mass: k={k}, p={p}"


def test_sleeper_diesel_mass_in_plausible_range(fleet):
    """Sleeper diesel loaded mass should be 20–55 t (literature target: 36–40 t)."""
    v = fleet.vehicles['sleeper', 'dice', START_YEAR]
    assert 20_000 <= v.total_mass[0] <= 55_000, \
        f"Sleeper diesel mass implausible: {v.total_mass[0] / 1000:.1f} t (expect 36–40 t)"


def test_zev_payload_positive(fleet):
    """ZEV payload must be positive. Comparison to diesel is omitted — ZEVs may
    carry more payload than diesel once GVWL exemptions are introduced."""
    for k in fleet.K:
        for p in ('be', 'fc'):
            if p not in fleet.P[k]:
                continue
            payload = float(np.asarray(fleet.vehicles[k, p, START_YEAR].mass['payload'])[0])
            assert payload > 0, f"{p} payload is zero or negative: k={k}"


# ---------------------------------------------------------------------------
# Fuel consumption
# ---------------------------------------------------------------------------

def test_fuel_consumption_positive(vehicles):
    for (k, p), v in vehicles.items():
        for f, fc in v.fuel_consumption.items():
            assert np.all(fc > 0), f"fuel_consumption[{f}] non-positive: k={k}, p={p}"


def test_dice_has_only_diesel(fleet):
    for k in fleet.K:
        v = fleet.vehicles[k, 'dice', START_YEAR]
        assert set(v.fuel_consumption) == {'diesel'}, \
            f"dice fuel keys wrong: {set(v.fuel_consumption)}"


def test_hice_has_only_h2(fleet):
    for k in fleet.K:
        v = fleet.vehicles[k, 'hice', START_YEAR]
        assert set(v.fuel_consumption) == {'h2'}, \
            f"hice fuel keys wrong: {set(v.fuel_consumption)}"


def test_dhice_has_diesel_and_h2(fleet):
    for k in fleet.K:
        v = fleet.vehicles[k, 'dhice', START_YEAR]
        assert set(v.fuel_consumption) == {'diesel', 'h2'}, \
            f"dhice fuel keys wrong: {set(v.fuel_consumption)}"


def test_be_has_a_charge_fuel(fleet):
    for k in fleet.K:
        v = fleet.vehicles[k, 'be', START_YEAR]
        assert any('charge' in f for f in v.fuel_consumption), \
            f"be has no charge fuel: {set(v.fuel_consumption)}"


def test_dhice_fuel_proportions_match_params(fleet):
    """
    DHICE energy split between diesel and H2 must match the proportions
    declared in data.json (within LHV conversion rounding).
    """
    DIESEL_LHV = float(PARAMS['fuels']['diesel']['lhv'])
    H2_LHV     = float(PARAMS['fuels']['h2']['lhv'])
    for k in fleet.K:
        v  = fleet.vehicles[k, 'dhice', START_YEAR]
        fp = v.params['fuels']
        d_prop = fp.get('diesel', {}).get('proportion', 0.75)
        h_prop = fp.get('h2',     {}).get('proportion', 0.25)
        # Convert tank-level FC back (undo refuel_efficiency) then to energy
        d_eff = float(PARAMS['fuels']['diesel'].get('refuel_efficiency', 1.0))
        h_eff = float(PARAMS['fuels']['h2'].get('refuel_efficiency', 1.0))
        e_diesel = v.fuel_consumption['diesel'][0] * d_eff * DIESEL_LHV
        e_h2     = v.fuel_consumption['h2'][0]     * h_eff * H2_LHV
        actual_d_frac = e_diesel / (e_diesel + e_h2)
        expected_d_frac = d_prop / (d_prop + h_prop)
        npt.assert_allclose(actual_d_frac, expected_d_frac, rtol=1e-4,
                            err_msg=f"dhice diesel energy fraction wrong: k={k}")


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------

def test_range_positive(vehicles):
    for (k, p), v in vehicles.items():
        assert np.all(v.range > 0), f"range <= 0: k={k}, p={p}"


def test_be_range_less_than_diesel_range(fleet):
    """BEV has a smaller tank (battery) than a diesel and should have a shorter range."""
    for k in fleet.K:
        be_range   = fleet.vehicles[k, 'be',   START_YEAR].range[0]
        dice_range = fleet.vehicles[k, 'dice', START_YEAR].range[0]
        assert be_range < dice_range, (
            f"BEV range ({be_range:.0f} km) >= diesel ({dice_range:.0f} km): k={k}"
        )


# ---------------------------------------------------------------------------
# Annual distance
# ---------------------------------------------------------------------------

def test_annual_distance_positive(vehicles):
    for (k, p), v in vehicles.items():
        assert np.all(v.annual_distance > 0), f"annual_distance <= 0: k={k}, p={p}"


def test_annual_distance_does_not_exceed_target(vehicles):
    """
    Annual km cannot exceed the target even with en-route stops — the time
    budget is fixed.  Allow 1 % margin for floating-point.
    """
    for (k, p), v in vehicles.items():
        target = np.asarray(v.params['target_distance'])
        assert np.all(v.annual_distance <= target * 1.01), \
            f"annual_distance exceeds target: k={k}, p={p}, " \
            f"max excess={(v.annual_distance / target).max():.4f}x"


# ---------------------------------------------------------------------------
# FC replacements
# ---------------------------------------------------------------------------

def test_non_fc_powertrains_have_no_replacements(fleet):
    for k in fleet.K:
        for p in ('dice', 'he', 'be', 'hice'):
            if p not in fleet.P[k]:
                continue
            v = fleet.vehicles[k, p, START_YEAR]
            assert np.all(v.fc_replacements == 0), \
                f"{p} vehicle has fc_replacements > 0: k={k}"


# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------

def test_embodied_nonnegative_and_positive_at_purchase(vehicles):
    """
    Manufacturing emissions must be positive at age 0 and non-negative at all ages.
    Ages > 0 may be nonzero once FC stack replacement embodied emissions are added.
    """
    for (k, p), v in vehicles.items():
        assert v.embodied[0] > 0, f"embodied[0] == 0: k={k}, p={p}"
        assert np.all(v.embodied >= 0), f"embodied < 0 at some age: k={k}, p={p}"


def test_zev_has_zero_tailpipe_emissions(fleet):
    for k in fleet.K:
        for p in ZEV_POWERTRAINS:
            if p not in fleet.P[k]:
                continue
            v = fleet.vehicles[k, p, START_YEAR]
            npt.assert_array_equal(v.emissions_use, 0,
                                   err_msg=f"ZEV {p} has tailpipe emissions: k={k}")


def test_diesel_has_positive_tailpipe_emissions(fleet):
    for k in fleet.K:
        v = fleet.vehicles[k, 'dice', START_YEAR]
        assert np.all(v.emissions_use > 0), \
            f"diesel has zero tailpipe emissions: k={k}"


def test_supply_emissions_nonnegative(vehicles):
    for (k, p), v in vehicles.items():
        assert np.all(v.emissions_supply >= 0), \
            f"emissions_supply < 0: k={k}, p={p}"


# ---------------------------------------------------------------------------
# Discount formula
# ---------------------------------------------------------------------------

def test_discount_matches_survival_weighted_formula(fleet):
    """
    _discount(constant C) must equal C × Σ_a survival_rate[a] / (1+r)^a
    — the exact formula from the docstring.
    """
    v        = fleet.vehicles['sleeper', 'dice', START_YEAR]
    C        = 10_000.0
    surv     = np.asarray(v.params['survival_rate'])
    expected = float(np.sum(C * surv / (1.0 + DISCOUNT_RATE) ** np.arange(MAX_AGE)))
    actual   = v._discount(np.full(MAX_AGE, C))
    npt.assert_allclose(actual, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# TCO and NPV
# ---------------------------------------------------------------------------

def test_tco_equals_sum_of_discounted_cost_components(vehicles):
    """TCO must equal the sum of _discount() over every cost component."""
    for (k, p), v in vehicles.items():
        expected = sum(v._discount(arr) for arr in v.annual_cost.values())
        npt.assert_allclose(v.tco, expected, rtol=1e-5,
                            err_msg=f"TCO decomposition wrong: k={k}, p={p}")


def test_npv_equals_discounted_revenue_minus_tco(vehicles):
    for (k, p), v in vehicles.items():
        expected = v._discount(v.annual_revenue) - v.tco
        npt.assert_allclose(v.npv, expected, rtol=1e-5,
                            err_msg=f"NPV formula wrong: k={k}, p={p}")


def test_diesel_tco_in_plausible_range(fleet):
    """
    Sleeper diesel TCO should be $200k–$2M in 2025 CAD
    (literature target: $350k–$600k; wide bounds accommodate parameter uncertainty).
    """
    v = fleet.vehicles['sleeper', 'dice', START_YEAR]
    assert 200_000 <= v.tco <= 2_000_000, \
        f"Sleeper diesel TCO implausible: ${v.tco:,.0f} (expect ~$350k–$600k)"
