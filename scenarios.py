"""
Policy scenario definitions for the fleet2 Monte Carlo runner.

Each entry in SCENARIOS is a name -> Policies object.  run.py runs the full MC
for each scenario using an identical random sample matrix, so results are directly
comparable across scenarios.

Policy parameter sources
------------------------
carbon_tax  BC scheduled trajectory (paused 2024; prior schedule retained as reference)
lcfs        BC LCFS regulation: 18.3% CI reduction in 2025 -> 76% by 2050, $300/tCO2e credit
zev_mandate BC HDV ZEV mandate proposal: 30% ZEV sales by 2030, 100% by 2040
gvwl        BC ZEV GVWL exemption: +5000 kg sleeper, +3000 kg day_cab, +2000 kg straight

Edit the values here to change scenario assumptions; model.py and run.py need no changes.
"""

from policies import CarbonTax, GVWLExemption, LCFS, ZEVMandate, Policies

SCENARIOS = {
    'baseline': Policies(),

    'carbon_tax': Policies(
        carbon_tax=CarbonTax({'2025': 95, '2030': 170, '2050': 170}),
    ),

    'lcfs': Policies(
        lcfs=LCFS(credit_price=300, start_target=0.183, end_target=0.76),
    ),

    'zev_mandate': Policies(
        zev_mandate=ZEVMandate(
            targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
            penalty=30_000,
            scope='fleet',
        ),
    ),

    'gvwl': Policies(
        gvwl_exemption=GVWLExemption(),
    ),

    'full_policy': Policies(
        carbon_tax=CarbonTax({'2025': 95, '2030': 170, '2050': 170}),
        lcfs=LCFS(credit_price=300, start_target=0.183, end_target=0.76),
        zev_mandate=ZEVMandate(
            targets={'2025': 0, '2030': 0.3, '2040': 1.0, '2050': 1.0},
            penalty=30_000,
            scope='fleet',
        ),
        gvwl_exemption=GVWLExemption(),
    ),
}
