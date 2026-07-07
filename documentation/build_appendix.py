"""
Generate Appendix A supplementary material tables from params.py.

Usage:
    python documentation/build_appendix.py

Output:
    documentation/appendix.md

Each section corresponds to a table in the paper's Appendix A. Values are read
directly from PARAM_DICT in params.py. Update params.py to change values or
citations; re-run this script to regenerate the tables. Do not edit appendix.md
by hand.

Convert to Word:
    pandoc documentation/appendix.md -o documentation/appendix.docx
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import PARAM_DICT, Param

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _val(p):
    """Render a Param value as a compact human-readable string."""
    if p is None:
        return "N/A"
    if not isinstance(p, Param):
        return str(p)
    v = p.value
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        kind = v.get("dist") or v.get("array")
        if kind == "const":
            return str(v["val"])
        if kind == "uniform":
            return f"{v['min']} - {v['max']}"
        if kind == "triangle":
            return f"triangular({v['min']}, {v['mode']}, {v['max']})"
        if kind == "interp":
            years = [k for k in v if k not in ("dist", "group")]
            years.sort()
            parts = []
            for yr in years:
                sub = v[yr]
                if isinstance(sub, dict):
                    sub_kind = sub.get("dist")
                    if sub_kind == "const":
                        parts.append(f"{yr}: {sub['val']}")
                    elif sub_kind == "uniform":
                        parts.append(f"{yr}: {sub['min']}-{sub['max']}")
                    elif sub_kind == "triangle":
                        parts.append(f"{yr}: triangular({sub['min']},{sub['mode']},{sub['max']})")
                    else:
                        parts.append(f"{yr}: {sub}")
                else:
                    parts.append(f"{yr}: {sub}")
            return "; ".join(parts)
        if kind == "linear":
            start = v.get("start", {})
            end   = v.get("end", {})
            s = start.get("val", f"{start.get('min')}-{start.get('max')}") if isinstance(start, dict) else start
            e = end.get("val",   f"{end.get('min')}-{end.get('max')}")     if isinstance(end,   dict) else end
            e_str = (f"triangular({end['min']},{end['mode']},{end['max']})"
                     if isinstance(end, dict) and end.get("dist") == "triangle"
                     else str(e))
            return f"2025: {s} -> 2050: {e_str}"
        if kind == "logistic":
            return f"logistic(max={v['max_val']}, min={v['min_val']}, k={v['k']}, x0={v['x0']})"
        if kind == "linspace":
            return f"linspace({v['start']}, {v['end']})"
        if kind == "step":
            return f"step at age {v['threshold']}: {v['below']} / {v['above']}"
        if kind == "constant":
            return str(v["value"])
        return str(v)
    return str(v)


def _src(p):
    if p is None or not isinstance(p, Param) or p.src is None:
        return "**UNREF**"
    return p.src


def _units(p, override=None):
    if override:
        return override
    if p is None or not isinstance(p, Param):
        return ""
    return p.units or ""


def _row(name, p, units_override=None):
    return f"| {name} | {_val(p)} | {_units(p, units_override)} | {_src(p)} |"


def _header(cols):
    sep = "|".join("---" for _ in cols)
    return f"| {' | '.join(cols)} |\n|{sep}|"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_general():
    D = PARAM_DICT
    lines = [
        "## Table A1 - General parameters\n",
        _header(["Parameter", "Value", "Units", "Source"]),
        _row("Growth rate (real)",                    _get(D, "settings", "growth_rate"),          "%"),
        _row("Private discount rate (real)",          _get(D, "settings", "discount_rate"),         "%"),
        _row("Market sensitivity (lambda)",           _get(D, "fleet",    "price_lambda")),
        _row("Activity (year 0)",                     _get(D, "fleet",    "initial_activity")),
        _row("Gravity",                               _get(D, "settings", "gravity"),               "m/s2"),
        _row("Start year",                            _get(D, "settings", "start_year")),
        _row("End year",                              _get(D, "settings", "end_year")),
        _row("Max age",                               _get(D, "settings", "max_age"),               "years"),
        "",
    ]
    return lines


def render_vehicle_activity():
    D = PARAM_DICT
    lines = [
        "## Table A2 - Vehicle activity\n",
        "Source: National Research Council, 2010; NRCan, 2025; Statistics Canada, 2009\n",
        _header(["Truck type", "Fleet share", "Avg annual distance (km)",
                 "Avg payload (kg)", "Source"]),
    ]
    for k, label in [("sleeper", "Sleeper-cab"), ("day_cab", "Day-cab"),
                     ("straight", "Straight truck")]:
        prop = _get(D, "vehicles", "types", k, "shared", "activity_proportion")
        dc_key = "long_haul" if k == "sleeper" else ("regional_haul" if k == "day_cab" else "short_haul")
        payload = _get(D, "drive_cycles", dc_key, "payload")
        lines.append(f"| {label} | {_val(prop)} | (from logistic fit) | {_val(payload)} | {_src(prop)} |")
    lines.append("")
    return lines


def render_fuel_properties():
    D = PARAM_DICT
    fuels = [
        ("diesel",      "Diesel",                   "L"),
        ("slow_charge", "Electricity (slow charge)", "kWh"),
        ("fast_charge", "Electricity (fast charge)", "kWh"),
        ("h2",          "Hydrogen (electrolysis)",   "kg"),
        ("h2_p",        "Hydrogen (pyrolysis)",      "kg"),
        ("h2_pe",       "Hydrogen (electrified pyrolysis)", "kg"),
    ]
    lines = [
        "## Table A3 - Fuel properties\n",
        _header(["Fuel", "Unit", "LHV (MJ/unit)", "Supply GHG (kgCO2e/unit)",
                 "Use GHG (kgCO2e/unit)", "Refuel eff.", "Water use (L/unit)",
                 "Elec. use (kWh/unit)", "Source"]),
    ]
    for key, label, unit in fuels:
        f = D["fuels"][key]
        lhv_mj = float(_val(_get(f, "lhv"))) / 1e6
        sup  = _val(_get(f, "emissions_intensity", "supply"))
        use  = _val(_get(f, "emissions_intensity", "use"))
        ref  = _val(_get(f, "refuel_efficiency"))
        wat  = _val(_get(f, "water_intensity"))
        elec = _val(_get(f, "electricity_intensity"))
        src  = _src(_get(f, "lhv"))
        lines.append(f"| {label} | {unit} | {lhv_mj} | {sup} | {use} | {ref} | {wat} | {elec} | {src} |")
    lines.append("")
    return lines


def render_shared_vehicle():
    D = PARAM_DICT
    lines = [
        "## Table A4 - Shared HDT technical parameters\n",
        _header(["Parameter", "Value", "Units", "Source"]),
        _row("Frontal area",          _get(D, "vehicles", "types", "sleeper", "shared", "frontal_area")),
        _row("Mass correction",       _get(D, "vehicles", "types", "sleeper", "shared", "mass_correction")),
        _row("Combustion transmission efficiency",
             _get(D, "vehicles", "components", "transmission", "combustion_transmission", "efficiency")),
        _row("Electric transmission efficiency",
             _get(D, "vehicles", "components", "transmission", "electric_transmission", "efficiency")),
        _row("Frame embodied emissions",
             _get(D, "vehicles", "components", "frame", "embodied_emissions")),
        _row("Li-ion BESS specific mass",
             _get(D, "vehicles", "components", "ess", "battery", "specific_mass")),
        _row("Battery degradation (per year)",
             _get(D, "vehicles", "components", "ess", "battery", "deg_per_year")),
        _row("Battery degradation (per cycle)",
             _get(D, "vehicles", "components", "ess", "battery", "deg_per_cycle")),
        _row("Autonomous t50",
             _get(D, "fleet", "autonomous_t50")),
        "",
    ]
    return lines


def render_vehicle_type(k, label, table_num):
    D = PARAM_DICT
    s = D["vehicles"]["types"][k]["shared"]
    lines = [
        f"## Table A{table_num} - Technical parameters: {label}\n",
        _header(["Parameter", "Value", "Units", "Source"]),
        _row("Drag coefficient",  _get(s, "drag_coef")),
        _row("Trailers per truck", _get(s, "trailers_per_truck")),
        _row("GVWL",              _get(s, "gvwl"),         "kg"),
        _row("Frame mass",        _get(s, "frame_mass"),   "kg"),
        _row("Trailer mass",      _get(s, "trailer_mass"), "kg"),
        _row("Weighed-out journeys", _get(s, "p_weighed_out")),
        _row("Base cost",         _get(s, "base_cost"),    "CAD"),
        _row("Driver cost",       _get(s, "driver_cost"),  "CAD/km"),
        _row("Revenue per t-km",  _get(s, "revenue_per_tkm"), "CAD/t-km"),
        "",
    ]
    return lines


def render_powertrain_params():
    D = PARAM_DICT
    # Battery/H2 capacities are plain ints inside components dicts (not Param-wrapped)
    # Wrap them in a bare Param for display purposes only
    def cap(kw):
        return Param(kw, notes="model internal")

    be_cap_sleeper  = cap(_get(D, "vehicles", "types", "sleeper",  "powertrains", "be",  "components", "battery",  "capacity"))
    be_cap_daycab   = cap(_get(D, "vehicles", "types", "day_cab",  "powertrains", "be",  "components", "battery",  "capacity"))
    be_cap_straight = cap(_get(D, "vehicles", "types", "straight", "powertrains", "be",  "components", "battery",  "capacity"))
    fc_cap_sleeper  = cap(_get(D, "vehicles", "types", "sleeper",  "powertrains", "fc",  "components", "h2_700bar","capacity"))
    hice_cap        = cap(_get(D, "vehicles", "types", "sleeper",  "powertrains", "hice","components", "h2_700bar","capacity"))

    lines = [
        "## Table A8 - Diesel and hybrid (dice/he/phe) powertrain parameters\n",
        _header(["Parameter", "Value", "Units", "Source"]),
        _row("ICE mass",          _get(D, "vehicles", "components", "converter", "ice", "mass"), "kg"),
        _row("Engine efficiency", _get(D, "vehicles", "components", "converter", "ice", "efficiency")),
        _row("Motor efficiency (battery discharge)",
             _get(D, "vehicles", "components", "converter", "motor", "he", "efficiency")),
        _row("Combustion transmission efficiency",
             _get(D, "vehicles", "components", "transmission", "combustion_transmission", "efficiency")),
        _row("Regen efficiency (dice)",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "dice", "regen_efficiency")),
        _row("Regen efficiency (he/phe)",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "he", "regen_efficiency")),
        _row("Accessory load (dice)",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "dice", "accessory_load"), "W"),
        _row("Accessory load (he/phe)",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "he", "accessory_load"), "W"),
        _row("O&M cost (dice/he/phe)",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "dice", "running_cost"), "CAD/km"),
        "",
        "## Table A9 - BET (be) powertrain parameters\n",
        _header(["Parameter", "Value", "Units", "Source"]),
        _row("Powertrain mass (motor)",
             _get(D, "vehicles", "components", "converter", "motor", "be", "mass"), "kg"),
        _row("Motor efficiency",
             _get(D, "vehicles", "components", "converter", "motor", "be", "efficiency")),
        _row("Battery capacity (sleeper / day-cab / straight)",
             Param(f"{be_cap_sleeper.value} / {be_cap_daycab.value} / {be_cap_straight.value}",
                   src="Table 15"), "kWh"),
        _row("Battery usable capacity",
             _get(D, "vehicles", "components", "ess", "battery", "usable_capacity")),
        _row("Charging rate",
             _get(D, "vehicles", "components", "ess", "battery", "refuel_rate"), "kW"),
        _row("Regen efficiency",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "be", "regen_efficiency")),
        _row("Accessory load",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "be", "accessory_load"), "W"),
        _row("O&M cost",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "be", "running_cost"), "CAD/km"),
        "",
        "## Table A10 - FCET (fc) powertrain parameters\n",
        _header(["Parameter", "Value", "Units", "Source"]),
        _row("Powertrain mass (motor)",
             _get(D, "vehicles", "components", "converter", "motor", "fc", "mass"), "kg"),
        _row("FC efficiency",
             _get(D, "vehicles", "components", "converter", "fc", "efficiency")),
        _row("FC lifetime",
             _get(D, "vehicles", "components", "converter", "fc", "lifetime"), "hours"),
        _row("H2 tank capacity (sleeper, 700 bar)",
             Param(fc_cap_sleeper.value, src="Table 16"), "kg"),
        _row("H2 tank specific mass (sleeper, 700 bar)",
             _get(D, "vehicles", "components", "ess", "h2_700bar", "specific_mass"), "kg/kgH2"),
        _row("H2 tank specific mass (day-cab/straight, 350 bar)",
             _get(D, "vehicles", "components", "ess", "h2_350bar", "specific_mass"), "kg/kgH2"),
        _row("H2 usable capacity",
             _get(D, "vehicles", "components", "ess", "h2_700bar", "usable_capacity")),
        _row("H2 refuel rate",
             _get(D, "vehicles", "components", "ess", "h2_700bar", "refuel_rate"), "kg/hr"),
        _row("Regen efficiency",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "fc", "regen_efficiency")),
        _row("Accessory load",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "fc", "accessory_load"), "W"),
        _row("O&M cost",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "fc", "running_cost"), "CAD/km"),
        "",
    ]
    return lines


def render_sales_growth():
    D = PARAM_DICT
    lines = [
        "## Table A10 - Maximum annual sales growth rates\n",
        _header(["Powertrain group", "Nascent (market share < 15%)", "Mature (market share > 15%)", "Source"]),
        _row("dice/he/phe/hice/dhice",
             _get(D, "vehicles", "types", "sleeper", "powertrains", "dice", "cagr_nacent")),
    ]
    nacent_dice = _val(_get(D, "vehicles", "types", "sleeper", "powertrains", "dice", "cagr_nacent"))
    mature_dice = _val(_get(D, "vehicles", "types", "sleeper", "powertrains", "dice", "cagr_mature"))
    src_dice    = _src(_get(D, "vehicles", "types", "sleeper", "powertrains", "dice", "cagr_nacent"))
    nacent_be   = _val(_get(D, "vehicles", "types", "sleeper", "powertrains", "be", "cagr_nacent"))
    mature_be   = _val(_get(D, "vehicles", "types", "sleeper", "powertrains", "be", "cagr_mature"))
    src_be      = _src(_get(D, "vehicles", "types", "sleeper", "powertrains", "be", "cagr_nacent"))
    lines = [
        "## Table A11 - Maximum annual sales growth rates\n",
        _header(["Powertrain group", "Nascent (< 15% share)", "Mature (> 15% share)", "Source"]),
        f"| dice/he/phe/hice/dhice | {nacent_dice} | {mature_dice} | {src_dice} |",
        f"| be/fc                  | {nacent_be}   | {mature_be}   | {src_be}   |",
        "",
    ]
    return lines


def render_costs():
    D = PARAM_DICT
    c = D["vehicles"]["costs"]
    lines = [
        "## Table A12 - Vehicle component costs\n",
        _header(["Component", "Cost", "Units", "Source"]),
        _row("Frame (sleeper)",   _get(D, "vehicles", "types", "sleeper",  "shared", "base_cost"), "CAD"),
        _row("Frame (day-cab)",   _get(D, "vehicles", "types", "day_cab",  "shared", "base_cost"), "CAD"),
        _row("Frame (straight)",  _get(D, "vehicles", "types", "straight", "shared", "base_cost"), "CAD"),
        _row("Diesel engine",           _get(c, "diesel_engine")),
        _row("Combustion transmission", _get(c, "combustion_transmission")),
        _row("Electric transmission",   _get(c, "electric_transmission")),
        _row("Diesel after-treatment",  _get(c, "after_treatment")),
        _row("Diesel tank",             _get(c, "tank"),      "CAD/L"),
        _row("BESS",                    _get(c, "battery"),   "CAD/kWh"),
        _row("H2 tank (700 bar)",       _get(c, "h2_700bar"), "CAD/kg"),
        _row("H2 tank (350 bar)",       _get(c, "h2_350bar"), "CAD/kg"),
        _row("Electric motor",          _get(c, "motor"),     "CAD/kW"),
        _row("Fuel cell",               _get(c, "fc"),        "CAD/kW"),
        _row("HICE engine",             _get(c, "hice_engine")),
        _row("Charger (50 kW)",         _get(c, "charger_50kw")),
        "",
    ]
    return lines


def render_fuel_costs():
    D = PARAM_DICT
    fuels = [
        ("diesel",      "Diesel",                    "L"),
        ("fast_charge", "Electricity (fast charge)", "kWh"),
        ("slow_charge", "Electricity (slow charge)", "kWh"),
        ("h2",          "Hydrogen (electrolysis)",   "kg"),
        ("h2_p",        "Hydrogen (pyrolysis)",      "kg"),
        ("h2_pe",       "Hydrogen (electrified pyrolysis)", "kg"),
    ]
    lines = [
        "## Table A13 - Fuel cost\n",
        "Uniformly distributed between low and high in Monte Carlo analysis.\n",
        _header(["Fuel", "Unit", "2025", "2030", "2035", "2040", "2045", "2050", "Source"]),
    ]
    for key, label, unit in fuels:
        cost_p = _get(D, "fuels", key, "cost")
        src = _src(cost_p)
        v = cost_p.value if isinstance(cost_p, Param) else {}
        def yr(y):
            s = v.get(str(y), {})
            if not s:
                return "-"
            if isinstance(s, dict):
                if s.get("dist") == "const":
                    return str(s["val"])
                if s.get("dist") == "uniform":
                    return f"{s['min']}-{s['max']}"
            return str(s)
        lines.append(f"| {label} | {unit} | {yr(2025)} | {yr(2030)} | {yr(2035)} | {yr(2040)} | {yr(2045)} | {yr(2050)} | {src} |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Discrepancy summary
# ---------------------------------------------------------------------------

def render_discrepancies():
    lines = [
        "## Discrepancy notes\n",
        "The following parameters differ between this model and the values shown in the paper's Appendix A.",
        "Each is flagged with UNREF in the source column above.\n",
        "| Parameter | Model value | Paper value | Location |",
        "|---|---|---|---|",
        "| initial_activity | 47,361,761,620 t-km | 48.9 Gt-km | Table A1 |",
        "| straight drag_coef (2025) | 0.65 | 0.60 | Table A6 |",
        "| charger_50kw (2030 max) | 54,000 CAD | 60,000 CAD | Table A11 |",
        "| electric_transmission efficiency | 0.97 | 96% | Table A4 |",
        "| sleeper hice accessory_load | 3,400 W | 4,250 W | Table A9 |",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build():
    sections = []
    sections += render_general()
    sections += render_vehicle_activity()
    sections += render_fuel_properties()
    sections += render_shared_vehicle()
    sections += render_vehicle_type("sleeper",  "Sleeper-cab trucks",  5)
    sections += render_vehicle_type("day_cab",  "Day-cab trucks",      6)
    sections += render_vehicle_type("straight", "Straight trucks",     7)  # reuses A5-A7 numbering
    sections += render_powertrain_params()
    sections += render_sales_growth()
    sections += render_costs()
    sections += render_fuel_costs()
    sections += render_discrepancies()

    header = [
        "# Appendix A - Model Parameters\n",
        "Generated automatically from `params.py`. Do not edit this file by hand.",
        "Re-run `python documentation/build_appendix.py` to update.\n",
        "**UNREF** in the Source column means no citation was found in the paper's Appendix A",
        "for that value, or the value differs from what the paper shows.\n",
    ]

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appendix.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + sections))
    print(f"Written: {out_path}")


if __name__ == "__main__":
    build()
