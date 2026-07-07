"""
Shared plotting utilities for the fleet2 HDT adoption model.

All style constants (colours, labels) and reusable helper functions live here
so that fleet_plots.py, vehicle_plots.py, and policy plot files stay consistent.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 11, 'ytick.labelsize': 11})

SAMPLE_YEARS = [2025, 2030, 2035, 2040, 2045, 2050]

PT_COLOR = {p: _CYCLE[i % len(_CYCLE)]
            for i, p in enumerate(['dice', 'he', 'phe', 'be', 'fc', 'hice', 'dhice'])}

# MJ of useful (traction) energy per unit of fuel consumed.
# Diesel/H2: LHV x powertrain efficiency. Electric: 3.6 MJ/kWh x drivetrain efficiency.
# h2 efficiency uses FC (~60%); hice/dhice are a minor share so the error is small.
FUEL_TO_MJ = {
    'diesel':      35.8 * 0.43,   # L   -> 15.4 MJ
    'h2':         120.0 * 0.60,   # kg  -> 72.0 MJ
    'h2_p':       120.0 * 0.60,
    'h2_pe':      120.0 * 0.60,
    'slow_charge':  3.6 * 0.90,   # kWh ->  3.2 MJ
    'fast_charge':  3.6 * 0.90,
}

FUEL_COLOR = {
    'diesel':      PT_COLOR['dice'],
    'slow_charge': PT_COLOR['be'],
    'fast_charge': _CYCLE[7 % len(_CYCLE)],
    'h2':          PT_COLOR['fc'],
    'h2_p':        PT_COLOR['hice'],
    'h2_pe':       PT_COLOR['dhice'],
}

EMIS_COLOR = {e: _CYCLE[i % len(_CYCLE)]
              for i, e in enumerate(['Use', 'Supply', 'Embodied'])}

COST_COLOR = {c: _CYCLE[i % len(_CYCLE)]
              for i, c in enumerate(['Capital', 'Operational', 'Fuel', 'Driver', 'Carbon Tax'])}

POLICY_COLOR = {
    'carbon_tax': _CYCLE[5 % len(_CYCLE)],
}

# ---------------------------------------------------------------------------
# Label dicts
# ---------------------------------------------------------------------------

PT_LABELS = {
    'dice':  'DICE',
    'he':    'HE',
    'phe':   'PHE',
    'be':    'BE',
    'fc':    'FC',
    'hice':  'HICE',
    'dhice': 'DHICE',
}

K_LABELS = {
    'sleeper':  'Sleeper',
    'day_cab':  'Day Cab',
    'straight': 'Straight',
}

# Merged display labels -- covers powertrains, vehicle types, cost/emission keys
DISPLAY_LABELS = {
    **PT_LABELS,
    **K_LABELS,
    'Capital':     'Capital',
    'Operational': 'Operational',
    'Fuel':        'Fuel',
    'Driver':      'Driver',
    'Carbon Tax':  'Carbon Tax',
    'Use':         'Use',
    'Supply':      'Supply',
    'Embodied':    'Embodied',
}

KEY_LABELS = {
    'frame':                   'Frame',
    'trailer':                 'Trailer',
    'payload':                 'Payload',
    'ice':                     'ICE',
    'motor':                   'Motor',
    'battery':                 'Battery',
    'fc':                      'Fuel Cell',
    'diesel_tank':             'Diesel Tank',
    'h2_700bar':               'H2 Tank (700 bar)',
    'h2_350bar':               'H2 Tank (350 bar)',
    'tire':                    'Tire',
    'trailer_tire':            'Trailer Tire',
    'electronic_controller':   'Electronic Controller',
    'combustion_transmission': 'Combustion Transmission',
    'electric_transmission':   'Electric Transmission',
    'after_treatment':         'Aftertreatment',
    'engine':                  'Engine',
    'h2_tank':                 'H2 Tank',
    'tank':                    'Diesel Tank',
    'charger':                 'Charger',
    'capital':                 'Capital',
    'operational':             'Operational',
    'fuel':                    'Fuel',
    'driver':                  'Driver',
    'fc_replacements':         'FC Replacements',
    'carbon_tax':              'Carbon Tax',
    'revenue':                 'Revenue',
    'embodied':                'Embodied',
    'supply':                  'Supply',
    'use':                     'Use',
}

# ---------------------------------------------------------------------------
# Bar / stacked-bar helpers
# ---------------------------------------------------------------------------

def _unique_keys(fleet, k, attr):
    """All unique non-zero component keys across every vehicle of type k, in first-appearance order."""
    seen = []
    for p in fleet.P[k]:
        for y in SAMPLE_YEARS:
            if (k, p, y) not in fleet.vehicles:
                continue
            for key, val in getattr(fleet.vehicles[k, p, y], attr).items():
                if key not in seen and np.any(np.asarray(val) != 0):
                    seen.append(key)
    return seen

def _colours(keys):
    """Standard cycle color for each key, assigned by position."""
    return {key: _CYCLE[i % len(_CYCLE)] for i, key in enumerate(keys)}

def _bar_layout(n, year_gap=5, fill=0.80, internal_gap=0.10):
    width   = (year_gap * fill - (n - 1) * internal_gap) / max(n, 1)
    offsets = [i * (width + internal_gap) - (year_gap * fill) / 2 + width / 2 for i in range(n)]
    return width, offsets

def _stacked_bar(ax, x, comps, width, col):
    """Stacked bar using pre-built name->color dict col."""
    bottom = 0.0
    for key, val in comps.items():
        val = float(np.asarray(val).flat[0])
        if val == 0:
            continue
        ax.bar(x, val, bottom=bottom, width=width, color=col[key], label=key)
        bottom += val
    return bottom

def _legend(ax, keys, col, **kw):
    """Legend in key order, one entry per key, using pre-built colors."""
    handles = [plt.Rectangle((0, 0), 1, 1, color=col[k]) for k in keys]
    labels  = [KEY_LABELS.get(k, k) for k in keys]
    ax.legend(handles, labels, **kw)

# ---------------------------------------------------------------------------
# Plotting class -- fleet-level line/area layouts
# ---------------------------------------------------------------------------

class Plotting:
    def __init__(self, start_year=None, end_year=None):
        # Import lazily to avoid circular imports when plot_utils is loaded before model
        from model import START_YEAR, END_YEAR
        self.T = np.arange(start_year or START_YEAR, (end_year or END_YEAR) + 1)
        self.sample_years = np.array(SAMPLE_YEARS)

    @staticmethod
    def _sum_by_outer(result):
        """Sum across outer keys (vehicle types k), keep inner keys (categories)."""
        summed = {}
        for subdict in result.values():
            for cat, arr in subdict.items():
                arr = np.asarray(arr, dtype=float)
                summed[cat] = summed[cat] + arr if cat in summed else arr.copy()
        return summed

    @staticmethod
    def plot_lines(t, result, ax, x_label=None, y_label=None,
                   add_total=False, color_map=None):
        """
        Plot each category in result as a line.
        If v is 1-D (single run): plain line.
        If v is 2-D (MC, axis-0 = runs): mean line + p5-p95 fill band.
        Returns the local y-axis maximum for shared-axis sizing.
        """
        local_max = 0
        for k, v in result.items():
            v   = np.asarray(v, dtype=float)
            col = (color_map or {}).get(k)
            if v.ndim == 1:
                ax.plot(t, v, label=DISPLAY_LABELS.get(k, k), color=col)
                local_max = max(local_max, v.max() if v.size else 0)
            else:
                p5, p95 = np.percentile(v, [5, 95], axis=0)
                mean    = np.mean(v, axis=0)
                ax.fill_between(t, p5, p95, alpha=0.2, color=col)
                ax.plot(t, mean, label=DISPLAY_LABELS.get(k, k), color=col)
                local_max = max(local_max, float(p95.max()))

        if add_total:
            arrays = [np.asarray(v, dtype=float) for v in result.values()]
            total  = sum(arrays)
            if total.ndim == 1:
                ax.plot(t, total, label='Total', linewidth=2, color='black')
                local_max = max(local_max, total.max() if total.size else 0)
            else:
                p5, p95 = np.percentile(total, [5, 95], axis=0)
                ax.fill_between(t, p5, p95, alpha=0.1, color='black')
                ax.plot(t, np.mean(total, axis=0), label='Total', linewidth=2, color='black')
                local_max = max(local_max, float(p95.max()))

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(0, None)
        return local_max

    def plot_by_both(self, result, title=None, x_label=None, y_label=None,
                     add_total=False, color_map=None):
        """One subplot per vehicle type k; one line per category (powertrain/component)."""
        n    = len(result)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharex=True, sharey=True,
                                 constrained_layout=True, dpi=150)
        axes = np.atleast_1d(axes)

        all_handles, all_labels = [], []
        global_max = 0
        for ax, (k, subdict) in zip(axes, result.items()):
            local_max = self.plot_lines(self.T, subdict, ax, x_label, y_label,
                                        add_total, color_map)
            global_max = max(global_max, local_max)
            ax.set_title(DISPLAY_LABELS.get(k, k))
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in all_labels:
                    all_handles.append(h); all_labels.append(l)

        for ax in axes:
            ax.set_ylim(0, global_max * 1.1)
        axes[-1].legend(handles=all_handles, labels=all_labels,
                        bbox_to_anchor=(1.05, 1), loc='upper left',
                        borderaxespad=0., fontsize=8)
        if title:
            fig.suptitle(title)
        return fig, axes

    def plot_by_inner(self, result, title=None, x_label=None, y_label=None,
                      add_total=False, color_map=None):
        """Single plot; sum across vehicle types k, one line per category."""
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
        summed  = self._sum_by_outer(result)
        self.plot_lines(self.T, summed, ax, x_label, y_label, add_total, color_map)
        ax.legend(fontsize=8)
        if title:
            fig.suptitle(title)
        return fig, ax
