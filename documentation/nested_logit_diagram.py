"""
Generates figures illustrating the NEST_TREE nesting structure (model.py) for the
methodology write-up. Reads the tree shape and nest scales live from model.py / data.py,
so the figures can't drift out of sync if NEST_TREE or params.py: fleet.nest_lambdas changes.

Two styles are produced (pick whichever suits the write-up; both carry identical content):
  tree  -> documentation/figures/nested_logit_tree.png
           Dendrogram, root at left, all powertrains aligned in one column at right.
  boxes -> documentation/figures/nested_logit_boxes.png
           Containment diagram: each nest drawn as a box literally containing its children.
           More compact / single-column friendly; shows nesting without connector lines.

Run: C:\\Users\\ivana\\anaconda3\\python.exe documentation\\nested_logit_diagram.py [tree|boxes|both]
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import NEST_TREE
from data import PARAMS
from plots.plot_utils import PT_LABELS, PT_COLOR

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_HERE, 'figures')

NEST_LAMBDAS = PARAMS['fleet']['nest_lambdas']

# Display-only relabeling for the write-up (the NEST_TREE node itself is still named
# 'Liquid' in model.py, matching its own docs/comments -- this override only affects
# what these figures show, not the model).
_DISPLAY_NAME = {'Liquid': 'Diesel'}
_ROOT_LABEL = 'New vehicle purchase'
_ROOT_LABEL_WRAPPED = 'New vehicle\npurchase'  # 'tree' style sets it beside the root, so it wraps

# Expanded powertrain names, so the figure is readable without the abbreviation key.
# Falls back to PT_LABELS for any powertrain not listed (e.g. a future addition).
_PT_FULL = {
    'dice':  'Diesel ICE',
    'he':    'Hybrid electric',
    'phe':   'Plug-in hybrid electric',
    'be':    'Battery electric',
    'fc':    'Fuel cell',
    'hice':  'Hydrogen ICE',
    'dhice': 'Dual-fuel hydrogen / diesel ICE',
}

# Built via chr() (not a literal char) to keep this source file ASCII-only per project convention.
_LAMBDA = chr(0x03bb)  # Greek small letter lambda -- nest scale parameter

_INK, _MUTED, _RULE = '#1a1a1a', '#6b6b6b', '#b0b0b0'
_NEST_FACE = ['#ffffff', '#f5f6f8', '#eaecf0']  # by depth, for the 'boxes' style
_NEST_EDGE = '#c9ccd1'


def _tint(colour, frac=0.16):
    """colour blended towards white -- a fill light enough to carry dark text."""
    r, g, b = to_rgb(colour)
    return (1 - frac + frac * r, 1 - frac + frac * g, 1 - frac + frac * b)


def _shade(colour, frac=0.72):
    """colour darkened -- readable text in the same hue as the box outline."""
    r, g, b = to_rgb(colour)
    return (r * frac, g * frac, b * frac)


def _leaves(node):
    if isinstance(node, str):
        yield node
    else:
        for child in node[2]:
            yield from _leaves(child)


def _leaf_text(p):
    return PT_LABELS.get(p, p), _PT_FULL.get(p, '')


def _nest_text(name, lambda_key):
    label = _ROOT_LABEL if lambda_key is None and name == 'root' else _DISPLAY_NAME.get(name, name)
    lam = 1.0 if lambda_key is None else NEST_LAMBDAS[lambda_key]
    return label, f'{_LAMBDA} = {lam:.2g}'


def _leaf_pill(ax, x, y, w, h, p, z=2, label_inset=None):
    """Rounded pill for one powertrain: tinted fill, full-strength outline, dark-hue text."""
    colour = PT_COLOR.get(p, '#777777')
    abbr, full = _leaf_text(p)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0,rounding_size=0.05',
        facecolor=_tint(colour), edgecolor=colour, linewidth=0.9, zorder=z))
    if label_inset is None:
        ax.text(x + w / 2, y + h / 2, abbr, ha='center', va='center',
                fontsize=8, fontweight='bold', color=_shade(colour), zorder=z + 1)
    else:
        ax.text(x + label_inset[0], y + h / 2, abbr, ha='left', va='center',
                fontsize=8, fontweight='bold', color=_shade(colour), zorder=z + 1)
        if full:
            ax.text(x + label_inset[1], y + h / 2, full, ha='left', va='center',
                    fontsize=8, color=_MUTED, zorder=z + 1)


def _new_axes(fig_w, fig_h, xlim, ylim):
    """Axes spanning the whole figure with 1 data unit = 1 inch, so every size below
    can be reasoned about directly in inches -- no unit-conversion guesswork."""
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')
    return fig, ax


# ---------------------------------------------------------------------------
# Style 1: dendrogram
# ---------------------------------------------------------------------------
def plot_tree(path, full_names=True):
    """
    Root at left, powertrains in one aligned column at right (a leaf sits in that column
    whatever its depth, so the tree reads as rows rather than a ragged staircase).
    Nests are junctions on the connector, labelled above/below the line -- no nest boxes,
    which keeps the only filled shapes in the figure the things being chosen between.
    """
    order = list(_leaves(NEST_TREE))
    y_gap = 0.44
    y_of = {p: -i * y_gap for i, p in enumerate(order)}

    max_depth = 0

    def depth_of(node, d=0):
        nonlocal max_depth
        max_depth = max(max_depth, d)
        if not isinstance(node, str):
            for c in node[2]:
                depth_of(c, d + 1)

    depth_of(NEST_TREE)
    x_step = 1.15
    x_stop = [d * x_step for d in range(max_depth)]   # junction x by nest depth
    stub = 0.32                                        # parent junction -> elbow
    pill_w, pill_h = 0.82, 0.31
    x_leaf = x_stop[-1] + 0.65                          # left edge of the leaf column
    junctions = []

    def place(node, depth):
        if isinstance(node, str):
            return y_of[node]
        name, lambda_key, kids = node
        ys = [place(c, depth + 1) for c in kids]
        y = sum(ys) / len(ys)
        junctions.append((x_stop[depth], y, depth, name, lambda_key, kids, ys))
        return y

    place(NEST_TREE, 0)

    left = -1.02 if full_names else -1.02
    right = x_leaf + pill_w + (2.45 if full_names else 0.10)
    top = 0.28
    bottom = -(len(order) - 1) * y_gap - 0.26
    m = 0.06
    fig, ax = _new_axes(right - left + 2 * m, top - bottom + 2 * m,
                        (left - m, right + m), (bottom - m, top + m))

    for x, y, depth, name, lambda_key, kids, ys in junctions:
        elbow = x + stub
        ax.plot([x, elbow], [y, y], color=_RULE, lw=1.0, solid_capstyle='round', zorder=1)
        if len(ys) > 1:
            ax.plot([elbow, elbow], [min(ys), max(ys)], color=_RULE, lw=1.0,
                    solid_capstyle='round', zorder=1)
        for child, y_c in zip(kids, ys):
            x_c = x_leaf if isinstance(child, str) else x_stop[depth + 1]
            ax.plot([elbow, x_c], [y_c, y_c], color=_RULE, lw=1.0,
                    solid_capstyle='round', zorder=1)
        ax.plot([x], [y], marker='o', ms=2.6, color=_RULE, zorder=2)
        label, lam = _nest_text(name, lambda_key)
        if lambda_key is None:
            ax.text(x - 0.10, y, _ROOT_LABEL_WRAPPED, ha='right', va='center',
                    fontsize=8.5, fontweight='bold', color=_INK, linespacing=1.3)
        else:
            ax.text(x - 0.05, y + 0.045, label, ha='right', va='bottom',
                    fontsize=8.5, fontweight='bold', color=_INK)
            ax.text(x - 0.05, y - 0.05, lam, ha='right', va='top',
                    fontsize=7.5, color=_MUTED)

    for p in order:
        y = y_of[p]
        _leaf_pill(ax, x_leaf, y - pill_h / 2, pill_w, pill_h, p)
        if full_names and _PT_FULL.get(p):
            ax.text(x_leaf + pill_w + 0.14, y, _PT_FULL[p], ha='left', va='center',
                    fontsize=8, color=_MUTED)

    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Style 2: containment boxes
# ---------------------------------------------------------------------------
def plot_boxes(path):
    """
    Each nest is a box containing its children, so the nesting is shown by containment
    rather than by connector lines. Sizes are computed bottom-up, so the layout adapts
    to any NEST_TREE shape.
    """
    leaf_w, leaf_h = 2.90, 0.32
    pad_x, pad_top, pad_bot, gap = 0.13, 0.31, 0.13, 0.09

    def size(node):
        if isinstance(node, str):
            return leaf_w, leaf_h
        sizes = [size(c) for c in node[2]]
        return (max(s[0] for s in sizes) + 2 * pad_x,
                sum(s[1] for s in sizes) + gap * (len(node[2]) - 1) + pad_top + pad_bot)

    total_w, total_h = size(NEST_TREE)
    m = 0.08
    fig, ax = _new_axes(total_w + 2 * m, total_h + 2 * m,
                        (-m, total_w + m), (-m, total_h + m))

    def draw(node, x, y, w, h, depth):
        """(x, y) is the lower-left corner, in inches."""
        if isinstance(node, str):
            _leaf_pill(ax, x, y, w, h, node, z=depth + 2, label_inset=(0.12, 0.78))
            return
        name, lambda_key, kids = node
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0,rounding_size=0.07',
            facecolor=_NEST_FACE[min(depth, len(_NEST_FACE) - 1)], edgecolor=_NEST_EDGE,
            linewidth=0.9, zorder=depth + 1))
        label, lam = _nest_text(name, lambda_key)
        ax.text(x + pad_x + 0.02, y + h - pad_top + 0.08, label, ha='left', va='bottom',
                fontsize=8.5, fontweight='bold', color=_INK, zorder=depth + 3)
        ax.text(x + w - pad_x - 0.02, y + h - pad_top + 0.08, lam, ha='right', va='bottom',
                fontsize=7.5, color=_MUTED, zorder=depth + 3)
        cy = y + h - pad_top
        for child in kids:
            ch = size(child)[1]
            cy -= ch
            draw(child, x + pad_x, cy, w - 2 * pad_x, ch, depth + 1)
            cy -= gap

    draw(NEST_TREE, 0, 0, total_w, total_h, 0)

    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    which = (sys.argv[1] if len(sys.argv) > 1 else 'both').lower()
    os.makedirs(_OUT_DIR, exist_ok=True)
    if which in ('tree', 'both'):
        plot_tree(os.path.join(_OUT_DIR, 'nested_logit_tree.png'))
    if which in ('boxes', 'both'):
        plot_boxes(os.path.join(_OUT_DIR, 'nested_logit_boxes.png'))
    print(f'Wrote figure(s) to {_OUT_DIR}')
