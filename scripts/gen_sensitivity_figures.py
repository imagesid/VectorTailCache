#!/usr/bin/env python3
"""
gen_sensitivity_figures.py — Generate figures from PCS sensitivity analysis.
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


# ── Configurable figure settings ──────────────────────────────────────────────

# Distance between each point and its annotation, measured in screen points.
# Increase this value to move annotations farther below the points.
ANNOTATION_DISTANCE = 12

# Additional distance when multiple annotations share the same x position.
ANNOTATION_STACK_GAP = 11

# Font sizes.
BASE_FONT_SIZE       = 15
TITLE_FONT_SIZE      = 15
AXIS_LABEL_FONT_SIZE = 14
TICK_FONT_SIZE       = 14
LEGEND_FONT_SIZE     = 14
ANNOTATION_FONT_SIZE = 15
BAR_LABEL_FONT_SIZE  = 11

# Figure sizes.
P999_FIGSIZE      = (10, 6)
REDUCTION_FIGSIZE = (10, 6)

# Space between rows and columns.
GRID_HEIGHT_SPACE = 0.68
GRID_WIDTH_SPACE  = 0.50


# ── Args ──────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python3 gen_sensitivity_figures.py <results_dir>")
    sys.exit(1)

RUN_DIR = sys.argv[1].rstrip('/')
TSV = os.path.join(RUN_DIR, "results.tsv")
OUT = os.path.join(RUN_DIR, "figures")

os.makedirs(OUT, exist_ok=True)

if not os.path.exists(TSV):
    print(f"ERROR: results.tsv not found at {TSV}")
    sys.exit(1)

print(f"Run dir : {RUN_DIR}")
print(f"TSV     : {TSV}")
print(f"Output  : {OUT}")


# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(TSV, sep='\t')

for col in [
    'mean_us',
    'p999_us',
    'recall',
    'cache_size',
    'L',
    'percentile',
    'run',
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Rows    : {len(df)}")
print(f"Policies: {sorted(df['policy'].dropna().unique())}")


# ── Aggregate ─────────────────────────────────────────────────────────────────

agg = (
    df.groupby(
        ['policy', 'percentile', 'L'],
        dropna=False
    )
    .agg(
        mean_ms=('mean_us', lambda x: x.mean() / 1000),
        mean_std=('mean_us', lambda x: x.std() / 1000),
        p999_ms=('p999_us', lambda x: np.median(x) / 1000),
        p999_std=('p999_us', lambda x: x.std() / 1000),
        recall=('recall', 'mean'),
        runs=('run', 'count'),
    )
    .reset_index()
)

L_vals = sorted(df['L'].dropna().unique())

pcts = sorted(
    df.loc[
        df['policy'] == 'pcs',
        'percentile'
    ].dropna().unique()
)

bfs = agg[agg['policy'] == 'bfs']

print(f"L values    : {L_vals}")
print(f"Percentiles : {pcts}")


# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': BASE_FONT_SIZE,

    'axes.titlesize': TITLE_FONT_SIZE,
    'axes.labelsize': AXIS_LABEL_FONT_SIZE,

    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,

    'legend.fontsize': LEGEND_FONT_SIZE,

    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',

    'axes.grid': True,
    'grid.alpha': 0.25,

    'axes.spines.top': False,
    'axes.spines.right': False,

    'lines.linewidth': 2.2,
    'lines.markersize': 8,
})


# ── General helpers ───────────────────────────────────────────────────────────

def save(fig, name):
    """Save one figure as PDF and PNG."""

    for ext in ['pdf', 'png']:
        path = os.path.join(OUT, f"{name}.{ext}")

        fig.savefig(
            path,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.15,
        )

        print(f"  Saved: {path}")

    plt.close(fig)


def format_number(value):
    """Format percentile values without unnecessary trailing zeros."""

    try:
        return f'{float(value):g}'
    except (TypeError, ValueError):
        return str(value)


def create_centered_axes(number_of_axes, figsize):
    """
    Create a two-column layout.

    For three subfigures:

        ┌────────────┐  ┌────────────┐
        │  Plot 1    │  │  Plot 2    │
        └────────────┘  └────────────┘

               ┌────────────┐
               │  Plot 3    │
               └────────────┘

    The bottom subplot occupies the middle of the second row.
    """

    if number_of_axes <= 0:
        raise ValueError("At least one subplot is required.")

    if number_of_axes == 1:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(figsize[0] / 2, figsize[1] / 2),
            constrained_layout=True,
        )
        return fig, [ax]

    if number_of_axes == 2:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(figsize[0], figsize[1] / 2),
            constrained_layout=True,
        )
        return fig, list(np.atleast_1d(axes))

    if number_of_axes == 3:
        fig = plt.figure(
            figsize=figsize,
            constrained_layout=False,
        )

        gs = fig.add_gridspec(
            2,
            4,
            hspace=GRID_HEIGHT_SPACE,
            wspace=GRID_WIDTH_SPACE,
        )

        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[1, 1:3]),
        ]

        # Manual margins are more predictable than tight_layout with this
        # deliberately centred GridSpec arrangement.
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            bottom=0.10,
            top=0.96,
        )

        return fig, axes

    # Fallback for more than three L values.
    number_of_columns = 2
    number_of_rows = int(np.ceil(number_of_axes / number_of_columns))

    fig, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(
            figsize[0],
            figsize[1] * number_of_rows / 2,
        ),
        constrained_layout=True,
    )

    axes = list(np.atleast_1d(axes).flatten())

    # Remove unused subplot slots.
    for unused_ax in axes[number_of_axes:]:
        unused_ax.remove()

    return fig, axes[:number_of_axes]


def should_show_ylabel(axis_index, number_of_axes):
    """
    Show the y-axis label on the left subplot of each row.

    For a three-subfigure centred layout, the first and third plots receive
    the y-axis label.
    """

    if number_of_axes == 3:
        return axis_index in (0, 2)

    return axis_index % 2 == 0


# ── Annotation helper ─────────────────────────────────────────────────────────

def _annotate_grouped(
    ax,
    groups,
    distance=ANNOTATION_DISTANCE,
    stack_gap=ANNOTATION_STACK_GAP,
    fontsize=ANNOTATION_FONT_SIZE,
):
    """
    Place every annotation below its corresponding point.

    Parameters
    ----------
    ax:
        Matplotlib axes.

    groups:
        Dictionary in the form:
            x_position -> [(y_value, colour, text), ...]

    distance:
        Initial distance below the point in screen points.

    stack_gap:
        Additional downward distance when multiple labels share the same
        x position.

    fontsize:
        Annotation font size.
    """

    for x, points in groups.items():
        points_sorted = sorted(points, key=lambda item: item[0])

        for annotation_index, (y, colour, text) in enumerate(points_sorted):
            downward_distance = (
                distance
                + annotation_index * stack_gap
            )

            ax.annotate(
                text,
                xy=(x, y),
                xytext=(0, -downward_distance),
                textcoords='offset points',

                ha='center',
                va='top',

                fontsize=fontsize,
                fontweight='bold',
                color=colour,

                annotation_clip=False,
                clip_on=False,
                zorder=10,
            )


# ── Colours ───────────────────────────────────────────────────────────────────

pcs_colors = plt.cm.Reds(
    np.linspace(
        0.35,
        0.90,
        len(pcts),
    )
)

pct_to_color = {
    pct: colour
    for pct, colour in zip(pcts, pcs_colors)
}


# ── Fig 1: P999 vs percentile ─────────────────────────────────────────────────

fig, axes = create_centered_axes(
    number_of_axes=len(L_vals),
    figsize=P999_FIGSIZE,
)

for axis_index, (ax, L) in enumerate(zip(axes, L_vals)):
    groups = {}
    pcs_p999 = []

    for percentile_index, pct in enumerate(pcts):
        sub = agg[
            (agg['policy'] == 'pcs')
            & (agg['percentile'] == pct)
            & (agg['L'] == L)
        ]

        if sub.empty:
            pcs_p999.append(np.nan)
            continue

        row = sub.iloc[0]

        pcs_p999.append(row.p999_ms)

        ax.errorbar(
            [percentile_index],
            [row.p999_ms],
            yerr=[row.p999_std],

            color=pct_to_color[pct],
            marker='^',

            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            linewidth=1.8,
            markersize=10,

            zorder=4,
        )

        groups.setdefault(percentile_index, []).append(
            (
                row.p999_ms,
                pct_to_color[pct],
                f'{row.p999_ms:.1f}',
            )
        )

    # Connect PCS points.
    valid_points = [
        (index, value)
        for index, value in enumerate(pcs_p999)
        if not np.isnan(value)
    ]

    if valid_points:
        x_values, y_values = zip(*valid_points)

        ax.plot(
            list(x_values),
            list(y_values),

            color='#E53935',
            linewidth=2.0,
            alpha=0.55,

            zorder=2,
        )

    # BFS baseline as a horizontal dashed line.
    bfs_sub = bfs[bfs['L'] == L]

    if not bfs_sub.empty:
        bfs_value = bfs_sub.iloc[0].p999_ms
        bfs_std = bfs_sub.iloc[0].p999_std

        ax.axhline(
            bfs_value,

            color='#777777',
            linestyle='--',
            linewidth=2.3,

            label=f'BFS ({bfs_value:.1f} ms)',
            zorder=1,
        )

        ax.fill_between(
            [-0.5, len(pcts) - 0.5],

            [bfs_value - bfs_std, bfs_value - bfs_std],
            [bfs_value + bfs_std, bfs_value + bfs_std],

            color='#9E9E9E',
            alpha=0.14,
            zorder=0,
        )

    ax.set_xticks(range(len(pcts)))

    ax.set_xticklabels(
        [format_number(pct) for pct in pcts],
        rotation=30,
        ha='right',
        fontsize=TICK_FONT_SIZE,
    )

    ax.tick_params(
        axis='both',
        labelsize=TICK_FONT_SIZE,
    )

    ax.set_xlabel(
        r'Tail Percentile Threshold ($\tau$)',
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=8,
    )

    if should_show_ylabel(axis_index, len(axes)):
        ax.set_ylabel(
            'P999 Latency (ms)\n(median, 5 runs)',
            fontsize=AXIS_LABEL_FONT_SIZE,
            labelpad=8,
        )

    recall_value = agg.loc[
        agg['L'] == L,
        'recall'
    ].mean()

    ax.set_title(
        f'L={int(L)}  (Recall={recall_value:.1f}%)',
        fontsize=TITLE_FONT_SIZE,
        pad=12,
    )

    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc='lower right',
        frameon=True,
    )

    ax.margins(
        x=0.12,
        y=0.18,
    )

    # Add extra upper space while keeping the lower limit at zero.
    _, current_ymax = ax.get_ylim()

    ax.set_ylim(
        bottom=0,
        top=current_ymax * 1.18,
    )

    # Annotate only after the final y-axis limit is established.
    _annotate_grouped(
        ax,
        groups,
        distance=ANNOTATION_DISTANCE,
        stack_gap=ANNOTATION_STACK_GAP,
        fontsize=ANNOTATION_FONT_SIZE,
    )

save(
    fig,
    'sensitivity_percentile_p999',
)


# ── Fig 2: P999 reduction vs BFS ─────────────────────────────────────────────

fig, axes = create_centered_axes(
    number_of_axes=len(L_vals),
    figsize=REDUCTION_FIGSIZE,
)

for axis_index, (ax, L) in enumerate(zip(axes, L_vals)):
    bfs_sub = bfs[bfs['L'] == L]

    if bfs_sub.empty:
        ax.text(
            0.5,
            0.5,
            'BFS data unavailable',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=AXIS_LABEL_FONT_SIZE,
        )
        continue

    bfs_p999 = bfs_sub.iloc[0].p999_ms
    reductions = []

    for pct in pcts:
        sub = agg[
            (agg['policy'] == 'pcs')
            & (agg['percentile'] == pct)
            & (agg['L'] == L)
        ]

        if sub.empty:
            reductions.append(np.nan)
            continue

        pcs_value = sub.iloc[0].p999_ms

        if bfs_p999 == 0 or np.isnan(bfs_p999):
            reductions.append(np.nan)
        else:
            reductions.append(
                (bfs_p999 - pcs_value)
                / bfs_p999
                * 100
            )

    bar_colours = [
        pct_to_color[pct]
        for pct in pcts
    ]

    bars = ax.bar(
        range(len(pcts)),
        reductions,

        color=bar_colours,
        alpha=0.88,
        width=0.62,

        zorder=3,
    )

    for bar, value in zip(bars, reductions):
        if np.isnan(value):
            continue

        vertical_offset = 0.7 if value >= 0 else -0.7
        vertical_alignment = 'bottom' if value >= 0 else 'top'

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + vertical_offset,

            f'{value:.1f}%',

            ha='center',
            va=vertical_alignment,

            fontsize=BAR_LABEL_FONT_SIZE,
            fontweight='bold',

            clip_on=False,
            zorder=5,
        )

    ax.axhline(
        0,
        color='black',
        linewidth=1.0,
        zorder=2,
    )

    ax.set_xticks(range(len(pcts)))

    ax.set_xticklabels(
        [format_number(pct) for pct in pcts],
        rotation=30,
        ha='right',
        fontsize=TICK_FONT_SIZE,
    )

    ax.tick_params(
        axis='both',
        labelsize=TICK_FONT_SIZE,
    )

    ax.set_xlabel(
        r'Tail Percentile Threshold ($\tau$)',
        fontsize=AXIS_LABEL_FONT_SIZE,
        labelpad=8,
    )

    if should_show_ylabel(axis_index, len(axes)):
        ax.set_ylabel(
            'P999 Reduction vs BFS (%)',
            fontsize=AXIS_LABEL_FONT_SIZE,
            labelpad=8,
        )

    ax.set_title(
        f'L={int(L)}',
        fontsize=TITLE_FONT_SIZE,
        pad=12,
    )

    valid_reductions = [
        value
        for value in reductions
        if not np.isnan(value)
    ]

    if valid_reductions:
        minimum_value = min(valid_reductions)
        maximum_value = max(valid_reductions)

        if minimum_value >= 0:
            ax.set_ylim(
                bottom=0,
                top=max(maximum_value * 1.30, maximum_value + 3),
            )
        else:
            value_range = maximum_value - minimum_value

            if value_range == 0:
                value_range = max(abs(maximum_value), 1)

            ax.set_ylim(
                bottom=minimum_value - value_range * 0.15,
                top=maximum_value + value_range * 0.25,
            )
    else:
        ax.set_ylim(0, 1)

    ax.margins(x=0.10)

save(
    fig,
    'sensitivity_percentile_reduction',
)


# ── Summary table ─────────────────────────────────────────────────────────────

SEP = "─" * 80

for L in L_vals:
    print(f"\n{SEP}")
    print(f"  L={int(L)} — P999 sensitivity (cache=100K, T=1)")
    print(SEP)

    print(
        f"  {'Policy/Pct':<14}  "
        f"{'Mean(ms)':>10}  "
        f"{'P999(ms)':>10}  "
        f"{'P999std':>9}  "
        f"{'vs BFS':>8}  "
        f"{'Recall':>8}"
    )

    print(f"  {'─' * 76}")

    bfs_sub = bfs[bfs['L'] == L]

    bfs_p999 = (
        bfs_sub.iloc[0].p999_ms
        if not bfs_sub.empty
        else float('nan')
    )

    if not bfs_sub.empty:
        row = bfs_sub.iloc[0]

        print(
            f"  {'BFS':<14}  "
            f"{row.mean_ms:>10.1f}  "
            f"{row.p999_ms:>10.1f}  "
            f"{row.p999_std:>9.2f}  "
            f"{'baseline':>8}  "
            f"{row.recall:>8.2f}%"
        )

    for pct in pcts:
        sub = agg[
            (agg['policy'] == 'pcs')
            & (agg['percentile'] == pct)
            & (agg['L'] == L)
        ]

        if sub.empty:
            continue

        row = sub.iloc[0]

        if bfs_p999 == 0 or np.isnan(bfs_p999):
            delta_text = 'N/A'
        else:
            delta = (
                (bfs_p999 - row.p999_ms)
                / bfs_p999
                * 100
            )

            delta_text = (
                f"+{delta:.1f}%"
                if delta > 0
                else f"{delta:.1f}%"
            )

        policy_text = f'PCS p={format_number(pct)}'

        print(
            f"  {policy_text:<14}  "
            f"{row.mean_ms:>10.1f}  "
            f"{row.p999_ms:>10.1f}  "
            f"{row.p999_std:>9.2f}  "
            f"{delta_text:>8}  "
            f"{row.recall:>8.2f}%"
        )

    print(SEP)

print(f"\nFigures saved to: {OUT}/")