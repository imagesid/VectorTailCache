#!/usr/bin/env python3
"""
gen_sata_figures.py — Generate figures from SATA BFS vs PCS results
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

# ── Args ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python3 gen_sata_figures.py <results_dir>")
    sys.exit(1)

RESULTS_DIR = sys.argv[1].rstrip('/')
TSV  = os.path.join(RESULTS_DIR, "results.tsv")
FIGS = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGS, exist_ok=True)

print(f"Results dir : {RESULTS_DIR}")
print(f"TSV         : {TSV}")
print(f"Figures     : {FIGS}")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(TSV, sep='\t')
print(f"\nColumns : {list(df.columns)}")
print(f"Rows    : {len(df)}")

for col in [
    'qps', 'mean_us', 'p999_us', 'mean_ios', 'mean_io_us',
    'recall', 'tail_amp', 'cache_size', 'L'
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Aggregate ─────────────────────────────────────────────────────────────────
agg = df.groupby(['policy', 'cache_size', 'L']).agg(
    mean_us  = ('mean_us',   'mean'),
    mean_std = ('mean_us',   'std'),
    p999_us  = ('p999_us',   'mean'),
    p999_std = ('p999_us',   'std'),
    tail_amp = ('tail_amp',  'mean'),
    recall   = ('recall',    'mean'),
    runs     = ('run',       'count'),
).reset_index()

print(f"\nAggregated ({len(agg)} rows):")
print(
    agg[
        ['policy', 'cache_size', 'L', 'mean_us', 'p999_us', 'tail_amp']
    ].to_string(index=False)
)

cache_sizes = sorted(agg['cache_size'].unique())
L_vals      = sorted(agg['L'].unique())

PC = {'bfs': '#9E9E9E', 'pcs': '#E53935'}
PM = {'bfs': 'o',       'pcs': '^'}
PL = {'bfs': 'BFS',     'pcs': 'PCS'}

# ── Global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   11,
    'axes.labelsize':   11,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'legend.fontsize':  11,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'lines.linewidth':  2.2,
    'lines.markersize': 7,
})

def xlabels(vals):
    return [f'{int(v)//1000}K' if v >= 1000 else '0' for v in vals]

def save(fig, name):
    for ext in ['pdf', 'png']:
        path = f"{FIGS}/{name}.{ext}"
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)

def finish_row(fig):
    fig.tight_layout(pad=0.35, w_pad=0.25)
    fig.subplots_adjust(wspace=0.25)

# ── Fixed-direction value labels + raw-run scatter dots ──────────────────────
# (ported from compare_policies.py / compare_policies_all_datasets.py)
# bfs is always labeled above its point, pcs always below, regardless of
# which policy has the higher value at a given x.
_ANNOT_DY = {'bfs': 9, 'pcs': -9}
_ANNOT_VA = {'bfs': 'bottom', 'pcs': 'top'}
_SCATTER_OFFSET = {'bfs': -0.06, 'pcs': 0.06}

def _pad_ylim_for_labels(ax, top_frac=0.22, bottom_frac=0.12):
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.set_ylim(ymin - yrange * bottom_frac, ymax + yrange * top_frac)

def _annotate_fixed(ax, points):
    """points: dict of x -> list of (policy, value) tuples."""
    for x, pts in points.items():
        for policy, v in pts:
            ax.annotate(
                f'{v:.1f}', (x, v),
                textcoords='offset points',
                xytext=(0, _ANNOT_DY[policy]),
                ha='center', va=_ANNOT_VA[policy],
                fontsize=8.5, color=PC[policy],
                fontweight='bold'
            )

def _scatter_raw_runs(ax, x_index, policy, raw_vals_ms):
    """Plot individual raw-run values as small jittered, semi-transparent
    dots underneath the mean/median line."""
    raw_vals_ms = np.asarray(raw_vals_ms, dtype=float)
    raw_vals_ms = raw_vals_ms[~np.isnan(raw_vals_ms)]

    if len(raw_vals_ms) == 0:
        return

    jitter = np.linspace(-0.025, 0.025, len(raw_vals_ms))
    x_center = x_index + _SCATTER_OFFSET[policy]

    ax.scatter(
        np.full(len(raw_vals_ms), x_center) + jitter,
        raw_vals_ms,
        color=PC[policy],
        alpha=0.35,
        s=18,
        edgecolors='none',
        zorder=2,
    )

xs    = range(len(cache_sizes))
xlbls = xlabels(cache_sizes)

# ── Fig 1: P999 per L value ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(L_vals), figsize=(8, 3))
if len(L_vals) == 1:
    axes = [axes]

for i, (ax, L) in enumerate(zip(axes, L_vals)):
    points1 = {}
    for policy in ['bfs', 'pcs']:
        sub = agg[
            (agg['policy'] == policy) &
            (agg['L'] == L)
        ].set_index('cache_size')

        vals = [sub['p999_us'].get(c, np.nan) / 1000 for c in cache_sizes]
        errs = [sub['p999_std'].get(c, np.nan) / 1000 for c in cache_sizes]

        ax.errorbar(
            xs,
            vals,
            yerr=errs,
            color=PC[policy],
            marker=PM[policy],
            label=PL[policy],
            capsize=4
        )

        # Raw per-run dots
        raw_sub = df[
            (df['policy'] == policy) &
            (df['L'] == L)
        ]

        for j, c in enumerate(cache_sizes):
            raw_vals = raw_sub[raw_sub['cache_size'] == c]['p999_us'].values / 1000
            _scatter_raw_runs(ax, j, policy, raw_vals)

        for j, v in enumerate(vals):
            if np.isnan(v):
                continue
            points1.setdefault(j, []).append((policy, v))

    recall_val = agg[agg['L'] == L]['recall'].mean()

    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls)
    ax.set_xlabel('Nodes Cached in DRAM')

    if i == 0:
        ax.set_ylabel('P999 Latency (ms)')

    ax.set_title(f'L={int(L)}  (Recall≈{recall_val:.1f}%)')
    ax.legend(fontsize=8)

    ax.margins(x=0.12)
    _pad_ylim_for_labels(ax)
    _annotate_fixed(ax, points1)

# fig.suptitle(
#     'SATA SSD: P999 Latency — BFS vs PCS\nSIFT1M, T=1, 3 cold-start runs',
#     fontsize=11,
#     y=1.02
# )

finish_row(fig)
save(fig, 'sata_fig1_p999')

# ── Fig 2: P999 reduction % bars ──────────────────────────────────────────────
fig, axes = plt.subplots(1, len(L_vals), figsize=(8, 3))
if len(L_vals) == 1:
    axes = [axes]

for i, (ax, L) in enumerate(zip(axes, L_vals)):
    bfs_d = agg[
        (agg['policy'] == 'bfs') &
        (agg['L'] == L)
    ].set_index('cache_size')['p999_us']

    pcs_d = agg[
        (agg['policy'] == 'pcs') &
        (agg['L'] == L)
    ].set_index('cache_size')['p999_us']

    reductions = []
    valid_xs = []
    valid_xlbls = []

    for j, c in enumerate(cache_sizes):
        if c == 0:
            continue

        bv = bfs_d.get(c, np.nan)
        pv = pcs_d.get(c, np.nan)

        if not np.isnan(bv) and not np.isnan(pv) and bv > 0:
            reductions.append((bv - pv) / bv * 100)
            valid_xs.append(j)
            valid_xlbls.append(xlbls[j])

    if not reductions:
        ax.set_title(f'L={int(L)} — no data')
        continue

    colors = [
        '#4CAF50' if r > 2 else ('#F44336' if r < -2 else '#FF9800')
        for r in reductions
    ]

    bars = ax.bar(
        valid_xs,
        reductions,
        color=colors,
        alpha=0.85,
        width=0.5
    )

    for bar, val in zip(bars, reductions):
        ypos = bar.get_height() * 1.02 if val >= 0 else bar.get_height() - 1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f'{val:+.1f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(valid_xs)
    ax.set_xticklabels(valid_xlbls)
    ax.set_xlabel('Nodes Cached')

    if i == 0:
        ax.set_ylabel('P999 Reduction vs BFS (%)')

    ax.set_title(f'L={int(L)}  (positive = PCS better)')

    yabs = max(abs(r) for r in reductions)
    if yabs == 0:
        yabs = 1.0
    ax.set_ylim(-yabs * 1.4, yabs * 1.4)

# fig.suptitle(
#     'SATA SSD: P999 Reduction of PCS over BFS\nSIFT1M, T=1',
#     fontsize=11,
#     y=1.02
# )

finish_row(fig)
save(fig, 'sata_fig2_reduction')

# ── Fig 3: Mean latency ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(L_vals), figsize=(8, 3))
if len(L_vals) == 1:
    axes = [axes]

for i, (ax, L) in enumerate(zip(axes, L_vals)):
    points3 = {}
    for policy in ['bfs', 'pcs']:
        sub = agg[
            (agg['policy'] == policy) &
            (agg['L'] == L)
        ].set_index('cache_size')

        vals = [sub['mean_us'].get(c, np.nan) / 1000 for c in cache_sizes]
        errs = [sub['mean_std'].get(c, np.nan) / 1000 for c in cache_sizes]

        ax.errorbar(
            xs,
            vals,
            yerr=errs,
            color=PC[policy],
            marker=PM[policy],
            label=PL[policy],
            capsize=4
        )

        # Raw per-run dots
        raw_sub = df[
            (df['policy'] == policy) &
            (df['L'] == L)
        ]

        for j, c in enumerate(cache_sizes):
            raw_vals = raw_sub[raw_sub['cache_size'] == c]['mean_us'].values / 1000
            _scatter_raw_runs(ax, j, policy, raw_vals)

        for j, v in enumerate(vals):
            if np.isnan(v):
                continue
            points3.setdefault(j, []).append((policy, v))

    ax.set_xticks(xs)
    ax.set_xticklabels(xlbls)
    ax.set_xlabel('Nodes Cached in DRAM')

    if i == 0:
        ax.set_ylabel('Mean Latency (ms)')

    ax.set_title(f'L={int(L)}')
    ax.legend(fontsize=8)

    ax.margins(x=0.12)
    _pad_ylim_for_labels(ax)
    _annotate_fixed(ax, points3)

# fig.suptitle(
#     'SATA SSD: Mean Latency — BFS vs PCS\nSIFT1M, T=1, 3 cold-start runs',
#     fontsize=11,
#     y=1.02
# )

finish_row(fig)
save(fig, 'sata_fig3_mean')



# ── Summary table ─────────────────────────────────────────────────────────────
SEP = "─" * 72
print(f"\n{SEP}")
print("  SATA SSD: BFS vs PCS SUMMARY")
print(SEP)

print(
    f"  {'Cache':>8}  {'L':>5}  {'Policy':<8}  "
    f"{'Mean(ms)':>10}  {'P999(ms)':>10}  {'Tail':>7}  {'Recall':>8}"
)
print(f"  {'─' * 68}")

for cache in cache_sizes:
    for L in L_vals:
        for policy in ['pcs', 'bfs']:
            sub = agg[
                (agg['policy'] == policy) &
                (agg['cache_size'] == cache) &
                (agg['L'] == L)
            ]

            if len(sub) == 0:
                continue

            r = sub.iloc[0]
            clbl = f'{int(cache)//1000}K' if cache >= 1000 else '0'

            print(
                f"  {clbl:>8}  {int(L):>5}  {policy:<8}  "
                f"{r.mean_us / 1000:>10.1f}  {r.p999_us / 1000:>10.1f}  "
                f"{r.tail_amp:>7.2f}x  {r.recall:>8.2f}%"
            )

        bv = agg[
            (agg['policy'] == 'bfs') &
            (agg['cache_size'] == cache) &
            (agg['L'] == L)
        ]['p999_us'].values

        pv = agg[
            (agg['policy'] == 'pcs') &
            (agg['cache_size'] == cache) &
            (agg['L'] == L)
        ]['p999_us'].values

        if len(bv) and len(pv) and bv[0] > 0:
            delta = (bv[0] - pv[0]) / bv[0] * 100
            sym = "✓ PCS wins" if delta > 2 else ("✗ BFS wins" if delta < -2 else "≈ tie")

            print(
                f"  {'':>8}  {'':>5}  {'':>8}  "
                f"P999 reduction: {delta:+.1f}%  {sym}"
            )

        print()

print(SEP)
print(f"\nFigures saved to: {FIGS}/")