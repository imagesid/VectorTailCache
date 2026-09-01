#!/usr/bin/env python3
"""
gen_beamwidth_figures.py — Figures from BFS vs PCS beamwidth comparison

Usage:
    python3 gen_beamwidth_figures.py <results_dir>
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
    print("Usage: python3 gen_beamwidth_figures.py <results_dir>")
    sys.exit(1)

RUN_DIR = sys.argv[1].rstrip('/')
TSV     = os.path.join(RUN_DIR, "results.tsv")
OUT     = os.path.join(RUN_DIR, "figures")
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(TSV):
    print(f"ERROR: results.tsv not found at {TSV}")
    sys.exit(1)

print(f"Run dir : {RUN_DIR}")
print(f"TSV     : {TSV}")
print(f"Output  : {OUT}")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(TSV, sep='\t')
for col in ['mean_us','p999_us','recall','beamwidth','cache_size','L']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Rows      : {len(df)}")
print(f"Policies  : {sorted(df['policy'].unique())}")
print(f"Beamwidths: {sorted(df['beamwidth'].unique())}")
print(f"L values  : {sorted(df['L'].unique())}")

# ── Aggregate ─────────────────────────────────────────────────────────────────
agg = df.groupby(['policy','beamwidth','L']).agg(
    mean_ms  = ('mean_us',  lambda x: x.mean()/1000),
    mean_std = ('mean_us',  lambda x: x.std()/1000),
    p999_ms  = ('p999_us',  lambda x: np.median(x)/1000),
    p999_std = ('p999_us',  lambda x: x.std()/1000),
    recall   = ('recall',   'mean'),
    runs     = ('run',      'count'),
).reset_index()

print(f"\nAggregated:")
print(agg[['policy','beamwidth','L','mean_ms','p999_ms','runs']].to_string(index=False))

beamwidths = sorted(agg['beamwidth'].unique())
L_vals     = sorted(agg['L'].unique())
xs         = range(len(beamwidths))
xlbls      = [f'W={int(bw)}' for bw in beamwidths]

PC  = {'bfs': '#9E9E9E', 'pcs': '#E53935'}
PM  = {'bfs': 'o',       'pcs': '^'}
PL  = {'bfs': 'BFS',     'pcs': 'PCS'}
PLS = {'bfs': '--',      'pcs': '-'}

# ── Annotation helper (no overlap) ───────────────────────────────────────────
def _position_frac(y, ylo, yhi):
    return (y - ylo) / (yhi - ylo) if (yhi - ylo) != 0 else 0.5

def _annotate_grouped(ax, groups, fmt='{:.1f}'):
    ylo, yhi = ax.get_ylim()
    for x, pts in groups.items():
        pts_sorted = sorted(pts, key=lambda p: p[0])
        n = len(pts_sorted)
        center_rank = (n - 1) / 2
        for i, (y, color, text) in enumerate(pts_sorted):
            rank = i - center_rank
            frac = _position_frac(y, ylo, yhi)
            if frac > 0.85:
                sign = -1
            elif frac < 0.15:
                sign = 1
            else:
                sign = 1 if rank >= 0 else -1
            magnitude = 3 + abs(rank) * 11
            dy = sign * magnitude
            va = 'bottom' if sign > 0 else 'top'
            ax.annotate(text, (x, y),
                        textcoords='offset points', xytext=(0, dy),
                        ha='center', va=va, fontsize=8.5, color=color,
                        fontweight='bold')

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.titlesize':    11,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'lines.linewidth':   2.2,
    'lines.markersize':  8,
})

def save(fig, name):
    for ext in ['pdf', 'png']:
        path = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.12)
        print(f"  Saved: {path}")
    plt.close(fig)

# ── Fig 1: P999 per L ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(L_vals), figsize=(8, 3))
if len(L_vals) == 1: axes = [axes]

def get_raw_bw_p999(policy, bw, L):
    rows = df[(df['policy']==policy) & (df['beamwidth']==bw) & (df['L']==L)]['p999_us']
    return rows.values / 1000.0

def is_outlier(vals, threshold=1.5):
    if len(vals) < 3:
        return np.zeros(len(vals), dtype=bool)
    q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    iqr = q3 - q1
    lo, hi = q1 - threshold*iqr, q3 + threshold*iqr
    return (vals < lo) | (vals > hi)

for i, (ax, L) in enumerate(zip(axes, L_vals)):
    sub_L = agg[agg['L'] == L]
    groups = {}
    for policy in ['bfs', 'pcs']:
        sub  = sub_L[sub_L['policy'] == policy].set_index('beamwidth')
        vals = [sub['p999_ms'].get(bw, np.nan) for bw in beamwidths]

        # IQR band excluding outliers
        q25_vals, q75_vals = [], []
        for bw in beamwidths:
            raw  = get_raw_bw_p999(policy, bw, L)
            mask = ~is_outlier(raw)
            clean = raw[mask]
            if len(clean) > 0:
                q25_vals.append(np.percentile(clean, 25))
                q75_vals.append(np.percentile(clean, 75))
            else:
                q25_vals.append(np.nan)
                q75_vals.append(np.nan)

        errs_lo = [v-q for v,q in zip(vals,q25_vals) if not (np.isnan(v) or np.isnan(q))]
        errs_hi = [q-v for v,q in zip(vals,q75_vals) if not (np.isnan(v) or np.isnan(q))]
        valid_xi = [j for j,v in enumerate(vals) if not np.isnan(v)]

        ax.errorbar(valid_xi, [vals[j] for j in valid_xi],
                    yerr=[errs_lo, errs_hi] if errs_lo else None,
                    color=PC[policy], marker=PM[policy],
                    linestyle=PLS[policy], label=PL[policy],
                    capsize=4)

        # Outlier points as x markers
        # Show all raw runs as small scattered points
        for j, bw in enumerate(beamwidths):
            raw = get_raw_bw_p999(policy, bw, L)
            jitter = np.linspace(-0.08, 0.08, len(raw))
            ax.scatter(
                np.full(len(raw), j) + jitter,
                raw,
                color=PC[policy],
                alpha=0.35,
                s=18,
                edgecolors='none',
                zorder=2,
            )

        for j, v in enumerate(vals):
            if np.isnan(v): continue
            groups.setdefault(j, []).append((v, PC[policy], f'{v:.1f}'))
    r = sub_L['recall'].mean()
    ax.set_xticks(xs); ax.set_xticklabels(xlbls)
    ax.set_xlabel('Beam Width')
    if i == 0: ax.set_ylabel('P999 Latency (ms)\n(median, raw runs as dots)')
    ax.set_title(f'L={int(L)}  (Recall={r:.1f}%)')
    ax.legend(fontsize=9)
    # set ylim with padding BEFORE annotating so _position_frac is correct
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.22)
    ax.margins(x=0.12)
    _annotate_grouped(ax, groups)

# fig.suptitle(f'BFS vs PCS across Beamwidths — SIFT1M, cache=100K, T=1',
#              fontsize=11, y=1.02)
plt.tight_layout(pad=0.8, w_pad=1.0, h_pad=0.8)
save(fig, 'beamwidth_fig1_p999')

# ── Fig 2: Mean latency per L ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(L_vals), figsize=(8, 3))
if len(L_vals) == 1: axes = [axes]

for i, (ax, L) in enumerate(zip(axes, L_vals)):
    sub_L = agg[agg['L'] == L]
    groups2 = {}
    for policy in ['bfs', 'pcs']:
        sub  = sub_L[sub_L['policy'] == policy].set_index('beamwidth')
        vals = [sub['mean_ms'].get(bw, np.nan) for bw in beamwidths]
        errs = [sub['mean_std'].get(bw, np.nan) for bw in beamwidths]
        ax.errorbar(xs, vals, yerr=errs,
                    color=PC[policy], marker=PM[policy],
                    linestyle=PLS[policy], label=PL[policy],
                    capsize=4)
        for j, v in enumerate(vals):
            if np.isnan(v): continue
            groups2.setdefault(j, []).append((v, PC[policy], f'{v:.1f}'))
    r = sub_L['recall'].mean()
    ax.set_xticks(xs); ax.set_xticklabels(xlbls)
    ax.set_xlabel('Beam Width')
    if i == 0: ax.set_ylabel('Mean Latency (ms)')
    ax.set_title(f'L={int(L)}  (Recall={r:.1f}%)')
    ax.legend(fontsize=9)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax * 1.22)
    ax.margins(x=0.12)
    _annotate_grouped(ax, groups2)

# fig.suptitle(f'BFS vs PCS across Beamwidths — SIFT1M, cache=100K, T=1',
#              fontsize=11, y=1.02)
plt.tight_layout(pad=0.8, w_pad=1.0, h_pad=0.8)
save(fig, 'beamwidth_fig2_mean')

# ── Fig 3: P999 reduction vs BFS across beamwidths ───────────────────────────
fig, axes = plt.subplots(1, len(L_vals), figsize=(8, 3))
if len(L_vals) == 1: axes = [axes]

for i, (ax, L) in enumerate(zip(axes, L_vals)):
    sub_L = agg[agg['L'] == L]
    bfs_d = sub_L[sub_L['policy']=='bfs'].set_index('beamwidth')['p999_ms']
    pcs_d = sub_L[sub_L['policy']=='pcs'].set_index('beamwidth')['p999_ms']
    reductions = []; valid_xs = []; valid_xlbls = []
    for j, bw in enumerate(beamwidths):
        bv = bfs_d.get(bw, np.nan); pv = pcs_d.get(bw, np.nan)
        if not np.isnan(bv) and not np.isnan(pv) and bv > 0:
            reductions.append((bv-pv)/bv*100)
            valid_xs.append(j); valid_xlbls.append(xlbls[j])
    colors = ['#4CAF50' if r > 2 else ('#F44336' if r < -2 else '#FF9800')
              for r in reductions]
    bars = ax.bar(valid_xs, reductions, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, reductions):
        ypos = bar.get_height()*1.02 if val >= 0 else bar.get_height()-1.5
        ax.text(bar.get_x()+bar.get_width()/2, ypos,
                f'{val:+.1f}%', ha='center', va='bottom',
                fontsize=7, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(valid_xs); ax.set_xticklabels(valid_xlbls)
    ax.set_xlabel('Beam Width')
    if i == 0: ax.set_ylabel('P999 Reduction vs BFS (%)')
    ax.set_title(f'L={int(L)}  (positive = PCS better)')
    yabs = max(abs(r) for r in reductions) if reductions else 1
    ax.set_ylim(-yabs*1.4, yabs*1.4)

# fig.suptitle('PCS P999 Reduction over BFS across Beamwidths\nSIFT1M, cache=100K, T=1',
#              fontsize=11, y=1.02)
plt.tight_layout(pad=0.8, w_pad=1.0, h_pad=0.8)
save(fig, 'beamwidth_fig3_reduction')

# ── Summary table ─────────────────────────────────────────────────────────────
SEP = "─" * 72
for L in L_vals:
    print(f"\n{SEP}")
    print(f"  Summary at L={int(L)}")
    print(SEP)
    print(f"  {'BW':>4}  {'Policy':<6}  {'Mean(ms)':>10}  {'P999(ms)':>10}  "
          f"{'Recall':>8}  {'Runs':>5}")
    print(f"  {'─'*68}")
    for bw in beamwidths:
        for policy in ['pcs', 'bfs']:
            sub = agg[(agg['policy']==policy) &
                      (agg['beamwidth']==bw) &
                      (agg['L']==L)]
            if len(sub) == 0: continue
            r = sub.iloc[0]
            print(f"  {int(bw):>4}  {policy:<6}  "
                  f"{r.mean_ms:>10.1f}  {r.p999_ms:>10.1f}  "
                  f"{r.recall:>8.2f}%  {int(r.runs):>5}")
        bv = agg[(agg['policy']=='bfs') & (agg['beamwidth']==bw) &
                 (agg['L']==L)]['p999_ms'].values
        pv = agg[(agg['policy']=='pcs') & (agg['beamwidth']==bw) &
                 (agg['L']==L)]['p999_ms'].values
        if len(bv) and len(pv) and bv[0] > 0:
            delta = (bv[0]-pv[0])/bv[0]*100
            sym = "✓ PCS wins" if delta > 2 else ("✗ BFS wins" if delta < -2 else "≈ tie")
            print(f"  {'':>4}  {'':>6}  P999 reduction: {delta:+.1f}%  {sym}")
        print()
    print(SEP)

print(f"\nFigures saved to: {OUT}/")