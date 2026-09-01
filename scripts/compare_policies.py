#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# compare_policies.py 
# Generate the mean-latency figure (BFS vs PCS) from
# an existing results.tsv, with error bars (std dev across runs).
# =============================================================================
import sys
import os
import glob
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# =============================================================================
# PATH SETUP
# =============================================================================
OUT_BASE = os.environ.get("VTC_RESULTS", "/home/agung/vector/results")
if len(sys.argv) >= 2:
    OUT = sys.argv[1].rstrip("/")
else:
    candidates = glob.glob(os.path.join(OUT_BASE, "final_comparison_*"))
    candidates = [p for p in candidates if os.path.isdir(p)]
    candidates.sort(key=os.path.getmtime, reverse=True)
    if candidates:
        OUT = candidates[0].rstrip("/")
    else:
        print(f"ERROR: no final_comparison_* directory found under {OUT_BASE}", file=sys.stderr)
        sys.exit(1)
TSV = os.path.join(OUT, "results.tsv")
FIGS = os.path.join(OUT, "figures2")
LOG = os.path.join(OUT, "figure_generation2.log")
os.makedirs(FIGS, exist_ok=True)
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
if not os.path.isfile(TSV):
    log(f"ERROR: results.tsv not found: {TSV}")
    sys.exit(1)
log("Generating mean-latency figure only")
log(f"Results: {TSV}")
log(f"Figures: {FIGS}")
# =============================================================================
# FIGURE STYLE
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13,
    'axes.titlesize': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'lines.linewidth': 2.2, 'lines.markersize': 8,
})
PC = {'bfs': '#4E4B51', 'pcs': '#E53935'}
PM = {'bfs': 'o',       'pcs': '^'}
PL = {'bfs': 'BFS', 'pcs': 'TCS'}
def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(f"{FIGS}/{name}.{ext}")
    plt.close(fig)
    print(f"  Saved: {name}")
def xlabels(vals):
    out = []
    for v in vals:
        v = int(v)
        out.append(f"{v//1000}K" if v >= 1000 else str(v))
    return out
# =============================================================================
# LOAD RESULTS
# =============================================================================
df = pd.read_csv(TSV, sep='\t')
required = {'policy', 'cache_size', 'L', 'mean_us'}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in results.tsv: {sorted(missing)}")
for col in ['mean_us', 'L', 'cache_size']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['policy', 'cache_size', 'L', 'mean_us'])
agg = df.groupby(['policy', 'cache_size', 'L']).agg(
    mean_us=('mean_us', 'mean'),
    mean_std=('mean_us', 'std'),
    n=('mean_us', 'count'),
).reset_index()
agg['mean_std'] = agg['mean_std'].fillna(0.0)
cache_sizes = [int(x) for x in sorted(agg['cache_size'].unique())]
L_vals = [int(x) for x in sorted(agg['L'].unique())]
xs = range(len(cache_sizes))
xlbls = xlabels(cache_sizes)
if not cache_sizes or not L_vals:
    raise SystemExit("No usable rows found in results.tsv")

# =============================================================================
# PRINT: verify paper text numbers (mean latency only)
# =============================================================================
SEP = "=" * 70
print(f"\n{SEP}")
print("PAPER TEXT VERIFICATION — MEAN LATENCY")
print(SEP)

print("\n[Mean §Mean] At cache=100K:")
for L in L_vals:
    for policy in ['bfs', 'pcs']:
        sub = agg[(agg['policy']==policy) & (agg['L']==L) &
                  (agg['cache_size']==100000)]
        if len(sub) == 0: continue
        r = sub.iloc[0]
        print(f"  {policy} L={L}: mean={r.mean_us/1000:.1f}ms (±{r.mean_std/1000:.2f}ms)")
    bfs = agg[(agg['policy']=='bfs') & (agg['L']==L) &
              (agg['cache_size']==100000)]
    pcs = agg[(agg['policy']=='pcs') & (agg['L']==L) &
              (agg['cache_size']==100000)]
    if len(bfs) and len(pcs):
        bv = bfs.iloc[0].mean_us
        pv = pcs.iloc[0].mean_us
        print(f"  → mean reduction: {(bv-pv)/bv*100:.1f}%")
    print()
print(f"{SEP}\n")

# =============================================================================
# Fig: Mean latency, with error bars (std dev across runs)
# =============================================================================
fig, axes = plt.subplots(1, len(L_vals), figsize=(8, 3))
axes = np.atleast_1d(axes).flatten()
_ANNOT_DY = {'bfs': 8, 'pcs': -8}
_ANNOT_VA = {'bfs': 'bottom', 'pcs': 'top'}
for ax, L in zip(axes, L_vals):
    points5 = {}
    for policy in ['bfs', 'pcs']:
        sub = agg[(agg['policy'] == policy) & (agg['L'] == L)].set_index('cache_size')
        vals = np.array([sub['mean_us'].get(c, np.nan) / 1000 for c in cache_sizes])
        errs = np.array([sub['mean_std'].get(c, 0.0) / 1000 for c in cache_sizes])
        ax.errorbar(
            xs, vals, yerr=errs,
            color=PC[policy], marker=PM[policy], label=PL[policy],
            capsize=4, capthick=1.3, linewidth=1.8, markersize=6,
        )
        print(policy, L, vals, "±", errs)
        for j, v in enumerate(vals):
            if np.isnan(v): continue
            points5.setdefault(j, []).append((policy, v))
    ax.set_xticks(xs); ax.set_xticklabels(xlbls, rotation=30)
    ax.set_xlabel('Nodes Cached')
    if ax is axes[0]: ax.set_ylabel('Mean Latency (ms)')
    else: ax.set_ylabel('')
    ax.set_title(f'L={L}')
    # ax.legend()
    ax.legend(loc='lower left')
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.set_ylim(ymin - yrange * 0.15, ymax + yrange * 0.25)
    ax.margins(x=0.12)
    for j, pts in points5.items():
        for policy, v in pts:
            ax.annotate(f'{v:.1f}', (j, v),
                        textcoords='offset points',
                        xytext=(0, _ANNOT_DY[policy]),
                        ha='center', va=_ANNOT_VA[policy],
                        fontsize=10, color=PC[policy], fontweight='bold')
plt.tight_layout()
fig.subplots_adjust(wspace=0.20)
save(fig, 'fig5_mean_latency')

# =============================================================================
# Summary table
# =============================================================================
print("\n" + "=" * 60)
print("MEAN LATENCY SUMMARY")
print("=" * 60)
print(f"{'Policy':<8} {'Cache':>8} {'L':>5} {'Mean(ms)':>10} {'Std(ms)':>9}")
print("-" * 60)
for L in L_vals:
    for cache in cache_sizes:
        for policy in ['pcs', 'bfs']:
            r = agg[(agg['policy'] == policy) &
                    (agg['L'] == L) &
                    (agg['cache_size'] == cache)]
            if len(r) == 0: continue
            r = r.iloc[0]
            print(f"{policy:<8} {int(cache):>8,} {L:>5} "
                  f"{r.mean_us / 1000:>10.2f} {r.mean_std / 1000:>9.2f}")
        bfs = agg[(agg['policy'] == 'bfs') & (agg['L'] == L) & (agg['cache_size'] == cache)]
        pcs = agg[(agg['policy'] == 'pcs') & (agg['L'] == L) & (agg['cache_size'] == cache)]
        if len(bfs) and len(pcs):
            bv = bfs.iloc[0]['mean_us']; pv = pcs.iloc[0]['mean_us']
            delta = (bv - pv) / bv * 100
            sym = "✓ PCS wins" if delta > 2 else ("✗ BFS wins" if delta < -2 else "≈ tie")
            print(f"  → mean reduction: {delta:+.1f}%  {sym}")
        print()
# =============================================================================
# DONE
# =============================================================================
log("=======================================================")
log("DONE")
log(f"Figures: {FIGS}")
log("=======================================================")
print("")
print("Copy figure:")
print(f"  cp {FIGS}/fig5_mean_latency.* ~/vtc_final_figures/")