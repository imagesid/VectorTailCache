#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# gen_concurrency_tradeoff.py 
# =============================================================================
import sys
import os
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
if len(sys.argv) < 2:
    print("Usage: python3 gen_concurrency_tradeoff.py <results_dir>")
    sys.exit(1)
OUT = sys.argv[1].rstrip("/")
TSV = os.path.join(OUT, "results.tsv")
FIGS = os.path.join(OUT, "figures")
LOG = os.path.join(OUT, "figure_generation.log")
os.makedirs(FIGS, exist_ok=True)
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
if not os.path.isfile(TSV):
    log(f"ERROR: results.tsv not found: {TSV}")
    sys.exit(1)
log("Generating concurrency x beamwidth tradeoff figure")
log(f"Results: {TSV}")
log(f"Figures: {FIGS}")
# =============================================================================
# FIGURE STYLE
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
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
# =============================================================================
# LOAD RESULTS
# =============================================================================
df = pd.read_csv(TSV, sep='\t')
required = {'policy', 'beamwidth', 'threads', 'qps', 'mean_us'}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in results.tsv: {sorted(missing)}")
for col in ['beamwidth', 'threads', 'qps', 'mean_us', 'L']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['policy', 'beamwidth', 'threads', 'qps', 'mean_us'])
agg = df.groupby(['policy', 'beamwidth', 'threads']).agg(
    qps=('qps', 'mean'),
    qps_std=('qps', 'std'),
    mean_us=('mean_us', 'mean'),
    mean_std=('mean_us', 'std'),
    n=('qps', 'count'),
).reset_index()
agg['qps_std'] = agg['qps_std'].fillna(0.0)
agg['mean_std'] = agg['mean_std'].fillna(0.0)
beamwidths = [int(x) for x in sorted(agg['beamwidth'].unique())]
threads_list = [int(x) for x in sorted(agg['threads'].unique())]
xs = range(len(threads_list))
xlbls = [str(t) for t in threads_list]
if not beamwidths or not threads_list:
    raise SystemExit("No usable rows found in results.tsv")

# =============================================================================
# PRINT: verify paper text numbers (QPS scaling efficiency)
# =============================================================================
SEP = "=" * 70
print(f"\n{SEP}")
print("PAPER TEXT VERIFICATION — CONCURRENCY x BEAMWIDTH")
print(SEP)

print("\n[Scaling §Concurrency] QPS efficiency vs ideal linear (baseline=threads=1):")
for bw in beamwidths:
    for policy in ['bfs', 'pcs']:
        sub = agg[(agg['policy']==policy) & (agg['beamwidth']==bw)].set_index('threads')
        if 1 not in sub.index: continue
        base = sub.loc[1, 'qps']
        r = sub.loc[threads_list[-1]] if threads_list[-1] in sub.index else None
        if r is None: continue
        eff = r['qps'] / (base * threads_list[-1]) * 100
        print(f"  {policy} W={bw}: qps@t=1={base:.1f}  qps@t={threads_list[-1]}={r['qps']:.1f}  efficiency={eff:.1f}%")
    bfs = agg[(agg['policy']=='bfs') & (agg['beamwidth']==bw) & (agg['threads']==threads_list[-1])]
    pcs = agg[(agg['policy']=='pcs') & (agg['beamwidth']==bw) & (agg['threads']==threads_list[-1])]
    if len(bfs) and len(pcs):
        bv, pv = bfs.iloc[0].qps, pcs.iloc[0].qps
        print(f"  → PCS advantage at t={threads_list[-1]}: {(pv-bv)/bv*100:+.1f}%")
    print()
print(f"{SEP}\n")

# =============================================================================
# Fig: QPS (row 1) + Mean latency (row 2), with error bars (std dev across runs)
#
# Annotation direction differs by row:
#   Row 1 (QPS): PCS labels above the line, BFS labels below.
#   Row 2 (Mean latency): BFS labels above the line, PCS labels below.
# =============================================================================
n_bw = len(beamwidths)
fig, axes = plt.subplots(2, n_bw, figsize=(9, 5))
axes = np.atleast_2d(axes)
if axes.shape[0] == 1:
    axes = axes.reshape(2, -1)

# Row 1 (QPS): PCS labels above the line, BFS labels below.
_ANNOT_DY_ROW1 = {'bfs': -8, 'pcs': 8}
_ANNOT_VA_ROW1 = {'bfs': 'top', 'pcs': 'bottom'}

# Row 2 (Mean latency): BFS above, PCS below.
_ANNOT_DY_ROW2 = {'bfs': 8, 'pcs': -8}
_ANNOT_VA_ROW2 = {'bfs': 'bottom', 'pcs': 'top'}

for col_idx, bw in enumerate(beamwidths):
    # ── Row 1: QPS vs threads ──
    ax = axes[0, col_idx]
    points_qps = {}
    for policy in ['bfs', 'pcs']:
        sub = agg[(agg['policy'] == policy) & (agg['beamwidth'] == bw)].set_index('threads')
        vals = np.array([sub['qps'].get(t, np.nan) for t in threads_list])
        errs = np.array([sub['qps_std'].get(t, 0.0) for t in threads_list])
        ax.errorbar(
            xs, vals, yerr=errs,
            color=PC[policy], marker=PM[policy], label=PL[policy],
            capsize=4, capthick=1.3, linewidth=1.8, markersize=6,
        )
        print(policy, 'W='+str(bw), 'qps', vals, "±", errs)
        for j, v in enumerate(vals):
            if np.isnan(v): continue
            points_qps.setdefault(j, []).append((policy, v))
    ax.set_xticks(xs); ax.set_xticklabels(xlbls)
    ax.set_xlabel('Threads')
    if col_idx == 0: ax.set_ylabel('QPS')
    else: ax.set_ylabel('')
    ax.set_title(f'W={bw}')
    ax.legend()
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.set_ylim(ymin - yrange * 0.15, ymax + yrange * 0.25)
    ax.margins(x=0.12)
    for j, pts in points_qps.items():
        for policy, v in pts:
            ax.annotate(f'{v:.0f}', (j, v),
                        textcoords='offset points',
                        xytext=(0, _ANNOT_DY_ROW1[policy]),
                        ha='center', va=_ANNOT_VA_ROW1[policy],
                        fontsize=8.5, color=PC[policy], fontweight='bold')

    # ── Row 2: Mean latency vs threads ──
    ax2 = axes[1, col_idx]
    points_mean = {}
    for policy in ['bfs', 'pcs']:
        sub = agg[(agg['policy'] == policy) & (agg['beamwidth'] == bw)].set_index('threads')
        vals = np.array([sub['mean_us'].get(t, np.nan) / 1000 for t in threads_list])
        errs = np.array([sub['mean_std'].get(t, 0.0) / 1000 for t in threads_list])
        ax2.errorbar(
            xs, vals, yerr=errs,
            color=PC[policy], marker=PM[policy], label=PL[policy],
            capsize=4, capthick=1.3, linewidth=1.8, markersize=6,
        )
        print(policy, 'W='+str(bw), 'mean_ms', vals, "±", errs)
        for j, v in enumerate(vals):
            if np.isnan(v): continue
            points_mean.setdefault(j, []).append((policy, v))
    ax2.set_xticks(xs); ax2.set_xticklabels(xlbls)
    ax2.set_xlabel('Threads')
    if col_idx == 0: ax2.set_ylabel('Mean Latency (ms)')
    else: ax2.set_ylabel('')
    ax2.legend()
    ymin, ymax = ax2.get_ylim()
    yrange = ymax - ymin
    ax2.set_ylim(ymin - yrange * 0.15, ymax + yrange * 0.25)
    ax2.margins(x=0.12)
    for j, pts in points_mean.items():
        for policy, v in pts:
            ax2.annotate(f'{v:.2f}', (j, v),
                         textcoords='offset points',
                         xytext=(0, _ANNOT_DY_ROW2[policy]),
                         ha='center', va=_ANNOT_VA_ROW2[policy],
                         fontsize=8.5, color=PC[policy], fontweight='bold')

plt.tight_layout()
fig.subplots_adjust(wspace=0.3, hspace=0.4)
save(fig, 'concurrency_tradeoff')

# =============================================================================
# Summary table
# =============================================================================
print("\n" + "=" * 78)
print("CONCURRENCY x BEAMWIDTH SUMMARY")
print("=" * 78)
print(f"{'Policy':<8} {'BW':>4} {'Threads':>8} {'QPS':>9} {'Std':>7} {'Mean(ms)':>10} {'Std(ms)':>8}")
print("-" * 78)
for bw in beamwidths:
    for t in threads_list:
        for policy in ['pcs', 'bfs']:
            r = agg[(agg['policy'] == policy) & (agg['beamwidth'] == bw) & (agg['threads'] == t)]
            if len(r) == 0: continue
            r = r.iloc[0]
            print(f"{policy:<8} {bw:>4} {t:>8} "
                  f"{r.qps:>9.1f} {r.qps_std:>7.1f} "
                  f"{r.mean_us/1000:>10.2f} {r.mean_std/1000:>8.2f}")
        bfs = agg[(agg['policy'] == 'bfs') & (agg['beamwidth'] == bw) & (agg['threads'] == t)]
        pcs = agg[(agg['policy'] == 'pcs') & (agg['beamwidth'] == bw) & (agg['threads'] == t)]
        if len(bfs) and len(pcs):
            bv, pv = bfs.iloc[0]['qps'], pcs.iloc[0]['qps']
            delta = (pv - bv) / bv * 100
            sym = "✓ PCS wins" if delta > 2 else ("✗ BFS wins" if delta < -2 else "≈ tie")
            print(f"  → PCS QPS advantage: {delta:+.1f}%  {sym}")
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
print(f"  cp {FIGS}/concurrency_tradeoff.* ~/vtc_final_figures/")