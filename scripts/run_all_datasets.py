#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# run_all_datasets.py 
# Generate a 5-subfigure comparison (BFS vs PCS) from
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

if len(sys.argv) < 2:
    print("Usage: python3 run_all_datasets.py <log_folder>", file=sys.stderr)
    candidates = glob.glob(os.path.join(OUT_BASE, "all_datasets_*"))
    candidates = [p for p in candidates if os.path.isdir(p)]
    candidates.sort(key=os.path.getmtime, reverse=True)
    if candidates:
        print(f"Hint: most recent run dir is {candidates[0]}", file=sys.stderr)
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

log("Generating all-datasets mean + P999 latency figure")
log(f"Results: {TSV}")
log(f"Figures: {FIGS}")

L_TARGET = 80

# Dataset display order + labels for the 5 subfigure columns
DATASET_ORDER = ["sift1m", "gist1m", "deep10m", "glove1.2m", "msturing1m"]
DATASET_LABEL = {
    "sift1m": "SIFT1M",
    "gist1m": "GIST1M",
    "deep10m": "DEEP-10M",
    "glove1.2m": "GloVe-1.2M",
    "msturing1m": "MSTuring-1M",
}

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
_ANNOT_DY = {'bfs': 8, 'pcs': -8}
_ANNOT_VA = {'bfs': 'bottom', 'pcs': 'top'}

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
required = {'dataset', 'policy', 'cache_size', 'L', 'mean_us'}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in results.tsv: {sorted(missing)}")

HAS_P999 = 'p999_us' in df.columns
if not HAS_P999:
    log("WARNING: 'p999_us' column not found in results.tsv — P999 row will be skipped")

for col in ['L', 'cache_size', 'qps', 'mean_us', 'p999_us', 'mean_ios', 'recall', 'tail_amp']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['dataset', 'policy', 'cache_size', 'L', 'mean_us'])
df = df[df['L'] == L_TARGET]
if df.empty:
    raise SystemExit(f"No rows found with L={L_TARGET} in {TSV}")

agg_kwargs = dict(
    mean_us=('mean_us', 'mean'),
    mean_std=('mean_us', 'std'),
    n=('mean_us', 'count'),
)
if HAS_P999:
    agg_kwargs['p999_us'] = ('p999_us', 'mean')
    agg_kwargs['p999_std'] = ('p999_us', 'std')

agg = df.groupby(['dataset', 'policy', 'cache_size']).agg(**agg_kwargs).reset_index()
agg['mean_std'] = agg['mean_std'].fillna(0.0)
if HAS_P999:
    agg['p999_std'] = agg['p999_std'].fillna(0.0)

present_datasets = set(agg['dataset'].unique())
ordered_datasets = [d for d in DATASET_ORDER if d in present_datasets]
extra_datasets = sorted(present_datasets - set(ordered_datasets))
ordered_datasets += extra_datasets
if not ordered_datasets:
    raise SystemExit("No usable dataset rows found in results.tsv")
if extra_datasets:
    log(f"NOTE: dataset(s) not in the expected 5 found and appended: {extra_datasets}")
if len(ordered_datasets) < 5:
    missing_ds = [d for d in DATASET_ORDER if d not in present_datasets]
    log(f"WARNING: only {len(ordered_datasets)}/5 expected datasets present "
        f"(missing: {missing_ds}) — plotting what's available")

cache_sizes = [int(x) for x in sorted(agg['cache_size'].unique())]
xs = range(len(cache_sizes))
xlbls = xlabels(cache_sizes)

# =============================================================================
# Fig: Mean latency (row 1) + P999 latency (row 2), one column per dataset
# (L=80 fixed)
# =============================================================================
n = len(ordered_datasets)
n_rows = 2 if HAS_P999 else 1
fig, axes = plt.subplots(n_rows, n, figsize=(2.4 * n, 2.8 * n_rows), squeeze=False)


def plot_row(row_idx, value_col, std_col, ylabel):
    for col_idx, dataset in enumerate(ordered_datasets):
        ax = axes[row_idx, col_idx]
        points = {}
        for policy in ['bfs', 'pcs']:
            sub = agg[(agg['dataset'] == dataset) & (agg['policy'] == policy)].set_index('cache_size')
            vals = np.array([sub[value_col].get(c, np.nan) / 1000 for c in cache_sizes])
            errs = np.array([sub[std_col].get(c, 0.0) / 1000 for c in cache_sizes])
            ax.errorbar(
                xs, vals, yerr=errs,
                color=PC[policy], marker=PM[policy], label=PL[policy],
                capsize=4, capthick=1.3, linewidth=1.8, markersize=6,
            )
            print(dataset, value_col, policy, vals, "±", errs)
            for j, v in enumerate(vals):
                if np.isnan(v):
                    continue
                points.setdefault(j, []).append((policy, v))
        ax.set_xticks(xs)
        ax.set_xticklabels(xlbls)
        ax.set_xlabel('Nodes Cached in DRAM')
        if col_idx == 0:
            ax.set_ylabel(ylabel)
        title = DATASET_LABEL.get(dataset, dataset)
        if row_idx == 0:
            ax.set_title(f'{title}  (L={L_TARGET})')
        ax.legend(frameon=False)
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        ax.set_ylim(ymin - yrange * 0.12, ymax + yrange * 0.22)
        ax.margins(x=0.12)
        for j, pts in points.items():
            for policy, v in pts:
                ax.annotate(f'{v:.1f}', (j, v),
                            textcoords='offset points',
                            xytext=(0, _ANNOT_DY[policy]),
                            ha='center', va=_ANNOT_VA[policy],
                            fontsize=8, color=PC[policy], fontweight='bold')


plot_row(0, 'mean_us', 'mean_std', 'Mean Latency (ms)')
if HAS_P999:
    plot_row(1, 'p999_us', 'p999_std', 'P999 Latency (ms)')

plt.tight_layout()
fig.subplots_adjust(wspace=0.28, hspace=0.45 if HAS_P999 else 0.0)
save(fig, 'fig_all_datasets_mean_latency')

# =============================================================================
# Summary table
# =============================================================================
print("\n" + "=" * 75)
print(f"LATENCY SUMMARY (L={L_TARGET})")
print("=" * 75)
if HAS_P999:
    print(f"{'Dataset':<12} {'Cache':>8} {'Policy':>7} {'Mean(ms)':>10} {'P999(ms)':>10}")
else:
    print(f"{'Dataset':<12} {'Cache':>8} {'Policy':>7} {'Mean(ms)':>10}")
print("-" * 75)
for dataset in ordered_datasets:
    for cache in cache_sizes:
        for policy in ['pcs', 'bfs']:
            r = agg[(agg['dataset'] == dataset) &
                    (agg['policy'] == policy) &
                    (agg['cache_size'] == cache)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            if HAS_P999:
                print(f"{dataset:<12} {int(cache):>8,} {policy:>7} "
                      f"{r.mean_us / 1000:>10.2f} {r.p999_us / 1000:>10.2f}")
            else:
                print(f"{dataset:<12} {int(cache):>8,} {policy:>7} {r.mean_us / 1000:>10.2f}")
        bfs = agg[(agg['dataset'] == dataset) & (agg['policy'] == 'bfs') & (agg['cache_size'] == cache)]
        pcs = agg[(agg['dataset'] == dataset) & (agg['policy'] == 'pcs') & (agg['cache_size'] == cache)]
        if len(bfs) and len(pcs):
            bv = bfs.iloc[0]['mean_us']
            pv = pcs.iloc[0]['mean_us']
            delta = (bv - pv) / bv * 100
            line = f"  → mean reduction: {delta:+.1f}%"
            if HAS_P999:
                bv999 = bfs.iloc[0]['p999_us']
                pv999 = pcs.iloc[0]['p999_us']
                delta999 = (bv999 - pv999) / bv999 * 100
                line += f"   p999 reduction: {delta999:+.1f}%"
            print(line)
        print()

# =============================================================================
# DONE
# =============================================================================
log("=======================================================")
log("DONE")
log(f"Figure: {FIGS}/fig_all_datasets_mean_latency.png")
log("=======================================================")
print("")