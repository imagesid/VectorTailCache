#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# compare_policies_all_datasets.py — Generate figures from existing extended_eval data
# =============================================================================

import os
import sys
import glob
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate VectorTailCache extended figures from existing results.tsv files only."
    )

    parser.add_argument(
        "--resume",
        "--run-dir",
        dest="run_dir",
        default="",
        help="Path to extended_eval_<timestamp> run directory."
    )

    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use latest extended_eval_* directory under ${VTC_RESULTS:-/home/agung/vector/results}."
    )

    return parser.parse_args()


args = parse_args()

OUT_BASE = os.environ.get("VTC_RESULTS", "/home/agung/vector/results")

RUN_DIR = ""

if args.latest:
    candidates = glob.glob(os.path.join(OUT_BASE, "extended_eval_*"))
    candidates = [p for p in candidates if os.path.isdir(p)]
    candidates = sorted(candidates)

    if candidates:
        RUN_DIR = candidates[-1]

elif args.run_dir:
    RUN_DIR = args.run_dir.rstrip("/")

else:
    print("ERROR: provide --resume /path/to/run_dir or --latest")
    sys.exit(1)


if not RUN_DIR:
    print("ERROR: no extended_eval_* directory found under {}".format(OUT_BASE))
    sys.exit(1)

if not os.path.isdir(RUN_DIR):
    print("ERROR: run dir not found: {}".format(RUN_DIR))
    sys.exit(1)


# =============================================================================
# PATHS
# =============================================================================

LOG = os.path.join(RUN_DIR, "figure_generation.log")

EXP1_DIR = os.path.join(RUN_DIR, "exp1_sift1m_multithread")
EXP2_DIR = os.path.join(RUN_DIR, "exp2_gist1m")
EXP3_DIR = os.path.join(RUN_DIR, "exp3_deep10m")
FIG_DIR = os.path.join(RUN_DIR, "extended_figures3")

os.makedirs(FIG_DIR, exist_ok=True)


def log(msg):
    line = "[{}] {}".format(datetime.now().strftime("%H:%M:%S"), msg)
    print(line)

    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


log("============================================================")
log("VectorTailCache figure-only generation")
log("Run dir : {}".format(RUN_DIR))
log("Figures : {}".format(FIG_DIR))
log("Mode    : existing data only")
log("============================================================")


# =============================================================================
# FIGURE STYLE — unchanged from original script
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.2,
    'lines.markersize': 8,
})

PC = {'bfs': '#9E9E9E', 'pcs': '#E53935'}
PM = {'bfs': 'o',       'pcs': '^'}
PL = {'bfs': 'BFS', 'pcs': 'PCS'}

NUMERIC_COLS = [
    'cache_size', 'L', 'threads', 'run',
    'qps', 'mean_us', 'p999_us', 'mean_ios', 'recall', 'tail_amp'
]


# =============================================================================
# HELPERS
# =============================================================================

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig("{}/{}.{}".format(FIG_DIR, name, ext))

    plt.close(fig)
    print("  Saved: {}".format(name))


def xlabels(vals):
    return [
        "{}K".format(int(v) // 1000) if v >= 1000 else str(int(v))
        for v in vals
    ]


def nearest_cache(cache_sizes, target):
    """Return the cache_size value in cache_sizes closest to target (excluding 0)."""
    candidates = [c for c in cache_sizes if c > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - target))


def get_p999_ms(agg, policy, cache, L, threads=1):
    """Look up mean P999 (in ms) for a given policy/cache/L/threads combo.
    Returns NaN if the combo isn't present."""
    d = agg[
        (agg['policy'] == policy) &
        (agg['cache_size'] == cache) &
        (agg['L'] == L) &
        (agg['threads'] == threads)
    ]
    if len(d) == 0:
        return float('nan')
    return d['p999_us'].values[0] / 1000.0


def get_recall(agg, policy, cache, L, threads=1):
    d = agg[
        (agg['policy'] == policy) &
        (agg['cache_size'] == cache) &
        (agg['L'] == L) &
        (agg['threads'] == threads)
    ]
    if len(d) == 0:
        return float('nan')
    return d['recall'].values[0]


def pct_reduction(bfs_v, pcs_v):
    if bfs_v is None or pcs_v is None:
        return float('nan')
    if np.isnan(bfs_v) or np.isnan(pcs_v) or bfs_v <= 0:
        return float('nan')
    return (bfs_v - pcs_v) / bfs_v * 100.0


# Fixed-direction point labels for P999 line plots: bfs always labeled
# above its point, pcs always labeled below, regardless of which policy
# has the higher value at a given x. Also pads the y-axis top and bottom
# so labels don't collide with the axis frame.
_ANNOT_DY = {'bfs': 9, 'pcs': -9}
_ANNOT_VA = {'bfs': 'bottom', 'pcs': 'top'}


def _pad_ylim_for_labels(ax, top_frac=0.22, bottom_frac=0.12):
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.set_ylim(ymin - yrange * bottom_frac, ymax + yrange * top_frac)


def _annotate_fixed(ax, points):
    """points: dict of x -> list of (policy, value) tuples."""
    for x, pts in points.items():
        for policy, v in pts:
            ax.annotate(
                '{:.1f}'.format(v), (x, v),
                textcoords='offset points',
                xytext=(0, _ANNOT_DY[policy]),
                ha='center', va=_ANNOT_VA[policy],
                fontsize=8.5, color=PC[policy],
                fontweight='bold'
            )


_SCATTER_OFFSET = {'bfs': -0.06, 'pcs': 0.06}


def _scatter_raw_runs(ax, x_index, policy, raw_vals_ms):
    """Plot individual raw-run values as small jittered, semi-transparent
    dots underneath the mean/median line, matching compare_policies.py."""
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


def load_tsv(exp_dir):
    tsv = os.path.join(exp_dir, 'results.tsv')

    if not os.path.exists(tsv):
        print("  [SKIP] missing: {}".format(tsv))
        return None

    df = pd.read_csv(tsv, sep='\t')

    if len(df) == 0:
        print("  [SKIP] empty: {}".format(tsv))
        return None

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    required = {
        'policy', 'dataset', 'cache_size', 'L', 'threads',
        'p999_us', 'mean_us', 'recall', 'tail_amp'
    }

    missing = required - set(df.columns)

    if missing:
        print("  [SKIP] {} missing columns: {}".format(tsv, sorted(missing)))
        return None

    df = df.dropna(subset=['policy', 'cache_size', 'L', 'threads', 'p999_us'])

    if len(df) == 0:
        print("  [SKIP] no usable rows: {}".format(tsv))
        return None

    print("  Loaded {} rows from {}".format(len(df), tsv))
    return df


def agg_df(df):
    return df.groupby(['policy', 'cache_size', 'L', 'threads']).agg(
        p999_us=('p999_us', 'mean'),
        p999_std=('p999_us', 'std'),
        mean_us=('mean_us', 'mean'),
        mean_std=('mean_us', 'std'),
        recall=('recall', 'mean'),
        tail_amp=('tail_amp', 'mean'),
    ).reset_index()


def headline_bars(ax, agg, L, cache_sizes, label='', is_first=False):
    bfs_d = agg[
        (agg['policy'] == 'bfs') &
        (agg['L'] == L) &
        (agg['threads'] == 1)
    ].set_index('cache_size')['p999_us']

    pcs_d = agg[
        (agg['policy'] == 'pcs') &
        (agg['L'] == L) &
        (agg['threads'] == 1)
    ].set_index('cache_size')['p999_us']

    reds = []
    vxs = []
    vxlbls = []

    for i, c in enumerate(cache_sizes):
        if c == 0:
            continue

        bv = bfs_d.get(c, np.nan)
        pv = pcs_d.get(c, np.nan)

        if not np.isnan(bv) and not np.isnan(pv) and bv > 0:
            reds.append((bv - pv) / bv * 100)
            vxs.append(i)
            vxlbls.append(xlabels(cache_sizes)[i])

    colors = [
        '#4CAF50' if r > 2 else ('#F44336' if r < -2 else '#FF9800')
        for r in reds
    ]

    bars = ax.bar(vxs, reds, color=colors, alpha=0.85, width=0.6)

    for bar, val in zip(bars, reds):
        ypos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            '{:+.1f}%'.format(val),
            ha='center',
            va='bottom',
            fontsize=7,
            fontweight='bold'
        )

        print(bar, '{:+.1f}%'.format(val))

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(vxs)
    ax.set_xticklabels(vxlbls)
    ax.set_xlabel('Nodes Cached')

    if is_first:
        ax.set_ylabel('P999 Reduction vs BFS (%)')
    else:
        ax.set_ylabel('')

    ax.set_title('L={}  {}'.format(L, label))
    ax.set_ylim([-15, 50])

    return dict(zip([cache_sizes[i] for i in vxs], reds))


# =============================================================================
# LOAD DATA
# =============================================================================

df1 = load_tsv(EXP1_DIR)
df2 = load_tsv(EXP2_DIR)
df3 = load_tsv(EXP3_DIR)


# =============================================================================
# Fig E1 + E2: SIFT1M multithread
# =============================================================================

if df1 is not None:
    agg1 = agg_df(df1)

    thread_vals = sorted(agg1['threads'].dropna().unique())
    cache_sizes1 = sorted(agg1['cache_size'].dropna().unique())
    L_vals1 = sorted(agg1['L'].dropna().unique())

    cache_100k = nearest_cache(cache_sizes1, 100000)

    # ── PRINT: concurrency numbers for paper text ─────────────────────
    print(f"\n[Concurrency] thread_vals={thread_vals}")
    print(f"[Concurrency] cache_sizes={cache_sizes1}")
    print(f"[Concurrency] L_vals={L_vals1}")

    print(f"\n[Concurrency] BFS P999 at L=80 across threads and cache sizes:")
    print(f"  {'Cache':>8}  {'T=1':>8}  {'T=4':>8}  {'T=8':>8}  {'T=16':>8}  {'spike at':>10}")
    for cache in cache_sizes1:
        sub = agg1[(agg1['policy']=='bfs') & (agg1['L']==80) & (agg1['cache_size']==cache)]
        vals = [sub[sub['threads']==t]['p999_us'].values[0]/1000
                if len(sub[sub['threads']==t]) > 0 else float('nan')
                for t in thread_vals]
        spike = thread_vals[vals.index(max(v for v in vals if v==v))]
        clbl = f'{int(cache)//1000}K' if cache >= 1000 else '0'
        print(f"  {clbl:>8}  " + "  ".join(f"{v:>8.2f}" for v in vals) +
              f"  {'T='+str(int(spike)):>10}")

    print(f"\n[Concurrency] Tail amplification (P999/Mean) BFS at L=80:")
    print(f"  {'Cache':>8}  {'T=1':>8}  {'T=4':>8}  {'T=8':>8}  {'T=16':>8}")
    for cache in cache_sizes1:
        sub = agg1[(agg1['policy']=='bfs') & (agg1['L']==80) & (agg1['cache_size']==cache)]
        amps = []
        for t in thread_vals:
            row = sub[sub['threads']==t]
            if len(row) == 0:
                amps.append(float('nan'))
                continue
            p999 = row['p999_us'].values[0]
            mean = row['mean_us'].values[0] if 'mean_us' in row.columns else float('nan')
            amps.append(p999/mean if mean > 0 else float('nan'))
        clbl = f'{int(cache)//1000}K' if cache >= 1000 else '0'
        print(f"  {clbl:>8}  " + "  ".join(f"{a:>8.2f}" for a in amps))

    print(f"\n[Concurrency] Min tail_amp across all configs:")
    min_amp = agg1[agg1['policy']=='bfs']['tail_amp'].min() if 'tail_amp' in agg1.columns else 'N/A'
    print(f"  BFS min tail_amp = {min_amp}")
    # ──────────────────────────────────────────────────────────────────

    # ══════════════════════════════════════════════════════════════════
    # TEXT NUMBERS — Section eval:concurrency (matches sec:eval:concurrency paragraph)
    # ══════════════════════════════════════════════════════════════════
    log("")
    log("=== TEXT NUMBERS: sec:eval:concurrency (SIFT1M, L=80) ===")
    if cache_100k is not None:
        for t in [1, 4, 8, 16]:
            if t not in thread_vals:
                continue
            bfs_v = get_p999_ms(agg1, 'bfs', cache_100k, 80, t)
            pcs_v = get_p999_ms(agg1, 'pcs', cache_100k, 80, t)
            red = pct_reduction(bfs_v, pcs_v)
            log("  T={:<3} cache={}: PCS={:.2f}ms  BFS={:.2f}ms  reduction={:.1f}%".format(
                t, xlabels([cache_100k])[0], pcs_v, bfs_v, red))
    else:
        log("  [WARN] no ~100K cache size found in SIFT1M data")

    # "reductions of up to X% at 100K nodes across L in {40,80,120}" (Fig E2 / multithread_reduction)
    log("  -- max/min reduction at 100K-node cache across L values (T=1, headline_bars basis) --")
    if cache_100k is not None:
        reds_100k = []
        for L in L_vals1:
            bfs_v = get_p999_ms(agg1, 'bfs', cache_100k, L, 1)
            pcs_v = get_p999_ms(agg1, 'pcs', cache_100k, L, 1)
            red = pct_reduction(bfs_v, pcs_v)
            if not np.isnan(red):
                reds_100k.append(red)
            log("     L={:<4} reduction={:.1f}%".format(int(L), red))
        if reds_100k:
            log("  => reduction range at 100K nodes across L: {:.1f}% to {:.1f}%".format(
                min(reds_100k), max(reds_100k)))

    fig, axes = plt.subplots(1, len(cache_sizes1), figsize=(8, 3))

    if len(cache_sizes1) == 1:
        axes = [axes]

    for ax, cache in zip(axes, cache_sizes1):
        sub = agg1[
            (agg1['L'] == 80) &
            (agg1['cache_size'] == cache)
        ]

        points_e1 = {}
        for policy in ['bfs', 'pcs']:
            d = sub[sub['policy'] == policy].set_index('threads')

            vals = [
                d['p999_us'].get(t, np.nan) / 1000
                for t in thread_vals
            ]

            errs = [
                d['p999_std'].get(t, np.nan) / 1000
                for t in thread_vals
            ]

            ax.errorbar(
                range(len(thread_vals)),
                vals,
                yerr=errs,
                color=PC[policy],
                marker=PM[policy],
                label=PL[policy],
                capsize=4
            )

            print(policy, cache, vals, errs)

            # Raw per-run dots
            raw_sub = df1[
                (df1['policy'] == policy) &
                (df1['L'] == 80) &
                (df1['cache_size'] == cache)
            ]

            for j, t in enumerate(thread_vals):
                raw_vals = raw_sub[raw_sub['threads'] == t]['p999_us'].values / 1000
                _scatter_raw_runs(ax, j, policy, raw_vals)

            for j, v in enumerate(vals):
                if np.isnan(v):
                    continue
                points_e1.setdefault(j, []).append((policy, v))

        ax.set_xticks(range(len(thread_vals)))
        ax.set_xticklabels([str(int(t)) for t in thread_vals])
        ax.set_xlabel('Concurrent Threads')

        if ax is axes[0]:
            ax.set_ylabel('P999 Latency (ms)')
        else:
            ax.set_ylabel('')

        cache_label = "{}K".format(int(cache) // 1000) if cache >= 1000 else str(int(cache))
        ax.set_title('Cache={} nodes'.format(cache_label))
        ax.legend(fontsize=9)

        ax.margins(x=0.12)
        _pad_ylim_for_labels(ax)
        _annotate_fixed(ax, points_e1)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.18)
    save(fig, 'E1_sift1m_multithread_p999')

    fig, axes = plt.subplots(1, len(L_vals1), figsize=(8, 3))

    if len(L_vals1) == 1:
        axes = [axes]

    all_reds_e2 = []
    for i, (ax, L) in enumerate(zip(axes, L_vals1)):
        reds_for_L = headline_bars(
            ax,
            agg1,
            L,
            cache_sizes1,
            is_first=(i == 0)
        )
        all_reds_e2.extend(reds_for_L.values())

    if all_reds_e2:
        log("  -- E2 headline_bars full range (all L, all nonzero cache sizes, T=1) --")
        log("     min={:.1f}%  max={:.1f}%".format(min(all_reds_e2), max(all_reds_e2)))

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.18)
    save(fig, 'E2_sift1m_multithread_reduction')


# =============================================================================
# Fig E3 + E4: GIST1M
# =============================================================================

if df2 is not None:
    agg2 = agg_df(df2)

    cache_sizes2 = sorted(agg2['cache_size'].dropna().unique())
    L_vals2 = sorted(agg2['L'].dropna().unique())
    xs2 = range(len(cache_sizes2))

    cache_100k_g = nearest_cache(cache_sizes2, 100000)
    cache_0_g = 0 if 0 in cache_sizes2 else nearest_cache(cache_sizes2, 0)

    # ══════════════════════════════════════════════════════════════════
    # TEXT NUMBERS — Section eval:gist (matches sec:eval:gist paragraph)
    # ══════════════════════════════════════════════════════════════════
    log("")
    log("=== TEXT NUMBERS: sec:eval:gist (GIST1M) ===")
    if cache_100k_g is not None:
        for L in [80, 40]:
            if L not in L_vals2:
                continue
            no_cache_v = get_p999_ms(agg2, 'pcs', cache_0_g, L, 1) if cache_0_g is not None else float('nan')
            full_cache_v = get_p999_ms(agg2, 'pcs', cache_100k_g, L, 1)
            red = pct_reduction(no_cache_v, full_cache_v)
            recall_v = get_recall(agg2, 'pcs', cache_100k_g, L, 1)
            log("  L={:<3}: PCS no-cache={:.2f}ms -> PCS @ {}={:.2f}ms  reduction={:.1f}%  recall@{}nodes={:.1f}%".format(
                L, no_cache_v, xlabels([cache_100k_g])[0], full_cache_v, red, xlabels([cache_100k_g])[0], recall_v))

        log("  -- peak reduction range at 100K nodes across all L (BFS vs PCS, Fig E4 basis) --")
        peak_reds = []
        for L in L_vals2:
            bfs_v = get_p999_ms(agg2, 'bfs', cache_100k_g, L, 1)
            pcs_v = get_p999_ms(agg2, 'pcs', cache_100k_g, L, 1)
            red = pct_reduction(bfs_v, pcs_v)
            if not np.isnan(red):
                peak_reds.append(red)
            log("     L={:<4} reduction={:.1f}%".format(int(L), red))
        if peak_reds:
            log("  => peak reduction range at 100K nodes: {:.1f}% to {:.1f}%".format(
                min(peak_reds), max(peak_reds)))
    else:
        log("  [WARN] no ~100K cache size found in GIST1M data")

    fig, axes = plt.subplots(1, len(L_vals2), figsize=(8, 3))

    if len(L_vals2) == 1:
        axes = [axes]

    for ax, L in zip(axes, L_vals2):
        sub = agg2[agg2['threads'] == 1]

        points_e3 = {}
        for policy in ['bfs', 'pcs']:
            d = sub[
                (sub['policy'] == policy) &
                (sub['L'] == L)
            ].set_index('cache_size')

            vals = [
                d['p999_us'].get(c, np.nan) / 1000
                for c in cache_sizes2
            ]

            errs = [
                d['p999_std'].get(c, np.nan) / 1000
                for c in cache_sizes2
            ]

            ax.errorbar(
                xs2,
                vals,
                yerr=errs,
                color=PC[policy],
                marker=PM[policy],
                label=PL[policy],
                capsize=4
            )

            print(policy, L, vals, errs)

            # Raw per-run dots
            raw_sub = df2[
                (df2['policy'] == policy) &
                (df2['threads'] == 1) &
                (df2['L'] == L)
            ]

            for j, c in enumerate(cache_sizes2):
                raw_vals = raw_sub[raw_sub['cache_size'] == c]['p999_us'].values / 1000
                _scatter_raw_runs(ax, j, policy, raw_vals)

            for j, v in enumerate(vals):
                if np.isnan(v):
                    continue
                points_e3.setdefault(j, []).append((policy, v))

        r = agg2[agg2['L'] == L]['recall'].mean()

        ax.set_xticks(xs2)
        ax.set_xticklabels(xlabels(cache_sizes2))
        ax.set_xlabel('Nodes Cached')

        if ax is axes[0]:
            ax.set_ylabel('P999 Latency (ms)')
        else:
            ax.set_ylabel('')

        ax.set_title('L={}  (Recall≈{:.1f}%)'.format(int(L), r))
        ax.legend()

        ax.margins(x=0.12)
        _pad_ylim_for_labels(ax)
        _annotate_fixed(ax, points_e3)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.25)
    save(fig, 'E3_gist1m_p999')

    fig, axes = plt.subplots(1, len(L_vals2), figsize=(8, 3))

    if len(L_vals2) == 1:
        axes = [axes]

    for i, (ax, L) in enumerate(zip(axes, L_vals2)):
        headline_bars(
            ax,
            agg2,
            L,
            cache_sizes2,
            label='(GIST1M)',
            is_first=(i == 0)
        )

        ax.set_ylim([-15, 90])

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.18)
    save(fig, 'E4_gist1m_reduction')


# =============================================================================
# Fig E5: DEEP-10M
# =============================================================================

if df3 is not None:
    agg3 = agg_df(df3)

    cache_sizes3 = sorted(agg3['cache_size'].dropna().unique())
    L_vals3 = sorted(agg3['L'].dropna().unique())
    xs3 = range(len(cache_sizes3))

    cache_500k_d = nearest_cache(cache_sizes3, 500000)
    cache_10k_d = nearest_cache(cache_sizes3, 10000)

    # ══════════════════════════════════════════════════════════════════
    # TEXT NUMBERS — Section eval:deep (matches sec:eval:deep paragraph)
    # ══════════════════════════════════════════════════════════════════
    log("")
    log("=== TEXT NUMBERS: sec:eval:deep (DEEP-10M, L=80) ===")
    if cache_500k_d is not None:
        bfs_v = get_p999_ms(agg3, 'bfs', cache_500k_d, 80, 1)
        pcs_v = get_p999_ms(agg3, 'pcs', cache_500k_d, 80, 1)
        red = pct_reduction(bfs_v, pcs_v)
        log("  cache={}: PCS={:.2f}ms  BFS={:.2f}ms  reduction={:.1f}%".format(
            xlabels([cache_500k_d])[0], pcs_v, bfs_v, red))
    else:
        log("  [WARN] no ~500K cache size found in DEEP-10M data")

    if cache_10k_d is not None:
        bfs_v = get_p999_ms(agg3, 'bfs', cache_10k_d, 80, 1)
        pcs_v = get_p999_ms(agg3, 'pcs', cache_10k_d, 80, 1)
        red = pct_reduction(bfs_v, pcs_v)
        log("  cache={}: PCS={:.2f}ms  BFS={:.2f}ms  reduction={:.1f}%".format(
            xlabels([cache_10k_d])[0], pcs_v, bfs_v, red))
    else:
        log("  [WARN] no ~10K cache size found in DEEP-10M data")

    fig, axes = plt.subplots(1, len(L_vals3), figsize=(8, 3))

    if len(L_vals3) == 1:
        axes = [axes]

    for ax, L in zip(axes, L_vals3):
        sub = agg3[agg3['threads'] == 1]

        points_e5 = {}
        for policy in ['bfs', 'pcs']:
            d = sub[
                (sub['policy'] == policy) &
                (sub['L'] == L)
            ].set_index('cache_size')

            vals = [
                d['p999_us'].get(c, np.nan) / 1000
                for c in cache_sizes3
            ]

            errs = [
                d['p999_std'].get(c, np.nan) / 1000
                for c in cache_sizes3
            ]

            ax.errorbar(
                xs3,
                vals,
                yerr=errs,
                color=PC[policy],
                marker=PM[policy],
                label=PL[policy],
                capsize=4
            )

            print(policy, L, vals, errs)

            # Raw per-run dots
            raw_sub = df3[
                (df3['policy'] == policy) &
                (df3['threads'] == 1) &
                (df3['L'] == L)
            ]

            for j, c in enumerate(cache_sizes3):
                raw_vals = raw_sub[raw_sub['cache_size'] == c]['p999_us'].values / 1000
                _scatter_raw_runs(ax, j, policy, raw_vals)

            for j, v in enumerate(vals):
                if np.isnan(v):
                    continue
                points_e5.setdefault(j, []).append((policy, v))

        r = agg3[agg3['L'] == L]['recall'].mean()

        ax.set_xticks(xs3)
        ax.set_xticklabels(xlabels(cache_sizes3))
        ax.set_xlabel('Nodes Cached')

        if ax is axes[0]:
            ax.set_ylabel('P999 Latency (ms)')
        else:
            ax.set_ylabel('')

        ax.set_title('L={}  (Recall≈{:.1f}%)'.format(int(L), r))
        ax.legend()

        ax.margins(x=0.12)
        _pad_ylim_for_labels(ax)
        _annotate_fixed(ax, points_e5)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.18)
    save(fig, 'E5_deep10m_p999')


# =============================================================================
# Fig E6: Cross-dataset summary
# =============================================================================

fig, ax = plt.subplots(figsize=(4, 3))

datasets = []
improvements = []
errors = []
colors_bar = []

# ══════════════════════════════════════════════════════════════════
# TEXT NUMBERS — Section eval:summary (matches sec:eval:summary paragraph)
# ══════════════════════════════════════════════════════════════════
log("")
log("=== TEXT NUMBERS: sec:eval:summary (cross-dataset, ~10K nodes, L=80, T=1) ===")

for df, name in [
    (df1, 'SIFT1M'),
    (df2, 'GIST1M'),
    (df3, 'DEEP-10M'),
]:
    if df is None or len(df) == 0:
        continue

    cache_candidates = sorted([
        c for c in df['cache_size'].dropna().unique()
        if c > 0
    ])

    if not cache_candidates:
        continue

    target = min(cache_candidates, key=lambda x: abs(x - 10000))

    bfs = df[
        (df['policy'] == 'bfs') &
        (df['L'] == 80) &
        (df['cache_size'] == target) &
        (df['threads'] == 1)
    ]

    pcs = df[
        (df['policy'] == 'pcs') &
        (df['L'] == 80) &
        (df['cache_size'] == target) &
        (df['threads'] == 1)
    ]

    if len(bfs) == 0 or len(pcs) == 0:
        log("  {}: [WARN] missing bfs/pcs rows at cache={}".format(name, target))
        continue

    bv = bfs['p999_us'].mean()
    pv = pcs['p999_us'].mean()

    if bv <= 0 or np.isnan(bv) or np.isnan(pv):
        log("  {}: [WARN] invalid bfs/pcs values at cache={}".format(name, target))
        continue

    imp = (bv - pv) / bv * 100

    err = np.sqrt(
        np.nan_to_num(bfs['p999_us'].std(), nan=0.0) ** 2 +
        np.nan_to_num(pcs['p999_us'].std(), nan=0.0) ** 2
    ) / bv * 100

    log("  {}: cache={} BFS={:.2f}ms PCS={:.2f}ms reduction={:+.1f}% (err ~{:.1f}%)".format(
        name, target, bv / 1000, pv / 1000, imp, err))

    datasets.append("{}\n(cache≈{}K)".format(name, int(target) // 1000))
    improvements.append(imp)
    errors.append(err)
    colors_bar.append('#4CAF50' if imp > 2 else '#F44336')

if improvements:
    log("  => monotonic check: {}".format(
        " < ".join("{:.1f}%".format(v) for v in improvements)
        if improvements == sorted(improvements)
        else "NOT monotonic — values: " + ", ".join("{:.1f}%".format(v) for v in improvements)
    ))

if datasets:
    bars = ax.bar(
        range(len(datasets)),
        improvements,
        color=colors_bar,
        alpha=0.85,
        width=0.5,
        yerr=errors,
        capsize=6,
        ecolor='black'
    )

    for bar, val in zip(bars, improvements):
        ypos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            '{:+.1f}%'.format(val),
            ha='center',
            va='bottom',
            fontsize=7,
            fontweight='bold'
        )

        print(bar, '{:+.1f}%'.format(val))

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel('P999 Reduction vs BFS (%)')

    ax.set_title(
        '(L=80, T=1, existing results.tsv data only)'
    )

    ax.set_ylim([-10, 90])

else:
    ax.text(
        0.5,
        0.5,
        'No cross-dataset summary data available',
        ha='center',
        va='center',
        transform=ax.transAxes,
        fontsize=7
    )

    ax.set_axis_off()

plt.tight_layout()
save(fig, 'E6_cross_dataset_summary')

print("\nAll figures written to: {}".format(FIG_DIR))


# =============================================================================
# DONE
# =============================================================================

png_count = len(glob.glob(os.path.join(FIG_DIR, "*.png")))
pdf_count = len(glob.glob(os.path.join(FIG_DIR, "*.pdf")))

log("")
log("============================================================")
log("COMPLETE")
log("Run dir : {}".format(RUN_DIR))
log("Figures : {}".format(FIG_DIR))
log("PNG files: {}".format(png_count))
log("PDF files: {}".format(pdf_count))
log("============================================================")