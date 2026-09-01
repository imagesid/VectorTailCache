#!/usr/bin/env python3
"""
04_analyze.py
Reads results.tsv from the sweep and generates all paper figures.

Usage:
    python3 /workspace/scripts/04_analyze.py --results /workspace/results/sweep_sift1m_<timestamp>
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Style — clean paper style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 7,
})

COLORS = {
    40:  '#2196F3',   # blue
    80:  '#FF9800',   # orange
    120: '#F44336',   # red
}
MARKERS = {40: 'o', 80: 's', 120: '^'}

# ---------------------------------------------------------------------------
# Load & aggregate
# ---------------------------------------------------------------------------
def load_results(results_dir: str) -> pd.DataFrame:
    tsv_path = os.path.join(results_dir, 'results.tsv')
    if not os.path.exists(tsv_path):
        print(f"ERROR: {tsv_path} not found")
        sys.exit(1)
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"Loaded {len(df)} rows from {tsv_path}")
    print(df.dtypes)
    print(df.head())
    return df

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average across repeat runs."""
    group_cols = ['dataset', 'cache_size', 'L', 'threads', 'K']
    agg = df.groupby(group_cols).agg(
        qps=('qps', 'mean'),
        recall=('recall', 'mean'),
        p50_us=('p50_us', 'mean'),
        p95_us=('p95_us', 'mean'),
        p99_us=('p99_us', 'mean'),
        p999_us=('p999_us', 'mean'),
        avg_us=('avg_us', 'mean'),
        tail_amp_p99=('tail_amp_p99', 'mean'),
        tail_amp_p999=('tail_amp_p999', 'mean'),
    ).reset_index()
    return agg

# ---------------------------------------------------------------------------
# Figure 1: Tail Amplification vs Cache Size (the key motivating figure)
# ---------------------------------------------------------------------------
def fig1_tail_amplification(df: pd.DataFrame, out_dir: str, threads: int = 1):
    sub = df[df['threads'] == threads].copy()
    cache_sizes = sorted(sub['cache_size'].unique())
    L_values = sorted(sub['L'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: P99/P50 tail amplification
    ax = axes[0]
    for L in L_values:
        d = sub[sub['L'] == L].groupby('cache_size')['tail_amp_p99'].mean()
        ax.plot(range(len(cache_sizes)),
                [d.get(c, np.nan) for c in cache_sizes],
                color=COLORS[L], marker=MARKERS[L], label=f'L={L}')

    ax.set_xticks(range(len(cache_sizes)))
    ax.set_xticklabels([f'{c//1000}K' if c >= 1000 else str(c) for c in cache_sizes])
    ax.set_xlabel('Nodes Cached (in memory)')
    ax.set_ylabel('Tail Amplification  P99 / P50')
    ax.set_title('(a) P99/P50 Amplification vs. Cache Budget')
    ax.legend(title='Search L')
    ax.set_ylim(bottom=1)

    # Right: P999/P50 — even more dramatic
    ax = axes[1]
    for L in L_values:
        d = sub[sub['L'] == L].groupby('cache_size')['tail_amp_p999'].mean()
        ax.plot(range(len(cache_sizes)),
                [d.get(c, np.nan) for c in cache_sizes],
                color=COLORS[L], marker=MARKERS[L], label=f'L={L}')

    ax.set_xticks(range(len(cache_sizes)))
    ax.set_xticklabels([f'{c//1000}K' if c >= 1000 else str(c) for c in cache_sizes])
    ax.set_xlabel('Nodes Cached (in memory)')
    ax.set_ylabel('Tail Amplification  P999 / P50')
    ax.set_title('(b) P999/P50 Amplification vs. Cache Budget')
    ax.legend(title='Search L')
    ax.set_ylim(bottom=1)

    fig.suptitle(f'Tail Amplification in DiskANN (threads={threads})', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_tail_amplification.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"[Fig 1] Saved: {path}")
    plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Absolute P99 latency vs cache size
# ---------------------------------------------------------------------------
def fig2_p99_vs_cache(df: pd.DataFrame, out_dir: str, threads: int = 1):
    sub = df[df['threads'] == threads].copy()
    cache_sizes = sorted(sub['cache_size'].unique())
    L_values = sorted(sub['L'].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for L in L_values:
        d = sub[sub['L'] == L].groupby('cache_size')['p99_us'].mean()
        ax.plot(range(len(cache_sizes)),
                [d.get(c, np.nan) for c in cache_sizes],
                color=COLORS[L], marker=MARKERS[L], label=f'L={L} (P99)')

        d_avg = sub[sub['L'] == L].groupby('cache_size')['avg_us'].mean()
        ax.plot(range(len(cache_sizes)),
                [d_avg.get(c, np.nan) for c in cache_sizes],
                color=COLORS[L], marker=MARKERS[L], label=f'L={L} (avg)',
                linestyle='--', alpha=0.5)

    ax.set_xticks(range(len(cache_sizes)))
    ax.set_xticklabels([f'{c//1000}K' if c >= 1000 else str(c) for c in cache_sizes])
    ax.set_xlabel('Nodes Cached (in memory)')
    ax.set_ylabel('Latency (µs)')
    ax.set_title(f'P99 vs. Avg Latency  —  Average diverges from P99 at small cache\n(threads={threads})')
    ax.legend(ncol=2, fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_p99_vs_cache.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"[Fig 2] Saved: {path}")
    plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Recall@K vs cache size
# ---------------------------------------------------------------------------
def fig3_recall_vs_cache(df: pd.DataFrame, out_dir: str, threads: int = 1):
    sub = df[df['threads'] == threads].copy()
    cache_sizes = sorted(sub['cache_size'].unique())
    L_values = sorted(sub['L'].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for L in L_values:
        d = sub[sub['L'] == L].groupby('cache_size')['recall'].mean()
        ax.plot(range(len(cache_sizes)),
                [d.get(c, np.nan) * 100 for c in cache_sizes],
                color=COLORS[L], marker=MARKERS[L], label=f'L={L}')

    ax.set_xticks(range(len(cache_sizes)))
    ax.set_xticklabels([f'{c//1000}K' if c >= 1000 else str(c) for c in cache_sizes])
    ax.set_xlabel('Nodes Cached (in memory)')
    ax.set_ylabel('Recall@10 (%)')
    ax.set_title(f'Recall@10 vs. Cache Budget (threads={threads})')
    ax.set_ylim([0, 101])
    ax.axhline(y=95, color='gray', linestyle=':', label='95% recall target')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_recall_vs_cache.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"[Fig 3] Saved: {path}")
    plt.close()

# ---------------------------------------------------------------------------
# Figure 4: Tail amplification vs thread count (concurrency effect)
# ---------------------------------------------------------------------------
def fig4_tail_vs_concurrency(df: pd.DataFrame, out_dir: str, cache_size: int = 0):
    sub = df[df['cache_size'] == cache_size].copy()
    thread_counts = sorted(sub['threads'].unique())
    L_values = sorted(sub['L'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, metric in enumerate(['p99_us', 'tail_amp_p99']):
        ax = axes[ax_idx]
        for L in L_values:
            d = sub[sub['L'] == L].groupby('threads')[metric].mean()
            ax.plot(range(len(thread_counts)),
                    [d.get(t, np.nan) for t in thread_counts],
                    color=COLORS[L], marker=MARKERS[L], label=f'L={L}')

        ax.set_xticks(range(len(thread_counts)))
        ax.set_xticklabels([str(t) for t in thread_counts])
        ax.set_xlabel('Concurrent Threads')
        if metric == 'p99_us':
            ax.set_ylabel('P99 Latency (µs)')
            ax.set_title(f'(a) P99 Latency vs. Concurrency (cache={cache_size} nodes)')
        else:
            ax.set_ylabel('Tail Amplification P99/P50')
            ax.set_title(f'(b) Tail Amplification vs. Concurrency (cache={cache_size} nodes)')
        ax.legend(title='Search L')

    fig.suptitle('Concurrency Amplifies Tail Latency', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f'fig4_tail_vs_concurrency_cache{cache_size}.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"[Fig 4] Saved: {path}")
    plt.close()

# ---------------------------------------------------------------------------
# Figure 5: Latency percentile staircase (P50, P95, P99, P999 together)
# ---------------------------------------------------------------------------
def fig5_percentile_staircase(df: pd.DataFrame, out_dir: str, L: int = 80, threads: int = 1):
    sub = df[(df['L'] == L) & (df['threads'] == threads)].copy()
    cache_sizes = sorted(sub['cache_size'].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    pct_cols = [('p50_us',  'P50',  'o', '-'),
                ('p95_us',  'P95',  's', '--'),
                ('p99_us',  'P99',  '^', '-.'),
                ('p999_us', 'P999', 'D', ':')]
    pct_colors = ['#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for (col, label, marker, ls), color in zip(pct_cols, pct_colors):
        d = sub.groupby('cache_size')[col].mean()
        ax.plot(range(len(cache_sizes)),
                [d.get(c, np.nan) for c in cache_sizes],
                color=color, marker=marker, linestyle=ls, label=label)

    ax.set_xticks(range(len(cache_sizes)))
    ax.set_xticklabels([f'{c//1000}K' if c >= 1000 else str(c) for c in cache_sizes])
    ax.set_xlabel('Nodes Cached (in memory)')
    ax.set_ylabel('Latency (µs)')
    ax.set_title(f'Latency Percentile Spread  (L={L}, threads={threads})\nP99 and P999 diverge sharply at small cache')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, f'fig5_percentile_staircase_L{L}.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"[Fig 5] Saved: {path}")
    plt.close()

# ---------------------------------------------------------------------------
# Figure 6: QPS vs cache size (throughput impact)
# ---------------------------------------------------------------------------
def fig6_qps_vs_cache(df: pd.DataFrame, out_dir: str):
    L_values = sorted(df['L'].unique())
    thread_counts = [1, 8, 16]
    thread_counts = [t for t in thread_counts if t in df['threads'].unique()]

    fig, axes = plt.subplots(1, len(thread_counts), figsize=(5*len(thread_counts), 4.5), sharey=False)
    if len(thread_counts) == 1:
        axes = [axes]

    cache_sizes = sorted(df['cache_size'].unique())

    for ax, threads in zip(axes, thread_counts):
        sub = df[df['threads'] == threads]
        for L in L_values:
            d = sub[sub['L'] == L].groupby('cache_size')['qps'].mean()
            ax.plot(range(len(cache_sizes)),
                    [d.get(c, np.nan) for c in cache_sizes],
                    color=COLORS[L], marker=MARKERS[L], label=f'L={L}')
        ax.set_xticks(range(len(cache_sizes)))
        ax.set_xticklabels([f'{c//1000}K' if c >= 1000 else str(c) for c in cache_sizes])
        ax.set_xlabel('Nodes Cached')
        ax.set_ylabel('QPS')
        ax.set_title(f'Throughput (threads={threads})')
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig6_qps_vs_cache.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f"[Fig 6] Saved: {path}")
    plt.close()

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def summary_table(df: pd.DataFrame, out_dir: str):
    # Key configurations: zero cache vs full cache, each L
    rows = []
    for L in sorted(df['L'].unique()):
        for cache in [df['cache_size'].min(), df['cache_size'].max()]:
            sub = df[(df['L'] == L) & (df['cache_size'] == cache) & (df['threads'] == 1)]
            if len(sub) == 0:
                continue
            rows.append({
                'L': L,
                'Cache': f"{cache:,} nodes",
                'QPS': f"{sub['qps'].mean():.0f}",
                'Recall@10': f"{sub['recall'].mean()*100:.1f}%",
                'P50 (µs)': f"{sub['p50_us'].mean():.0f}",
                'P99 (µs)': f"{sub['p99_us'].mean():.0f}",
                'P999 (µs)': f"{sub['p999_us'].mean():.0f}",
                'P99/P50': f"{sub['tail_amp_p99'].mean():.1f}×",
            })

    table_df = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'summary_table.tsv')
    table_df.to_csv(path, sep='\t', index=False)
    print("\n=== Summary Table ===")
    print(table_df.to_string(index=False))
    print(f"\nSaved: {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True,
                        help='Path to sweep output directory (containing results.tsv)')
    parser.add_argument('--threads', type=int, default=1,
                        help='Thread count for single-thread figures (default: 1)')
    args = parser.parse_args()

    out_dir = os.path.join(args.results, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    df_raw = load_results(args.results)
    df = aggregate(df_raw)

    print(f"\nDatasets: {df['dataset'].unique()}")
    print(f"Cache sizes: {sorted(df['cache_size'].unique())}")
    print(f"L values: {sorted(df['L'].unique())}")
    print(f"Thread counts: {sorted(df['threads'].unique())}")
    print(f"Output dir: {out_dir}\n")

    threads = args.threads
    if threads not in df['threads'].unique():
        threads = sorted(df['threads'].unique())[0]
        print(f"[WARN] --threads {args.threads} not in data; using {threads}")

    fig1_tail_amplification(df, out_dir, threads=threads)
    fig2_p99_vs_cache(df, out_dir, threads=threads)
    fig3_recall_vs_cache(df, out_dir, threads=threads)
    fig4_tail_vs_concurrency(df, out_dir, cache_size=df['cache_size'].min())
    fig5_percentile_staircase(df, out_dir, L=80, threads=threads)
    fig6_qps_vs_cache(df, out_dir)
    summary_table(df, out_dir)

    print(f"\nAll figures saved to: {out_dir}")

if __name__ == '__main__':
    main()
