#!/usr/bin/env python3
"""
Generate paper figures from a results.tsv file.

"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Policy styling ──────────────────────────────────────────────────────────
PC = {'bfs': '#9E9E9E', 'frequency': '#1E88E5', 'pcs': '#E53935'}
PM = {'bfs': 'o',       'frequency': 's',       'pcs': '^'}
PL = {'bfs': 'BFS',     'frequency': 'Frequency', 'pcs': 'PCS'}

# ── Global plot style ────────────────────────────────────────────────────────
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

# Cache sizes to include, and how to label them on the x-axis
CACHE_SIZES = [0, 50000, 100000]
CACHE_LABELS = {0: "0", 50000: "50K", 100000: "100K"}


def load_data(folder):
    path = os.path.join(folder, "results.tsv")
    if not os.path.isfile(path):
        sys.exit(f"Error: could not find 'results.tsv' inside '{folder}'")
    df = pd.read_csv(path, sep="\t")
    return df


def aggregate(df):
    """Average over 'run', keep std for normal mean/std panels."""
    metrics = ["qps", "mean_us", "p999_us", "mean_ios", "mean_io_us", "recall", "tail_amp"]
    group_cols = ["policy", "cache_size"]
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg


def aggregate_p999_robust(df):
    """Robust aggregation for p999 only: median + IQR."""
    group_cols = ["policy", "cache_size"]

    rob = (
        df.groupby(group_cols)
        .agg(
            p999_med=("p999_us", "median"),
            p999_q25=("p999_us", lambda x: x.quantile(0.25)),
            p999_q75=("p999_us", lambda x: x.quantile(0.75)),
            p999_mean=("p999_us", "mean"),
            p999_std=("p999_us", "std"),
            n=("p999_us", "count"),
        )
        .reset_index()
    )

    return rob


def _position_frac(y, ylo, yhi, log_y):
    if log_y:
        return (np.log10(y) - np.log10(ylo)) / (np.log10(yhi) - np.log10(ylo))
    return (y - ylo) / (yhi - ylo)


def _annotate_grouped(ax, groups, log_y, fmt):
    """Annotate every point with its value."""
    ylo, yhi = ax.get_ylim()

    for x, pts in groups.items():
        pts_sorted = sorted(pts, key=lambda p: p[0])
        n = len(pts_sorted)
        center_rank = (n - 1) / 2

        for i, (y, color, text) in enumerate(pts_sorted):
            rank = i - center_rank
            frac = _position_frac(y, ylo, yhi, log_y)

            if frac > 0.85:
                sign = -1
            elif frac < 0.15:
                sign = 1
            else:
                sign = 1 if rank >= 0 else -1

            magnitude = 3 + abs(rank) * 11
            dy = sign * magnitude
            va = "bottom" if sign > 0 else "top"

            ax.annotate(
                text,
                (x, y),
                textcoords="offset points",
                xytext=(0, dy),
                ha="center",
                va=va,
                fontsize=8.5,
                color=color,
            )


def plot_meanstd_panel(ax, agg, ycol, ylabel, unit_scale=1.0, log_y=False, fmt="{:.0f}"):
    """Normal panel: mean ± std."""
    groups = {}

    for pol in PC:
        sub = agg[agg["policy"] == pol].sort_values("cache_size")

        if sub.empty:
            continue

        x = sub["cache_size"].values
        y = sub[f"{ycol}_mean"].values * unit_scale
        yerr = sub[f"{ycol}_std"].fillna(0.0).values * unit_scale

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=PM[pol],
            color=PC[pol],
            label=PL[pol],
            capsize=3,
        )

        for xi, yi in zip(x, y):
            groups.setdefault(xi, []).append((yi, PC[pol], fmt.format(yi)))

    ax.set_xlabel("Cache size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CACHE_SIZES)
    ax.set_xticklabels([CACHE_LABELS[c] for c in CACHE_SIZES])

    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(False, which="minor")

    ax.margins(x=0.15, y=0.25)
    _annotate_grouped(ax, groups, log_y, fmt)


def plot_p999_robust_panel(ax, df, ylabel, unit_scale=1.0, log_y=True, fmt="{:.1f}"):
    """
    p999 panel only:
        line = median
        error bar = IQR, q25 to q75
        faint dots = individual raw runs
    """
    rob = aggregate_p999_robust(df)
    groups = {}

    # Small horizontal offsets for raw dots so policies do not fully overlap
    if len(CACHE_SIZES) >= 2:
        step = min(np.diff(sorted(CACHE_SIZES)))
        base_offset = step * 0.035
        jitter_width = step * 0.012
    else:
        base_offset = 1.0
        jitter_width = 0.2

    policy_offsets = {
        'bfs': -base_offset,
        'frequency': 0.0,
        'pcs': base_offset,
    }

    for pol in PC:
        sub = rob[rob["policy"] == pol].sort_values("cache_size")

        if sub.empty:
            continue

        x = sub["cache_size"].values
        y = sub["p999_med"].values * unit_scale
        q25 = sub["p999_q25"].values * unit_scale
        q75 = sub["p999_q75"].values * unit_scale

        yerr = np.vstack([
            y - q25,
            q75 - y,
        ])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=PM[pol],
            color=PC[pol],
            label=PL[pol],
            capsize=3,
        )

        # Show individual raw runs lightly
        raw_sub = df[df["policy"] == pol]

        for xi in CACHE_SIZES:
            pts = raw_sub[raw_sub["cache_size"] == xi]["p999_us"].values * unit_scale

            if len(pts) == 0:
                continue

            jitter = np.linspace(-jitter_width, jitter_width, len(pts))
            x_raw = np.full(len(pts), xi + policy_offsets.get(pol, 0.0)) + jitter

            ax.scatter(
                x_raw,
                pts,
                color=PC[pol],
                alpha=0.35,
                s=18,
                edgecolors="none",
                zorder=2,
            )

        for xi, yi in zip(x, y):
            groups.setdefault(xi, []).append((yi, PC[pol], fmt.format(yi)))

    ax.set_xlabel("Cache size")
    ax.set_ylabel(ylabel)
    ax.set_xticks(CACHE_SIZES)
    ax.set_xticklabels([CACHE_LABELS[c] for c in CACHE_SIZES])

    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(False, which="minor")

    ax.margins(x=0.15, y=0.25)
    _annotate_grouped(ax, groups, log_y, fmt)


def make_figure(df, agg, figsize, out_dir, out_name):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Mean latency: unchanged, mean ± std
    plot_meanstd_panel(
        axes[0],
        agg,
        ycol="mean_us",
        ylabel="Mean latency (ms)",
        unit_scale=1e-3,
        log_y=False,
        fmt="{:.1f}",
    )

    # p999 latency: changed only here, median + IQR + raw runs
    plot_p999_robust_panel(
        axes[1],
        df,
        ylabel="p999 latency (ms)",
        unit_scale=1e-3,
        log_y=True,
        fmt="{:.1f}",
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out_path = os.path.join(out_dir, f"{out_name}.{ext}")
        fig.savefig(out_path)
        print(f"{os.path.basename(out_path)}\t{os.path.abspath(out_path)}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from results.tsv")
    parser.add_argument("folder", help="Folder containing results.tsv")
    args = parser.parse_args()

    folder = args.folder
    out_dir = os.path.join(folder, "figures")
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(folder)
    df = df[df["cache_size"].isin(CACHE_SIZES)]

    agg = aggregate(df)
    p999_rob = aggregate_p999_robust(df)

    # ── Figure: mean latency + tail p999 latency ────────────────────────────
    make_figure(
        df=df,
        agg=agg,
        figsize=(6, 3),
        out_dir=out_dir,
        out_name="mean_p999_latency",
    )

    csv_path = os.path.join(out_dir, "aggregated_results.csv")
    agg.to_csv(csv_path, index=False)
    print(f"{os.path.basename(csv_path)}\t{os.path.abspath(csv_path)}")

    p999_csv_path = os.path.join(out_dir, "p999_robust_results.csv")
    p999_rob.to_csv(p999_csv_path, index=False)
    print(f"{os.path.basename(p999_csv_path)}\t{os.path.abspath(p999_csv_path)}")


if __name__ == "__main__":
    main()