#!/usr/bin/env python3
"""
gen_device_figures.py
Generate paper-ready figures from device comparison results.tsv

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
import matplotlib.patches as mpatches

# ── Args ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python3 gen_device_figures.py <results_dir>")
    sys.exit(1)

RESULTS_DIR = sys.argv[1].rstrip('/')
TSV         = os.path.join(RESULTS_DIR, "results.tsv")
FIGS        = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGS, exist_ok=True)

print(f"Results dir : {RESULTS_DIR}")
print(f"TSV file    : {TSV}")
print(f"Figures out : {FIGS}")

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(TSV, sep='\t')
print(f"\nRaw rows    : {len(df)}")
print(f"Columns     : {list(df.columns)}")
print(f"\nSample:")
print(df.head(6).to_string(index=False))

for col in ['qps','mean_us','p999_us','mean_ios','mean_io_us','recall','tail_amp','L']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Device config ─────────────────────────────────────────────────────────────
DEVICE_COLOR  = {'nvme': '#2196F3', 'sata': '#FF9800', 'hdd': '#E53935'}
DEVICE_MARKER = {'nvme': 'o',       'sata': 's',       'hdd': '^'}
DEVICE_LABEL  = {
    'nvme': 'NVMe SSD (Samsung 980 PRO, PCIe 4.0)',
    'sata': 'SATA SSD (Samsung 870, SATA III)',
    'hdd':  'HDD (Seagate Exos, 7200 RPM)',
}
DEVICE_LABEL2  = {
    'nvme': 'NVMe SSD',
    'sata': 'SATA SSD',
    'hdd':  'HDD',
}
DEVICE_ORDER = ['nvme', 'sata', 'hdd']

# ── Aggregate ─────────────────────────────────────────────────────────────────
agg = df.groupby(['device','L']).agg(
    mean_us  =('mean_us',    'mean'),
    mean_std =('mean_us',    'std'),
    p999_us  =('p999_us',   'mean'),
    p999_std =('p999_us',   'std'),
    io_us    =('mean_io_us','mean'),
    tail_amp =('tail_amp',  'mean'),
    tail_std =('tail_amp',  'std'),
    recall   =('recall',    'mean'),
    runs     =('run',       'count'),
).reset_index()

devices = [d for d in DEVICE_ORDER if d in agg['device'].unique()]
L_vals  = sorted(agg['L'].unique())
xs      = list(range(len(L_vals)))
xlbls   = [f'L={int(L)}' for L in L_vals]

print(f"\nDevices found : {devices}")
print(f"L values      : {L_vals}")
print(f"\nAggregated data:")
print(agg.to_string(index=False))

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'serif',
    'font.size':      13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize':12,
    'ytick.labelsize':12,
    'legend.fontsize':12,
    'figure.dpi':     150,
    'savefig.dpi':    300,
    'savefig.bbox':  'tight',
    'axes.grid':      True,
    'grid.alpha':     0.3,
    'lines.linewidth':2.2,
    'lines.markersize':8,
})

def save(fig, name):
    for ext in ['pdf', 'png']:
        path = f"{FIGS}/{name}.{ext}"
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)

def get_vals(dev, metric, scale=1.0):
    sub = agg[agg['device']==dev].set_index('L')
    return [sub[metric].get(L, np.nan) * scale for L in L_vals]

def get_errs(dev, metric, scale=1.0):
    sub = agg[agg['device']==dev].set_index('L')
    col = metric.replace('mean','std').replace('p999','p999_std') \
          if metric+'_std' not in agg.columns else metric+'_std'
    err_col = metric + '_std' if metric + '_std' in agg.columns else None
    if err_col is None:
        return [0] * len(L_vals)
    return [sub[err_col].get(L, 0) * scale for L in L_vals]

# # ── Fig 1: Mean latency ───────────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(4,5))
# for dev in devices:
#     vals = get_vals(dev, 'mean_us', 1/1000)
#     errs = get_errs(dev, 'mean_us', 1/1000)
#     ax.errorbar(xs, vals, yerr=errs,
#                 color=DEVICE_COLOR.get(dev,'#777'),
#                 marker=DEVICE_MARKER.get(dev,'o'),
#                 label=DEVICE_LABEL2.get(dev, dev),
#                 capsize=4)
# ax.set_xticks(xs); ax.set_xticklabels(xlbls)
# ax.set_xlabel('Search Complexity (L)')
# ax.set_ylabel('Mean Latency (ms)')
# # ax.set_title('Mean Query Latency\nSIFT1M, no cache, T=1')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28),
#           ncol=3, fontsize=10, framealpha=0.9)
# plt.tight_layout(rect=[0, 0.18, 1, 1])
# save(fig, 'fig1_mean_latency_devices')

# # ── Fig 2: P999 latency ───────────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(4, 5))
# for dev in devices:
#     vals = get_vals(dev, 'p999_us', 1/1000)
#     errs = get_errs(dev, 'p999_us', 1/1000)
#     ax.errorbar(xs, vals, yerr=errs,
#                 color=DEVICE_COLOR.get(dev,'#777'),
#                 marker=DEVICE_MARKER.get(dev,'o'),
#                 label=DEVICE_LABEL2.get(dev, dev),
#                 capsize=4)
# ax.set_xticks(xs); ax.set_xticklabels(xlbls)
# ax.set_xlabel('Search Complexity (L)')
# ax.set_ylabel('P999 Latency (ms)')
# # ax.set_title('P999 Tail Latency\nSIFT1M, no cache, T=1')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28),
#           ncol=3, fontsize=10, framealpha=0.9)
# plt.tight_layout(rect=[0, 0.18, 1, 1])
# save(fig, 'fig2_p999_latency_devices')

# # ── Fig 3: Tail amplification ─────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(4, 5))
# for dev in devices:
#     vals = get_vals(dev, 'tail_amp')
#     errs = get_errs(dev, 'tail_amp')
#     ax.errorbar(xs, vals, yerr=errs,
#                 color=DEVICE_COLOR.get(dev,'#777'),
#                 marker=DEVICE_MARKER.get(dev,'o'),
#                 label=DEVICE_LABEL2.get(dev, dev),
#                 capsize=4)
# ax.set_xticks(xs); ax.set_xticklabels(xlbls)
# ax.set_xlabel('Search Complexity (L)')
# ax.set_ylabel('Tail Amplification (P999 / Mean)')
# # ax.set_title('Tail Amplification\nSIFT1M, no cache, T=1')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28),
#           ncol=3, fontsize=10, framealpha=0.9)
# ax.set_ylim(bottom=1.0)
# ax.axhline(1.0, color='black', lw=0.8, linestyle='--', alpha=0.5)
# plt.tight_layout(rect=[0, 0.18, 1, 1])
# save(fig, 'fig3_tail_amp_devices')
# ── Fig 1-3 combined: Mean latency, P999 latency, Tail amplification ────────
fig, axes = plt.subplots(1, 3, figsize=(8, 3))

panel_specs = [
    ('mean_us', 1/1000, 'Mean Latency (ms)',            False),
    ('p999_us', 1/1000, 'P999 Latency (ms)',             False),
    ('tail_amp', 1.0,   'Tail Amplification (P999/Mean)', True),
]

for ax, (metric, scale, ylabel, is_tail) in zip(axes, panel_specs):
    for dev in devices:
        vals = get_vals(dev, metric, scale)
        errs = get_errs(dev, metric, scale)
        ax.errorbar(xs, vals, yerr=errs,
                    color=DEVICE_COLOR.get(dev, '#777'),
                    marker=DEVICE_MARKER.get(dev, 'o'),
                    label=DEVICE_LABEL2.get(dev, dev),
                    capsize=4)
    ax.set_xticks(xs); ax.set_xticklabels(xlbls)
    ax.set_xlabel('Search Complexity (L)')
    ax.set_ylabel(ylabel)
    if is_tail:
        ax.set_ylim(bottom=1.0)
        ax.axhline(1.0, color='black', lw=0.8, linestyle='--', alpha=0.5)

# Single shared legend below all three panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
           ncol=3, fontsize=10, framealpha=0.9)

plt.tight_layout(rect=[0, 0.1, 1, 1])
save(fig, 'fig123_latency_and_tail_devices')

# ── Fig 4: Mean vs P999 per device (side by side) ────────────────────────────
# NVMe SSD and SATA SSD share the same y-axis max (both flash, comparable scale).
# HDD uses its own y-axis (mechanical, much higher latency range).
# figsize 8x3 to fit paper column width.

flash_devs = [d for d in devices if d in ('nvme', 'sata')]
hdd_devs   = [d for d in devices if d == 'hdd']
n_panels   = len(devices)

fig, axes = plt.subplots(1, n_panels, figsize=(7, 2.5))
if n_panels == 1:
    axes = [axes]

# Shared y-max for NVMe + SATA
flash_p999_max = 0
for dev in flash_devs:
    vals = [v for v in get_vals(dev, 'p999_us', 1/1000) if not np.isnan(v)]
    if vals:
        flash_p999_max = max(flash_p999_max, max(vals))
flash_ymax = flash_p999_max * 1.18 if flash_p999_max > 0 else 20

# Separate y-max for HDD
hdd_p999_max = 0
for dev in hdd_devs:
    vals = [v for v in get_vals(dev, 'p999_us', 1/1000) if not np.isnan(v)]
    if vals:
        hdd_p999_max = max(hdd_p999_max, max(vals))
hdd_ymax = hdd_p999_max * 1.18 if hdd_p999_max > 0 else 600

for ax, dev in zip(axes, devices):
    mean_v = get_vals(dev, 'mean_us', 1/1000)
    p999_v = get_vals(dev, 'p999_us', 1/1000)
    c = DEVICE_COLOR.get(dev, '#777')

    ax.plot(xs, mean_v, 'o--', color='#4CAF50', lw=1.8, ms=5, label='Mean')
    ax.plot(xs, p999_v, '^-',  color=c,          lw=1.8, ms=5, label='P999')
    ax.fill_between(xs, mean_v, p999_v, alpha=0.12, color=c, label='Tail gap')

    ax.set_xticks(xs); ax.set_xticklabels(xlbls, fontsize=12)
    ax.set_xlabel('L', fontsize=12)
    ax.set_ylim(0, hdd_ymax if dev == 'hdd' else flash_ymax)

    # annotate P999 values above each point
    for i, pv in enumerate(p999_v):
        if not np.isnan(pv):
            ax.annotate(f'{pv:.0f}', xy=(i, pv),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=9, color=c)

    short = DEVICE_LABEL2.get(dev, dev).split('(')[0].strip()
    ax.set_title(short, fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax.tick_params(labelsize=8)

    # note shared scale on SATA panel
    # if dev == 'sata' and flash_ymax > 0:
    #     ax.text(0.98, 0.97, 'same scale as NVMe SSD',
    #             transform=ax.transAxes, fontsize=6, color='#888',
    #             ha='right', va='top', style='italic')

axes[0].set_ylabel('Latency (ms)', fontsize=9)
# fig.suptitle(
#     'Mean vs P999 — NVMe SSD & SATA SSD: shared y-axis  |  HDD: separate y-axis',
#     fontsize=8, y=1.02)
plt.tight_layout(pad=0.3, w_pad=0.05)
save(fig, 'fig4_mean_vs_p999_per_device')


# ── Fig 5: Grouped bar — headline comparison at L=80 ─────────────────────────
L_focus = 80
if L_focus in L_vals:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    metrics   = ['mean_us', 'p999_us']
    ylabels   = ['Mean Latency (ms)', 'P999 Latency (ms)']
    titles    = ['Mean Latency at L=80', 'P999 Latency at L=80']

    for ax, metric, ylabel, title in zip(axes, metrics, ylabels, titles):
        sub = agg[agg['L']==L_focus]
        bar_vals  = []
        bar_errs  = []
        bar_cols  = []
        bar_labs  = []
        err_col   = 'mean_std' if metric == 'mean_us' else 'p999_std'

        for dev in devices:
            row = sub[sub['device']==dev]
            if len(row) == 0: continue
            bar_vals.append(row.iloc[0][metric] / 1000)
            bar_errs.append(row.iloc[0].get(err_col, 0) / 1000)
            bar_cols.append(DEVICE_COLOR.get(dev,'#777'))
            bar_labs.append(DEVICE_LABEL2.get(dev,dev).split('(')[0].strip())

        bx = range(len(bar_vals))
        bars = ax.bar(bx, bar_vals, color=bar_cols, alpha=0.85,
                      yerr=bar_errs, capsize=5, width=0.5)
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(bar_errs)*0.1 + 0.02,
                    f'{val:.1f} ms', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        ax.set_xticks(bx); ax.set_xticklabels(bar_labs)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(bar_vals) + max(bar_errs) * 2 + max(bar_vals) * 0.18)

    # fig.suptitle('Storage Device Comparison at L=80 — SIFT1M, no cache, T=1',
    #              fontsize=12)
    plt.tight_layout()
    save(fig, 'fig5_headline_comparison_L80')

# ── Print summary table ───────────────────────────────────────────────────────
SEP = "─" * 78
print(f"\n{SEP}")
print("  SUMMARY TABLE")
print(SEP)
print(f"  {'Device':<35} {'L':>4} {'Mean(ms)':>10} {'P999(ms)':>10} "
      f"{'Tail':>8} {'Recall':>8} {'Runs':>5}")
print(f"  {'─'*74}")
for dev in devices:
    for L in L_vals:
        sub = agg[(agg['device']==dev) & (agg['L']==L)]
        if len(sub) == 0: continue
        r = sub.iloc[0]
        label = DEVICE_LABEL.get(dev, dev)
        print(f"  {label:<35} {int(L):>4} "
              f"{r.mean_us/1000:>10.1f} {r.p999_us/1000:>10.1f} "
              f"{r.tail_amp:>8.2f}x {r.recall:>8.2f}% {int(r.runs):>5}")
    print()

