#!/usr/bin/env bash
# =============================================================================
# sensitivity_percentile.sh — PCS sensitivity analysis across tail percentiles
#
# Runs PCS with tail_percentile in {0.75, 0.80, 0.85, 0.90, 0.95, 0.99}
# and compares P999 and mean latency at cache=100K, L=80, T=1.
#
# Run order: ALL PCS percentile runs first (coldest), then BFS baseline.
# =============================================================================
set -euo pipefail

SEARCH="${DISKANN_SEARCH:-/home/agung/vector/diskann/build/apps/search_disk_index}"
DATA_DIR="${VTC_DATA:-/mnt/nvme/vtc_data}"
INDEX_DIR="${VTC_INDEX:-/home/agung/vector/index}"
OUT_BASE="${VTC_RESULTS:-/home/agung/vector/results}"
PYTHON="${VIRTUAL_ENV:-/home/agung/vector/diskann/venv}/bin/python3"

INDEX="${INDEX_DIR}/sift1m/disk_index"
QUERY="${DATA_DIR}/sift1m/sift_query.bin"
GT="${DATA_DIR}/sift1m/sift_groundtruth.bin"

# ── Config ────────────────────────────────────────────────────────────────────
PERCENTILES=(0.75 0.80 0.85 0.90 0.95 0.99)
CACHE_SIZE=100000
L_VALUES=(40 80 120)
THREADS=1
RUNS=5

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_BASE}/sensitivity_percentile_${TIMESTAMP}"
RAW="${OUT}/raw"
mkdir -p "${RAW}"

LOG="${OUT}/pipeline.log"
TSV="${OUT}/results.tsv"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

# drop_cache() {
#     sync
#     echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
#     sleep 10
# }
drop_cache() {
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 2 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    sleep 30
}

# ── Checks ────────────────────────────────────────────────────────────────────
[[ -f "${SEARCH}" ]]           || { log "ERROR: binary not found: ${SEARCH}"; exit 1; }
[[ -f "${INDEX}_disk.index" ]] || { log "ERROR: index not found: ${INDEX}"; exit 1; }
[[ -f "${QUERY}" ]]            || { log "ERROR: query not found: ${QUERY}"; exit 1; }
[[ -f "${GT}" ]]               || { log "ERROR: gt not found: ${GT}"; exit 1; }

# ── TSV header ────────────────────────────────────────────────────────────────
printf "policy\tpercentile\tcache_size\trun\tL\tqps\tmean_us\tp999_us\tmean_ios\tmean_io_us\trecall\ttail_amp\n" \
    > "${TSV}"

# ── Log parser ────────────────────────────────────────────────────────────────
parse_log() {
    local logfile="$1" policy="$2" pct="$3" cache="$4" run="$5"
    "${PYTHON}" - "$logfile" "$policy" "$pct" "$cache" "$run" "$TSV" << 'PYEOF'
import sys, re
logfile, policy, pct, cache, run, tsv = sys.argv[1:]
with open(logfile) as f:
    content = f.read()
pat = re.compile(
    r'^\s*(\d+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+))?',
    re.MULTILINE
)
rows = []
for m in pat.finditer(content):
    L = m.group(1)
    try:
        if not (10 <= int(L) <= 500): continue
    except ValueError:
        continue
    qps, mean, p999, ios, io_us, cpu = m.group(2,3,4,5,6,7)
    recall = m.group(8) if m.group(8) else "0"
    tail = float(p999)/float(mean) if float(mean) > 0 else 0
    rows.append(
        f"{policy}\t{pct}\t{cache}\t{run}\t{L}\t"
        f"{qps}\t{mean}\t{p999}\t{ios}\t{io_us}\t{recall}\t{tail:.3f}"
    )
if not rows:
    print(f"  [WARN] no rows parsed from {logfile}", file=__import__('sys').stderr)
else:
    with open(tsv, 'a') as f:
        f.write('\n'.join(rows) + '\n')
    print(f"  parsed {len(rows)} L-values")
PYEOF
}

L_ARGS=$(printf " -L %s" "${L_VALUES[@]}")
TOTAL_PCS=$(( ${#PERCENTILES[@]} * RUNS ))
TOTAL_BFS=${RUNS}

log "============================================================"
log "PCS Tail Percentile Sensitivity Analysis"
log "Percentiles: ${PERCENTILES[*]}"
log "Cache      : ${CACHE_SIZE}"
log "L values   : ${L_VALUES[*]}"
log "Runs       : ${RUNS}"
log "Output     : ${OUT}"
log "============================================================"

# =============================================================================
# PHASE 1 — ALL PCS RUNS (all percentiles)
# =============================================================================
log ""
log "─── PHASE 1: PCS across all percentiles ────────────────────"
n=0
for pct in "${PERCENTILES[@]}"; do
    for run in $(seq 1 $RUNS); do
        n=$(( n+1 ))
        pct_label=$(echo "$pct" | tr '.' '_')
        logfile="${RAW}/pcs_p${pct_label}_r${run}.log"
        log "[PCS ${n}/${TOTAL_PCS}] percentile=${pct} run=${run}"
        drop_cache
        "${SEARCH}" \
            --data_type float --dist_fn l2 \
            --index_path_prefix "${INDEX}" \
            --query_file "${QUERY}" \
            --gt_file "${GT}" \
            --result_path "${RAW}/res_pcs_p${pct_label}_r${run}" \
            --num_nodes_to_cache "${CACHE_SIZE}" \
            --cache_policy pcs \
            --pcs_tail_percentile "${pct}" \
            -K 10 -T "${THREADS}" \
            ${L_ARGS} \
            > "${logfile}" 2>&1
        parse_log "${logfile}" "pcs" "${pct}" "${CACHE_SIZE}" "${run}"
        log "  Done."
    done
done
log "Phase 1 complete."

# =============================================================================
# PHASE 2 — BFS BASELINE
# =============================================================================
log ""
log "─── PHASE 2: BFS baseline ───────────────────────────────────"
n=0
for run in $(seq 1 $RUNS); do
    n=$(( n+1 ))
    logfile="${RAW}/bfs_r${run}.log"
    log "[BFS ${n}/${TOTAL_BFS}] run=${run}"
    drop_cache
    "${SEARCH}" \
        --data_type float --dist_fn l2 \
        --index_path_prefix "${INDEX}" \
        --query_file "${QUERY}" \
        --gt_file "${GT}" \
        --result_path "${RAW}/res_bfs_r${run}" \
        --num_nodes_to_cache "${CACHE_SIZE}" \
        --cache_policy bfs \
        -K 10 -T "${THREADS}" \
        ${L_ARGS} \
        > "${logfile}" 2>&1
    parse_log "${logfile}" "bfs" "baseline" "${CACHE_SIZE}" "${run}"
    log "  Done."
done
log "Phase 2 complete."

# =============================================================================
# ANALYSIS
# =============================================================================
log ""
log "─── Analysis ────────────────────────────────────────────────"

"${PYTHON}" - "${TSV}" "${OUT}" << 'PYEOF'
import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TSV, OUT = sys.argv[1], sys.argv[2]
FIGS = os.path.join(OUT, "figures")
os.makedirs(FIGS, exist_ok=True)

df = pd.read_csv(TSV, sep='\t')
for col in ['mean_us','p999_us','recall','cache_size','L']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Aggregate
agg = df.groupby(['policy','percentile','L']).agg(
    mean_ms  = ('mean_us',  lambda x: x.mean()/1000),
    mean_std = ('mean_us',  lambda x: x.std()/1000),
    p999_ms  = ('p999_us',  lambda x: np.median(x)/1000),
    p999_std = ('p999_us',  lambda x: x.std()/1000),
    recall   = ('recall',   'mean'),
    runs     = ('run',      'count'),
).reset_index()

L_vals = sorted(df['L'].dropna().unique())
pcts   = sorted(df[df['policy']=='pcs']['percentile'].dropna().unique())
bfs    = agg[agg['policy']=='bfs']

# ── Print summary table ───────────────────────────────────────────────────────
SEP = "─" * 80
for L in L_vals:
    print(f"\n{SEP}")
    print(f"  L={int(L)} — P999 sensitivity to tail percentile (cache=100K, T=1)")
    print(SEP)
    print(f"  {'Policy/Pct':<14}  {'Mean(ms)':>10}  {'P999(ms)':>10}  "
          f"{'P999std':>9}  {'vs BFS':>8}  {'Recall':>8}")
    print(f"  {'─'*76}")

    bfs_p999 = bfs[bfs['L']==L]['p999_ms'].values
    bfs_mean = bfs[bfs['L']==L]['mean_ms'].values
    if len(bfs_p999):
        print(f"  {'BFS':<14}  {bfs_mean[0]:>10.1f}  {bfs_p999[0]:>10.1f}  "
              f"{bfs[bfs['L']==L]['p999_std'].values[0]:>9.2f}  "
              f"{'baseline':>8}  "
              f"{bfs[bfs['L']==L]['recall'].values[0]:>8.2f}%")

    for pct in pcts:
        sub = agg[(agg['policy']=='pcs') & (agg['percentile']==pct) & (agg['L']==L)]
        if len(sub) == 0: continue
        r = sub.iloc[0]
        delta = ((bfs_p999[0] - r.p999_ms) / bfs_p999[0] * 100
                 if len(bfs_p999) else float('nan'))
        sym = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        print(f"  {'PCS p='+str(pct):<14}  {r.mean_ms:>10.1f}  {r.p999_ms:>10.1f}  "
              f"{r.p999_std:>9.2f}  {sym:>8}  {r.recall:>8.2f}%")
    print(SEP)

# ── Figure: P999 vs percentile ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 11, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.dpi': 300, 'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
    'lines.linewidth': 2.2, 'lines.markersize': 8,
})

fig, axes = plt.subplots(1, len(L_vals), figsize=(8, 3))
if len(L_vals) == 1: axes = [axes]

colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(pcts)))

for ax, L in zip(axes, L_vals):
    pcs_p999 = []
    for i, pct in enumerate(pcts):
        sub = agg[(agg['policy']=='pcs') & (agg['percentile']==pct) & (agg['L']==L)]
        if len(sub) == 0: pcs_p999.append(np.nan); continue
        r = sub.iloc[0]
        pcs_p999.append(r.p999_ms)
        ax.errorbar([pct], [r.p999_ms], yerr=[r.p999_std],
                    color=colors[i], marker='^', capsize=4,
                    linewidth=1.5, markersize=7)

    # BFS baseline as horizontal line
    bfs_sub = bfs[bfs['L']==L]
    if len(bfs_sub):
        bv = bfs_sub.iloc[0].p999_ms
        ax.axhline(bv, color='#9E9E9E', ls='--', lw=2,
                   label=f'BFS ({bv:.1f}ms)')

    ax.plot(pcts, pcs_p999, color='#E53935', lw=1.5, alpha=0.5)
    ax.set_xlabel('Tail Percentile Threshold')
    if ax is axes[0]: ax.set_ylabel('P999 Latency (ms)\n(median, 5 runs)')
    r_val = agg[agg['L']==L]['recall'].mean()
    ax.set_title(f'L={int(L)}  (Recall={r_val:.1f}%)')
    ax.set_xticks(pcts)
    ax.set_xticklabels([str(p) for p in pcts], rotation=30, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)

fig.suptitle('PCS Sensitivity to Tail Percentile Threshold\nSIFT1M, cache=100K, T=1',
             fontsize=11, y=1.02)
plt.tight_layout(pad=0.5, w_pad=0.8)

for ext in ['pdf', 'png']:
    path = os.path.join(FIGS, f'sensitivity_percentile.{ext}')
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.12)
    print(f"Saved: {path}")
plt.close(fig)
PYEOF

log ""
log "============================================================"
log "DONE"
log "Results : ${TSV}"
log "Figures : ${OUT}/figures/"
log "============================================================"
