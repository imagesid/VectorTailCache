#!/usr/bin/env bash
# =============================================================================
# compare_three_policies.sh — BFS vs Frequency vs PCS on SIFT1M
#
# Compares all three cache admission policies:
#   BFS       — DiskANN default (hop distance from medoid)
#   Frequency — visit count from sample queries (commented out in original)
#   PCS       — Path Criticality Score (VectorTailCache)
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

CACHE_SIZES=(0 10000 50000 100000)
L_VALUES=(80)
THREADS=1
RUNS=5

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_BASE}/three_policy_${TIMESTAMP}"
RAW="${OUT}/raw"
FIGS="${OUT}/figures"
mkdir -p "${RAW}" "${FIGS}"

LOG="${OUT}/pipeline.log"
TSV="${OUT}/results.tsv"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

drop_cache() {
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 2 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    sleep 30
}

# ── Checks ────────────────────────────────────────────────────────────────────
[[ -f "${SEARCH}" ]]           || { log "ERROR: binary not found"; exit 1; }
[[ -f "${INDEX}_disk.index" ]] || { log "ERROR: index not found"; exit 1; }
[[ -f "${QUERY}" ]]            || { log "ERROR: query not found"; exit 1; }
[[ -f "${GT}" ]]               || { log "ERROR: gt not found"; exit 1; }

# ── TSV header ────────────────────────────────────────────────────────────────
printf "policy\tcache_size\trun\tL\tqps\tmean_us\tp999_us\tmean_ios\tmean_io_us\trecall\ttail_amp\n" \
    > "${TSV}"

# ── Log parser ────────────────────────────────────────────────────────────────
parse_log() {
    local logfile="$1" policy="$2" cache="$3" run="$4"
    "${PYTHON}" - "$logfile" "$policy" "$cache" "$run" "$TSV" << 'PYEOF'
import sys, re
logfile, policy, cache, run, tsv = sys.argv[1:]
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
        f"{policy}\t{cache}\t{run}\t{L}\t"
        f"{qps}\t{mean}\t{p999}\t{ios}\t{io_us}\t{recall}\t{tail:.3f}"
    )
if not rows:
    print(f"  [WARN] no rows from {logfile}", file=__import__('sys').stderr)
else:
    with open(tsv, 'a') as f:
        f.write('\n'.join(rows) + '\n')
    print(f"  parsed {len(rows)} L-values")
PYEOF
}

# ── Main sweep ────────────────────────────────────────────────────────────────
L_ARGS=$(printf " -L %s" "${L_VALUES[@]}")
TOTAL=$(( ${#CACHE_SIZES[@]} * RUNS ))
n=0

log "============================================================"
log "Three Policy Comparison: BFS vs Frequency vs PCS"
log "Policy order per run: PCS → Frequency → BFS"
log "Runs: ${RUNS} | Threads: ${THREADS} | sleep: 5s between runs"
log "Cache sizes: ${CACHE_SIZES[*]}"
log "L values   : ${L_VALUES[*]}"
log "Output     : ${OUT}"
log "============================================================"

for cache in "${CACHE_SIZES[@]}"; do
    for run in $(seq 1 $RUNS); do
        n=$(( n+1 ))
        log "[$n/$TOTAL] cache=${cache} run=${run}"

        # ── PCS (always first — cold) ─────────────────────────────────────────
        log "  → PCS (cold)"
        drop_cache
        "${SEARCH}" \
            --data_type float --dist_fn l2 \
            --index_path_prefix "${INDEX}" \
            --query_file "${QUERY}" \
            --gt_file "${GT}" \
            --result_path "${RAW}/res_pcs_c${cache}_r${run}" \
            --num_nodes_to_cache "${cache}" \
            --cache_policy pcs \
            --pcs_tail_percentile 0.90 \
            -K 10 -T "${THREADS}" \
            ${L_ARGS} \
            > "${RAW}/pcs_c${cache}_r${run}.log" 2>&1
        parse_log "${RAW}/pcs_c${cache}_r${run}.log" "pcs" "${cache}" "${run}"

        # ── Frequency policy ──────────────────────────────────────────────────
        # Uses same query file as PCS for fair comparison
        # (generate_cache_list_from_sample_queries with same query sample)
        log "  → Frequency (cold)"
        drop_cache
        "${SEARCH}" \
            --data_type float --dist_fn l2 \
            --index_path_prefix "${INDEX}" \
            --query_file "${QUERY}" \
            --gt_file "${GT}" \
            --result_path "${RAW}/res_freq_c${cache}_r${run}" \
            --num_nodes_to_cache "${cache}" \
            --cache_policy frequency \
            -K 10 -T "${THREADS}" \
            ${L_ARGS} \
            > "${RAW}/freq_c${cache}_r${run}.log" 2>&1
        parse_log "${RAW}/freq_c${cache}_r${run}.log" "frequency" "${cache}" "${run}"

        # ── BFS (last — most conservative for BFS) ────────────────────────────
        log "  → BFS (cold)"
        drop_cache
        "${SEARCH}" \
            --data_type float --dist_fn l2 \
            --index_path_prefix "${INDEX}" \
            --query_file "${QUERY}" \
            --gt_file "${GT}" \
            --result_path "${RAW}/res_bfs_c${cache}_r${run}" \
            --num_nodes_to_cache "${cache}" \
            --cache_policy bfs \
            -K 10 -T "${THREADS}" \
            ${L_ARGS} \
            > "${RAW}/bfs_c${cache}_r${run}.log" 2>&1
        parse_log "${RAW}/bfs_c${cache}_r${run}.log" "bfs" "${cache}" "${run}"

        log "  Run ${run} done."
    done
done

log "Sweep complete — $(wc -l < ${TSV}) rows"

# ── Analysis ──────────────────────────────────────────────────────────────────
"${PYTHON}" - "${OUT}" "${TSV}" "${FIGS}" << 'PYEOF'
import sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT, TSV, FIGS = sys.argv[1], sys.argv[2], sys.argv[3]

df = pd.read_csv(TSV, sep='\t')
for col in ['qps','mean_us','p999_us','mean_ios','mean_io_us','recall','tail_amp','cache_size','L']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

agg = df.groupby(['policy','cache_size','L']).agg(
    mean_us  =('mean_us',  'mean'), mean_std =('mean_us',  'std'),
    p999_us  =('p999_us',  'mean'), p999_std =('p999_us',  'std'),
    tail_amp =('tail_amp', 'mean'),
    recall   =('recall',   'mean'),
).reset_index()

cache_sizes = sorted(agg['cache_size'].unique())
L_vals      = sorted(agg['L'].unique())
xs          = range(len(cache_sizes))
xlbls       = [f'{int(c)//1000}K' if c >= 1000 else '0' for c in cache_sizes]

PC = {'bfs': '#9E9E9E', 'frequency': '#FF9800', 'pcs': '#E53935'}
PM = {'bfs': 'o',       'frequency': 's',        'pcs': '^'}
PL = {'bfs': 'BFS',     'frequency': 'Frequency','pcs': 'PCS'}
POLICIES = ['bfs', 'frequency', 'pcs']

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 11, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'lines.linewidth': 2.2, 'lines.markersize': 7,
})

def save(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(f"{FIGS}/{name}.{ext}")
    plt.close(fig)
    print(f"  Saved: {name}")

def finish(fig):
    fig.tight_layout(pad=0.4, w_pad=0.3)

# ── Fig 1: P999 per L ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(L_vals), figsize=(8, 3))
if len(L_vals)==1: axes=[axes]
for i, (ax, L) in enumerate(zip(axes, L_vals)):
    for policy in POLICIES:
        sub = agg[(agg['policy']==policy) & (agg['L']==L)].set_index('cache_size')
        vals = [sub['p999_us'].get(c, np.nan)/1000 for c in cache_sizes]
        errs = [sub['p999_std'].get(c, np.nan)/1000 for c in cache_sizes]
        ax.errorbar(xs, vals, yerr=errs,
                    color=PC[policy], marker=PM[policy],
                    label=PL[policy], capsize=4)
    r = agg[agg['L']==L]['recall'].mean()
    ax.set_xticks(xs); ax.set_xticklabels(xlbls)
    ax.set_xlabel('Nodes Cached')
    if i==0: ax.set_ylabel('P999 Latency (ms)')
    ax.set_title(f'L={L}  (Recall≈{r:.1f}%)')
    ax.legend(fontsize=8)
finish(fig)
save(fig, 'fig1_p999_three_policies')

# ── Fig 2: P999 reduction vs BFS ────────────────────────────────────────────
fig, axes = plt.subplots(1, len(L_vals), figsize=(8, 3))
if len(L_vals)==1: axes=[axes]
for i, (ax, L) in enumerate(zip(axes, L_vals)):
    bfs_d = agg[(agg['policy']=='bfs') & (agg['L']==L)].set_index('cache_size')['p999_us']
    width = 0.35
    for j, policy in enumerate(['frequency','pcs']):
        pcs_d = agg[(agg['policy']==policy) & (agg['L']==L)].set_index('cache_size')['p999_us']
        reductions=[]; valid_xs=[]; valid_xlbls=[]
        for k, c in enumerate(cache_sizes):
            if c==0: continue
            bv=bfs_d.get(c,np.nan); pv=pcs_d.get(c,np.nan)
            if not np.isnan(bv) and not np.isnan(pv) and bv>0:
                reductions.append((bv-pv)/bv*100)
                valid_xs.append(k); valid_xlbls.append(xlbls[k])
        offset = (j - 0.5) * width
        bars = ax.bar([x+offset for x in valid_xs], reductions,
                      width=width, color=PC[policy], alpha=0.85,
                      label=PL[policy])
        for bar, val in zip(bars, reductions):
            if abs(val) > 1:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()*1.02 if val>=0 else bar.get_height()-2,
                        f'{val:+.0f}%', ha='center', va='bottom',
                        fontsize=7, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(valid_xs); ax.set_xticklabels(valid_xlbls)
    ax.set_xlabel('Nodes Cached')
    if i==0: ax.set_ylabel('P999 Reduction vs BFS (%)')
    ax.set_title(f'L={L}')
    ax.legend(fontsize=8)
finish(fig)
save(fig, 'fig2_reduction_vs_bfs')

# ── Summary table ─────────────────────────────────────────────────────────────
SEP = "─" * 75
print(f"\n{SEP}")
print("  THREE POLICY SUMMARY (T=1)")
print(SEP)
print(f"  {'Cache':>8}  {'L':>5}  {'Policy':<12}  "
      f"{'Mean(ms)':>10}  {'P999(ms)':>10}  {'Tail':>7}  {'Recall':>8}")
print(f"  {'─'*71}")
for cache in cache_sizes:
    for L in L_vals:
        for policy in POLICIES:
            sub = agg[(agg['policy']==policy) &
                      (agg['cache_size']==cache) & (agg['L']==L)]
            if len(sub)==0: continue
            r = sub.iloc[0]
            clbl = f'{int(cache)//1000}K' if cache>=1000 else '0'
            print(f"  {clbl:>8}  {int(L):>5}  {policy:<12}  "
                  f"{r.mean_us/1000:>10.1f}  {r.p999_us/1000:>10.1f}  "
                  f"{r.tail_amp:>7.2f}x  {r.recall:>8.2f}%")
        # BFS vs Freq reduction
        bv = agg[(agg['policy']=='bfs') & (agg['cache_size']==cache) &
                 (agg['L']==L)]['p999_us'].values
        fv = agg[(agg['policy']=='frequency') & (agg['cache_size']==cache) &
                 (agg['L']==L)]['p999_us'].values
        pv = agg[(agg['policy']=='pcs') & (agg['cache_size']==cache) &
                 (agg['L']==L)]['p999_us'].values
        if len(bv) and len(fv) and len(pv) and bv[0]>0:
            df_bfs  = (bv[0]-fv[0])/bv[0]*100
            dp_bfs  = (bv[0]-pv[0])/bv[0]*100
            print(f"  {'':>8}  {'':>5}  {'':>12}  "
                  f"Freq vs BFS: {df_bfs:+.1f}%  PCS vs BFS: {dp_bfs:+.1f}%")
        print()
print(SEP)
print(f"Figures: {FIGS}/")
PYEOF

log ""
log "============================================================"
log "DONE — results in ${OUT}/"
log "Run figures: python3 scripts/gen_three_policy_figures.py ${OUT}"
log "============================================================"
