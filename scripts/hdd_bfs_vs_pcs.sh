#!/usr/bin/env bash
# =============================================================================
# hdd_bfs_vs_pcs.sh — Quick BFS vs PCS comparison on HDD
#
# Uses existing SIFT1M index already copied to /mnt/hdd2/vtc_index_sift1m
# L=80 only, T=1, 3 cold runs each, cache=10K and 100K nodes
# =============================================================================
set -euo pipefail

SEARCH="${DISKANN_SEARCH:-/home/agung/vector/diskann/build/apps/search_disk_index}"
DATA_DIR="${VTC_DATA:-/mnt/nvme/vtc_data}"
HDD_MOUNT="/mnt/hdd2"
INDEX="${HDD_MOUNT}/vtc_index_sift1m/disk_index"
QUERY="${DATA_DIR}/sift1m/sift_query.bin"
GT="${DATA_DIR}/sift1m/sift_groundtruth.bin"
PYTHON="${VIRTUAL_ENV:-/home/agung/vector/diskann/venv}/bin/python3"

OUT_BASE="${VTC_RESULTS:-/home/agung/vector/results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_BASE}/hdd_bfs_vs_pcs_${TIMESTAMP}"
RAW="${OUT}/raw"
mkdir -p "${RAW}"

LOG="${OUT}/run.log"
TSV="${OUT}/results.tsv"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

drop_cache() {
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 2 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    sleep 5
}

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_SIZES=(0 100000)
POLICIES=(pcs bfs)
RUNS=3
L=80

# Use 1K query subset — 10x faster than full 10K on HDD
QUERY_1K="${DATA_DIR}/sift1m/sift_query_1k.bin"
if [[ ! -f "${QUERY_1K}" ]]; then
    log "Creating 1K query subset..."
    "${PYTHON}" - "${DATA_DIR}/sift1m/sift_query.bin" "${QUERY_1K}" << SUBSETEOF
import sys, numpy as np, struct
src2, dst = sys.argv[1], sys.argv[2]
with open(src2, 'rb') as f:
    num = struct.unpack('<i', f.read(4))[0]
    dim = struct.unpack('<i', f.read(4))[0]
    data = np.frombuffer(f.read(num * dim * 4), dtype=np.float32).reshape(num, dim)
n_out = min(1000, num)
data = data[:n_out]
with open(dst, 'wb') as f:
    f.write(struct.pack('<i', n_out))
    f.write(struct.pack('<i', dim))
    f.write(data.astype(np.float32).tobytes())
print(f"  1K subset created: {dst} ({n_out} queries x {dim}d)")
SUBSETEOF
fi
QUERY="${QUERY_1K}"

# ── Checks ────────────────────────────────────────────────────────────────────
[[ -f "${SEARCH}" ]]             || { log "ERROR: binary not found"; exit 1; }
[[ -f "${INDEX}_disk.index" ]]   || { log "ERROR: HDD index not found at ${INDEX}"; exit 1; }
[[ -f "${QUERY}" ]]              || { log "ERROR: query not found"; exit 1; }
[[ -f "${GT}" ]]                 || { log "ERROR: groundtruth not found"; exit 1; }

# ── TSV header ────────────────────────────────────────────────────────────────
printf "policy\tcache_size\trun\tqps\tmean_us\tp999_us\tmean_ios\tmean_io_us\trecall\ttail_amp\n" \
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
    tail = float(p999) / float(mean) if float(mean) > 0 else 0
    rows.append(
        f"{policy}\t{cache}\t{run}\t{qps}\t{mean}\t{p999}\t"
        f"{ios}\t{io_us}\t{recall}\t{tail:.3f}"
    )
if not rows:
    print(f"  [WARN] no rows parsed", file=sys.stderr)
else:
    with open(tsv, 'a') as f:
        f.write('\n'.join(rows) + '\n')
    print(f"  parsed {len(rows)} rows")
PYEOF
}

# ── Main sweep ────────────────────────────────────────────────────────────────
log "================================================="
log "HDD BFS vs PCS  —  SIFT1M, L=${L}, T=1"
log "Index  : ${INDEX}"
log "Caches : ${CACHE_SIZES[*]}"
log "Runs   : ${RUNS}"
log "Output : ${OUT}"
log "================================================="

total=$(( ${#CACHE_SIZES[@]} * ${#POLICIES[@]} * RUNS ))
n=0

for cache in "${CACHE_SIZES[@]}"; do
    for run in $(seq 1 $RUNS); do
        for policy in "${POLICIES[@]}"; do
            n=$(( n + 1 ))
            logfile="${RAW}/${policy}_c${cache}_r${run}.log"
            log "[$n/$total] policy=${policy} cache=${cache} run=${run}"

            drop_cache

            extra=""
            [[ "$policy" == "pcs" ]] && extra="--pcs_tail_percentile 0.90"

            "${SEARCH}" \
                --data_type float --dist_fn l2 \
                --index_path_prefix "${INDEX}" \
                --query_file "${QUERY}" \
                --gt_file "${GT}" \
                --result_path "${RAW}/res_${policy}_c${cache}_r${run}" \
                --num_nodes_to_cache "${cache}" \
                --cache_policy "${policy}" \
                -K 10 -T 1 -L ${L} \
                ${extra} \
                > "${logfile}" 2>&1

            parse_log "${logfile}" "${policy}" "${cache}" "${run}"
        done
    done
done

log "Sweep done. Generating figures..."

# ── Analysis ──────────────────────────────────────────────────────────────────
"${PYTHON}" - "${OUT}" "${TSV}" << 'PYEOF'
import sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT, TSV = sys.argv[1], sys.argv[2]

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'lines.linewidth': 2.2, 'lines.markersize': 7,
})

PC = {'bfs': '#9E9E9E', 'pcs': '#E53935'}
PM = {'bfs': 'o',       'pcs': '^'}
PL = {'bfs': 'BFS (DiskANN default)', 'pcs': 'PCS (VectorTailCache)'}

df = pd.read_csv(TSV, sep='\t')
for col in ['qps','mean_us','p999_us','mean_ios','mean_io_us','recall','tail_amp','cache_size']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

agg = df.groupby(['policy','cache_size']).agg(
    mean_us  =('mean_us',  'mean'), mean_std =('mean_us',  'std'),
    p999_us  =('p999_us',  'mean'), p999_std =('p999_us',  'std'),
    tail_amp =('tail_amp', 'mean'),
    recall   =('recall',   'mean'),
).reset_index()

cache_sizes = sorted(agg['cache_size'].unique())
xs    = range(len(cache_sizes))
xlbls = [f'{int(c)//1000}K' if c >= 1000 else '0' for c in cache_sizes]

def save(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  Saved: {name}")

# ── Fig 1: P999 ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
for policy in ['bfs','pcs']:
    sub = agg[agg['policy']==policy].set_index('cache_size')
    vals = [sub['p999_us'].get(c, np.nan)/1000 for c in cache_sizes]
    errs = [sub['p999_std'].get(c, np.nan)/1000 for c in cache_sizes]
    ax.errorbar(xs, vals, yerr=errs, color=PC[policy], marker=PM[policy],
                label=PL[policy], capsize=4)
ax.set_xticks(xs); ax.set_xticklabels(xlbls)
ax.set_xlabel('Nodes Cached in DRAM')
ax.set_ylabel('P999 Latency (ms)')
ax.set_title(f'HDD: P999 Latency — BFS vs PCS\nSIFT1M, L=80, T=1')
ax.legend()
save(fig, 'hdd_p999_bfs_vs_pcs')

# ── Fig 2: P999 reduction % ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
bfs_d = agg[agg['policy']=='bfs'].set_index('cache_size')['p999_us']
pcs_d = agg[agg['policy']=='pcs'].set_index('cache_size')['p999_us']
reductions = []; valid_xs = []; valid_xlbls = []
for i, c in enumerate(cache_sizes):
    if c == 0: continue
    bv = bfs_d.get(c, np.nan); pv = pcs_d.get(c, np.nan)
    if not np.isnan(bv) and not np.isnan(pv) and bv > 0:
        reductions.append((bv - pv) / bv * 100)
        valid_xs.append(i); valid_xlbls.append(xlbls[i])
colors = ['#4CAF50' if r > 2 else ('#F44336' if r < -2 else '#FF9800')
          for r in reductions]
bars = ax.bar(valid_xs, reductions, color=colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, reductions):
    ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{val:+.1f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(valid_xs); ax.set_xticklabels(valid_xlbls)
ax.set_xlabel('Nodes Cached')
ax.set_ylabel('P999 Reduction vs BFS (%)')
ax.set_title('HDD: P999 Reduction of PCS over BFS\n(positive = PCS better)')
save(fig, 'hdd_p999_reduction')

# ── Fig 3: Mean latency ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
for policy in ['bfs','pcs']:
    sub = agg[agg['policy']==policy].set_index('cache_size')
    vals = [sub['mean_us'].get(c, np.nan)/1000 for c in cache_sizes]
    errs = [sub['mean_std'].get(c, np.nan)/1000 for c in cache_sizes]
    ax.errorbar(xs, vals, yerr=errs, color=PC[policy], marker=PM[policy],
                label=PL[policy], capsize=4)
ax.set_xticks(xs); ax.set_xticklabels(xlbls)
ax.set_xlabel('Nodes Cached in DRAM')
ax.set_ylabel('Mean Latency (ms)')
ax.set_title('HDD: Mean Latency — BFS vs PCS\nSIFT1M, L=80, T=1')
ax.legend()
save(fig, 'hdd_mean_bfs_vs_pcs')

# ── Summary ───────────────────────────────────────────────────────────────────
SEP = "─" * 65
print(f"\n{SEP}")
print("  HDD: BFS vs PCS SUMMARY  (L=80, T=1, SIFT1M)")
print(SEP)
print(f"  {'Cache':>8}  {'Policy':<8}  {'Mean(ms)':>10}  "
      f"{'P999(ms)':>10}  {'Tail':>7}  {'Recall':>8}")
print(f"  {'─'*61}")
for c in cache_sizes:
    for policy in ['pcs','bfs']:
        sub = agg[(agg['policy']==policy) & (agg['cache_size']==c)]
        if len(sub) == 0: continue
        r = sub.iloc[0]
        clbl = f'{int(c)//1000}K' if c >= 1000 else '0'
        print(f"  {clbl:>8}  {policy:<8}  "
              f"{r.mean_us/1000:>10.1f}  {r.p999_us/1000:>10.1f}  "
              f"{r.tail_amp:>7.2f}x  {r.recall:>8.2f}%")
    if 0 < cache_sizes.index(c):
        bv = agg[(agg['policy']=='bfs') & (agg['cache_size']==c)]['p999_us'].values
        pv = agg[(agg['policy']=='pcs') & (agg['cache_size']==c)]['p999_us'].values
        if len(bv) and len(pv) and bv[0] > 0:
            delta = (bv[0]-pv[0])/bv[0]*100
            sym = "✓ PCS wins" if delta > 2 else ("✗ BFS wins" if delta < -2 else "≈ tie")
            print(f"  {'':>8}  P999 reduction: {delta:+.1f}%  {sym}")
    print()
print(SEP)
print(f"Figures: {OUT}/")
PYEOF

log ""
log "================================================="
log "DONE — results in ${OUT}/"
log "================================================="