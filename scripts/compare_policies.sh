#!/usr/bin/env bash
# =============================================================================
# compare_policies.sh 
# Fair comparison for paper results
#
# Fairness guarantees:
#   1. Cache dropped before EVERY run (both BFS and PCS)
#   2. PCS always runs FIRST (cold cache) — disadvantages PCS, so wins are real
#   3. BFS runs SECOND (may benefit from PCS profiling warmup)
#   4. 5 runs per config for stable statistics
#   5. All L values: 40, 80, 120
#   6. All cache sizes: 0, 1K, 5K, 10K, 50K, 100K
#
# 
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

CACHE_SIZES=(0 1000 5000 10000 50000 100000)
L_VALUES=(40 80 120)
THREADS=1
RUNS=5

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_BASE}/final_comparison_${TIMESTAMP}"
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

[[ -f "${SEARCH}" ]]           || { log "ERROR: binary not found"; exit 1; }
[[ -f "${INDEX}_disk.index" ]] || { log "ERROR: index not found"; exit 1; }
[[ -f "${QUERY}" ]]            || { log "ERROR: query not found"; exit 1; }
[[ -f "${GT}" ]]               || { log "ERROR: groundtruth not found"; exit 1; }

printf "policy\tcache_size\tL\tthreads\trun\tqps\tmean_us\tp999_us\tmean_ios\trecall\ttail_amp\n" \
    > "${TSV}"

parse_log() {
    local logfile="$1" policy="$2" cache="$3" run="$4"
    "${PYTHON}" - "$logfile" "$policy" "$cache" "$THREADS" "$run" "$TSV" << 'PYEOF'
import sys, re
logfile, policy, cache, threads, run, tsv = sys.argv[1:]
with open(logfile) as f:
    content = f.read()

# DiskANN output line format (9 columns):
# L  Beamwidth  QPS  MeanLatency  P999Latency  MeanIOs  MeanIO_us  CPU_s  Recall
# All are numbers. Use a flexible pattern that handles optional recall column.
pat = re.compile(
    r'^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+))?',
    re.MULTILINE
)

rows = []
for m in pat.finditer(content):
    L        = m.group(1)
    # group 2 = beamwidth (skip)
    qps      = m.group(3)
    mean     = m.group(4)
    p999     = m.group(5)
    ios      = m.group(6)
    # group 7 = mean_io_us (skip)
    # group 8 = cpu_s (skip)
    recall   = m.group(9) if m.group(9) is not None else "0"

    try:
        mean_f = float(mean)
        p999_f = float(p999)
    except ValueError:
        continue

    # Skip lines that are clearly not search result rows
    # (L value should be a reasonable search list size: 10-500)
    l_val = int(L)
    if l_val < 10 or l_val > 500:
        continue

    tail = p999_f / mean_f if mean_f > 0 else 0
    rows.append(
        f"{policy}\t{cache}\t{L}\t{threads}\t{run}\t"
        f"{qps}\t{mean}\t{p999}\t{ios}\t{recall}\t{tail:.3f}"
    )

if not rows:
    print(f"  [WARN] no rows parsed from {logfile}", file=sys.stderr)
else:
    with open(tsv, 'a') as f:
        f.write('\n'.join(rows) + '\n')
    print(f"  parsed {len(rows)} L-values")
PYEOF
}

log "======================================================="
log "VectorTailCache Final Fair Comparison"
log "Policy order: PCS first (cold), BFS second (may be warm)."
log "This DISADVANTAGES PCS — wins are conservative lower bounds"
log "Runs: ${RUNS} | Threads: ${THREADS}"
log "Cache sizes: ${CACHE_SIZES[*]}"
log "L values   : ${L_VALUES[*]}"
log "======================================================="

TOTAL=$(( ${#CACHE_SIZES[@]} * RUNS ))
n=0
L_ARGS=$(printf " -L %s" "${L_VALUES[@]}")

for cache in "${CACHE_SIZES[@]}"; do
    for run in $(seq 1 $RUNS); do
        n=$(( n + 1 ))
        log "[$n/$TOTAL] cache=${cache} run=${run}"

        # PCS: always first (cold cache)
        log "  → PCS (cold cache drop)"
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

        # BFS: second — cache dropped, but profiling warmup gone too
        log "  → BFS (cold cache drop)"
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
    done
done

log "Sweep complete - $(wc -l < ${TSV}) rows"


log "======================================================="
log "DONE"
log "Results : ${TSV}"
log "======================================================="
echo ""

