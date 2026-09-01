#!/usr/bin/env bash
# =============================================================================
# multi-threads.sh 
# =============================================================================
set -euo pipefail

# ── parse args ────────────────────────────────────────────────────────────────
RESUME_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume) RESUME_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SEARCH="${DISKANN_SEARCH:-/home/agung/vector/diskann/build/apps/search_disk_index}"
BUILD="${DISKANN_BUILD:-/home/agung/vector/diskann/build}/apps/build_disk_index"
DATA_DIR="${VTC_DATA:-/mnt/nvme/vtc_data}"
INDEX_DIR="${VTC_INDEX:-/home/agung/vector/index}"
OUT_BASE="${VTC_RESULTS:-/home/agung/vector/results}"
PYTHON="${VIRTUAL_ENV:-/home/agung/vector/diskann/venv}/bin/python3"

# ── set run dir ───────────────────────────────────────────────────────────────
if [[ -n "${RESUME_DIR}" ]]; then
    RUN_DIR="${RESUME_DIR}"
    if [[ ! -d "${RUN_DIR}" ]]; then
        echo "ERROR: Resume dir not found: ${RUN_DIR}"
        exit 1
    fi
    echo "Resuming from: ${RUN_DIR}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${OUT_BASE}/extended_eval_${TIMESTAMP}"
    mkdir -p "${RUN_DIR}"
fi

LOG="${RUN_DIR}/master.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

# ── paths ─────────────────────────────────────────────────────────────────────
INDEX="${INDEX_DIR}/sift1m/disk_index"
# INDEX="/mnt/nvme/vtc_index_sift1m/disk_index"
QUERY="${DATA_DIR}/sift1m/sift_query.bin"
GT="${DATA_DIR}/sift1m/sift_groundtruth.bin"

GIST_INDEX="${INDEX_DIR}/gist1m/disk_index"
GIST_QUERY="${DATA_DIR}/gist1m/gist_query.bin"
GIST_GT="${DATA_DIR}/gist1m/gist_groundtruth.bin"

DEEP_DATA="${DATA_DIR}/deep10m"
DEEP_INDEX="${INDEX_DIR}/deep10m/disk_index"
DEEP_QUERY="${DEEP_DATA}/deep10m_query.bin"
DEEP_GT="${DEEP_DATA}/deep10m_groundtruth.bin"

drop_cache() {
    sync
    for lvl in 3 1 2 3; do
        echo $lvl > /proc/sys/vm/drop_caches 2>/dev/null || true
    done
    sleep 5
}

# ── shared log parser ─────────────────────────────────────────────────────────
parse_log() {
    local logfile="$1" policy="$2" dataset="$3" cache="$4" threads="$5" run="$6" tsv="$7"
    "${PYTHON}" - "$logfile" "$policy" "$dataset" "$cache" "$threads" "$run" "$tsv" << 'PYEOF'
import sys, re
logfile, policy, dataset, cache, threads, run, tsv = sys.argv[1:]
with open(logfile) as f:
    content = f.read()
pat = re.compile(
    r'^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+))?',
    re.MULTILINE
)
rows = []
for m in pat.finditer(content):
    L = int(m.group(1))
    if L < 10 or L > 500:
        continue
    qps    = m.group(3); mean   = m.group(4)
    p999   = m.group(5); ios    = m.group(6)
    recall = m.group(9) if m.group(9) else "0"
    try:
        tail = float(p999) / float(mean) if float(mean) > 0 else 0
    except:
        continue
    rows.append(f"{policy}\t{dataset}\t{cache}\t{L}\t{threads}\t{run}\t"
                f"{qps}\t{mean}\t{p999}\t{ios}\t{recall}\t{tail:.3f}")
if rows:
    with open(tsv, 'a') as f:
        f.write('\n'.join(rows) + '\n')
    print(f"  parsed {len(rows)} L-values")
else:
    print(f"  [WARN] no rows from {logfile}", file=sys.stderr)
PYEOF
}

# ── run one PCS vs BFS pair ───────────────────────────────────────────────────
run_pair() {
    local dataset="$1" index="$2" query="$3" gt="$4"
    local cache="$5" threads="$6" run="$7" tsv="$8" raw_dir="$9"
    local l_args="${10}"

    local pcs_log="${raw_dir}/pcs_c${cache}_t${threads}_r${run}.log"
    local bfs_log="${raw_dir}/bfs_c${cache}_t${threads}_r${run}.log"

    # PCS
    if [[ ! -f "${pcs_log}" ]]; then
        drop_cache
        "${SEARCH}" \
            --data_type float --dist_fn l2 \
            --index_path_prefix "${index}" \
            --query_file "${query}" --gt_file "${gt}" \
            --result_path "${raw_dir}/res_pcs_c${cache}_t${threads}_r${run}" \
            --num_nodes_to_cache "${cache}" \
            --cache_policy pcs --pcs_tail_percentile 0.90 \
            -K 10 -T "${threads}" ${l_args} \
            > "${pcs_log}" 2>&1
    else
        log "  [SKIP] PCS already done: $(basename ${pcs_log})"
    fi
    parse_log "${pcs_log}" "pcs" "${dataset}" \
        "${cache}" "${threads}" "${run}" "${tsv}"

    # BFS
    if [[ ! -f "${bfs_log}" ]]; then
        drop_cache
        "${SEARCH}" \
            --data_type float --dist_fn l2 \
            --index_path_prefix "${index}" \
            --query_file "${query}" --gt_file "${gt}" \
            --result_path "${raw_dir}/res_bfs_c${cache}_t${threads}_r${run}" \
            --num_nodes_to_cache "${cache}" \
            --cache_policy bfs \
            -K 10 -T "${threads}" ${l_args} \
            > "${bfs_log}" 2>&1
    else
        log "  [SKIP] BFS already done: $(basename ${bfs_log})"
    fi
    parse_log "${bfs_log}" "bfs" "${dataset}" \
        "${cache}" "${threads}" "${run}" "${tsv}"
}

log "============================================================"
log "VectorTailCache Extended Evaluation"
log "Run dir: ${RUN_DIR}."
log "Mode   : $([ -n '${RESUME_DIR}' ] && echo 'RESUME' || echo 'FRESH')"
log "============================================================"

# =============================================================================
# EXP 1 — SIFT1M MULTI-THREAD
# =============================================================================
EXP1_DIR="${RUN_DIR}/exp1_sift1m_multithread"
EXP1_RAW="${EXP1_DIR}/raw"
EXP1_TSV="${EXP1_DIR}/results.tsv"

if [[ -f "${EXP1_TSV}" ]] && [[ $(wc -l < "${EXP1_TSV}") -gt 10 ]]; then
    log "EXP 1: SIFT1M multithread — already complete ($(wc -l < ${EXP1_TSV}) rows), skipping"
else
    log ""
    log "============================================================"
    log "EXP 1: SIFT1M Multi-thread (~2 hours)"
    log "============================================================"
    mkdir -p "${EXP1_RAW}"
    [[ -f "${EXP1_TSV}" ]] || \
        printf "policy\tdataset\tcache_size\tL\tthreads\trun\tqps\tmean_us\tp999_us\tmean_ios\trecall\ttail_amp\n" \
            > "${EXP1_TSV}"

    CACHE_SIZES_1=(0 10000 100000)
    THREAD_COUNTS_1=(1 4 8 16)
    RUNS_1=3
    L_ARGS_1="-L 40 -L 80 -L 120"
    TOTAL_1=$(( ${#CACHE_SIZES_1[@]} * ${#THREAD_COUNTS_1[@]} * RUNS_1 ))
    n=0

    for cache in "${CACHE_SIZES_1[@]}"; do
        for threads in "${THREAD_COUNTS_1[@]}"; do
            for run in $(seq 1 $RUNS_1); do
                n=$(( n+1 ))
                log "[Exp1 $n/$TOTAL_1] SIFT1M cache=${cache} T=${threads} run=${run}"
                run_pair "sift1m" "${INDEX}" "${QUERY}" "${GT}" \
                    "${cache}" "${threads}" "${run}" \
                    "${EXP1_TSV}" "${EXP1_RAW}" "${L_ARGS_1}"
            done
        done
    done
    log "Exp 1 done — $(wc -l < ${EXP1_TSV}) rows"
fi



log ""
log "============================================================"
log "COMPLETE"
log "Run dir: ${RUN_DIR}"
log "============================================================"
