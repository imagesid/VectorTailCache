#!/usr/bin/env bash
# =============================================================================
# compare_beamwidth_concurrency.sh 
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
DATA_DIR="${VTC_DATA:-/mnt/nvme/vtc_data}"
INDEX_DIR="${VTC_INDEX:-/home/agung/vector/index}"
OUT_BASE="${VTC_RESULTS:-/home/agung/vector/results}"
PYTHON="${VIRTUAL_ENV:-/home/agung/vector/diskann/venv}/bin/python3"

# INDEX="/mnt/nvme/vtc_index_sift1m/disk_index"
INDEX="${INDEX_DIR}/sift1m/disk_index"
QUERY="${DATA_DIR}/sift1m/sift_query.bin"
GT="${DATA_DIR}/sift1m/sift_groundtruth.bin"

# 
BEAMWIDTHS=(2 4 8 16)
THREADS_LIST=(1 4 8 16)
CACHE_SIZE=100000
L_VALUE=80
RUNS=5

# Point --resume at a previous beamwidth_concurrency_<timestamp> dir (e.g.
# one that only covered W=2,8) to reuse its results.tsv and skip any
# (bw, threads, run) combo whose log already exists — only the new W=4,16
# combos actually run. Without --resume, starts a fresh timestamped dir
# and runs everything.
if [[ -n "${RESUME_DIR}" ]]; then
    OUT="${RESUME_DIR}"
    if [[ ! -d "${OUT}" ]]; then
        echo "ERROR: Resume dir not found: ${OUT}"
        exit 1
    fi
    echo "Resuming from: ${OUT}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUT="${OUT_BASE}/beamwidth_concurrency_${TIMESTAMP}"
fi
RAW="${OUT}/raw"
mkdir -p "${RAW}"

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

# ── TSV header (adds `threads` vs. compare_beamwidth.sh's schema) ───────────
[[ -f "${TSV}" ]] || \
    printf "policy\tbeamwidth\tthreads\tcache_size\trun\tL\tqps\tmean_us\tp999_us\tmean_ios\tmean_io_us\trecall\ttail_amp\n" \
        > "${TSV}"

# ── Log parser ────────────────────────────────────────────────────────────────
parse_log() {
    local logfile="$1" policy="$2" bw="$3" threads="$4" cache="$5" run="$6"
    "${PYTHON}" - "$logfile" "$policy" "$bw" "$threads" "$cache" "$run" "$TSV" << 'PYEOF'
import sys, re
logfile, policy, bw, threads, cache, run, tsv = sys.argv[1:]
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
        f"{policy}\t{bw}\t{threads}\t{cache}\t{run}\t{L}\t"
        f"{qps}\t{mean}\t{p999}\t{ios}\t{io_us}\t{recall}\t{tail:.3f}"
    )
if not rows:
    print(f"  [WARN] no rows parsed from {logfile}", file=sys.stderr)
else:
    with open(tsv, 'a') as f:
        f.write('\n'.join(rows) + '\n')
    print(f"  parsed {len(rows)} row(s)")
PYEOF
}

TOTAL_PER_POLICY=$(( ${#BEAMWIDTHS[@]} * ${#THREADS_LIST[@]} * RUNS ))

log "============================================================"
log "Beamwidth x Concurrency: BFS vs PCS"
log "Beamwidths : ${BEAMWIDTHS[*]}"
log "Threads    : ${THREADS_LIST[*]}"
log "L (fixed)  : ${L_VALUE}"
log "Cache size : ${CACHE_SIZE}"
log "Runs       : ${RUNS}"
log "Run order  : ALL PCS first, then ALL BFS"
log "Output     : ${OUT}"
log "============================================================"

# =============================================================================
# PHASE 1 — ALL PCS RUNS
# =============================================================================
log ""
log "--- PHASE 1: PCS (all beamwidths x all threads x all runs) ---"
n=0
for bw in "${BEAMWIDTHS[@]}"; do
    for threads in "${THREADS_LIST[@]}"; do
        for run in $(seq 1 $RUNS); do
            n=$(( n+1 ))
            logfile="${RAW}/pcs_bw${bw}_t${threads}_r${run}.log"
            if [[ -f "${logfile}" ]]; then
                log "[PCS ${n}/${TOTAL_PER_POLICY}] [SKIP] already done: beamwidth=${bw} threads=${threads} run=${run}"
                continue
            fi
            log "[PCS ${n}/${TOTAL_PER_POLICY}] beamwidth=${bw} threads=${threads} run=${run}"
            drop_cache
            "${SEARCH}" \
                --data_type float --dist_fn l2 \
                --index_path_prefix "${INDEX}" \
                --query_file "${QUERY}" \
                --gt_file "${GT}" \
                --result_path "${RAW}/res_pcs_bw${bw}_t${threads}_r${run}" \
                --num_nodes_to_cache "${CACHE_SIZE}" \
                --cache_policy pcs \
                --pcs_tail_percentile 0.90 \
                -K 10 -T "${threads}" -W "${bw}" \
                -L "${L_VALUE}" \
                > "${logfile}" 2>&1
            parse_log "${logfile}" "pcs" "${bw}" "${threads}" "${CACHE_SIZE}" "${run}"
        done
    done
done
log "Phase 1 complete — PCS done."

# =============================================================================
# PHASE 2 — ALL BFS RUNS
# =============================================================================
log ""
log "--- PHASE 2: BFS (all beamwidths x all threads x all runs) ---"
n=0
for bw in "${BEAMWIDTHS[@]}"; do
    for threads in "${THREADS_LIST[@]}"; do
        for run in $(seq 1 $RUNS); do
            n=$(( n+1 ))
            logfile="${RAW}/bfs_bw${bw}_t${threads}_r${run}.log"
            if [[ -f "${logfile}" ]]; then
                log "[BFS ${n}/${TOTAL_PER_POLICY}] [SKIP] already done: beamwidth=${bw} threads=${threads} run=${run}"
                continue
            fi
            log "[BFS ${n}/${TOTAL_PER_POLICY}] beamwidth=${bw} threads=${threads} run=${run}"
            drop_cache
            "${SEARCH}" \
                --data_type float --dist_fn l2 \
                --index_path_prefix "${INDEX}" \
                --query_file "${QUERY}" \
                --gt_file "${GT}" \
                --result_path "${RAW}/res_bfs_bw${bw}_t${threads}_r${run}" \
                --num_nodes_to_cache "${CACHE_SIZE}" \
                --cache_policy bfs \
                -K 10 -T "${threads}" -W "${bw}" \
                -L "${L_VALUE}" \
                > "${logfile}" 2>&1
            parse_log "${logfile}" "bfs" "${bw}" "${threads}" "${CACHE_SIZE}" "${run}"
        done
    done
done
log "Phase 2 complete — BFS done."

log ""
log "============================================================"
log "DONE — $(wc -l < ${TSV}) rows written to ${TSV}"
log "============================================================"