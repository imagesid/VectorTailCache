#!/usr/bin/env bash
# =============================================================================
# run_all_datasets.sh
# BFS-vs-PCS comparison across ALL five datasets:
#   sift1m, gist1m, deep10m, glove1.2m, msturing1m
# =============================================================================
set -euo pipefail


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

# Which dataset(s) to run: comma-separated subset of
# sift1m,gist1m,deep10m,glove,msturing  (default: all)
DATASETS="${DATASETS:-all}"

CACHE_SIZES=(0 50000 100000)
L_VALUE=80
THREADS=1
RUNS="${RUNS:-5}"

# ── run dir / resume ─────────────────────────────────────────────────────────
if [[ -n "${RESUME_DIR}" ]]; then
    RUN_DIR="${RESUME_DIR}"
    if [[ ! -d "${RUN_DIR}" ]]; then
        echo "ERROR: Resume dir not found: ${RUN_DIR}"
        exit 1
    fi
    echo "Resuming from: ${RUN_DIR}"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="${OUT_BASE}/all_datasets_${TIMESTAMP}"
    mkdir -p "${RUN_DIR}"
fi

RAW="${RUN_DIR}/raw"
mkdir -p "${RAW}"
LOG="${RUN_DIR}/master.log"
TSV="${RUN_DIR}/results.tsv"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

[[ -f "${TSV}" ]] || \
    printf "dataset\tpolicy\tcache_size\tL\tthreads\trun\tqps\tmean_us\tp999_us\tmean_ios\trecall\ttail_amp\n" \
        > "${TSV}"

drop_cache() {
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 2 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    sleep 30
}

dataset_selected() {
    local name="$1"
    [[ "${DATASETS}" == "all" ]] && return 0
    [[ ",${DATASETS}," == *",${name},"* ]] && return 0
    return 1
}

# =============================================================================
# Dataset paths
# =============================================================================

# ---- SIFT1M (assumed already built) ----------------------------------------
SIFT_INDEX="${INDEX_DIR}/sift1m/disk_index"
# SIFT_INDEX="/mnt/nvme/vtc_index_sift1m/disk_index"
SIFT_QUERY="${DATA_DIR}/sift1m/sift_query.bin"
SIFT_GT="${DATA_DIR}/sift1m/sift_groundtruth.bin"
SIFT_DIST="l2"

# ---- GIST1M -----------------------------------------------------------------
GIST_INDEX="${INDEX_DIR}/gist1m/disk_index"
GIST_QUERY="${DATA_DIR}/gist1m/gist_query.bin"
GIST_GT="${DATA_DIR}/gist1m/gist_groundtruth.bin"
GIST_DIST="l2"

# ---- DEEP-10M -----------------------------------------------------------------
DEEP_DATA="${DATA_DIR}/deep10m"
DEEP_INDEX="${INDEX_DIR}/deep10m/disk_index"
DEEP_QUERY="${DEEP_DATA}/deep10m_query.bin"
DEEP_GT="${DEEP_DATA}/deep10m_groundtruth.bin"
DEEP_DIST="l2"

# ---- GloVe-1.2M ---------------------------------------------------------------
GLOVE_DIR="${DATA_DIR}/glove1.2m"
GLOVE_QUERY="${GLOVE_DIR}/glove_query.fbin"
GLOVE_GT="${GLOVE_DIR}/glove_groundtruth.bin"
GLOVE_INDEX="${INDEX_DIR}/glove1.2m/disk_index"
GLOVE_DIST="cosine"

# ---- MSTuring-ANNS 1M subset ---------------------------------------------------
MSTURING_DIR="${DATA_DIR}/msturing1m"
MSTURING_QUERY="${MSTURING_DIR}/query100K.fbin"
MSTURING_GT="${MSTURING_DIR}/msturing-gt-1M"
MSTURING_INDEX="${INDEX_DIR}/msturing1m/disk_index"
MSTURING_DIST="l2"

# =============================================================================
# Dataset prep — all data/indexes are assumed already downloaded and built.
# We only verify the expected files are present; nothing is downloaded or
# built here.
# =============================================================================

check_index_and_files() {
    local name="$1" idx_prefix="$2"; shift 2
    local missing=0
    if [[ ! -f "${idx_prefix}_disk.index" ]]; then
        log "ERROR: ${name} index not found: ${idx_prefix}_disk.index"
        missing=1
    fi
    for f in "$@"; do
        if [[ ! -f "${f}" ]]; then
            log "ERROR: ${name} file not found: ${f}"
            missing=1
        fi
    done
    if [[ "${missing}" -eq 1 ]]; then
        log "ERROR: ${name} is missing required files (see above). Skipping this dataset."
        return 1
    fi
    log "${name}: index and data files found."
    return 0
}

# =============================================================================
# Result parsing (same DiskANN search_disk_index log format used everywhere)
# =============================================================================
parse_log() {
    local logfile="$1" dataset="$2" policy="$3" cache="$4" run="$5"
    "${PYTHON}" - "$logfile" "$dataset" "$policy" "$cache" "$THREADS" "$run" "$TSV" << 'PYEOF'
import sys, re
logfile, dataset, policy, cache, threads, run, tsv = sys.argv[1:]
with open(logfile) as f:
    content = f.read()

pat = re.compile(
    r'^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s+([\d.]+))?',
    re.MULTILINE
)

rows = []
for m in pat.finditer(content):
    L      = m.group(1)
    qps    = m.group(3)
    mean   = m.group(4)
    p999   = m.group(5)
    ios    = m.group(6)
    recall = m.group(9) if m.group(9) is not None else "0"
    try:
        mean_f = float(mean)
        p999_f = float(p999)
    except ValueError:
        continue
    l_val = int(L)
    if l_val < 10 or l_val > 500:
        continue
    tail = p999_f / mean_f if mean_f > 0 else 0
    rows.append(
        f"{dataset}\t{policy}\t{cache}\t{L}\t{threads}\t{run}\t"
        f"{qps}\t{mean}\t{p999}\t{ios}\t{recall}\t{tail:.3f}"
    )

if not rows:
    print(f"  [WARN] no rows parsed from {logfile}", file=sys.stderr)
else:
    with open(tsv, 'a') as f:
        f.write('\n'.join(rows) + '\n')
    print(f"  parsed {len(rows)} row(s)")
PYEOF
}

# =============================================================================
# Search sweep (L=80 only, cache in {0, 50K, 100K}) — resumable per run
# =============================================================================
run_policy() {
    local dataset="$1" policy="$2" cache="$3" run="$4"
    local index="$5" query="$6" gt="$7" dist="$8"

    local logfile="${RAW}/${dataset}_${policy}_c${cache}_r${run}.log"
    if [[ -f "${logfile}" ]]; then
        log "  [SKIP] already done: $(basename "${logfile}")"
    else
        local extra_args=()
        if [[ "${policy}" == "pcs" ]]; then
            extra_args=(--pcs_tail_percentile 0.90)
        fi
        drop_cache
        "${SEARCH}" \
            --data_type float --dist_fn "${dist}" \
            --index_path_prefix "${index}" \
            --query_file "${query}" \
            --gt_file "${gt}" \
            --result_path "${RAW}/res_${dataset}_${policy}_c${cache}_r${run}" \
            --num_nodes_to_cache "${cache}" \
            --cache_policy "${policy}" \
            "${extra_args[@]}" \
            -K 10 -T "${THREADS}" \
            -L "${L_VALUE}" \
            > "${logfile}" 2>&1
    fi
    parse_log "${logfile}" "${dataset}" "${policy}" "${cache}" "${run}"
}

sweep_dataset() {
    local dataset="$1" index="$2" query="$3" gt="$4" dist="$5"
    local total=$(( ${#CACHE_SIZES[@]} * RUNS * 2 ))
    local n=0

    log "--- ${dataset}: Phase 1/2 — PCS sweep (all cache sizes x all runs) ---"
    for cache in "${CACHE_SIZES[@]}"; do
        for run in $(seq 1 "${RUNS}"); do
            n=$(( n + 1 ))
            log "[${dataset} ${n}/${total}] PCS cache=${cache} run=${run}"
            run_policy "${dataset}" "pcs" "${cache}" "${run}" "${index}" "${query}" "${gt}" "${dist}"
        done
    done

    log "--- ${dataset}: Phase 2/2 — BFS sweep (all cache sizes x all runs) ---"
    for cache in "${CACHE_SIZES[@]}"; do
        for run in $(seq 1 "${RUNS}"); do
            n=$(( n + 1 ))
            log "[${dataset} ${n}/${total}] BFS cache=${cache} run=${run}"
            run_policy "${dataset}" "bfs" "${cache}" "${run}" "${index}" "${query}" "${gt}" "${dist}"
        done
    done
    log "${dataset} sweep done."
}

# =============================================================================
# Main
# =============================================================================
[[ -f "${SEARCH}" ]] || { log "ERROR: search_disk_index binary not found at ${SEARCH}"; exit 1; }

log "======================================================="
log "ALL-DATASETS BFS vs PCS comparison"
log "Run dir: ${RUN_DIR}"
log "Datasets: sift1m, gist1m, deep10m, glove1.2m, msturing1m (selected: ${DATASETS})"
log "Cache sizes: ${CACHE_SIZES[*]} | L: ${L_VALUE} | Runs: ${RUNS} | Threads: ${THREADS}"
log "Policy order per dataset: ALL PCS runs first (cold), then ALL BFS runs (cold)"
log "======================================================="

# ---- prep phase: verify everything needed is already in place -------------
RUN_SIFT=0; RUN_GIST=0; RUN_DEEP=0; RUN_GLOVE=0; RUN_MSTURING=0

if dataset_selected sift1m; then
    check_index_and_files "SIFT1M" "${SIFT_INDEX}" "${SIFT_QUERY}" "${SIFT_GT}" && RUN_SIFT=1
fi
if dataset_selected gist1m; then
    check_index_and_files "GIST1M" "${GIST_INDEX}" "${GIST_QUERY}" "${GIST_GT}" && RUN_GIST=1
fi
if dataset_selected deep10m; then
    check_index_and_files "DEEP-10M" "${DEEP_INDEX}" "${DEEP_QUERY}" "${DEEP_GT}" && RUN_DEEP=1
fi
if dataset_selected glove; then
    check_index_and_files "GloVe-1.2M" "${GLOVE_INDEX}" "${GLOVE_QUERY}" "${GLOVE_GT}" && RUN_GLOVE=1
fi
if dataset_selected msturing; then
    check_index_and_files "MSTuring-1M" "${MSTURING_INDEX}" "${MSTURING_QUERY}" "${MSTURING_GT}" && RUN_MSTURING=1
fi

# ---- sweep phase --------------------------------------------------------------
if [[ "${RUN_SIFT}" -eq 1 ]]; then
    log "=== Running sweep: SIFT1M ==="
    sweep_dataset "sift1m" "${SIFT_INDEX}" "${SIFT_QUERY}" "${SIFT_GT}" "${SIFT_DIST}"
else
    log "=== Skipping sweep: SIFT1M ==="
fi

if [[ "${RUN_GIST}" -eq 1 ]]; then
    log "=== Running sweep: GIST1M ==="
    sweep_dataset "gist1m" "${GIST_INDEX}" "${GIST_QUERY}" "${GIST_GT}" "${GIST_DIST}"
else
    log "=== Skipping sweep: GIST1M ==="
fi

if [[ "${RUN_DEEP}" -eq 1 ]]; then
    log "=== Running sweep: DEEP-10M ==="
    sweep_dataset "deep10m" "${DEEP_INDEX}" "${DEEP_QUERY}" "${DEEP_GT}" "${DEEP_DIST}"
else
    log "=== Skipping sweep: DEEP-10M ==="
fi

if [[ "${RUN_GLOVE}" -eq 1 ]]; then
    log "=== Running sweep: GloVe-1.2M ==="
    sweep_dataset "glove1.2m" "${GLOVE_INDEX}" "${GLOVE_QUERY}" "${GLOVE_GT}" "${GLOVE_DIST}"
else
    log "=== Skipping sweep: GloVe-1.2M ==="
fi

if [[ "${RUN_MSTURING}" -eq 1 ]]; then
    log "=== Running sweep: MSTuring-1M ==="
    sweep_dataset "msturing1m" "${MSTURING_INDEX}" "${MSTURING_QUERY}" "${MSTURING_GT}" "${MSTURING_DIST}"
else
    log "=== Skipping sweep: MSTuring-1M ==="
fi

log "======================================================="
log "DONE"
log "Results : ${TSV}  ($(wc -l < "${TSV}") rows)"
log "Figures : ${PYTHON} run_all_datasets.py ${RUN_DIR}"
log "======================================================="
