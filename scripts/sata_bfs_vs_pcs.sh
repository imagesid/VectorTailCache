#!/usr/bin/env bash
# =============================================================================
# sata_bfs_vs_pcs.sh — BFS vs PCS on SATA SSD (Samsung 870, /mnt/hdd1)
#
# Runs both policies from scratch with cache sweep.
# =============================================================================
set -euo pipefail

SEARCH="${DISKANN_SEARCH:-/home/agung/vector/diskann/build/apps/search_disk_index}"
DATA_DIR="${VTC_DATA:-/mnt/nvme/vtc_data}"
INDEX="/mnt/hdd1/vtc_index_sift1m/disk_index"
QUERY="${DATA_DIR}/sift1m/sift_query.bin"
GT="${DATA_DIR}/sift1m/sift_groundtruth.bin"
PYTHON="${VIRTUAL_ENV:-/home/agung/vector/diskann/venv}/bin/python3"
OUT_BASE="${VTC_RESULTS:-/home/agung/vector/results}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_BASE}/sata_bfs_vs_pcs_${TIMESTAMP}"
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
CACHE_SIZES=(0 10000 100000)
POLICIES=(pcs bfs)
RUNS=3
L_VALUES=(40 80 120)

# ── Checks ────────────────────────────────────────────────────────────────────
[[ -f "${SEARCH}" ]]           || { log "ERROR: binary not found: ${SEARCH}"; exit 1; }
[[ -f "${INDEX}_disk.index" ]] || { log "ERROR: index not found: ${INDEX}"; exit 1; }
[[ -f "${QUERY}" ]]            || { log "ERROR: query not found: ${QUERY}"; exit 1; }
[[ -f "${GT}" ]]               || { log "ERROR: gt not found: ${GT}"; exit 1; }

# ── TSV header — cache_size column included ───────────────────────────────────
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
    tail = float(p999) / float(mean) if float(mean) > 0 else 0
    rows.append(
        f"{policy}\t{cache}\t{run}\t{L}\t"
        f"{qps}\t{mean}\t{p999}\t{ios}\t{io_us}\t{recall}\t{tail:.3f}"
    )
if not rows:
    print(f"  [WARN] no rows parsed from {logfile}", file=sys.stderr)
else:
    with open(tsv, 'a') as f:
        f.write('\n'.join(rows) + '\n')
    print(f"  parsed {len(rows)} L-values")
PYEOF
}

# ── Main sweep ────────────────────────────────────────────────────────────────
L_ARGS=$(printf " -L %s" "${L_VALUES[@]}")
total=$(( ${#CACHE_SIZES[@]} * ${#POLICIES[@]} * RUNS ))
n=0

log "================================================="
log "SATA SSD: BFS vs PCS — SIFT1M, T=1"
log "Index   : ${INDEX}"
log "Caches  : ${CACHE_SIZES[*]}"
log "Policies: ${POLICIES[*]}"
log "Runs    : ${RUNS}"
log "L values: ${L_VALUES[*]}"
log "Output  : ${OUT}"
log "================================================="

for cache in "${CACHE_SIZES[@]}"; do
    for run in $(seq 1 $RUNS); do
        for policy in "${POLICIES[@]}"; do
            n=$(( n + 1 ))
            logfile="${RAW}/${policy}_c${cache}_r${run}.log"
            log "[${n}/${total}] policy=${policy} cache=${cache} run=${run}"

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
                -K 10 -T 1 \
                ${L_ARGS} \
                ${extra} \
                > "${logfile}" 2>&1

            parse_log "${logfile}" "${policy}" "${cache}" "${run}"
            log "  Done."
        done
    done
done

log ""
log "Sweep complete — $(wc -l < "${TSV}") rows written to ${TSV}"
log "Run the figure script:"
log "  python3 gen_sata_figures.py ${OUT}"
log "================================================="
