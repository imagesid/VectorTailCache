#!/usr/bin/env bash
# =============================================================================
# compare_devices.sh — Compare tail latency across storage devices
#
# Devices:
#   NVMe : /mnt/nvme   Samsung 980 PRO (PCIe 4.0, 3D TLC)
#   SATA : /mnt/hdd1   Samsung 870 SATA SSD (SATA III, 3D TLC)
#   HDD  : /mnt/hdd2   Seagate Exos (7200 RPM mechanical)
# =============================================================================
set -euo pipefail

SEARCH="${DISKANN_SEARCH:-/home/agung/vector/diskann/build/apps/search_disk_index}"
DATA_DIR="${VTC_DATA:-/mnt/nvme/vtc_data}"
INDEX_SRC="${VTC_INDEX:-/home/agung/vector/index}/sift1m"
OUT_BASE="${VTC_RESULTS:-/home/agung/vector/results}"
PYTHON="${VIRTUAL_ENV:-/home/agung/vector/diskann/venv}/bin/python3"

QUERY="${DATA_DIR}/sift1m/sift_query.bin"
GT="${DATA_DIR}/sift1m/sift_groundtruth.bin"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_BASE}/device_compare_${TIMESTAMP}"
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
    sleep 5
}

# ── Device config ─────────────────────────────────────────────────────────────
# label → mount point
declare -A DEVICE_MOUNT
DEVICE_MOUNT[nvme]="/mnt/nvme"
DEVICE_MOUNT[sata]="/mnt/hdd1"
DEVICE_MOUNT[hdd]="/mnt/hdd2"

RUNS=3
L_VALUES=(40 80 120)

# ── Sanity checks ─────────────────────────────────────────────────────────────
[[ -f "${SEARCH}" ]]     || { log "ERROR: binary not found: ${SEARCH}"; exit 1; }
[[ -f "${QUERY}" ]]      || { log "ERROR: query not found: ${QUERY}";   exit 1; }
[[ -f "${GT}" ]]         || { log "ERROR: gt not found: ${GT}";         exit 1; }
[[ -f "${INDEX_SRC}/disk_index_disk.index" ]] || \
    { log "ERROR: source index not found: ${INDEX_SRC}/disk_index_disk.index"; exit 1; }

# ── TSV header ────────────────────────────────────────────────────────────────
printf "device\trun\tL\tqps\tmean_us\tp999_us\tmean_ios\tmean_io_us\trecall\ttail_amp\n" \
    > "${TSV}"

# ── Log parser ────────────────────────────────────────────────────────────────
parse_log() {
    local logfile="$1" device="$2" run="$3"
    "${PYTHON}" - "$logfile" "$device" "$run" "$TSV" << 'PYEOF'
import sys, re
logfile, device, run, tsv = sys.argv[1:]
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
        L_int = int(L)
        if L_int < 10 or L_int > 500: continue
    except ValueError:
        continue
    qps, mean, p999, ios, io_us, cpu = m.group(2,3,4,5,6,7)
    recall = m.group(8) if m.group(8) else "0"
    tail = float(p999) / float(mean) if float(mean) > 0 else 0
    rows.append(
        f"{device}\t{run}\t{L}\t{qps}\t{mean}\t{p999}\t"
        f"{ios}\t{io_us}\t{recall}\t{tail:.3f}"
    )
if not rows:
    print(f"  [WARN] no rows parsed from {logfile}", file=sys.stderr)
else:
    with open(tsv, 'a') as f:
        f.write('\n'.join(rows) + '\n')
    print(f"  parsed {len(rows)} L-values")
PYEOF
}

# ── Main loop ─────────────────────────────────────────────────────────────────
log "======================================================="
log "VectorTailCache — Device Comparison"
log "Devices : ${!DEVICE_MOUNT[*]}"
log "L values: ${L_VALUES[*]}"
log "Runs    : ${RUNS}"
log "Output  : ${OUT}"
log "======================================================="

for device in nvme sata hdd; do
#for device in nvme sata; do
    mount="${DEVICE_MOUNT[$device]}"
    # index_dst="${mount}/vtc_index_sift1m"
    if [[ "$device" == "nvme" ]]; then
        index_dst="${INDEX_SRC}"          # use original path directly, no copy
    else
        index_dst="${mount}/vtc_index_sift1m"
    fi

    log ""
    log "─────────────────────────────────────────────────────"
    log "Device: ${device}  →  ${mount}"
    log "─────────────────────────────────────────────────────"

    # Check mount point exists and is accessible
    if [[ ! -d "${mount}" ]]; then
        log "  SKIP: ${mount} does not exist"
        continue
    fi

    # Check it is actually mounted (not just a directory)
    if ! mountpoint -q "${mount}" 2>/dev/null; then
        log "  WARNING: ${mount} may not be mounted — continuing anyway"
    fi

    # Check free space (need ~3GB for SIFT1M index)
    FREE_GB=$(df -BG "${mount}" | awk 'NR==2{gsub("G","",$4); print $4}')
    if (( FREE_GB < 3 )); then
        log "  SKIP: not enough free space on ${mount} (${FREE_GB}GB free, need 3GB)"
        continue
    fi

    # Copy index to device (skip if already there and complete)
    log "  Checking index at ${index_dst}..."
    if [[ -f "${index_dst}/disk_index_disk.index" ]]; then
        log "  Index already on ${device}, skipping copy."
    else
        log "  Copying index to ${device} (may take a few minutes for HDD)..."
        mkdir -p "${index_dst}"
        cp "${INDEX_SRC}/"* "${index_dst}/"
        sync
        log "  Index copy complete."
    fi

    # Run searches
    L_ARGS=$(printf " -L %s" "${L_VALUES[@]}")

    for run in $(seq 1 $RUNS); do
        logfile="${RAW}/${device}_run${run}.log"
        log "  Run ${run}/${RUNS}..."

        drop_cache

        "${SEARCH}" \
            --data_type float --dist_fn l2 \
            --index_path_prefix "${index_dst}/disk_index" \
            --query_file "${QUERY}" \
            --gt_file "${GT}" \
            --result_path "${RAW}/res_${device}_r${run}" \
            --num_nodes_to_cache 0 \
            --cache_policy bfs \
            -K 10 -T 1 \
            ${L_ARGS} \
            > "${logfile}" 2>&1

        parse_log "${logfile}" "${device}" "${run}"
        log "  Run ${run} done."
    done

    log "  ✓ ${device} complete."
done


log ""
log "======================================================="
log "DONE"
log "Results : ${TSV}"
log "======================================================="
