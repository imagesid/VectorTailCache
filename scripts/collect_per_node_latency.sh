#!/usr/bin/env bash
# =============================================================================
# collect_per_node_latency.sh
# Runs ONE cold-start search with per-node logging enabled,
# then generates the paper table comparing hub nodes vs tail-critical nodes.
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT_BASE}/per_node_${TIMESTAMP}"
mkdir -p "${OUT}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Drop OS page cache for cold reads
log "Dropping OS page cache..."
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
sleep 5

log "Running search with per-node latency logging (no cache, L=80)..."

# Run with no cache so ALL node reads go to SSD
# --log_per_node_latency writes per_node_latency.tsv in CWD
cd "${OUT}"

"${SEARCH}" \
    --data_type float --dist_fn l2 \
    --index_path_prefix "${INDEX}" \
    --query_file "${QUERY}" \
    --gt_file "${GT}" \
    --result_path "${OUT}/res" \
    --num_nodes_to_cache 0 \
    --cache_policy bfs \
    -K 10 -T 1 -L 80 \
    > "${OUT}/search.log" 2>&1

log "Search complete. Per-node data: ${OUT}/per_node_latency.tsv"
wc -l "${OUT}/per_node_latency.tsv"


log "Done. Output in ${OUT}/"
