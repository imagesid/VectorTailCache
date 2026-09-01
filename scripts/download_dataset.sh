#!/usr/bin/env bash
set -Eeuo pipefail

DATASET="${1:-}"
OUT_ROOT="${2:-./datasets}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_dataset.sh <dataset|all> [output_dir]

Datasets:
  sift1m
  gist1m
  deep10m
  glove1.2m
  msturing1m
  all

Aliases:
  sift, gist, deep, glove, glove1.2, msturing, turing

Standard output:
  <output_dir>/<dataset>/<dataset>.bin
  <output_dir>/<dataset>/<dataset>_query.bin
  <output_dir>/<dataset>/<dataset>_groundtruth.bin

Exception:
  GloVe provides only glove1.2m.bin because the official archive has
  no standard ANN query or ground-truth files.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

download() {
  local url="$1"
  local output="$2"

  mkdir -p "$(dirname "$output")"

  if [[ -s "$output" ]]; then
    echo "Using existing file: $output"
    return
  fi

  echo "Downloading: $url"
  curl -fL \
    --retry 5 \
    --retry-delay 3 \
    --continue-at - \
    --output "$output" \
    "$url"
}

normalize_name() {
  case "${1,,}" in
    sift|sift1m) echo "sift1m" ;;
    gist|gist1m) echo "gist1m" ;;
    deep|deep10m|deep-10m) echo "deep10m" ;;
    glove|glove1.2|glove1.2m|glove-1.2m) echo "glove1.2m" ;;
    msturing|msturing1m|msturing-1m|turing|turing1m) echo "msturing1m" ;;
    all) echo "all" ;;
    *) return 1 ;;
  esac
}

verify_vector_bin() {
  python3 - "$1" <<'PY'
import os, struct, sys
path = sys.argv[1]
with open(path, "rb") as f:
    header = f.read(8)
if len(header) != 8:
    raise SystemExit(f"Invalid vector file: {path}")
count, dim = struct.unpack("<II", header)
expected = 8 + count * dim * 4
actual = os.path.getsize(path)
if actual != expected:
    raise SystemExit(
        f"Invalid size for {path}: header={count}x{dim}, "
        f"expected={expected}, actual={actual}"
    )
print(f"Verified vector file: {path}")
print(f"  vectors    : {count:,}")
print(f"  dimensions : {dim}")
print(f"  size       : {actual / (1024**3):.3f} GiB")
PY
}

verify_truthset() {
  python3 - "$1" <<'PY'
import os, struct, sys
path = sys.argv[1]
with open(path, "rb") as f:
    header = f.read(8)
if len(header) != 8:
    raise SystemExit(f"Invalid ground-truth file: {path}")
nq, k = struct.unpack("<II", header)
actual = os.path.getsize(path)
expected = 8 + nq * k * 4 + nq * k * 4
if actual != expected:
    raise SystemExit(
        f"Invalid truthset size for {path}: header={nq}x{k}, "
        f"expected={expected}, actual={actual}"
    )
print(f"Verified ground truth: {path}")
print(f"  queries   : {nq:,}")
print(f"  neighbors : {k}")
print(f"  size      : {actual / (1024**2):.2f} MiB")
PY
}

convert_fvecs() {
  local source="$1"
  local output="$2"

  python3 - "$source" "$output" <<'PY'
import os, struct, sys
src, dst = sys.argv[1], sys.argv[2]
size = os.path.getsize(src)

with open(src, "rb") as f:
    raw = f.read(4)
if len(raw) != 4:
    raise SystemExit(f"Invalid fvecs file: {src}")

dim = struct.unpack("<i", raw)[0]
record_size = 4 + dim * 4

if dim <= 0 or size % record_size:
    raise SystemExit(
        f"Invalid fvecs layout: size={size}, dim={dim}, "
        f"record_size={record_size}"
    )

count = size // record_size
tmp = dst + ".tmp"

with open(src, "rb") as fin, open(tmp, "wb") as fout:
    fout.write(struct.pack("<II", count, dim))
    for i in range(count):
        raw_dim = fin.read(4)
        if len(raw_dim) != 4:
            raise SystemExit(f"Unexpected EOF at vector {i}")
        current_dim = struct.unpack("<i", raw_dim)[0]
        if current_dim != dim:
            raise SystemExit(
                f"Inconsistent dimension at vector {i}: "
                f"expected {dim}, got {current_dim}"
            )
        vector = fin.read(dim * 4)
        if len(vector) != dim * 4:
            raise SystemExit(f"Unexpected EOF in vector {i}")
        fout.write(vector)

os.replace(tmp, dst)
print(f"Converted {count:,} vectors x {dim} dimensions")
PY
}

convert_ivecs_truthset() {
  local source="$1"
  local output="$2"

  python3 - "$source" "$output" <<'PY'
import os, struct, sys, tempfile
src, dst = sys.argv[1], sys.argv[2]
size = os.path.getsize(src)

with open(src, "rb") as f:
    raw = f.read(4)
if len(raw) != 4:
    raise SystemExit(f"Invalid ivecs file: {src}")

k = struct.unpack("<i", raw)[0]
record_size = 4 + k * 4

if k <= 0 or size % record_size:
    raise SystemExit(
        f"Invalid ivecs layout: size={size}, k={k}, "
        f"record_size={record_size}"
    )

nq = size // record_size
tmp = dst + ".tmp"

with tempfile.TemporaryFile() as ids:
    with open(src, "rb") as fin:
        for q in range(nq):
            raw_k = fin.read(4)
            if len(raw_k) != 4:
                raise SystemExit(f"Unexpected EOF at query {q}")
            current_k = struct.unpack("<i", raw_k)[0]
            if current_k != k:
                raise SystemExit(
                    f"Inconsistent k at query {q}: expected {k}, got {current_k}"
                )
            row = fin.read(k * 4)
            if len(row) != k * 4:
                raise SystemExit(f"Unexpected EOF at query {q}")
            ids.write(row)

    ids.seek(0)

    with open(tmp, "wb") as fout:
        fout.write(struct.pack("<II", nq, k))

        while True:
            block = ids.read(16 * 1024 * 1024)
            if not block:
                break
            fout.write(block)

        remaining = nq * k * 4
        zero_block = b"\x00" * min(16 * 1024 * 1024, max(1, remaining))

        while remaining:
            amount = min(len(zero_block), remaining)
            fout.write(zero_block[:amount])
            remaining -= amount

os.replace(tmp, dst)
print(f"Converted ground truth: {nq:,} queries x {k} neighbors")
PY
}

download_sift1m() {
  local dir="$OUT_ROOT/sift1m"
  local archive="$dir/sift.tar.gz"
  local extracted="$dir/sift"
  local base="$dir/sift1m.bin"
  local query="$dir/sift_query.bin"
  local gt="$dir/sift_groundtruth.bin"

  mkdir -p "$dir"

  if [[ ! -d "$extracted" ]]; then
    download "https://corpus-texmex.irisa.fr/sift.tar.gz" "$archive"
    tar -xzf "$archive" -C "$dir"
  fi

  [[ -s "$base" ]] || convert_fvecs "$extracted/sift_base.fvecs" "$base"
  [[ -s "$query" ]] || convert_fvecs "$extracted/sift_query.fvecs" "$query"
  [[ -s "$gt" ]] || convert_ivecs_truthset "$extracted/sift_groundtruth.ivecs" "$gt"

  verify_vector_bin "$base"
  verify_vector_bin "$query"
  verify_truthset "$gt"
}

download_gist1m() {
  local dir="$OUT_ROOT/gist1m"
  local archive="$dir/gist.tar.gz"
  local extracted="$dir/gist"
  local base="$dir/gist1m.bin"
  local query="$dir/gist_query.bin"
  local gt="$dir/gist_groundtruth.bin"

  mkdir -p "$dir"

  if [[ ! -d "$extracted" ]]; then
    download "https://corpus-texmex.irisa.fr/gist.tar.gz" "$archive"
    tar -xzf "$archive" -C "$dir"
  fi

  [[ -s "$base" ]] || convert_fvecs "$extracted/gist_base.fvecs" "$base"
  [[ -s "$query" ]] || convert_fvecs "$extracted/gist_query.fvecs" "$query"
  [[ -s "$gt" ]] || convert_ivecs_truthset "$extracted/gist_groundtruth.ivecs" "$gt"

  verify_vector_bin "$base"
  verify_vector_bin "$query"
  verify_truthset "$gt"
}

download_deep10m() {
  local dir="$OUT_ROOT/deep10m"
  local base="$dir/deep10m.bin"
  local query="$dir/deep_query.bin"
  local gt="$dir/deep_groundtruth.bin"

  mkdir -p "$dir"

  download \
    "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin" \
    "$base"

  download \
    "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin" \
    "$query"

  download \
    "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/deep-10M" \
    "$gt"

  verify_vector_bin "$base"
  verify_vector_bin "$query"
  verify_truthset "$gt"
}

download_glove1_2m() {
  local dir="$OUT_ROOT/glove1.2m"
  local archive="$dir/glove.twitter.27B.zip"
  local text_file="$dir/glove.twitter.27B.100d.txt"
  local base="$dir/glove1.2m.bin"

  mkdir -p "$dir"

  if [[ ! -s "$base" ]]; then
    download "https://nlp.stanford.edu/data/glove.twitter.27B.zip" "$archive"

    if [[ ! -s "$text_file" ]]; then
      unzip -j -o "$archive" "glove.twitter.27B.100d.txt" -d "$dir"
    fi

    python3 - "$text_file" "$base" <<'PY'
import os, struct, sys, tempfile
src, dst = sys.argv[1], sys.argv[2]
dim = 100
count = 0
tmp = dst + ".tmp"

with tempfile.TemporaryFile() as vectors:
    with open(src, "r", encoding="utf-8", errors="strict") as fin:
        for line_number, line in enumerate(fin, 1):
            fields = line.rstrip().split()
            if not fields:
                continue

            values = fields[-dim:]
            if len(values) != dim:
                raise SystemExit(
                    f"Invalid GloVe row {line_number}: expected {dim} values"
                )

            try:
                vector = [float(v) for v in values]
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid numeric value on row {line_number}"
                ) from exc

            vectors.write(struct.pack(f"<{dim}f", *vector))
            count += 1

    vectors.seek(0)

    with open(tmp, "wb") as fout:
        fout.write(struct.pack("<II", count, dim))
        while True:
            block = vectors.read(16 * 1024 * 1024)
            if not block:
                break
            fout.write(block)

os.replace(tmp, dst)
print(f"Converted GloVe: {count:,} vectors x {dim} dimensions")
PY
  fi

  verify_vector_bin "$base"
  echo "NOTE: GloVe has no official ANN query or ground-truth files."
}

download_msturing1m() {
  local dir="$OUT_ROOT/msturing1m"
  local base="$dir/msturing1m.bin"
  local query="$dir/msturing_query.bin"
  local gt="$dir/msturing_groundtruth.bin"
  local tmp="$dir/msturing1m.range.tmp"
  local base_url="https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS"

  mkdir -p "$dir"

  if [[ ! -s "$base" ]]; then
    local last_byte=400000007

    echo "Downloading the first 1M MS Turing vectors..."
    curl -fL \
      --retry 5 \
      --retry-delay 3 \
      --range "0-${last_byte}" \
      --output "$tmp" \
      "$base_url/base1b.fbin"

    python3 - "$tmp" "$base" <<'PY'
import os, struct, sys
src, dst = sys.argv[1], sys.argv[2]
target_count = 1_000_000
expected_dim = 100
expected_size = 8 + target_count * expected_dim * 4
actual_size = os.path.getsize(src)

if actual_size != expected_size:
    raise SystemExit(
        f"Expected {expected_size} bytes, received {actual_size}. "
        "The server may not support the requested byte range."
    )

with open(src, "r+b") as f:
    original_count, dim = struct.unpack("<II", f.read(8))

    if dim != expected_dim:
        raise SystemExit(
            f"Unexpected dimension: expected {expected_dim}, got {dim}"
        )

    if original_count < target_count:
        raise SystemExit(
            f"Source contains only {original_count} vectors"
        )

    f.seek(0)
    f.write(struct.pack("<II", target_count, dim))

os.replace(src, dst)
print(f"Created MS Turing subset: {target_count:,} x {expected_dim}")
PY
  fi

  download "$base_url/query100K.fbin" "$query"

  download \
    "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/msturing-gt-1M" \
    "$gt"

  verify_vector_bin "$base"
  verify_vector_bin "$query"
  verify_truthset "$gt"
}

download_one() {
  case "$1" in
    sift1m) download_sift1m ;;
    gist1m) download_gist1m ;;
    deep10m) download_deep10m ;;
    glove1.2m) download_glove1_2m ;;
    msturing1m) download_msturing1m ;;
    *) die "Unsupported dataset: $1" ;;
  esac
}

main() {
  [[ -n "$DATASET" ]] || {
    usage
    exit 1
  }

  need_command curl
  need_command python3
  need_command tar
  need_command unzip

  local normalized
  normalized="$(normalize_name "$DATASET")" || {
    usage
    die "Unknown dataset: $DATASET"
  }

  mkdir -p "$OUT_ROOT"
  OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"

  if [[ "$normalized" == "all" ]]; then
    for name in sift1m gist1m deep10m glove1.2m msturing1m; do
      echo
      echo "============================================================"
      echo "Preparing $name"
      echo "============================================================"
      download_one "$name"
    done
  else
    download_one "$normalized"
  fi

  echo
  echo "Done."
}

main "$@"