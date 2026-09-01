# VectorTailCache

**Latency-Aware Cache Admission for Disk-Based ANN Search**

VectorTailCache is a workload-aware cache admission policy for disk-based approximate nearest-neighbor (ANN) search. It replaces DiskANN's default breadth-first-search (BFS) cache with a *Path Criticality Score (PCS)* that jointly accounts for a node's visit frequency and its average SSD miss latency, directly targeting the root cause of P999 tail latency rather than raw cache-miss count.

Built on the [DiskANN `cpp_main` branch](https://github.com/microsoft/DiskANN/tree/cpp_main).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Requirements](#requirements)
- [Build](#build)
- [Environment Setup](#environment-setup)
- [Dataset Download](#dataset-download)
- [Index Building](#index-building)
- [Reproducing the Paper's Results](#reproducing-the-papers-results)
  - [Measurement Study (Section 2, optional)](#measurement-study-section-2-optional)
  - [Main Evaluation (Section 4)](#main-evaluation-section-4)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Requirements

| Dependency | Version |
|---|---|
| OS | Ubuntu 20.04 / 22.04 |
| CMake | ≥ 3.16.3 |
| GCC | ≥ 9 |
| OpenMP | any |
| Boost | ≥ 1.71 |
| Python | ≥ 3.8 |
| Python packages | `numpy`, `pandas`, `matplotlib` |

```bash
pip install numpy pandas matplotlib --break-system-packages
```

---

## Build

```bash
git clone https://github.com/imagesid/VectorTailCache.git
cd VectorTailCache/diskann

cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target search_disk_index -j"$(nproc)"
cmake --build build --target build_disk_index -j"$(nproc)"
```

---

## Environment Setup

Set the following environment variables inside `diskann/env.sh`:

```bash
export BASE=/home/yourpath/VectorTailCache
export DISKANN_DIR=${BASE}/diskann
export DISKANN_BUILD=${DISKANN_DIR}/build
export DISKANN_SEARCH=${DISKANN_BUILD}/apps/search_disk_index

export VTC_DATA=/mnt/nvme/vtc_data
export VTC_INDEX=${BASE}/index
export VTC_RESULTS=${BASE}/results
export VTC_SCRIPTS=${BASE}/scripts
```

Load the variables into your shell before running any script below:

```bash
source /home/yourpath/VectorTailCache/diskann/env.sh
```

---

## Dataset Download

Use the provided `download_dataset.sh` script to download and convert datasets into DiskANN-compatible `.bin` files.

```bash
sudo -E bash scripts/download_dataset.sh <dataset> "$VTC_DATA"
```

Supported dataset names:

| Parameter | Dataset | Vectors | Dimensions |
|---|---|---:|---:|
| `sift1m` | SIFT1M | 1,000,000 | 128 |
| `gist1m` | GIST1M | 1,000,000 | 960 |
| `deep10m` | DEEP10M | 10,000,000 | 96 |
| `glove1.2m` | GloVe Twitter | ~1,200,000 | 100 |
| `msturing1m` | MS Turing ANNS subset | 1,000,000 | 100 |

Aliases such as `sift`, `gist`, `deep`, `glove`, and `turing` are also accepted.

Examples:

```bash
sudo -E bash scripts/download_dataset.sh sift1m "$VTC_DATA"
sudo -E bash scripts/download_dataset.sh glove1.2m "$VTC_DATA"
```

Download all supported datasets at once:

```bash
sudo -E bash scripts/download_dataset.sh all "$VTC_DATA"
```

Each dataset is stored in its own directory:

```text
$VTC_DATA/
├── sift1m/
│   └── sift1m.bin
├── gist1m/
│   └── gist1m.bin
├── deep10m/
│   └── deep10m.bin
├── glove1.2m/
│   └── glove1.2m.bin
└── msturing1m/
    └── msturing1m.bin
```

---

## Index Building

Build a Vamana disk index for a chosen dataset:

```bash
DATASET=sift1m   # one of: sift1m | gist1m | deep10m | glove1.2m | msturing1m

mkdir -p "$VTC_INDEX/$DATASET"

$DISKANN_BUILD/apps/build_disk_index \
  --data_type float \
  --dist_fn l2 \
  --data_path "$VTC_DATA/$DATASET/$DATASET.bin" \
  --index_path_prefix "$VTC_INDEX/$DATASET/disk_index" \
  -R 64 -L 100 -B 0.3 -M 1
```

---

## Reproducing the Paper's Results

### Main Evaluation (Section 4)

#### All Datasets

```bash
# 5 cold-start runs, 5 datasets
sudo -E bash scripts/run_all_datasets.sh
python3 scripts/run_all_datasets.py "$VTC_RESULTS/all_datasets_<timestamp>"
```

Produces: Figure 7.

#### SIFT1M

```bash
# 5 cold-start runs, 6 cache sizes, 3 L values
sudo -E bash scripts/compare_policies.sh
python3 scripts/compare_policies.py "$VTC_RESULTS/final_comparison_<timestamp>"
```

Produces: Figure 8.

#### Concurency Robustness

```bash
sudo -E bash scripts/multi-threads.sh
sudo python3 scripts/multithread_mean_latency.py --run-dir  "$VTC_RESULTS/extended_eval_<timestamp>"
```

```bash
sudo -E bash scripts/compare_beamwidth_concurrency.sh
sudo python3 scripts/gen_concurrency_tradeoff.py  "$VTC_RESULTS/beamwidth_concurrency_<timestamp>"
```

Produces: Figures 13 and 14.


#### Device-specific BFS vs. PCS

**SATA SSD:**

```bash
sudo -E bash scripts/sata_bfs_vs_pcs.sh
```

**HDD** (long-running — recommended inside `screen`):

```bash
screen -S hdd_exp
sudo -E bash scripts/hdd_bfs_vs_pcs_clean.sh
# Detach: Ctrl+A D   |   Reattach: screen -r hdd_exp
```

Produces: Table 1.

## License

This project is licensed under the [MIT License](LICENSE).

VectorTailCache is built on top of [DiskANN](https://github.com/microsoft/DiskANN) (`cpp_main` branch), which is also released under the MIT License, © Microsoft Corporation. See [NOTICE](NOTICE) for the full upstream attribution and license text.

---

## Acknowledgments

This work builds on [DiskANN](https://github.com/microsoft/DiskANN) by Microsoft Research. We thank the DiskANN authors and maintainers for releasing the codebase that made this work possible.