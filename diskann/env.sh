# Source before running experiments: 
#   source diskann/env.sh

export BASE=/home/sslab/VectorTailCache
export DISKANN_DIR=${BASE}/diskann
export DISKANN_BUILD=${DISKANN_DIR}/build
export DISKANN_SEARCH=${DISKANN_BUILD}/apps/search_disk_index

export VTC_DATA=/mnt/nvme/vtc_data
export VTC_INDEX=${BASE}/index
export VTC_RESULTS=${BASE}/results
export VTC_SCRIPTS=${BASE}/scripts

# Keep the currently active Conda or virtual environment.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export VIRTUAL_ENV="$CONDA_PREFIX"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    export VIRTUAL_ENV="$VIRTUAL_ENV"
elif [[ -d "${DISKANN_DIR}/venv" ]]; then
    export VIRTUAL_ENV="${DISKANN_DIR}/venv"
else
    unset VIRTUAL_ENV
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    export PATH="${VIRTUAL_ENV}/bin:${DISKANN_BUILD}/apps:${PATH}"
else
    export PATH="${DISKANN_BUILD}/apps:${PATH}"
fi

export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

echo "VectorTailCache env loaded."
echo "  SEARCH : $DISKANN_SEARCH"
echo "  DATA   : $VTC_DATA"
echo "  INDEX  : $VTC_INDEX"
echo "  PYTHON : $(which python3)"
echo "  VIRTUAL_ENV : ${VIRTUAL_ENV:-not set}"
