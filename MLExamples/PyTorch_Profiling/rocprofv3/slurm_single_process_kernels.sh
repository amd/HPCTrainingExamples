#!/bin/bash
#SBATCH --job-name=rpv3-single-kernels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=00:30:00
#SBATCH --output=rpv3_single_process_kernels_%j.out
#SBATCH --error=rpv3_single_process_kernels_%j.err

# ---------------------------------------------------------------------------
# SLURM script: profile the single-process CIFAR-100 workload with rocprofv3
# kernel tracing (--stats --kernel-trace).
#
# It sources ../setup_rocm.sh to activate the ROCm PyTorch venv and set the
# matching ROCm env vars.
# ---------------------------------------------------------------------------

set -e

# Resolve the directory of this script. Under sbatch, $0 points to a copy in
# the SLURM spool dir, so prefer SLURM_SUBMIT_DIR (the directory from which the
# job was submitted). Submit this script from
# `MLExamples/PyTorch_Profiling/rocprofv3/`.
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(dirname "$(readlink -fm "$0")")"
fi
PROFILER_TOP_DIR="$(dirname "${SCRIPT_DIR}")"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PROFILER_TOP_DIR=${PROFILER_TOP_DIR}"

# ---------------------------------------------------------------------------
# Software environment.
#
# ../setup_rocm.sh (i.e. ${PROFILER_TOP_DIR}/setup_rocm.sh) activates the ROCm
# PyTorch venv and exports the matching ROCm env vars.
# ---------------------------------------------------------------------------
source ${PROFILER_TOP_DIR}/setup_rocm.sh
rocprofv3 --version

# Distributed bootstrap variables expected by train_cifar_100.py (single rank).
export NPROCS=1
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
# Derive a per-job port so concurrent/leftover jobs don't collide on a shared
# node (a fixed port like 1234 can fail with EADDRINUSE).
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}

# Make sure the dataset is present before the profiled step.
if [ ! -d ${PROFILER_TOP_DIR}/data/cifar-100-python ]; then
    python3 ${PROFILER_TOP_DIR}/train_cifar_100.py \
        --data-path ${PROFILER_TOP_DIR}/data --download-only
fi

OUT_DIR=${SCRIPT_DIR}/single_process
rm -rf ${OUT_DIR}
cd ${SCRIPT_DIR}

# ---------------------------------------------------------------------------
# Profile kernels with rocprofv3.
# ---------------------------------------------------------------------------
echo
echo "==================================================================="
echo "rocprofv3 --stats --kernel-trace -- python3 train_cifar_100.py"
echo "==================================================================="
srun -n 1 --gpus=1 \
    rocprofv3 --stats --kernel-trace --output-format csv \
        --output-directory ${OUT_DIR} --output-file kernels \
        -- \
        python3 ${PROFILER_TOP_DIR}/train_cifar_100.py \
            --batch-size 32 --max-steps 5 \
            --data-path ${PROFILER_TOP_DIR}/data

echo
echo "==================================================================="
echo "Profile complete."
echo "Output directory: ${OUT_DIR}"
echo "==================================================================="
