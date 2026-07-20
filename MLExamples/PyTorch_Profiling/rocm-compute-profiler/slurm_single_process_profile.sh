#!/bin/bash
#SBATCH --job-name=rpc-single-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=02:00:00
# Reserve the whole node so no other job shares the GPUs during profiling.
#SBATCH --exclusive
#SBATCH --output=rpc_single_process_profile_%j.out
#SBATCH --error=rpc_single_process_profile_%j.err

# ---------------------------------------------------------------------------
# SLURM script: profile the single-process CIFAR-100 workload with
# rocprofiler-compute (rocprof-compute profile).
#
# It sources ../setup_rocm.sh to activate the ROCm PyTorch venv and set the
# matching ROCm env vars.
# ---------------------------------------------------------------------------

set -e

# Resolve the directory of this script. Under sbatch, $0 points to a copy in
# the SLURM spool dir, so prefer SLURM_SUBMIT_DIR (the directory from which the
# job was submitted). Submit this script from
# `MLExamples/PyTorch_Profiling/rocm-compute-profiler/`.
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
rocprof-compute --version

# Distributed bootstrap variables expected by train_cifar_100.py (single rank).
export NPROCS=1
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
# Derive a per-job port so concurrent/leftover jobs don't collide on a shared
# node (a fixed port like 1234 can fail with EADDRINUSE).
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}

# Make sure the dataset is present (the wrapper also does this, but doing it
# here keeps the profiled step from spending time on the download).
if [ ! -d ${PROFILER_TOP_DIR}/data/cifar-100-python ]; then
    python3 ${PROFILER_TOP_DIR}/train_cifar_100.py \
        --data-path ${PROFILER_TOP_DIR}/data --download-only
fi

WORKLOAD_NAME=cifar_100_single_proc
WORK_ROOT=${SCRIPT_DIR}/workloads
WORK_DIR=${WORK_ROOT}/${WORKLOAD_NAME}

mkdir -p ${WORK_ROOT}
rm -rf ${WORK_DIR}
cd ${SCRIPT_DIR}

# ---------------------------------------------------------------------------
# Profile.
#
# Notes:
#  * --no-roof skips roofline capture (faster; not needed for this demo).
#  * rocprof-compute replays the application multiple times to cover all
#    hardware counters, so keep the workload short.
#  * We profile python3 directly (not the no-profiling/*.sh bash wrapper).
#    rocprof-compute 3.7.0 / rocprofiler-sdk v1.3.1 injects into the bash
#    wrapper process too and aborts it with "Output path is empty", so the
#    application target must be the python interpreter itself. The environment
#    (modules) is already configured above and is inherited by the profiled
#    process.
# ---------------------------------------------------------------------------
echo
echo "==================================================================="
echo "rocprof-compute profile -- python3 train_cifar_100.py"
echo "==================================================================="
# With --exclusive the whole node is ours; give the single task every core on
# the node (otherwise srun would pin it to just one CPU).
srun -n 1 --gpus=1 --cpus-per-task=${SLURM_CPUS_ON_NODE} \
    rocprof-compute profile \
        --name ${WORKLOAD_NAME} \
        --no-roof \
        -- \
        python3 ${PROFILER_TOP_DIR}/train_cifar_100.py \
            --batch-size 32 --max-steps 5 \
            --data-path ${PROFILER_TOP_DIR}/data

# Resolve the workload subdirectory written by rocprof-compute. Depending on
# the version this is named after the architecture (e.g. `MI300A_*`) or
# numerically (`0`, `1`, ...). Pick the first (typically only) subdirectory.
ARCH_DIR=$(find ${WORK_DIR} -mindepth 1 -maxdepth 1 -type d | sort | head -1)
if [[ -z "${ARCH_DIR}" ]]; then
    echo "ERROR: no workload subdirectory found under ${WORK_DIR}" >&2
    exit 1
fi

echo
echo "==================================================================="
echo "Profile complete."
echo "Workload directory: ${ARCH_DIR}"
echo
echo "Analyze with, e.g.:"
echo "  rocprof-compute analyze -p ${ARCH_DIR}"
echo "==================================================================="
