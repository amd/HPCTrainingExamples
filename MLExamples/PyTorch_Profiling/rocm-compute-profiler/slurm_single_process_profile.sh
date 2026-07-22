#!/bin/bash
#SBATCH --job-name=rpc-single-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=rpc_single_process_profile_%j.out
#SBATCH --error=rpc_single_process_profile_%j.err

# ---------------------------------------------------------------------------
# Profile the single-process CIFAR-100 workload with rocprof-compute profile.
# Sources ../setup_rocm.sh to activate the ROCm PyTorch venv.
# ---------------------------------------------------------------------------

set -e

# Resolve this script's dir; under sbatch prefer SLURM_SUBMIT_DIR. Submit from
# MLExamples/PyTorch_Profiling/rocm-compute-profiler/.
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(dirname "$(readlink -fm "$0")")"
fi
PROFILER_TOP_DIR="$(dirname "${SCRIPT_DIR}")"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PROFILER_TOP_DIR=${PROFILER_TOP_DIR}"

# ../setup_rocm.sh activates the ROCm PyTorch venv and exports ROCm env vars.
source ${PROFILER_TOP_DIR}/setup_rocm.sh
rocprof-compute --version

# Distributed bootstrap vars for train_cifar_100.py (single rank). Derive a
# per-job port so concurrent jobs don't collide.
export NPROCS=1
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}

# Pre-download the dataset so the profiled step doesn't spend time on it.
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

# Profile. Notes:
#  * --no-roof skips roofline capture (not needed here).
#  * rocprof-compute replays the app multiple times to cover all counters, so
#    keep the workload short.
echo
echo "==================================================================="
echo "rocprof-compute profile -- python3 train_cifar_100.py"
echo "==================================================================="
srun -n 1 --gpus=1 --cpus-per-task=8 \
    rocprof-compute profile \
        --name ${WORKLOAD_NAME} \
        --no-roof \
        -- \
        python3 ${PROFILER_TOP_DIR}/train_cifar_100.py \
            --batch-size 32 --max-steps 5 \
            --data-path ${PROFILER_TOP_DIR}/data

# The profile subdir is named by arch (e.g. MI300A_*) or numerically; pick the first.
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
