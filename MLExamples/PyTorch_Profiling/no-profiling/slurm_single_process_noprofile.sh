#!/bin/bash
#SBATCH --job-name=rpc-single-noprofile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=rpc_single_process_noprofile_%j.out
#SBATCH --error=rpc_single_process_noprofile_%j.err

# ---------------------------------------------------------------------------
# Run the single-process CIFAR-100 workload directly with python3 (no profiler).
# Sources ../setup_rocm.sh to activate the ROCm PyTorch venv.
# ---------------------------------------------------------------------------

set -e

# Resolve this script's dir; under sbatch prefer SLURM_SUBMIT_DIR. Submit from
# MLExamples/PyTorch_Profiling/no-profiling/.
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

# Distributed bootstrap variables expected by train_cifar_100.py (single rank).
export NPROCS=1
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
# Derive a per-job port so concurrent jobs don't collide.
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}

# Make sure the dataset is present.
if [ ! -d ${PROFILER_TOP_DIR}/data/cifar-100-python ]; then
    python3 ${PROFILER_TOP_DIR}/train_cifar_100.py \
        --data-path ${PROFILER_TOP_DIR}/data --download-only
fi

cd ${SCRIPT_DIR}

# Run the workload directly (no profiler).
echo
echo "==================================================================="
echo "Running (no profiler): python3 train_cifar_100.py"
echo "==================================================================="
srun -n 1 --gpus=1 --cpus-per-task=8 \
    python3 ${PROFILER_TOP_DIR}/train_cifar_100.py \
        --batch-size 32 --max-steps 5 \
        --data-path ${PROFILER_TOP_DIR}/data

echo
echo "==================================================================="
echo "Workload run complete."
echo "==================================================================="
