#!/bin/bash
#SBATCH --job-name=roofline-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=02:00:00
#SBATCH --output=roofline_single_process_%j.out
#SBATCH --error=roofline_single_process_%j.err

# ---------------------------------------------------------------------------
# SLURM script: collect roofline plots for the single-process CIFAR-100
# workload with AMD-HPC's rooflineExtractor (https://github.com/AMD-HPC/rooflineExtractor).
#
# It sources ../setup_rocm.sh to activate the ROCm PyTorch venv and set the
# matching ROCm env vars.
#
# rooflineExtractor's profile_app.py automates the whole flow: it runs rocprofv3
# several times to collect hardware counters and a kernel trace, post-processes
# them, then runs rooflineExtractor.py to produce the per-kernel roofline
# analysis and an interactive HTML plot.
# ---------------------------------------------------------------------------

set -e

# Resolve the directory of this script. Under sbatch, $0 points to a copy in
# the SLURM spool dir, so prefer SLURM_SUBMIT_DIR (the directory from which the
# job was submitted). Submit this script from
# `MLExamples/PyTorch_Profiling/roofline-extractor/`.
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

# ---------------------------------------------------------------------------
# Fetch rooflineExtractor and install its Python dependencies (into the venv).
# ---------------------------------------------------------------------------
RE_DIR=${SCRIPT_DIR}/rooflineExtractor
if [ ! -d ${RE_DIR} ]; then
    git clone https://github.com/AMD-HPC/rooflineExtractor.git ${RE_DIR}
fi
python3 -m pip install -r ${RE_DIR}/requirements.txt

# Distributed bootstrap variables expected by train_cifar_100.py (single rank).
export NPROCS=1
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
# Derive a per-job port so concurrent/leftover jobs don't collide on a shared
# node (a fixed port like 1234 can fail with EADDRINUSE).
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}

# Make sure the dataset is present before the profiled runs (profile_app.py runs
# the application several times, so pre-downloading avoids repeated downloads).
if [ ! -d ${PROFILER_TOP_DIR}/data/cifar-100-python ]; then
    python3 ${PROFILER_TOP_DIR}/train_cifar_100.py \
        --data-path ${PROFILER_TOP_DIR}/data --download-only
fi

OUT_DIR=${SCRIPT_DIR}/output
rm -rf ${OUT_DIR}
cd ${SCRIPT_DIR}

# ---------------------------------------------------------------------------
# Collect roofline data and generate plots.
#
# --arch MI300A selects the matching counter set (gfx942). profile_app.py runs
# rocprofv3 itself, so it is the application target (not launched under another
# profiler).
# ---------------------------------------------------------------------------
echo
echo "==================================================================="
echo "profile_app.py --arch MI300A -- python3 train_cifar_100.py"
echo "==================================================================="
srun -n 1 --gpus=1 \
    python3 ${RE_DIR}/profile_app.py \
        --arch MI300A \
        -o ${OUT_DIR} \
        -- \
        python3 ${PROFILER_TOP_DIR}/train_cifar_100.py \
            --batch-size 32 --max-steps 5 \
            --data-path ${PROFILER_TOP_DIR}/data

echo
echo "==================================================================="
echo "Roofline analysis complete."
echo "Output directory (counters, traces, plots, analysis): ${OUT_DIR}"
echo "Open the generated .html file for the interactive roofline plot."
echo "==================================================================="
