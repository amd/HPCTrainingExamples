#!/bin/bash
#SBATCH --job-name=roofline-single
# Charge account and partition come from SBATCH_ACCOUNT / SBATCH_PARTITION,
# which env.sh exports from local.env (#SBATCH lines cannot expand variables).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=roofline_single_process_%j.out
#SBATCH --error=roofline_single_process_%j.err

# ---------------------------------------------------------------------------
# Collect roofline plots for the single-process CIFAR-100 workload with
# AMD-HPC's rooflineExtractor (https://github.com/AMD-HPC/rooflineExtractor).
# Its profile_app.py runs rocprofv3 several times to collect counters + a kernel
# trace, then produces the per-kernel roofline analysis and an HTML plot.
# Sources ../setup_rocm.sh to activate the ROCm venv.
#
# Notes:
#  * --arch comes from ROOFLINE_ARCH in local.env and selects the counter set
#    matching this cluster's GPU (e.g. MI250X for gfx90a, MI300A for gfx942).
#  * rooflineExtractor and its pip deps are PRE-STAGED by install_rocm_pytorch.sh
#    on a login node, for sites whose compute nodes have no direct internet.
# ---------------------------------------------------------------------------

set -e

# Resolve this script's dir; under sbatch prefer SLURM_SUBMIT_DIR. Submit from
# MLExamples/PyTorch_Profiling/roofline-extractor/.
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(dirname "$(readlink -fm "$0")")"
fi
PROFILER_TOP_DIR="$(dirname "${SCRIPT_DIR}")"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PROFILER_TOP_DIR=${PROFILER_TOP_DIR}"

# ../setup_rocm.sh activates the ROCm PyTorch venv and exports ROCm env vars
# (and, via env.sh, sets ROOFLINE_ARCH from local.env).
source ${PROFILER_TOP_DIR}/setup_rocm.sh
rocprofv3 --version

if [[ -z "${ROOFLINE_ARCH}" ]]; then
    echo "ERROR: ROOFLINE_ARCH is not set; define it in local.env." >&2
    exit 1
fi

# Use the copy pre-staged by install_rocm_pytorch.sh on a login node; fall back
# to cloning here, which requires internet on the compute node.
RE_DIR=${SCRIPT_DIR}/rooflineExtractor
if [ ! -d ${RE_DIR} ]; then
    echo "rooflineExtractor not pre-staged; cloning (requires internet on this node)"
    git clone https://github.com/AMD-HPC/rooflineExtractor.git ${RE_DIR}
    python3 -m pip install -r ${RE_DIR}/requirements.txt
fi

# Distributed bootstrap variables expected by train_cifar_100.py (single rank).
export NPROCS=1
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
# Derive a per-job port so concurrent jobs don't collide.
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

# Collect roofline data and generate plots. --arch selects this cluster's
# counter set; profile_app.py runs rocprofv3 itself, so it is the target.
echo
echo "==================================================================="
echo "profile_app.py --arch ${ROOFLINE_ARCH} -- python3 train_cifar_100.py"
echo "==================================================================="
srun -n 1 --gpus=1 --cpus-per-task=8 \
    python3 ${RE_DIR}/profile_app.py \
        --arch ${ROOFLINE_ARCH} \
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
