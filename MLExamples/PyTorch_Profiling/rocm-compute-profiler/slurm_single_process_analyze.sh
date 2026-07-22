#!/bin/bash
#SBATCH --job-name=rpc-single-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=rpc_single_process_analyze_%j.out
#SBATCH --error=rpc_single_process_analyze_%j.err

# ---------------------------------------------------------------------------
# Analyze the profiled workload with rocprof-compute analyze.
#
# rocprof-compute analyze needs numpy 1.26.x, which conflicts with the shared
# venv's numpy 2.x. So we use a separate venv holding only rocprof-compute's
# pinned requirements, while reusing the tool bundled in the shared venv (the
# shared venv is never modified). CPU-only; run after the profile job finishes.
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

# Locate the shared ROCm venv WITHOUT activating it; we only need the ROCm
# install inside it (bundled rocprof-compute launcher + ROCm libraries).
MAIN_VENV="${HOME}/venvs/rocm-pytorch-pip"
if [[ ! -x "${MAIN_VENV}/bin/python3" ]]; then
    echo "ERROR: shared ROCm venv not found at ${MAIN_VENV}" >&2
    exit 1
fi
# site-packages of the shared venv (avoids hard-coding the python3.x version).
MAIN_SP="$("${MAIN_VENV}/bin/python3" -c 'import site; print(site.getsitepackages()[0])')"
ROCM_PROFILER="${MAIN_SP}/_rocm_profiler"
ROCM_CORE="${MAIN_SP}/_rocm_sdk_core"
ROCM_DEVEL="${MAIN_SP}/_rocm_sdk_devel"
COMPUTE_LIBEXEC="${ROCM_PROFILER}/libexec/rocprofiler-compute"
REQ_FILE="${COMPUTE_LIBEXEC}/requirements.txt"
RPC="${ROCM_PROFILER}/bin/rocprof-compute"

for p in "${RPC}" "${REQ_FILE}" "${ROCM_CORE}" "${ROCM_DEVEL}"; do
    if [[ ! -e "${p}" ]]; then
        echo "ERROR: expected ROCm component not found: ${p}" >&2
        exit 1
    fi
done

# Create/refresh the dedicated analysis venv with rocprof-compute's pinned
# requirements (numpy 1.26.x). Reinstalled only when requirements.txt changes.
ANALYZE_VENV="${HOME}/venvs/rocprof-compute-analyze"
if [[ ! -x "${ANALYZE_VENV}/bin/python3" ]]; then
    echo "Creating analysis venv at ${ANALYZE_VENV}"
    # A venv stays isolated from its creator's site-packages, so using the
    # shared venv's python3 keeps this independent while avoiding system python3.
    "${MAIN_VENV}/bin/python3" -m venv "${ANALYZE_VENV}"
fi
source "${ANALYZE_VENV}/bin/activate"

STAMP="${ANALYZE_VENV}/.rocprof_compute_reqs_installed"
if [[ ! -f "${STAMP}" || "${REQ_FILE}" -nt "${STAMP}" ]]; then
    echo "Installing rocprof-compute requirements into ${ANALYZE_VENV}"
    python3 -m pip install --upgrade pip
    python3 -m pip install -r "${REQ_FILE}"
    cp "${REQ_FILE}" "${STAMP}"
else
    echo "Analysis venv already satisfies ${REQ_FILE}"
fi

# Point the isolated venv at the ROCm install in the shared venv (bundled
# rocprof-compute + libs). ROCm bin goes LAST on PATH so it can't shadow the
# analysis venv's python3, keeping numpy 1.26.x in effect.
export PYTHONPATH="${COMPUTE_LIBEXEC}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${ROCM_CORE}/lib:${ROCM_CORE}/lib/rocm_sysdeps/lib:${ROCM_DEVEL}/lib:${ROCM_DEVEL}/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH}"
export ROCM_PATH="${ROCM_DEVEL}"
export HIP_PATH="${ROCM_DEVEL}"
export PATH="${PATH}:${ROCM_DEVEL}/bin"

echo "Analysis python : $(which python3)  (numpy $(python3 -c 'import numpy; print(numpy.__version__)'))"
"${RPC}" --version

# Locate the workload written by slurm_single_process_profile.sh (must match
# WORKLOAD_NAME / WORK_ROOT there).
WORKLOAD_NAME=cifar_100_single_proc
WORK_ROOT=${SCRIPT_DIR}/workloads
WORK_DIR=${WORK_ROOT}/${WORKLOAD_NAME}

if [[ ! -d "${WORK_DIR}" ]]; then
    echo "ERROR: workload directory not found: ${WORK_DIR}" >&2
    echo "Run slurm_single_process_profile.sh first." >&2
    exit 1
fi

# The profile is written into a single subdirectory named after the architecture
# (e.g. `MI300A_*`) or numerically (`0`, `1`, ...). Pick the first one.
ARCH_DIR=$(find ${WORK_DIR} -mindepth 1 -maxdepth 1 -type d | sort | head -1)
if [[ -z "${ARCH_DIR}" ]]; then
    echo "ERROR: no workload subdirectory found under ${WORK_DIR}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Analyze.
# ---------------------------------------------------------------------------
# Optional: list per-kernel stats and dispatch IDs. Uncomment only if you want
# to narrow the analysis to a specific kernel/dispatch (see the NOTE at the
# bottom of this file).
# echo
# echo "==================================================================="
# echo "rocprof-compute analyze --list-stats -p ${ARCH_DIR}"
# echo "==================================================================="
# STATS_FILE=${SCRIPT_DIR}/stats_${SLURM_JOB_ID}.txt
# "${RPC}" analyze --list-stats -p ${ARCH_DIR} >& ${STATS_FILE}
# echo "Stats and dispatch IDs written to ${STATS_FILE}"

echo
echo "==================================================================="
echo "rocprof-compute analyze -p ${ARCH_DIR}"
echo "==================================================================="
ANALYSIS_FILE=${SCRIPT_DIR}/analysis_${SLURM_JOB_ID}.txt
"${RPC}" analyze -p ${ARCH_DIR} > ${ANALYSIS_FILE} 2>&1
echo "Analysis written to ${ANALYSIS_FILE}"

echo
echo "-------------------------------------------------------------------"
echo "NOTE: The analysis above aggregates all kernels. To focus on a"
echo "specific kernel, uncomment the --list-stats block above to list the"
echo "kernel names and dispatch IDs, then re-run rocprof-compute analyze"
echo "narrowing to that kernel with --kernel <kernel-id>, or to a specific"
echo "invocation with --dispatch <dispatch-id> (you may pass either, or both):"
echo "  ${RPC} analyze -p ${ARCH_DIR} --kernel <kernel-id>"
echo "  ${RPC} analyze -p ${ARCH_DIR} --dispatch <dispatch-id>"
echo "-------------------------------------------------------------------"
