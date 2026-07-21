#!/bin/bash
#SBATCH --job-name=rpc-single-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=00:15:00
#SBATCH --output=rpc_single_process_analyze_%j.out
#SBATCH --error=rpc_single_process_analyze_%j.err

# ---------------------------------------------------------------------------
# SLURM script: analyze the workload produced by slurm_single_process_profile.sh
# with rocprofiler-compute (rocprof-compute analyze).
#
# Instead of downgrading numpy inside the shared ROCm PyTorch venv (and
# restoring it afterwards), this uses a SEPARATE, isolated virtual environment
# that contains only rocprof-compute's own pinned requirements (numpy==1.26.4,
# pandas, dash, textual, ...). The shared venv is never modified, so this can
# run safely even while training/profiling jobs use the shared venv.
#
# rocprof-compute itself is NOT reinstalled into the analysis venv: the bundled
# tool that ships inside the shared venv (the `_rocm_profiler` package) is
# reused by pointing PATH / PYTHONPATH / LD_LIBRARY_PATH at it. Only the pure
# Python analysis dependencies live in the dedicated venv.
#
# Analysis is CPU-only (it parses the counter database), so no GPU is requested.
# Run this after slurm_single_process_profile.sh has completed.
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
# Locate the shared ROCm PyTorch venv (built via ROCM_PYTORCH_PIP_VENV_SETUP.md
# and activated by ../setup_rocm.sh) WITHOUT activating it. We only need the
# ROCm install that lives inside it: the bundled rocprof-compute launcher and
# the ROCm shared libraries.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Create / refresh the dedicated analysis venv with just rocprof-compute's
# pinned requirements (numpy==1.26.4 etc.). This is isolated from the shared
# venv (no --system-site-packages), so it can never be shadowed by the shared
# venv's numpy 2.x. Re-installation is skipped unless requirements.txt changed.
# ---------------------------------------------------------------------------
ANALYZE_VENV="${HOME}/venvs/rocprof-compute-analyze"
if [[ ! -x "${ANALYZE_VENV}/bin/python3" ]]; then
    echo "Creating analysis venv at ${ANALYZE_VENV}"
    /usr/bin/python3 -m venv "${ANALYZE_VENV}"
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

# ---------------------------------------------------------------------------
# Point the isolated venv at the ROCm install in the shared venv so the bundled
# rocprof-compute launcher and its shared libraries are found.
#   * PYTHONPATH  -> the rocprofiler-compute python sources (libexec).
#   * LD_LIBRARY_PATH -> ROCm core/devel libs (+ bundled sysdeps).
#   * PATH        -> append the ROCm devel bin LAST so it can never shadow the
#                    analysis venv's `python3` (isolation must be preserved).
# The bundled launcher (${RPC}) is invoked by absolute path and re-execs with
# `#!/usr/bin/env python3`, which resolves to the analysis venv's python since
# its bin dir is first on PATH after activation -> numpy==1.26.4 is used.
# ---------------------------------------------------------------------------
export PYTHONPATH="${COMPUTE_LIBEXEC}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${ROCM_CORE}/lib:${ROCM_CORE}/lib/rocm_sysdeps/lib:${ROCM_DEVEL}/lib:${ROCM_DEVEL}/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH}"
export ROCM_PATH="${ROCM_DEVEL}"
export HIP_PATH="${ROCM_DEVEL}"
export PATH="${PATH}:${ROCM_DEVEL}/bin"

echo "Analysis python : $(which python3)  (numpy $(python3 -c 'import numpy; print(numpy.__version__)'))"
"${RPC}" --version

# ---------------------------------------------------------------------------
# Locate the workload written by slurm_single_process_profile.sh. This must
# match WORKLOAD_NAME / WORK_ROOT in that script.
# ---------------------------------------------------------------------------
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
echo
echo "==================================================================="
echo "rocprof-compute analyze --list-stats -p ${ARCH_DIR}"
echo "==================================================================="
STATS_FILE=${SCRIPT_DIR}/stats_${SLURM_JOB_ID}.txt
"${RPC}" analyze --list-stats -p ${ARCH_DIR} >& ${STATS_FILE}
echo "Stats and dispatch IDs written to ${STATS_FILE}"

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
echo "specific kernel, inspect ${STATS_FILE} for the kernel names and"
echo "dispatch IDs, then re-run rocprof-compute analyze narrowing to that"
echo "kernel with --kernel <kernel-id>, or to a specific invocation with"
echo "--dispatch <dispatch-id> (you may pass either one, or both):"
echo "  ${RPC} analyze -p ${ARCH_DIR} --kernel <kernel-id>"
echo "  ${RPC} analyze -p ${ARCH_DIR} --dispatch <dispatch-id>"
echo "-------------------------------------------------------------------"
