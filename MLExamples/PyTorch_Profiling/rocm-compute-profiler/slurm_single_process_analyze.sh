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
# It sources ../setup_rocm.sh to activate the ROCm PyTorch venv and set the
# matching ROCm env vars.
#
# Analysis is CPU-only (it parses the counter database), so no GPU is requested.
#
# !!! IMPORTANT - WHEN TO RUN THIS SCRIPT !!!
#   1. Run this ONLY AFTER slurm_single_process_profile.sh has completed, so
#      the workload counter database exists.
#   2. Do NOT run this while any other job that uses the same venv is running.
#      This script temporarily pins numpy==1.26.4 (required by rocprof-compute
#      analyze) in the SHARED venv and restores it on exit. If a training or
#      profiling job runs concurrently, it will observe the wrong numpy version
#      and crash on import. Run all such jobs to completion first, then run
#      this analysis by itself.
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

# ---------------------------------------------------------------------------
# rocprof-compute analyze pins numpy==1.26.4, but other tools that share this
# venv (e.g. the roofline extractor's scipy) can bump numpy to 2.x, which makes
# analyze abort with a version-requirement error. Temporarily install the
# required numpy for the analysis, then restore whatever version was there
# before so the shared venv is left unchanged (restored even on failure).
# ---------------------------------------------------------------------------
REQUIRED_NUMPY=1.26.4
ORIG_NUMPY="$(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null)"

restore_numpy() {
    if [[ -n "${ORIG_NUMPY}" && "${ORIG_NUMPY}" != "${REQUIRED_NUMPY}" ]]; then
        echo "Restoring numpy==${ORIG_NUMPY}"
        pip install "numpy==${ORIG_NUMPY}"
    fi
}

if [[ "${ORIG_NUMPY}" != "${REQUIRED_NUMPY}" ]]; then
    echo "Installing numpy==${REQUIRED_NUMPY} for rocprof-compute analyze"
    pip install "numpy==${REQUIRED_NUMPY}"
    trap restore_numpy EXIT
fi

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
rocprof-compute analyze --list-stats -p ${ARCH_DIR} >& ${STATS_FILE}
echo "Stats and dispatch IDs written to ${STATS_FILE}"

echo
echo "==================================================================="
echo "rocprof-compute analyze -p ${ARCH_DIR}"
echo "==================================================================="
ANALYSIS_FILE=${SCRIPT_DIR}/analysis_${SLURM_JOB_ID}.txt
rocprof-compute analyze -p ${ARCH_DIR} > ${ANALYSIS_FILE} 2>&1
echo "Analysis written to ${ANALYSIS_FILE}"

echo
echo "-------------------------------------------------------------------"
echo "NOTE: The analysis above aggregates all kernels. To focus on a"
echo "specific kernel, inspect ${STATS_FILE} for the kernel names and"
echo "dispatch IDs, then re-run rocprof-compute analyze narrowing to that"
echo "kernel with --kernel <kernel-id>, or to a specific invocation with"
echo "--dispatch <dispatch-id> (you may pass either one, or both):"
echo "  rocprof-compute analyze -p ${ARCH_DIR} --kernel <kernel-id>"
echo "  rocprof-compute analyze -p ${ARCH_DIR} --dispatch <dispatch-id>"
echo "-------------------------------------------------------------------"
