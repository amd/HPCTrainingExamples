#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh  --  submit the whole PyTorch_Profiling suite.
#
# Run this on a LOGIN node after install_rocm_pytorch.sh has built the
# venv/module and pre-staged everything. Each script is submitted from its own
# directory (the scripts rely on SLURM_SUBMIT_DIR). The rocprof-compute analyze
# job is chained after its profile job with --dependency=afterok.
#
# Account and partition are taken from local.env and passed to sbatch through
# the SBATCH_ACCOUNT / SBATCH_PARTITION environment variables exported by env.sh.
#
# Prints one "<tool> <jobid>" line per submission.
# ---------------------------------------------------------------------------
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Works both from inside MLExamples/PyTorch_Profiling and from a separate
# working copy that keeps the scripts in a PyTorch_Profiling/ subdirectory.
if [[ -d "${SELF_DIR}/PyTorch_Profiling" ]]; then
    # shellcheck source=PyTorch_Profiling/env.sh
    source "${SELF_DIR}/PyTorch_Profiling/env.sh"
else
    # shellcheck source=env.sh
    source "${SELF_DIR}/env.sh"
fi

if [[ ! -d "${EXAMPLES_TOP}" ]]; then
    echo "ERROR: ${EXAMPLES_TOP} not found. Run install_rocm_pytorch.sh first." >&2
    exit 1
fi

submit() {
    # submit <subdir> <script> [extra sbatch args...]
    local subdir="$1"; shift
    local script="$1"; shift
    ( cd "${EXAMPLES_TOP}/${subdir}" && sbatch "$@" "${script}" ) \
        | grep -oE '[0-9]+' | tail -1
}

echo "Submitting from ${EXAMPLES_TOP}"
echo "  account=${SBATCH_ACCOUNT:-<cluster default>}  partition=${SBATCH_PARTITION:-<cluster default>}"

BASELINE_ID=$(submit no-profiling        slurm_single_process_noprofile.sh)
echo "baseline            ${BASELINE_ID}"

KERNELS_ID=$(submit rocprofv3            slurm_single_process_kernels.sh)
echo "rocprofv3-kernels   ${KERNELS_ID}"

TRACES_ID=$(submit rocprofv3             slurm_single_process_traces.sh)
echo "rocprofv3-traces    ${TRACES_ID}"

PROFILE_ID=$(submit rocm-compute-profiler slurm_single_process_profile.sh)
echo "rocm-compute-profile ${PROFILE_ID}"

# analyze must run only after the profile job succeeds.
ANALYZE_ID=$(submit rocm-compute-profiler slurm_single_process_analyze.sh \
                --dependency=afterok:${PROFILE_ID})
echo "rocm-compute-analyze ${ANALYZE_ID}  (afterok:${PROFILE_ID})"

SYSTEMS_ID=$(submit rocm-systems-profiler slurm_single_process.sh)
echo "rocm-systems         ${SYSTEMS_ID}"

ROOFLINE_ID=$(submit roofline-extractor  slurm_single_process.sh)
echo "roofline             ${ROOFLINE_ID}"

echo
echo "All submitted. Track with:  squeue -u ${USER}"
echo "Job IDs: ${BASELINE_ID} ${KERNELS_ID} ${TRACES_ID} ${PROFILE_ID} ${ANALYZE_ID} ${SYSTEMS_ID} ${ROOFLINE_ID}"
