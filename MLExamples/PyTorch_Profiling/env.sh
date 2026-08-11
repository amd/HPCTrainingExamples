#!/usr/bin/env bash
# Central configuration for running the PyTorch_Profiling suite on a SLURM
# cluster. Nothing here is site-specific: every machine-dependent value comes
# from a site config file, so the same scripts run unmodified on any cluster.
#
# Site config resolution order:
#   1. $SITE_ENV, if set (lets one checkout drive several clusters)
#   2. local.env next to this file (copy local.env.example and edit)
#
# Sourced by setup_rocm.sh, install_rocm_pytorch.sh, run_all.sh and the
# rocprof-compute analyze script.

_env_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SITE_ENV="${SITE_ENV:-${_env_here}/local.env}"
if [[ -f "${SITE_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${SITE_ENV}"
else
    echo "env.sh: WARNING: no site config at ${SITE_ENV}" >&2
    echo "env.sh: copy local.env.example to local.env and edit it for this cluster." >&2
fi

# --- scheduler -------------------------------------------------------------
# SLURM charge account and partition. Left empty means "use the cluster default".
PROJECT="${PROJECT:-}"
PARTITION="${PARTITION:-}"

# sbatch reads these at submit time. The slurm_*.sh headers carry no
# --account/--partition, since #SBATCH directives cannot expand variables.
if [[ -n "${PROJECT}" ]]; then
    export SBATCH_ACCOUNT="${PROJECT}"
else
    unset SBATCH_ACCOUNT
fi
if [[ -n "${PARTITION}" ]]; then
    export SBATCH_PARTITION="${PARTITION}"
else
    unset SBATCH_PARTITION
fi

# --- storage ---------------------------------------------------------------
# Base directory holding the venvs, modulefiles and the cloned examples.
# Prefer the fastest storage that is shared by login and compute nodes.
# Defaults to $HOME, matching the upstream setup instructions.
VENV_BASE="${VENV_BASE:-${HOME}}"

# --- target hardware -------------------------------------------------------
# GPU_ARCH drives the pip wheel extras (device-<arch>); ROOFLINE_ARCH selects
# the counter set passed to the roofline extractor's --arch.
GPU_ARCH="${GPU_ARCH:-}"
ROOFLINE_ARCH="${ROOFLINE_ARCH:-}"

# --- ROCm build ------------------------------------------------------------
# Nightly build to validate; see the index below for available versions.
ROCM_VERSION="${ROCM_VERSION:-}"
ROCM_INDEX_URL="${ROCM_INDEX_URL:-https://rocm.nightlies.amd.com/whl-multi-arch/}"

# --- base python used to create the venv -----------------------------------
# PYTHON_MODULE is loaded first when set (e.g. cray-python on Cray systems);
# BASE_PYTHON overrides the interpreter outright.
PYTHON_MODULE="${PYTHON_MODULE:-}"
BASE_PYTHON="${BASE_PYTHON:-}"

# --- Lmod module exposing the venv ----------------------------------------
MODULE_NAME="${MODULE_NAME:-rocm-pytorch-pip}"
MODULE_VERSION="${MODULE_VERSION:-${ROCM_VERSION}}"
MODULEROOT="${MODULEROOT:-${VENV_BASE}/modulefiles}"

# Lmod init script to source when `module` is not already defined (common in
# non-login shells and over `ssh host command`). Empty where Lmod is preloaded.
LMOD_INIT="${LMOD_INIT:-}"

# --- derived paths ---------------------------------------------------------
VENV="${VENV:-${VENV_BASE}/venvs/rocm-pytorch-pip}"
ANALYZE_VENV="${ANALYZE_VENV:-${VENV_BASE}/venvs/rocprof-compute-analyze}"

# Two layouts are supported. When this file sits inside the examples tree (the
# normal case: MLExamples/PyTorch_Profiling), the tests run from that tree in
# place. When it is kept in a separate working copy, install_rocm_pytorch.sh
# clones the examples under VENV_BASE and overlays these scripts onto them.
if [[ -f "${_env_here}/train_cifar_100.py" ]]; then
    EXAMPLES_TOP="${EXAMPLES_TOP:-${_env_here}}"
    EXAMPLES_REPO="${EXAMPLES_REPO:-$(cd "${_env_here}/../.." 2>/dev/null && pwd)}"
else
    EXAMPLES_REPO="${EXAMPLES_REPO:-${VENV_BASE}/HPCTrainingExamples}"
    EXAMPLES_TOP="${EXAMPLES_TOP:-${EXAMPLES_REPO}/MLExamples/PyTorch_Profiling}"
fi

# --- network ---------------------------------------------------------------
# Outbound proxy for sites whose nodes have no direct internet. Empty elsewhere;
# it is only exported when set, so it never breaks direct-access clusters.
PROXY="${PROXY:-}"

# --- MIOpen ----------------------------------------------------------------
# MIOpen keeps its perf/kernel databases as SQLite files. The default location
# is $HOME/.config/miopen, and SQLite file locking fails on NFS and Lustre,
# surfacing as "RuntimeError: miopenStatusInternalError" during conv2d. Keeping
# these databases on node-local storage avoids it. Set MIOPEN_LOCAL_DB=0 on
# sites where the shared filesystem supports real POSIX locks.
MIOPEN_LOCAL_DB="${MIOPEN_LOCAL_DB:-1}"
MIOPEN_TMP_BASE="${MIOPEN_TMP_BASE:-${TMPDIR:-/tmp}}"

# Make `module` usable when the shell did not inherit Lmod's shell function.
ensure_module_cmd() {
    if ! command -v module >/dev/null 2>&1; then
        if [[ -n "${LMOD_INIT}" && -r "${LMOD_INIT}" ]]; then
            # shellcheck disable=SC1090
            source "${LMOD_INIT}"
        fi
    fi
}

# Export the proxy only when the site defines one.
export_proxy() {
    if [[ -n "${PROXY}" ]]; then
        export http_proxy="${http_proxy:-${PROXY}}"
        export https_proxy="${https_proxy:-${PROXY}}"
    fi
}
