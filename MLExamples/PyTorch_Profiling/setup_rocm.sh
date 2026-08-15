#!/usr/bin/env bash
# Activate the ROCm + PyTorch environment for the PyTorch_Profiling tests:
#   source setup_rocm.sh
#
# All site-specific values come from local.env via env.sh. Where
# install_rocm_pytorch.sh has generated an Lmod modulefile, that module is
# loaded; otherwise the venv is activated directly.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${_here}/env.sh"

# Only exports http(s)_proxy on sites that define PROXY.
export_proxy

ensure_module_cmd
_modfile="${MODULEROOT}/${MODULE_NAME}/${MODULE_VERSION}.lua"
if [[ -f "${_modfile}" ]] && command -v module >/dev/null 2>&1; then
    module use "${MODULEROOT}"
    module load "${MODULE_NAME}/${MODULE_VERSION}"
    echo "setup_rocm.sh: loaded module ${MODULE_NAME}/${MODULE_VERSION}"
else
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    # Derive site-packages from the active venv so this works regardless of the
    # venv's Python minor version (e.g. python3.12 vs python3.13).
    DEVEL="$(python3 -c 'import site; print(site.getsitepackages()[0])')/_rocm_sdk_devel"
    export ROCM_PATH="$DEVEL"
    export HIP_PATH="$DEVEL"
    export HIP_DEVICE_LIB_PATH="${DEVEL}/lib/llvm/amdgcn/bitcode"
    export PATH="${DEVEL}/bin:$PATH"
    export LD_LIBRARY_PATH="${DEVEL}/lib:${DEVEL}/lib/rocm_sysdeps/lib:$LD_LIBRARY_PATH"
    echo "setup_rocm.sh: ROCm venv active: ${VENV}"
fi

# MIOpen keeps its perf/kernel databases as SQLite files, and SQLite locking is
# unreliable on NFS and Lustre (conv2d then fails with
# miopenStatusInternalError). Keep them on node-local storage.
if [[ "${MIOPEN_LOCAL_DB}" == "1" ]]; then
    _miopen_dir="${MIOPEN_TMP_BASE}/${USER}-miopen-${SLURM_JOB_ID:-$$}"
    mkdir -p "${_miopen_dir}" 2>/dev/null || true
    export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-${_miopen_dir}}"
    export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-${_miopen_dir}}"
fi

echo "setup_rocm.sh: MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-<unset>}"
echo "setup_rocm.sh: VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}  ROCM_PATH=${ROCM_PATH:-<unset>}"
