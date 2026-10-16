#!/usr/bin/env bash
# Shared settings for the rocBudAI-on-Kimi-K3 (ATOM) package. Source first:
#   source ./config.sh
# Every value is overridable: export it before sourcing to change it.

# Model / hardware / cluster modules.
export MODEL="${MODEL:-moonshotai/Kimi-K3}"
export PYTAG="${PYTAG:-cp312}"                                 # python ABI tag (3.12 -> cp312)
export ROCM_MODULE="${ROCM_MODULE:-rocm/7.2.2}"
export VLLM_MODULE="${VLLM_MODULE:-vllm/0.19.1-pt2.10.0}"
export SLURM_PARTITION="${SLURM_PARTITION:-mi355x}"
export GPU_GRES="${GPU_GRES:-gpu:amd_instinct_mi355_oam:8}"    # 8 GPUs = TP8

# ATOM engine, pinned to the validated commit. main moves and its Triton-facing
# kernels drift, so an unpinned clone is not reproducible. This commit is the one
# this recipe was validated against (patches/ target it).
export ATOM_REPO="${ATOM_REPO:-https://github.com/ROCm/ATOM.git}"
export ATOM_COMMIT="${ATOM_COMMIT:-67c54da4e}"

# AITER release (has topk_select; wheel matched to the ROCm module). The nightly
# manifest is the unpinned fallback the ATOM CI uses.
export AITER_RELEASE_TAG="${AITER_RELEASE_TAG:-v0.1.25}"
export AITER_ROCM="${AITER_ROCM:-rocm7.2}"
export AITER_MANIFEST="${AITER_MANIFEST:-https://rocm.frameworks-nightlies.amd.com/whl-staging/gfx942-gfx950/main/latest.json}"

# Shared, world-readable install reused by everyone. One root; the rest derive
# from it. Point SHARE_ROOT at your own published install to use a different one.
export SHARE_ROOT="${SHARE_ROOT:-/shared/gcapodag}"

# Serve command. Prefer a copy bundled next to this config (portable kit for
# another system); otherwise use the one blessed command from the shared install.
_cfg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "${_cfg_dir}/rocbudai-serve-kimi.sh" ]]; then
    export SERVE="${SERVE:-${_cfg_dir}/rocbudai-serve-kimi.sh}"
else
    export SERVE="${SERVE:-${SHARE_ROOT}/opt/rocbudai/bin/rocbudai-serve-kimi}"
fi
unset _cfg_dir
export ATOM_PYLIB="${ATOM_PYLIB:-${SHARE_ROOT}/opt/atom-pylib}"
export AITER_DIR="${AITER_DIR:-${SHARE_ROOT}/opt/aiter-wheel}"
export HF_HOME="${HF_HOME:-${SHARE_ROOT}/hf-cache}"           # weights cache (1.5 TB)

# Your per-user work dir (logs, and your own build if you reproduce). On /shared
# so compute nodes can read it. Always keyed to $USER, never hard-coded.
export BASE="${BASE:-/shared/${USER}/rocbudai-atom}"
export LOGDIR="${LOGDIR:-${BASE}/logs}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${BASE}/.uv-cache}"      # same FS as BASE -> uv hardlinks
# NB: sourcing this file only sets variables. The scripts that write create
# ${LOGDIR}/${UV_CACHE_DIR} themselves, so merely inspecting config is side-effect free.
