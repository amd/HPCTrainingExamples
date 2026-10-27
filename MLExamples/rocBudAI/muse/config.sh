#!/usr/bin/env bash
# Shared settings for the rocBudAI-on-Muse-Spark-Open (vLLM) package. Source first:
#   source ./config.sh
# Every value is overridable: export it before sourcing to change it.

# Model / hardware / cluster modules.
# Agent-facing name opencode sends (--served-model-name); MODEL_HF is served.
export SERVED_NAME="${SERVED_NAME:-musespark}"
# HF checkpoint to download + serve (AMD-default Quark MXFP4 export). FP8 variant:
# amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8; BF16: meta-models/Muse-Spark-Open-816B-A42B.
export MODEL_HF="${MODEL_HF:-amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4}"
# DSpark draft model for speculative decoding.
export DSPARK_MODEL="${DSPARK_MODEL:-Inferact/Muse-Spark-Open-DSpark}"

export ROCM_MODULE="${ROCM_MODULE:-rocm/7.2.2}"
export VLLM_MODULE="${VLLM_MODULE:-vllm/0.32.1-pt2.14.1}"      # muse-capable vLLM, built from source (2-build-engine.sh)
export SLURM_PARTITION="${SLURM_PARTITION:-mi355x}"
export GPU_GRES="${GPU_GRES:-gpu:amd_instinct_mi355_oam:8}"    # 8 GPUs = TP8

# AITER ref built from source into the vLLM module (gfx950 MXFP4 MoE backend,
# ROCm/aiter PR #3470 + MI355X tables PR #2616); vllm_setup.sh pins it via --aiter-ref.
export AITER_RELEASE_TAG="${AITER_RELEASE_TAG:-v0.1.25}"

# ── Build the engine from source (2-build-engine.sh) ──────────────────────────
# vLLM (+ AITER) is built from source at the merge commit with HPCTrainingDock's
# vllm_setup.sh (no container), against a `pytorch` Lmod module, emitting the
# `vllm/<token>` module (VLLM_MODULE above) the serve command loads.
export GFX_ARCH="${GFX_ARCH:-gfx950}"                           # compile target (MI355X = gfx950)
# pytorch Lmod module vLLM links against; MUST provide torch 2.14 (build pins it).
# Build one first (HPCTrainingDock pytorch_setup.sh) if your site lacks it.
export PYTORCH_MODULE="${PYTORCH_MODULE:-pytorch/2.14.1}"
# vLLM ref to build: the Muse-Spark-Open merge commit (PR #61879); branch/tag/SHA.
export VLLM_BUILD_REF="${VLLM_BUILD_REF:-0f6b26191064384aa35060c87f4ed570c7875e66}"
export VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
# Version label for the build. Required: vllm_setup.sh's torch->vLLM auto-derive
# gate tops out below torch 2.14. Also labels the module (vllm/<ver>-pt<ptver>)
# and seeds setuptools-scm; not a git ref.
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.32.1}"
# AMD builder: ROCm compile workarounds + AITER-from-source + pytorch ABI pinning.
export VLLM_SETUP_URL="${VLLM_SETUP_URL:-https://raw.githubusercontent.com/amd/HPCTrainingDock/main/extras/scripts/vllm_setup.sh}"

# Shared, world-readable install reused by everyone. One root; the rest derive
# from it. Point SHARE_ROOT at your own published install to use a different one.
export SHARE_ROOT="${SHARE_ROOT:-/shared/${USER}}"

# Serve command. Prefer a copy bundled next to this config (portable kit for
# another system); otherwise use the one blessed command from the shared install.
_cfg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "${_cfg_dir}/rocbudai-serve-muse.sh" ]]; then
    export SERVE="${SERVE:-${_cfg_dir}/rocbudai-serve-muse.sh}"
else
    export SERVE="${SERVE:-${SHARE_ROOT}/opt/rocbudai/muse/bin/rocbudai-serve-muse}"
fi
unset _cfg_dir
export HF_HOME="${HF_HOME:-${SHARE_ROOT}/hf-cache}"           # weights cache (~510 GB MXFP4)
# Lmod root the serve command `module use`s; builder drops vllm/<token> under it.
export VLLM_MODULEPATH="${VLLM_MODULEPATH:-${SHARE_ROOT}/modulefiles}"
export VLLM_BUILD_PREFIX="${VLLM_BUILD_PREFIX:-${SHARE_ROOT}/opt}"  # source-build outputs

# Your per-user work dir (logs, and your own build if you reproduce). On /shared
# so compute nodes can read it. Always keyed to $USER, never hard-coded.
export BASE="${BASE:-/shared/${USER}/rocbudai-muse}"
export LOGDIR="${LOGDIR:-${BASE}/logs}"
# NB: sourcing this file only sets variables. The scripts that write create
# ${LOGDIR} themselves, so merely inspecting config is side-effect free.
