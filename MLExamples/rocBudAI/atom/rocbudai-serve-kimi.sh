#!/usr/bin/env bash
# rocbudai-serve-kimi — bring up Kimi-K3 on the local MI355X node via ATOM
# (AMD's AiTer-Optimized, vLLM-like, OpenAI-compatible engine), serving on
# 127.0.0.1:8000 so `module load rocbudai` (ROCBUDAI_BACKEND=vllm) talks to it.
#
# Run this INSIDE your own exclusive MI355X allocation, e.g.:
#   salloc -p mi355x --exclusive --gres=gpu:amd_instinct_mi355_oam:8 -t 2:00:00
#   rocbudai-serve-kimi                 # foreground (Ctrl-C to stop)
#   rocbudai-serve-kimi &               # background, then: module load rocbudai
#
# Engine: ATOM, layered on a prebuilt vllm Lmod module (which provides torch 2.10
# + AITER kernels). Weights come from a shared, world-readable HF cache
# (downloaded once). Validated recipe: gfx950 / MI355X, TP8, MXFP4 MoE via
# online ptpc_fp8 quant (ATOM recipes/Kimi-K3.md + AMD Day-0 article).
#
# Tunables (env):
#   RBK_MODEL      (default moonshotai/Kimi-K3)
#   RBK_PORT       (default 8000)
#   RBK_TP         (default 8)
#   RBK_MAXLEN     (default 131072; Kimi-K3 supports up to 1048576)
#   RBK_GPU_UTIL   (default 0.93)
#   RBK_HF_HOME    (default /shared/$USER/hf-cache)
#   RBK_ATOM_PYLIB (default /shared/$USER/opt/atom-pylib)
#   RBK_EXTRA      (extra args appended to the server command)
set -euo pipefail

RBK_MODEL="${RBK_MODEL:-moonshotai/Kimi-K3}"
RBK_PORT="${RBK_PORT:-8000}"
# Bind on all interfaces so the agent (rocBudAI TUI) can reach this server from
# the LOGIN node — the agent must NOT run on this compute node or it contends
# for CPU with the model. Set RBK_HOST=127.0.0.1 to restrict to local-only.
RBK_HOST="${RBK_HOST:-0.0.0.0}"
RBK_TP="${RBK_TP:-8}"
RBK_MAXLEN="${RBK_MAXLEN:-131072}"   # agent prompts (persona + code) exceed 16k
RBK_GPU_UTIL="${RBK_GPU_UTIL:-0.93}"
RBK_HF_HOME="${RBK_HF_HOME:-/shared/${USER}/hf-cache}"
RBK_ATOM_PYLIB="${RBK_ATOM_PYLIB:-/shared/${USER}/opt/atom-pylib}"
# Newer AITER than the one bundled in the vllm module — ATOM needs its newer
# symbols (topk_select, etc.). Pinned release v0.1.25 built for ROCm 7.2 (matches
# the rocm/7.2.2 module), prebuilt gfx950 kernels. Prepended so it shadows the
# module's older aiter.
RBK_AITER="${RBK_AITER:-/shared/${USER}/opt/aiter-wheel}"
# Default reasoning effort (low|high|max). Kimi-K3 defaults to "max" (very long
# hidden reasoning, ~5x slower turns); "high" keeps quality with fast turnaround.
# A client's own reasoning_effort still wins per request. See README "Turnaround".
RBK_REASONING_EFFORT="${RBK_REASONING_EFFORT:-high}"
# Prefix caching (1=on, 0=off). The agent resends a large stable prefix (persona +
# addendum + history) every turn; caching its KV cuts per-turn prefill from
# seconds to ~0. ATOM supports it for Kimi-K3/KDA and defaults it on. Set 0 to disable.
RBK_PREFIX_CACHING="${RBK_PREFIX_CACHING:-1}"
RBK_EXTRA="${RBK_EXTRA:-}"

ROCM_MODULE="${ROCM_MODULE:-rocm/7.2.2}"
VLLM_MODULE="${VLLM_MODULE:-vllm/0.19.1-pt2.10.0}"

say() { echo "[serve-kimi] $*" >&2; }

# Must run on an MI355X GPU compute node, never a login node. A login node has no
# AMD GPUs, so ATOM/ROCm can't start and the run fails cryptically or appears to
# do nothing. The absence of the ROCm device nodes is the reliable signal — catch
# it here and fail loudly with the right next step.
if [[ ! -e /dev/kfd ]] || ! compgen -G "/dev/dri/renderD*" >/dev/null 2>&1; then
    say "ERROR: no AMD GPU detected on $(hostname -s) — this looks like a login node."
    say "rocbudai-serve-kimi must run INSIDE an MI355X compute allocation. Grab a node,"
    say "then run it there:"
    say "  salloc -p mi355x --exclusive --gres=gpu:amd_instinct_mi355_oam:8 --mem=0 -t 2:00:00"
    say "  $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    say "WARNING: no SLURM_JOB_ID — run this inside an MI355X allocation:"
    say "  salloc -p mi355x --exclusive --gres=gpu:amd_instinct_mi355_oam:8 --mem=0 -t 2:00:00"
fi

# Kimi-K3 needs (nearly) all of the node's RAM to load. --exclusive alone does
# NOT necessarily grant memory (on many clusters it stays capped, often 16G from a
# partition default), so a ModelRunner gets OOM-killed (exitcode -9) on load. Read the real
# allocation from scontrol and fail fast unless you used --mem=0 (whole node).
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    _tomb() { local n="${1%[KMGT]}" u="${1##*[0-9]}"
        case "$u" in T) echo $((n*1024*1024));; G) echo $((n*1024));; K) echo $((n/1024));; *) echo "$n";; esac; }
    _alloc=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oE 'AllocTRES=[^ ]+' | head -1 | grep -oE 'mem=[0-9]+[KMGT]?' | cut -d= -f2)
    _nodemem=$(scontrol show node "$(hostname -s)" 2>/dev/null | grep -oE 'RealMemory=[0-9]+' | head -1 | cut -d= -f2)
    if [[ -n "${_alloc}" && -n "${_nodemem}" ]]; then
        _allocmb=$(_tomb "${_alloc}")
        if (( _allocmb * 100 < _nodemem * 80 )); then
            say "ERROR: only $((_allocmb/1024)) GB of the node's $((_nodemem/1024)) GB RAM is allocated — Kimi-K3 will OOM on load."
            say "Re-allocate with --mem=0 (all node memory):"
            say "  salloc -p mi355x --exclusive --gres=gpu:amd_instinct_mi355_oam:8 --mem=0 -t 2:00:00"
            exit 1
        fi
    fi
fi

# Lmod may not be initialised in a non-login shell (e.g. 'rocbudai-serve-kimi &').
if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi

# torch 2.10 + AITER kernels come from the vllm module.
say "loading modules: ${ROCM_MODULE} ${VLLM_MODULE}"
module load "${ROCM_MODULE}"
module load "${VLLM_MODULE}"

# Layer the newer AITER + the ATOM engine (and its pinned deps, e.g.
# transformers 5.16.1) ahead of the module path so both win; torch still
# resolves from the module. Order matters: aiter-nightly must precede the
# module's dist-packages so its newer `aiter` shadows the bundled one.
export PYTHONPATH="${RBK_AITER}:${RBK_ATOM_PYLIB}:${PYTHONPATH:-}"

# Shared, world-readable weights cache; serve offline (no egress needed).
export HF_HOME="${RBK_HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Per-user AITER JIT cache (the module hard-codes one user's path).
export AITER_JIT_DIR="${AITER_JIT_DIR:-/tmp/${USER}/aiter-jit}"
mkdir -p "${AITER_JIT_DIR}"

# AMD-validated gfx950/MI355X backend selection for Kimi-K3 (Day-0 article).
# All overridable; these are the known-good defaults.
export ATOM_LOADER_USE_THREADPOOL="${ATOM_LOADER_USE_THREADPOOL:-1}"
export ATOM_LOADER_THREADPOOL_WORKERS="${ATOM_LOADER_THREADPOOL_WORKERS:-16}"
export ATOM_SYNC_AFTER_LOAD="${ATOM_SYNC_AFTER_LOAD:-1}"
export ATOM_DIST_TIMEOUT_SECONDS="${ATOM_DIST_TIMEOUT_SECONDS:-3600}"
export ATOM_USE_TRITON_GEMM="${ATOM_USE_TRITON_GEMM:-1}"
export AITER_USE_GROUPED_GEMM="${AITER_USE_GROUPED_GEMM:-0}"
export ATOM_USE_TRITON_MOE="${ATOM_USE_TRITON_MOE:-0}"
export AITER_FLYDSL_FORCE="${AITER_FLYDSL_FORCE:-1}"
export AITER_FORCE_GFX1250="${AITER_FORCE_GFX1250:-0}"
export ATOM_USE_UNIFIED_ATTN="${ATOM_USE_UNIFIED_ATTN:-1}"
export ATOM_FORCE_ATTN_TRITON="${ATOM_FORCE_ATTN_TRITON:-1}"
# Kimi Delta Attention: use the pure-Triton KDA kernels, NOT the experimental
# Gluon ones. The AITER wheel's Gluon kernels (flash_kda_k1/k2) target a Triton
# whose gluon API differs from the one in the vllm module's torch 2.10, so they
# fail to compile on the first forward pass with
# "'SwizzledSharedLayout' object has no attribute '_flatten_ir'". The Triton
# route is version-stable. Set AITER_FDA_USE_GLUON=1 to re-enable Gluon.
export AITER_FDA_USE_GLUON="${AITER_FDA_USE_GLUON:-0}"

# Online quant: run the MXFP4 MoE checkpoint with ptpc_fp8 (ATOM Kimi-K3 recipe).
ONLINE_QUANT='{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}'

# Apply the default effort to every request as a chat-template kwarg.
RBK_EFFORT_ARG=()
if [[ -n "${RBK_REASONING_EFFORT}" ]]; then
    RBK_EFFORT_ARG=(--default-chat-template-kwargs "{\"thinking_effort\": \"${RBK_REASONING_EFFORT}\"}")
fi

# Prefix caching toggle (ATOM's BooleanOptionalAction flag).
if [[ "${RBK_PREFIX_CACHING}" == "0" || "${RBK_PREFIX_CACHING,,}" == "false" ]]; then
    RBK_PREFIX_ARG=(--no-enable_prefix_caching)
else
    RBK_PREFIX_ARG=(--enable_prefix_caching)
fi

say "node=$(hostname)  model=${RBK_MODEL}  tp=${RBK_TP}  port=${RBK_PORT}  max_len=${RBK_MAXLEN}  reasoning=${RBK_REASONING_EFFORT:-max(default)}"
say "HF_HOME=${HF_HOME}  AITER_JIT_DIR=${AITER_JIT_DIR}"
say "engine=ATOM ($(python3 -c 'import atom,os;print(os.path.dirname(atom.__file__))' 2>/dev/null || echo "${RBK_ATOM_PYLIB}/atom"))"
say "aiter=$(python3 -c 'import aiter,os;print(os.path.dirname(aiter.__file__))' 2>/dev/null || echo "${RBK_AITER}/aiter")"
say "starting ATOM — first load of Kimi-K3 (1.56 TB) + JIT kernel build takes several minutes."
say "wait for: 'Application startup complete' / 'Uvicorn running on http://${RBK_HOST}:${RBK_PORT}'"
say "then, ON THE LOGIN NODE, point the agent here and launch it:"
say "    export ROCBUDAI_OPENAI_BASE=http://$(hostname -s):${RBK_PORT}/v1"
say "    cd <project-dir> && module load rocbudai"
echo

exec python3 -m atom.entrypoints.openai_server \
    --model "${RBK_MODEL}" \
    --served-model-name "${RBK_MODEL}" \
    --kv_cache_dtype fp8 \
    -tp "${RBK_TP}" \
    --trust-remote-code \
    --max-model-len "${RBK_MAXLEN}" \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization "${RBK_GPU_UTIL}" \
    --block-size 128 \
    "${RBK_PREFIX_ARG[@]}" \
    --host "${RBK_HOST}" \
    --port "${RBK_PORT}" \
    --online_quant_config "${ONLINE_QUANT}" \
    "${RBK_EFFORT_ARG[@]}" \
    ${RBK_EXTRA}
