#!/usr/bin/env bash
# rocbudai-serve-muse — bring up Muse-Spark-Open on the local MI355X node via
# mainline vLLM (OpenAI-compatible), serving on 0.0.0.0:8000 so
# `module load rocbudai` (ROCBUDAI_BACKEND=vllm) talks to it.
#
# Run this INSIDE your own exclusive MI355X allocation, e.g.:
#   salloc -p mi355x --exclusive --gres=gpu:amd_instinct_mi355_oam:8 --mem=0 -t 2:00:00
#   rocbudai-serve-muse                 # foreground (Ctrl-C to stop)
#   rocbudai-serve-muse &               # background, then: module load rocbudai
#
# Engine: a muse-capable vLLM built FROM SOURCE and exposed as an Lmod module
# (2-build-engine.sh; VLLM_MODULE), with the AMD AITER kernel library for
# attention + MXFP4 MoE. No container. Weights come from a shared, world-readable
# HF cache (downloaded once). Validated recipe: gfx950 / MI355X, TP8, MXFP4 MoE +
# AITER FA, optional DSpark speculative decoding (AMD Day-0 article + vLLM recipe).
#
# Tunables (env):
#   RBK_MODEL         HF checkpoint to serve (default amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4)
#   RBK_SERVED_NAME   agent-facing model id (default musespark)
#   RBK_PORT          (default 8000)
#   RBK_TP            tensor-parallel degree (default 8 = whole node)
#   RBK_EP            expert parallel: 1 for throughput >32 concurrent, 0 for lowest latency (default 0)
#   RBK_DSPARK        DSpark speculative decoding 1/0 (default 1)
#   RBK_DSPARK_MODEL  draft model (default Inferact/Muse-Spark-Open-DSpark)
#   RBK_MAXLEN        --max-model-len (default 131072; AMD recipe default is 32768, full ctx is 1048576)
#   RBK_MAXSEQS       --max-num-seqs (default 8; raise with lower context)
#   RBK_GPU_UTIL      --gpu-memory-utilization (default 0.90)
#   RBK_REASONING_EFFORT  server-wide default low|medium|high|xhigh (default high; Muse defaults to xhigh)
#   RBK_HF_HOME       weights cache (default /shared/$USER/hf-cache)
#   VLLM_MODULE       muse-capable vLLM Lmod module built from source by
#                     2-build-engine.sh (default vllm/0.32.1-pt2.14.1)
#   VLLM_MODULEPATH   dir to 'module use' so VLLM_MODULE resolves
#                     (default /shared/$USER/modulefiles)
#   RBK_EXTRA         extra args appended to the server command
set -euo pipefail

RBK_MODEL="${RBK_MODEL:-amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4}"
RBK_SERVED_NAME="${RBK_SERVED_NAME:-musespark}"
RBK_PORT="${RBK_PORT:-8000}"
# Bind on all interfaces so the agent (rocBudAI TUI) can reach this server from
# the LOGIN node — the agent must NOT run on this compute node or it contends
# for CPU with the model. Set RBK_HOST=127.0.0.1 to restrict to local-only.
RBK_HOST="${RBK_HOST:-0.0.0.0}"
RBK_TP="${RBK_TP:-8}"
RBK_EP="${RBK_EP:-0}"
RBK_DSPARK="${RBK_DSPARK:-1}"
RBK_DSPARK_MODEL="${RBK_DSPARK_MODEL:-Inferact/Muse-Spark-Open-DSpark}"
# 131072 (not the AMD recipe's 32768) so the ~28k-token rocBudAI persona leaves
# working room; the MXFP4 variant is validated up to 1048576 at low --max-num-seqs.
RBK_MAXLEN="${RBK_MAXLEN:-131072}"
RBK_MAXSEQS="${RBK_MAXSEQS:-8}"
RBK_GPU_UTIL="${RBK_GPU_UTIL:-0.90}"
# Muse-Spark is a reasoning model; reasoning_effort is minimal|low|medium|high|xhigh
# (its chat template defaults to medium). We pin "high" server-wide for quality with
# faster turnaround; a client's own reasoning_effort still wins per request. See
# README "Turnaround".
RBK_REASONING_EFFORT="${RBK_REASONING_EFFORT:-high}"
RBK_HF_HOME="${RBK_HF_HOME:-/shared/${USER}/hf-cache}"
RBK_EXTRA="${RBK_EXTRA:-}"

ROCM_MODULE="${ROCM_MODULE:-rocm/7.2.2}"
# muse-capable vLLM module built from source (2-build-engine.sh), and the Lmod
# root to `module use` so it resolves. Bare (not RBK_-prefixed) to match the
# sibling atom recipe's VLLM_MODULE and this kit's config.sh.
VLLM_MODULE="${VLLM_MODULE:-vllm/0.32.1-pt2.14.1}"
VLLM_MODULEPATH="${VLLM_MODULEPATH:-/shared/${USER}/modulefiles}"

say() { echo "[serve-muse] $*" >&2; }

# Must run on an MI355X GPU compute node, never a login node. A login node has no
# AMD GPUs, so vLLM/ROCm can't start and the run fails cryptically. The absence of
# the ROCm device nodes is the reliable signal — catch it and fail loudly.
if [[ ! -e /dev/kfd ]] || ! compgen -G "/dev/dri/renderD*" >/dev/null 2>&1; then
    say "ERROR: no AMD GPU detected on $(hostname -s) — this looks like a login node."
    say "rocbudai-serve-muse must run INSIDE an MI355X compute allocation. Grab a node,"
    say "then run it there:"
    say "  salloc -p mi355x --exclusive --gres=gpu:amd_instinct_mi355_oam:8 --mem=0 -t 2:00:00"
    say "  $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    say "WARNING: no SLURM_JOB_ID — run this inside an MI355X allocation:"
    say "  salloc -p mi355x --exclusive --gres=gpu:amd_instinct_mi355_oam:8 --mem=0 -t 2:00:00"
fi

# Muse-Spark-Open needs (nearly) all of the node's RAM to stage weights on load.
# --exclusive alone does NOT necessarily grant memory (on many clusters it stays capped), so a
# worker gets OOM-killed on load. Read the real allocation and fail fast unless
# you used --mem=0 (whole node).
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    _tomb() { local n="${1%[KMGT]}" u="${1##*[0-9]}"
        case "$u" in T) echo $((n*1024*1024));; G) echo $((n*1024));; K) echo $((n/1024));; *) echo "$n";; esac; }
    _alloc=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | grep -oE 'AllocTRES=[^ ]+' | head -1 | grep -oE 'mem=[0-9]+[KMGT]?' | cut -d= -f2)
    _nodemem=$(scontrol show node "$(hostname -s)" 2>/dev/null | grep -oE 'RealMemory=[0-9]+' | head -1 | cut -d= -f2)
    if [[ -n "${_alloc}" && -n "${_nodemem}" ]]; then
        _allocmb=$(_tomb "${_alloc}")
        if (( _allocmb * 100 < _nodemem * 80 )); then
            say "ERROR: only $((_allocmb/1024)) GB of the node's $((_nodemem/1024)) GB RAM is allocated — the model will OOM on load."
            say "Re-allocate with --mem=0 (all node memory):"
            say "  salloc -p mi355x --exclusive --gres=gpu:amd_instinct_mi355_oam:8 --mem=0 -t 2:00:00"
            exit 1
        fi
    fi
fi

# Lmod may not be initialised in a non-login shell (e.g. 'rocbudai-serve-muse &').
if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi

# ROCm runtime (rocminfo/rocm-smi, libs) from the rocm module; the muse-capable
# vLLM (and its bound torch-rocm + AITER) comes from the vLLM Lmod module built
# from source by 2-build-engine.sh. The vllm/<token> module prepends vLLM + its
# bound pytorch onto PYTHONPATH/PATH (and prereqs rocm). Load rocm first, then it.
say "loading module: ${ROCM_MODULE}"
module load "${ROCM_MODULE}"
if [[ -n "${VLLM_MODULEPATH}" ]]; then
    say "module use ${VLLM_MODULEPATH}"
    module use "${VLLM_MODULEPATH}"
fi
say "loading vLLM module: ${VLLM_MODULE}"
module load "${VLLM_MODULE}"
if ! command -v vllm >/dev/null 2>&1; then
    say "ERROR: 'vllm' not on PATH after loading ${VLLM_MODULE}."
    say "Build it first with ./2-build-engine.sh, and check VLLM_MODULEPATH"
    say "(${VLLM_MODULEPATH}) points at the modulefiles dir holding vllm/<token>."
    exit 1
fi

# Shared, world-readable weights cache; serve offline (no egress needed).
export HF_HOME="${RBK_HF_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Per-user AITER JIT cache so users don't collide on a shared path.
export AITER_JIT_DIR="${AITER_JIT_DIR:-/tmp/${USER}/aiter-jit}"
mkdir -p "${AITER_JIT_DIR}"

# ROCm import-time workaround (see patches/vllm-rocm-symbol-promotion.patch).
# vLLM's env_override._maybe_promote_torch_symbols_for_rocm() dlopens
# libtorch_cpu.so with RTLD_GLOBAL (so rocprofiler-sdk can find
# rocprofiler_configure for GPU profiling). With this torch wheel that promotes
# libtorch's bundled LLVM symbols into the global scope, and Triton's libtriton.so
# then binds its LLVM against torch's copy and SEGV-faults on `import vllm`. We do
# not profile inside the server, so the patch gates that promotion behind this
# flag and the serve command turns it off. Keep both together.
export VLLM_NO_ROCM_SYMBOL_PROMOTION="${VLLM_NO_ROCM_SYMBOL_PROMOTION:-1}"

# AMD-validated gfx950/MI355X backend selection for Muse-Spark-Open (Day-0 article
# + vLLM recipe). All overridable; these are the known-good defaults.
#
# Muse-Spark's MXFP4 MoE REQUIRES AITER's AITER_MXFP4_MXFP4 kernels; without them
# vLLM selects the EMULATION backend and aborts at load ("MuseSpark1x MXFP4 on ROCm
# requires the AITER_MXFP4_MXFP4 backend; got EMULATION"). That kernel is gated on
# BOTH VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1. We force them from an
# RBK_ knob (not VLLM_ROCM_USE_AITER directly) because the vllm module loaded above
# sets VLLM_ROCM_USE_AITER=0 for general use, so a ${VLLM_ROCM_USE_AITER:-1} default
# would keep that 0. Set RBK_USE_AITER=0 to opt out (serving will then fail for this
# MXFP4 model, but the knob exists for porting to non-MXFP4 checkpoints).
export VLLM_ROCM_USE_AITER="${RBK_USE_AITER:-1}"
export VLLM_ROCM_USE_AITER_MOE="${RBK_USE_AITER:-1}"
export VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE="${VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE:-bf16}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}"
# Python API-server frontend. vLLM's optional Rust frontend needs a `vllm-rs`
# binary that this from-source build does not produce (vllm_setup.sh does not run
# the setuptools-rust step), so enabling it aborts startup with FileNotFoundError.
# The Python frontend is the standard path; override to 1 only on a build that
# ships vllm-rs.
export VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-0}"

# Expert parallel (throughput) vs plain TP (lowest single-request latency).
RBK_EP_ARG=()
if [[ "${RBK_EP}" == "1" || "${RBK_EP,,}" == "true" ]]; then
    RBK_EP_ARG=(--enable-expert-parallel)
fi

# DSpark speculative decoding. Its short-query 3D attention schedule + KV layout
# are enabled via env; the draft/verify config goes on the command line.
RBK_SPEC_ARG=()
if [[ "${RBK_DSPARK}" == "1" || "${RBK_DSPARK,,}" == "true" ]]; then
    export AITER_UNIFIED_ATTN_SHORT_QUERY_3D="${AITER_UNIFIED_ATTN_SHORT_QUERY_3D:-1}"
    export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-LBHNC}"
    RBK_SPEC_ARG=(--speculative-config "{\"method\":\"dspark\",\"model\":\"${RBK_DSPARK_MODEL}\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false,\"attention_backend\":\"ROCM_AITER_FA\"}")
fi

# Server-wide default reasoning effort. Muse-Spark reads reasoning_effort as a
# chat-template kwarg, so set it with --default-chat-template-kwargs (vLLM merges
# this into every request; a client's own chat_template_kwargs.reasoning_effort
# still wins per request).
RBK_EFFORT_ARG=()
if [[ -n "${RBK_REASONING_EFFORT}" ]]; then
    RBK_EFFORT_ARG=(--default-chat-template-kwargs "{\"reasoning_effort\": \"${RBK_REASONING_EFFORT}\"}")
fi

say "node=$(hostname)  model=${RBK_MODEL}  served=${RBK_SERVED_NAME}  tp=${RBK_TP}  ep=${RBK_EP}  dspark=${RBK_DSPARK}"
say "port=${RBK_PORT}  max_len=${RBK_MAXLEN}  max_seqs=${RBK_MAXSEQS}  reasoning=${RBK_REASONING_EFFORT}"
say "HF_HOME=${HF_HOME}  AITER_JIT_DIR=${AITER_JIT_DIR}"
say "vllm=$(command -v vllm)  ($(vllm --version 2>/dev/null | head -1))"
say "starting vLLM — first load of Muse-Spark-Open (~510 GB MXFP4) + graph capture takes several minutes."
say "wait for: 'Application startup complete' / 'Uvicorn running on http://${RBK_HOST}:${RBK_PORT}'"
say "then, ON THE LOGIN NODE, point the agent here and launch it:"
say "    export ROCBUDAI_OPENAI_BASE=http://$(hostname -s):${RBK_PORT}/v1"
say "    cd <project-dir> && module load rocbudai"
echo

exec vllm serve "${RBK_MODEL}" \
    --served-model-name "${RBK_SERVED_NAME}" \
    --tokenizer-mode hf --config-format auto --load-format safetensors --generation-config vllm \
    --language-model-only --dtype bfloat16 \
    --tensor-parallel-size "${RBK_TP}" \
    --distributed-executor-backend mp \
    "${RBK_EP_ARG[@]}" \
    --attention-backend ROCM_AITER_FA \
    --max-model-len "${RBK_MAXLEN}" \
    --max-num-seqs "${RBK_MAXSEQS}" \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization "${RBK_GPU_UTIL}" \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching --enable-chunked-prefill \
    --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice \
    --host "${RBK_HOST}" \
    --port "${RBK_PORT}" \
    "${RBK_SPEC_ARG[@]}" \
    "${RBK_EFFORT_ARG[@]}" \
    ${RBK_EXTRA}
