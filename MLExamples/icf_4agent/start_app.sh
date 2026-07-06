#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Backend selection ---
# Set to "openai" for local vLLM, "anthropic" for cloud API
#export ICF_API_TYPE="${ICF_API_TYPE:-anthropic}"
export ICF_API_TYPE="${ICF_API_TYPE:-openai}"

if [ "$ICF_API_TYPE" = "openai" ]; then
    # Local vLLM defaults. To use a REMOTE OpenAI-compatible endpoint instead, override the
    # ICF_EXPLORER_* / ICF_FRONTIER_* variables below (e.g. export them before running, or edit
    # the defaults here).
    #
    # IMPORTANT: the *_API_KEY must be the raw token only -- do NOT prepend "Bearer ". The
    # OpenAI SDK adds the "Bearer " prefix automatically; including it yourself sends
    # "Authorization: Bearer Bearer <key>" and the endpoint rejects it (HTTP 403).
    #
    # Note: ICF_FRONTIER_* backs the PI, the Defect_Adversary, and the GroupChatManager -- if
    # it points at an unreachable endpoint the campaign aborts on the PI's first turn with a
    # connection error, so make sure BOTH roles point at a live server.
    VLLM_PORT="${ICF_VLLM_PORT:-8000}"
    export ICF_EXPLORER_BASE_URL="${ICF_EXPLORER_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
    export ICF_EXPLORER_MODEL="${ICF_EXPLORER_MODEL:-gptoss-20b-hedp}"
    export ICF_EXPLORER_API_KEY="${ICF_EXPLORER_API_KEY:-unused}"
    export ICF_FRONTIER_BASE_URL="${ICF_FRONTIER_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
    export ICF_FRONTIER_MODEL="${ICF_FRONTIER_MODEL:-gptoss-20b-hedp}"
    export ICF_FRONTIER_API_KEY="${ICF_FRONTIER_API_KEY:-unused}"
else
    # Anthropic cloud defaults (reads from ANTHROPIC_* env vars)
    export ICF_EXPLORER_MODEL="${ICF_EXPLORER_MODEL:-Claude-Sonnet-4.5}"
    export ICF_FRONTIER_MODEL="${ICF_FRONTIER_MODEL:-Claude-Sonnet-4.5}"
fi

echo "Starting ICF Multi-Agent Optimizer"
echo "  Backend:        $ICF_API_TYPE"
echo "  Explorer model: $ICF_EXPLORER_MODEL"
echo "  Frontier model: $ICF_FRONTIER_MODEL"
echo ""

source "$SCRIPT_DIR/aivenv/bin/activate"
exec python3 "$SCRIPT_DIR/app.py"
