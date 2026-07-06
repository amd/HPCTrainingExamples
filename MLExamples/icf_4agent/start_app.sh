#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Backend selection ---
# "openai"    = any OpenAI-compatible endpoint (local vLLM or a remote server)
# "anthropic" = Anthropic cloud API
export ICF_API_TYPE="${ICF_API_TYPE:-openai}"

# There is ONE base model, shared by every agent. Its settings are read from the environment
# (export them in ~/.bashrc so secrets never live in this repo) and fall back to the code
# defaults in app.py when unset. Recognized variables (first one set wins):
#
#   ICF_BASE_URL   / ICF_EXPLORER_BASE_URL   base URL of the OpenAI-compatible endpoint
#   ICF_MODEL      / ICF_EXPLORER_MODEL       served model name
#   ICF_API_KEY    / ICF_EXPLORER_API_KEY     raw API token (NO "Bearer " prefix)
#   ICF_TIMEOUT    / ICF_REQUEST_TIMEOUT      per-request timeout in seconds (default 180)
#   ICF_TEMPERATURE                           sampling temperature (default 0.5)
#
# For the anthropic backend, ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY are also honored.
# This script intentionally sets NO model variables and NO secrets — it just launches the app.

echo "Starting ICF Multi-Agent Optimizer"
echo "  Backend: $ICF_API_TYPE"
echo "  Model:   ${ICF_MODEL:-${ICF_EXPLORER_MODEL:-<app.py default>}}"
echo "  Endpoint:${ICF_BASE_URL:-${ICF_EXPLORER_BASE_URL:-<app.py default>}}"
echo ""

source "$SCRIPT_DIR/aivenv/bin/activate"
exec python3 "$SCRIPT_DIR/app.py"
