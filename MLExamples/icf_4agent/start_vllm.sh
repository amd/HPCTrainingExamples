#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CHAT_TEMPLATE="${ICF_VLLM_CHAT_TEMPLATE:-${SCRIPT_DIR}/chatml_simple.jinja}"
MODEL_PATH="${ICF_VLLM_MODEL_PATH:-/home/jbelof/hfmodels/gptoss-20b-hedp.03302026}"
MODEL_NAME="${ICF_VLLM_MODEL_NAME:-gptoss-20b-hedp}"
VLLM_IMAGE_TAR="${ICF_VLLM_IMAGE_TAR:-/home/jbelof/vllm_rocm_nightly_main_20260531.tar}"
VLLM_PORT="${ICF_VLLM_PORT:-8000}"
TP_SIZE="${ICF_VLLM_TP_SIZE:-1}"
MAX_MODEL_LEN="${ICF_VLLM_MAX_MODEL_LEN:-16384}"

CONTAINER_RT="podman"
if ! command -v podman &>/dev/null; then
    CONTAINER_RT="docker"
fi

# --- Fix rootless podman on HPC nodes without /run/user writable ---
if [ "$CONTAINER_RT" = "podman" ]; then
    _uid=$(id -u)
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/podman-run-${_uid}}"
    _PODMAN_ROOT="/tmp/podman-root-${_uid}"
    _PODMAN_RUNROOT="/tmp/podman-run-${_uid}/containers"
    _PODMAN_TMPDIR="/tmp/podman-tmp-${_uid}"
    mkdir -p "$XDG_RUNTIME_DIR" "$_PODMAN_ROOT" "$_PODMAN_RUNROOT" "$_PODMAN_TMPDIR"
    PODMAN_FLAGS="--root $_PODMAN_ROOT --runroot $_PODMAN_RUNROOT --tmpdir $_PODMAN_TMPDIR --storage-opt ignore_chown_errors=true"
fi

IMAGE_NAME="vllm_rocm_nightly"

# Load the image if not already present
if ! $CONTAINER_RT $PODMAN_FLAGS image exists "$IMAGE_NAME" 2>/dev/null; then
    echo "Loading vLLM container image from $VLLM_IMAGE_TAR ..."
    $CONTAINER_RT $PODMAN_FLAGS load -i "$VLLM_IMAGE_TAR"
    LOADED_IMAGE=$($CONTAINER_RT $PODMAN_FLAGS images --format "{{.Repository}}:{{.Tag}}" | head -1)
    if [ "$LOADED_IMAGE" != "$IMAGE_NAME" ] && [ -n "$LOADED_IMAGE" ]; then
        $CONTAINER_RT $PODMAN_FLAGS tag "$LOADED_IMAGE" "$IMAGE_NAME"
    fi
    echo "Image loaded as $IMAGE_NAME"
fi

echo "Starting vLLM server..."
echo "  Model:  $MODEL_PATH"
echo "  Name:   $MODEL_NAME"
echo "  Port:   $VLLM_PORT"
echo "  TP:     $TP_SIZE"
echo "  Runtime: $CONTAINER_RT"
echo ""
echo "Once running, set these env vars for app.py:"
echo "  export ICF_EXPLORER_BASE_URL=http://localhost:${VLLM_PORT}/v1"
echo "  export ICF_EXPLORER_API_KEY=unused"
echo "  export ICF_EXPLORER_MODEL=${MODEL_NAME}"
echo "  export ICF_API_TYPE=openai"
echo ""

exec $CONTAINER_RT $PODMAN_FLAGS run --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --ipc=host \
    -v "$(dirname "$MODEL_PATH"):/models:ro" \
    -v "${CHAT_TEMPLATE}:/app/chat_template.jinja:ro" \
    -p "${VLLM_PORT}:8000" \
    --name icf-vllm \
    "$IMAGE_NAME" \
    python -m vllm.entrypoints.openai.api_server \
    --model "/models/$(basename "$MODEL_PATH")" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --chat-template /app/chat_template.jinja \
    --tool-call-parser none \
    --port 8000
