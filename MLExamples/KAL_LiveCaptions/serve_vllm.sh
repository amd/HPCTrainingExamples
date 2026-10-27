#!/usr/bin/env bash
# Serve the translator with vLLM, exposing an OpenAI-compatible API on port 8000.
# On a batch system, wrap this invocation in a scheduler job script and request one GPU.
set -euo pipefail

MODEL="${KAL_MODEL_PATH:-SaptivaAI/KAL-24B-mx-v1}"

vllm serve "$MODEL" \
  --served-model-name kal \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --host 0.0.0.0 --port 8000
