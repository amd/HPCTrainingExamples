#!/bin/bash

export MASTER_PORT=$((10000 + ($$ % 50000)))

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
${REPO_DIR}/MLExamples/PyTorch_Profiling/rocm-systems-profiler/rocm-systems-profiler.sh
