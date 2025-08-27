#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
${REPO_DIR}/MLExamples/PyTorch_Profiling/rocm-compute-profiler/single_process.sh
