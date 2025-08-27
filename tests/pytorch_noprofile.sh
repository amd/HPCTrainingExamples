#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
${REPO_DIR}/MLExamples/PyTorch_Profiling/no-profiling/single_process.sh
