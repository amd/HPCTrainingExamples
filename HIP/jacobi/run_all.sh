#! /bin/bash

set -euo pipefail

source env.sh

FLAGS=("--timestamp on" "-i input.txt" "--roctx-trace" "--hsa-trace" "--hip-trace" "--stats")
for FLAG in "${FLAGS[@]}"; do
  export ROCPROF_FLAGS="$FLAG"
  eval ./run.sh
done
