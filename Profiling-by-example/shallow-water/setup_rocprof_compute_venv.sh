#!/bin/bash
# One-time setup for rocprof-compute analyze. Run on a login node:
#   cd Profiling-by-example/shallow-water
#   ./setup_rocprof_compute_venv.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

python3 -m venv "${ROCprof_COMPUTE_VENV}"
source "${ROCprof_COMPUTE_VENV}/bin/activate"
python3 -m pip install -r "${ROCM_PATH}/libexec/rocprofiler-compute/requirements.txt"

echo "Created ${ROCprof_COMPUTE_VENV}"
