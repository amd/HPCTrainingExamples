#!/bin/bash

# This test runs the rccl_example to verify
# hip-python can use RCCL for multi-GPU communication
# NOTE: Requires multiple GPUs to be available

# NOTE: this test assumes HIP-Python has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/hip-python_setup.sh

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load hip-python

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

python3 $REPO_DIR/Python/hip-python/rccl_example.py 2>/dev/null | grep -q 'ok' && echo 'Success' || echo 'Failure'
