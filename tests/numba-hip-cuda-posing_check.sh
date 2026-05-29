#!/bin/bash

# This test runs the numba-hip CUDA-posing example to verify
# numba-hip can execute CUDA-style kernels on AMD GPUs

# NOTE: this test assumes numba-hip has been installed according
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
EXAMPLE_DIR="$REPO_DIR/Python/hip-python"
WORK_DIR=$(mktemp -d)

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

# Copy files to work directory
cp "$EXAMPLE_DIR/numba-hip-cuda-posing.py" "$WORK_DIR/"

cd $WORK_DIR

python3 ./numba-hip-cuda-posing.py 2>/dev/null | grep -q 'PASSED' && echo 'Success' || echo 'Failure'
