#!/bin/bash

# This test compiles and runs the Cython+HIP example
# to verify Cython can be used to optimize host code for GPU interaction

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load hip-python

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
EXAMPLE_DIR="$REPO_DIR/Python/hip-python/cython_hip_example"
WORK_DIR=$(mktemp -d)

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

# Copy files to work directory
cp "$EXAMPLE_DIR/matrix_prep.pyx" "$WORK_DIR/"
cp "$EXAMPLE_DIR/setup.py" "$WORK_DIR/"
cp "$EXAMPLE_DIR/demo.py" "$WORK_DIR/"

cd "$WORK_DIR"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --quiet cython numpy

# Build the extension
python3 setup.py build_ext --inplace 2>/dev/null

# Run the demo and check for success
python3 demo.py 2>/dev/null | grep -q 'ok' && echo 'Success' || echo 'Failure'

deactivate
