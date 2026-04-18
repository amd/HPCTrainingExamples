#!/bin/bash

# This test compiles and runs the basic Cython array_sum example
# to verify Cython can be used for CPU-side optimizations

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
EXAMPLE_DIR="$REPO_DIR/Python/hip-python/cython_basic"
WORK_DIR=$(mktemp -d)

cleanup() {
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

# Copy files to work directory
cp "$EXAMPLE_DIR/array_sum.pyx" "$WORK_DIR/"
cp "$EXAMPLE_DIR/setup.py" "$WORK_DIR/"
cp "$EXAMPLE_DIR/cython_basic.py" "$WORK_DIR/"

cd "$WORK_DIR"

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --quiet cython numpy

# Build the extension
python3 setup.py build_ext --inplace 2>/dev/null

# Run the demo and check for success
python3 cython_basic.py 2>/dev/null | grep -q 'ok' && echo 'Success' || echo 'Failure'

deactivate
