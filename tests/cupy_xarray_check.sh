#!/bin/bash

# This test checks that the basic
# functionalities of CuPy Xarray work

# NOTE: this test assumes CuPy has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/cupy_setup.sh

module purge

module load cupy

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

pushd $REPO_DIR/Python/cupy

python3 cupy_xarray_test.py

module unload cupy

