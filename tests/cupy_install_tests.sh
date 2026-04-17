#!/bin/bash

# This test runs the install tests
# suite from the Cupy repo:
# https://github.com/ROCm/cupy/tree/master/tests/install_tests

# NOTE: this test assumes CuPy has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/cupy_setup.sh


module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load cupy

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cp * ${BUILD_DIR}

cd ${BUILD_DIR}

git clone -q --depth 1 --recursive https://github.com/ROCm/cupy.git

export CUPY_INSTALL_USE_HIP=1

cd cupy/tests/install_tests || exit 1

python3 -m pytest -vvv
