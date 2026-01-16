#!/bin/bash

# This test runs the install tests
# suite from the Cupy repo:
# https://github.com/ROCm/cupy/tree/master/tests/install_tests

# NOTE: this test assumes CuPy has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/cupy_setup.sh


module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load cupy

git clone -q --depth 1 --recursive https://github.com/ROCm/cupy.git

cd cupy/tests/install_tests

sed -i -e '23d' -e '31d' test_build.py

python3 -m pytest -vvv

cd ../../../
rm -rf cupy
