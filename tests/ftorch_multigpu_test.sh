#!/bin/bash

# This test runs the multi gpu test from FTorch 

# NOTE: this test assumes FTorch has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/ftorch_setup.sh

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load ftorch

git clone https://github.com/Cambridge-ICCS/FTorch.git ftorch_test
cd ftorch_test/examples/
python3 -m venv ftorch_test
source ftorch_test/bin/activate
cd 6_MultiGPU
python3 pt2ts.py --device_type hip
python3 multigpu_infer_python.py --device_type hip
mkdir build && cd build
cmake ..
make -j
./multigpu_infer_fortran hip ../saved_multigpu_model_hip.pt
deactivate
cd ../../../../
rm -rf ftorch_test

