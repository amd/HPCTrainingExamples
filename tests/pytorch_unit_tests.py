#!/bin/bash

# This test runs the unit tests from the PyTorch repo 

# NOTE: this test assumes PyTorch has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/sources/scripts/pytorch_setup.sh

# NOTE: test_ops are not included in the command below because of this open issue:
# https://github.com/pytorch/pytorch/issues/120986
# the error is: RuntimeError: MKL FFT error: Intel oneMKL DFTI ERROR: Inconsistent configuration parameters

module purge

module load pytorch

git clone --recursive https://github.com/pytorch/pytorch
      
cd pytorch

pip install -r .ci/docker/requirements-ci.txt

PYTORCH_TEST_WITH_ROCM=1 python3 test/run_test.py --verbose --keep-going --include test_nn test_torch test_cuda test_unary_ufuncs test_binary_ufuncs test_autograd

cd ..

rm -rf pytorch
