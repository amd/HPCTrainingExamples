#!/bin/bash

# This test checks that Tensorflow can
# run an MNIST classification test on GPU

# NOTE: this test assumes Tensorflow has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/tensorflow_setup.sh


if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load tensorflow

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

pushd $REPO_DIR/Python/tensorflow

python3 tensorflow_mnist.py

