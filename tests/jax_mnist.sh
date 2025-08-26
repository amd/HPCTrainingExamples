#!/bin/bash

# This test checks that JAX
# can be used to run an example

# NOTE: this test assumes JAX has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/jax_setup.sh


module load rocm
module load jax

rm -rf jax_mnist_test_dir
mkdir jax_mnist_test_dir
cd jax_mnist_test_dir

git clone https://github.com/google/jax.git jax
cp jax/examples/datasets.py .
cp jax/examples/mnist_classifier.py .
sed -i -e 's/from examples //' mnist_classifier.py
export PYTHONPATH=.:$PYTHONPATH
python3 mnist_classifier.py

cd ..
rm -rf jax_mnist_test_dir
