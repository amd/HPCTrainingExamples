#!/bin/bash

# this script runs a PyTorch test using the  MNIST 
# (Modified National Institute of Standards and Technology) database:
# a database of handwritten digits that can be used to train a 
# Convolutional Neural Network for handwriting recognition.

# NOTE: this test assumes PyTorch has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/sources/scripts/pytorch_setup.sh

module purge

module load pytorch

rm -rf pytorch_mnist

git clone https://github.com/pytorch/examples.git pytorch_mnist 2>/dev/null

cd pytorch_mnist/mnist

python3 main.py

module purge

rm -rf pytorch_mnist

