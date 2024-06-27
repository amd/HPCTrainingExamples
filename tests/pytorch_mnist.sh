#!/bin/bash

# this script runs a PyTorch test using the  MNIST (Modified National Institute of Standards and Technology) database:
# a database of handwritten digits that can be used to train a Convolutional Neural Network for handwriting recognition.

module purge

module load pytorch

rm -rf pytorch_mnist

git clone https://github.com/pytorch/examples.git pytorch_mnist

cd pytorch_mnist/mnist

python3 main.py

# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2

module purge

