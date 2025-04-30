#!/bin/bash

# this script runs a PyTorch test using the  MNIST 
# (Modified National Institute of Standards and Technology) database:
# a database of handwritten digits that can be used to train a 
# Convolutional Neural Network for handwriting recognition.

# NOTE: this test assumes PyTorch has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

module purge

module load pytorch

rm -rf pytorch_mnist

git clone https://github.com/pytorch/examples.git pytorch_mnist 2>/dev/null

pushd pytorch_mnist

cd data
mkdir MNIST
mkdir MNIST/raw
cd MNIST/raw

wget https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz

gunzip -k *.gz

popd

# use downloaded data instead of letting it download from broken mirror
sed -i 's/train=True, download=True/train=True, download=False/' mnist/main.py

python3 ./mnist/main.py

module purge

cd ..

rm -rf pytorch_mnist

