#!/bin/bash

# this script runs a PyTorch test using the  MNIST (Modified National Institute of Standards and Technology) database:
# a database of handwritten digits that can be used to train a Convolutional Neural Network for handwriting recognition.

module purge

module load pytorch

rm -rf pytorch_mnist

git clone https://github.com/pytorch/examples.git pytorch_mnist

cd pytorch_mnist/mnist

pip3 install -r requirements.txt

export PATH=$PATH:~/.local/bin
export PYTHONPATH=$PYTHONPATH:~/.local/lib/python3.10/site-packages

pip3 install torch==2.3.0+rocm6.0 torchvision==0.18+rocm6.0 torchaudio==2.3.0+rocm6.0 --index-url https://download.pytorch.org/whl/rocm6.0

python3 -c "import torch"
python3 -c "import torchvision"
python3 -c "import torchaudio"

python3 main.py

# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2

module purge

