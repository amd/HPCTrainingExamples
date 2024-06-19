#!/bin/bash

# This test imports the torch package in Python to test 
# if PyTorch is installed and accessible

module purge

module load pytorch

python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'


