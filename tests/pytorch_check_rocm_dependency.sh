#!/bin/bash

# This test checks that PyTorch
# has been built with ROCm

# NOTE: this test assumes PyTorch has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh


if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load pytorch

python3 -m torch.utils.collect_env | grep ROCM > output.txt
cut -d : -f 2 output.txt > new.txt

IMPORT_CHECK=`python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'`

ROCM_VERSION=`cat new.txt | tr -d " tnr"`

if [[ "${ROCM_VERSION}" != "N/A" ]]; then
   if [[ "${IMPORT_CHECK}" != "Failure" ]]; then
      echo "Success"
   else
      echo "Failure"
   fi
else
   echo "Failure"
fi


rm output.txt new.txt

