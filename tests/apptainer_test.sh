#!/bin/bash

#module -t list 2>&1 | grep -q "^rocm"
#if [ $? -eq 1 ]; then
#  echo "rocm module is not loaded"
#  echo "loading default rocm module"
#  module load rocm
#fi
# No --rocm necessary -- container has rocm included
# if want to use host rocm, load the rocm module and use a container without rocm
apptainer exec docker://rocm/dev-ubuntu-24.04:7.2.4 rocminfo

# to launch a shell session inside a container
# apptainer shell docker://rocm/dev-ubuntu-24.04:7.2.4
