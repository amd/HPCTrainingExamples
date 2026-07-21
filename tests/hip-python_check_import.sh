#!/bin/bash

# This test imports the hip-python package in Python to test
# if HIP-Python is installed and accessible

# NOTE: this test assumes HIP-Python has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/hip-python_setup.sh

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
fi

module load hip-python

python3 -c 'from hip import hiprtc' 2> /dev/null && echo 'Success' || echo 'Failure'
