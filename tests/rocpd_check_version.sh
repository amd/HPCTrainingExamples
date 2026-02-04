#!/bin/bash

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
if [ "$(printf '%s\n' "7.0" "$ROCM_VERSION" | sort -V | head -n1)" = "7.0" ]; then
   rocpd --version
else
   echo "Skip"
fi
