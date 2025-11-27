#!/bin/bash

# This test checks that hpcviewer
# returns the version

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

module load hpctoolkit
hpcviewer --version
