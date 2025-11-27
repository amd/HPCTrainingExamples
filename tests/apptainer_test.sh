#!/bin/bash

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
apptainer exec --rocm docker://rocm/dev-ubuntu-22.04:6.4.1 rocminfo

# to launch a shell session inside a container
# apptainer shell --rocm  docker://rocm/dev-ubuntu-22.04:6.4.1
