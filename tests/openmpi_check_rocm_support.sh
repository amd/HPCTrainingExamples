#!/bin/bash

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load openmpi

ompi_info | grep "MPI extensions"
