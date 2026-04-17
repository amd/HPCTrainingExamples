#!/usr/bin/env bash

# Replace the contents of this script with your software setup!

module -t list 2>&1 | grep -q "^rocm"
if [ $? -ne 0 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

if command -v rocprof-compute &> /dev/null; then
    echo "rocprof-compute found at: $(which rocprof-compute)"
else
    echo "loading rocprofiler-compute module"
    module load rocprofiler-compute
fi

if command -v rocprof-sys-avail &> /dev/null; then
    echo "rocprof-sys-avail found at: $(which rocprof-sys-avail)"
else
    echo "loading rocprofiler-systems module"
    module load rocprofiler-systems
fi

module load pytorch
module list

# Install TraceLens - automated trace analysis library for PyTorch, JAX, and rocprofv3 profiles
# https://github.com/AMD-AGI/TraceLens
if python -c "import TraceLens" &> /dev/null; then
    echo "TraceLens is already installed"
else
    echo "Installing TraceLens..."
    pip install git+https://github.com/AMD-AGI/TraceLens.git
fi

if [[ `which mpicc | wc -l` -eq 0 ]]; then
   echo " "
   echo " "
   echo "WARNING: could not find MPI in the system, tests involving MPI will fail"
   echo " "
   echo " "
fi
