#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/jacobi

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   export CXX="`which CC`"
   export HIP_PLATFORM=amd
fi
if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   module load libfabric
   MPIRUN=srun
else
   module load openmpi
   MPIRUN=mpirun
fi

rm -rf build
mkdir build && cd build
cmake ..
make

#salloc -p LocalQ --gpus=2 -n 2 -t 00:10:00
${MPIRUN} -n 2 ./Jacobi_hip -g 2

cd ..
rm -rf build
