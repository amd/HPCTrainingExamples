#!/bin/bash

export HSA_XNACK=1
module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   export CXX=`which CC`
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cray/pe/cdst-support/2.14.6/lib:${ROCM_PATH}/lib/rocprofiler-systems"
   export LIBS="${LIBS} -L/opt/cray/pe/cdst-support/2.14.6/lib -L${ROCM_PATH}/lib/rocprofiler-systems"
else
   module load amdclang
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIPStdPar/CXX/MixAndMatch/std_cpu_gpu

make
./final
make clean

popd
