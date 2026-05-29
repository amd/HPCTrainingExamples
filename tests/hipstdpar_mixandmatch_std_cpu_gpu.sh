#!/bin/bash

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
   if [ -z "$HIPCC" ]; then
      export HIPCC=`which hipcc`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cray/pe/cdst-support/2.14.6/lib:${ROCM_PATH}/lib/rocprofiler-systems"
   export LIBS="${LIBS} -L/opt/cray/pe/cdst-support/2.14.6/lib -L${ROCM_PATH}/lib/rocprofiler-systems"
fi

export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/HIPStdPar/CXX/MixAndMatch/std_cpu_gpu

# Build/run in a per-invocation scratch dir so concurrent invocations
# (e.g. parallel cdash array tasks on the same node) do not race in the
# shared in-tree build of ${SRC_DIR}.
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cp ${SRC_DIR}/Makefile ${SRC_DIR}/main.cpp \
   ${SRC_DIR}/stdpar_cpu_executor.cpp ${SRC_DIR}/StdParCpuExecutor.hpp \
   ${SRC_DIR}/stdpar_gpu_executor.cpp ${SRC_DIR}/StdParGpuExecutor.hpp \
   ${SRC_DIR}/ParallelExecutor.hpp ${BUILD_DIR}/
pushd ${BUILD_DIR}

make
./final

popd
