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
module load scorep

export HSA_XNACK=1

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/8_jacobi/2_jacobi_targetdata

BUILDDIR=$(mktemp -d -p "$(pwd)")
cleanup() {
   cd /
   wait
   for i in 1 2 3 4 5; do
      rm -rf "${BUILDDIR}" 2>/dev/null && return
      sleep 1
   done
   rm -rf "${BUILDDIR}"
}
trap cleanup EXIT

cp ${SRC_DIR}/*.f90 ${SRC_DIR}/Makefile ${BUILDDIR}/
cd ${BUILDDIR}

echo "Building GPU Jacobi (targetdata) with Score-P instrumentation..."
make FC=scorep-amdflang
if [ $? -ne 0 ]; then
   echo "FAIL: compilation with scorep-amdflang failed"
   exit 1
fi

echo "Running instrumented Jacobi (GPU targetdata, mesh 256)..."
./jacobi -m 256
if [ $? -ne 0 ]; then
   echo "FAIL: instrumented jacobi execution failed"
   exit 1
fi

SCOREP_DIR=$(ls -dt scorep-*/ 2>/dev/null | head -1)
if [ -z "${SCOREP_DIR}" ]; then
   echo "FAIL: no scorep output directory created"
   exit 1
fi
echo "Score-P output directory: ${SCOREP_DIR}"

PROFILE="${SCOREP_DIR}profile.cubex"
if [ ! -f "${PROFILE}" ]; then
   echo "FAIL: profile.cubex not found in ${SCOREP_DIR}"
   exit 1
fi
echo "Found profile: ${PROFILE}"

echo "Running scorep-score..."
SCORE_OUTPUT=$(scorep-score -r -s totaltime "${PROFILE}" 2>&1)
if [ $? -ne 0 ]; then
   echo "FAIL: scorep-score command failed"
   echo "${SCORE_OUTPUT}"
   exit 1
fi

echo "${SCORE_OUTPUT}"

echo "${SCORE_OUTPUT}" | grep -q '!\$omp target'
if [ $? -ne 0 ]; then
   echo "FAIL: scorep-score output missing expected OpenMP offload regions (e.g. !\$omp target/teams/distribute)"
   exit 1
fi

echo "PASS: Score-P GPU profiling test completed successfully"
