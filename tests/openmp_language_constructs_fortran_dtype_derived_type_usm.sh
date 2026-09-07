#!/bin/bash

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
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi

export HSA_XNACK=1
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran
cd 7_derived_types

SRC_DIR=$(pwd)
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cp * ${BUILD_DIR}

cd ${BUILD_DIR}

# Fail visibly if the build or run fails, then emit an explicit success
# marker from the real executable output so CTest matches the marker, not a
# bare "10" that any path/build-name containing 10 could satisfy.
set -euo pipefail

make dtype_derived_type_usm
out="$(./dtype_derived_type_usm)"
echo "$out"
echo "FORTRAN-DTYPE-RESULT: $(awk 'NF {last=$0} END {print last}' <<< "$out")"
