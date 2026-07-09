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

CLONE_DIR=$(mktemp -d -p /tmp Chapter13_XXXXXX)
trap "rm -rf ${CLONE_DIR}" EXIT
git clone --depth=1 --recurse-submodules --shallow-submodules https://github.com/EssentialsOfParallelComputing/Chapter13 ${CLONE_DIR}
pushd ${CLONE_DIR}/Kokkos/StreamTriad/Orig

sed -i -e 's/80000000/100000/' StreamTriad.cc

rm -rf build
mkdir build && cd build
cmake ..
make
./StreamTriad

popd
