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

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRCDIR=${REPO_DIR}/Pragma_Examples/OpenMP/Intro
BUILDDIR=$(mktemp -d)
trap 'rm -rf ${BUILDDIR}' EXIT
cp ${SRCDIR}/Makefile ${SRCDIR}/*.cpp ${SRCDIR}/*.F90 ${BUILDDIR}/
cd ${BUILDDIR}
make saxpy5
./saxpy5
