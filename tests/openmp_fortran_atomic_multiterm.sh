#!/bin/bash

# LCOMPILER-1829: amdflang rejects a multi-term OpenMP atomic update
# `x = x + a + b` ("Atomic update statement should be of form...") unless the
# extra terms are parenthesized as `x = x + (a + b)`. The Atomic example's
# atomic_multiterm program exercises the accepted parenthesized form; this test
# builds and runs it.

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

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRCDIR=${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/Atomic

BUILDDIR=$(mktemp -d)
trap 'rm -rf ${BUILDDIR}' EXIT
cp ${SRCDIR}/Makefile ${SRCDIR}/*.F ${BUILDDIR}/
cd ${BUILDDIR}

export HSA_XNACK=1
make atomic_multiterm
./atomic_multiterm

make clean
