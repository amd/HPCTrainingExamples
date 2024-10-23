#!/bin/bash

module load rocm
XNACK_COUNT=`rocminfo | grep xnack | wc -l`
if [ ${XNACK_COUNT} -lt 1 ]; then
   echo "Skip"
else
   module load amdclang
   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/Pragma_Examples/OpenMP/USM/vector_add_usm
   make
   make run

   make clean
fi
