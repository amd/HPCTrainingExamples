#! /bin/bash

set -euo pipefail

source env.sh

if [[ $DEPRICATED != 0 ]]; then
  ./wrapper.sh
else
  mpirun -np $((${NGPUS} * ${NPROC_PER_GPU})) ./wrapper.sh
fi
