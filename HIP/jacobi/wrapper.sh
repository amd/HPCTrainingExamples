#! /bin/bash

set -euo pipefail
pid="$$"

source env.sh

pid="$$"
outdir="rank_${pid}_${MPI_RANK}"
outfile="results_${pid}_${MPI_RANK}.csv"

source env.sh

if [[ $DEPRICATED != 0 ]]; then
  ${ROCPROF_HOME}/rocprof ${ROCPROF_FLAGS} mpirun -np $((${NGPUS} * ${NPROC_PER_GPU})) ./Jacobi_hip -g ${NPROC_PER_GPU} ${NGPUS}
else
  ${ROCPROF_HOME}/rocprof ${ROCPROF_FLAGS} -d ${outdir} -o ${outdir}/${outfile} ./Jacobi_hip -g ${NPROC_PER_GPU} ${NGPUS}
fi
