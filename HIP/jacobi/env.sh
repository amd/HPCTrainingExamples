#! /usr/bin/env bash

set -euo pipefail

export NGPUS=${NGPUS:-1}
export NPROC_PER_GPU=${NPROC_PER_GPU:-1}
export DEPRICATED=${DEPRICATED:-0}

export ROCPROF_FLAGS=${ROCPROF_FLAGS:-}
export ROCPROF_HOME=${ROCPROF_HOME:-/opt/rocm/bin}

if [[ -n ${OMPI_COMM_WORLD_RANK+z} ]]; then
  # mpich
  export MPI_RANK=${OMPI_COMM_WORLD_RANK}
elif [[ -n ${MV2_COMM_WORLD_RANK+z} ]]; then
  # ompi
  export MPI_RANK=${MV2_COMM_WORLD_RANK}
fi
