#!/bin/bash
# Per-rank GPU visibility for MI300A in CPX mode (single NUMA node, six GPUs).
#
# Usage, from a stage directory inside an --exclusive allocation:
#   mpirun -n 4 --map-by slot ../gpu_bind_cpx.sh ./shallow_mpi
#
# CPX nodes expose one NUMA domain, so use --map-by slot rather than ppr:1:numa.
# Each rank sees GPU local_rank via ROCR_VISIBLE_DEVICES.

LRANK=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${PMI_LOCAL_RANK:-0}}}

export ROCR_VISIBLE_DEVICES=$LRANK
exec "$@"
