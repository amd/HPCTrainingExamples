#!/bin/bash
# Per-rank GPU visibility for MI300A, where GPU i is local to NUMA node i.
#
# Usage, from a stage directory inside an --exclusive allocation:
#   mpirun -n 4 --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh ./shallow_mpi
#
# Open MPI's ppr:1:numa/--bind-to numa places the rank's CPUs and memory. This adds the one
# thing those flags cannot express: each rank sees a single GPU, the one local to its node.
# With one device visible, the solver's hipSetDevice(rank % device_count) reduces to device 0,
# which is that GPU. Leaving all four visible costs about 8 percent at four ranks.

LRANK=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${PMI_LOCAL_RANK:-0}}}

export ROCR_VISIBLE_DEVICES=$LRANK
exec "$@"
