#!/bin/bash
# Per-rank GPU and NUMA binding wrapper for MI300A, where GPU i is local to NUMA node i.
#
# Usage, from a stage directory inside an allocation that owns whole NUMA nodes:
#   mpirun -n 4 --bind-to none ../gpu_bind.sh ./shallow_mpi
#
# Each rank is given one visible GPU and is pinned to the CPUs and memory of that GPU's
# NUMA node. With a single GPU visible, the solver's hipSetDevice(rank % device_count)
# reduces to device 0, which is that GPU, whatever the launcher's rank numbering.

LRANK=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${PMI_LOCAL_RANK:-0}}}

export ROCR_VISIBLE_DEVICES=$LRANK
exec numactl --cpunodebind="$LRANK" --membind="$LRANK" "$@"
