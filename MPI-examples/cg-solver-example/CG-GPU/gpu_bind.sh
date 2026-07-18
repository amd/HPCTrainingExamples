#!/bin/bash
# =============================================================================
# Per-rank GPU + NUMA binding wrapper for MI300A (GPU i <-> NUMA node i).
#
# Usage (inside an allocation that owns whole NUMA nodes, e.g. --exclusive):
#   mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm rccl
#   srun  --mpi=pmix -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm rccl
#
# It pins each rank to exactly one GPU (ROCR_VISIBLE_DEVICES=<local rank>) and
# binds its CPU + memory to the matching NUMA node.  With one GPU visible, the
# solver's hipSetDevice(rank % dev_count) reduces to device 0 == that one GPU,
# so the GPU choice is unambiguous regardless of launcher or rank numbering.
# =============================================================================

# Local (per-node) rank from whichever launcher is in use.
LRANK=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${PMI_LOCAL_RANK:-0}}}

# One GPU per rank; on MI300A the closest NUMA node has the same index.
export ROCR_VISIBLE_DEVICES=$LRANK

if command -v numactl >/dev/null 2>&1; then
   exec numactl --cpunodebind="$LRANK" --membind="$LRANK" "$@"
else
   exec "$@"
fi
