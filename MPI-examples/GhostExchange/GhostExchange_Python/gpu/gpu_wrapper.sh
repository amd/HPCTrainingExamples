#!/bin/bash
# Bind each MPI rank / Slurm task to its own node-local GPU by exposing a
# single GPU to the process. Works under OpenMPI (mpirun) and Slurm (srun).
#
# Usage:  mpirun -n <N> ./gpu_wrapper.sh python3 <script>.py ...
#         srun   -n <N> ./gpu_wrapper.sh python3 <script>.py ...
export HIP_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}
exec "$@"
