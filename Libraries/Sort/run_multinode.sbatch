#!/bin/bash
#SBATCH --job-name=zipsort
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4        # 4 APUs per MI300A node (SPX mode)
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#
# Usage: sbatch run_multinode.sbatch [records_per_rank] [skew(0|1)]
#
# Launch with mpirun (NOT srun). The batch script runs on the first allocated
# compute node; mpirun places SLURM_NTASKS ranks (ppr:4:node -> one APU each).
# No GPU-binding wrapper is needed: the code derives a node-local rank from an
# MPI_COMM_TYPE_SHARED sub-communicator and binds hipSetDevice(local_rank % ndev),
# so the 4 ranks on each node land on distinct APUs (verify via the startup line
# that prints each rank's device + PCI-bus id).

module load rocm/7.2.4 openmpi
export HSA_XNACK=1                 # unified/managed memory on the APU

NP=${SLURM_NTASKS:-8}
mpirun -np "$NP" --map-by ppr:4:node -x HSA_XNACK \
       ./multinode_zip_sort "${1:-4000000}" "${2:-0}"
