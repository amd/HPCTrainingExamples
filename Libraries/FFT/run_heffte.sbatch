#!/bin/bash
#SBATCH --job-name=heffte3d
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#
# Usage: sbatch run_heffte.sbatch [N]     (global N x N x N cube)

module load rocm/7.2.4 openmpi
export HSA_XNACK=1
export LD_LIBRARY_PATH=$HOME/heffte-rocm/lib:$LD_LIBRARY_PATH

mpirun -np "${SLURM_NTASKS}" --map-by ppr:4:node \
       -x HSA_XNACK -x LD_LIBRARY_PATH \
       ./heffte_3d "${1:-512}"
