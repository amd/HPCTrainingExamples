#!/bin/bash
#SBATCH --job-name=hipfftmp3d
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#
# Usage: sbatch run_hipfftmp.sbatch [N]    (global N x N x N cube, N % nranks == 0)
#
# Native distributed 3D FFT with hipFFT-MP (ROCm 7.2.4+). No external solver:
# hipFFT ships the MPI multi-process path, so only the ROCm module is needed.

module load rocm/7.2.4 openmpi
export HSA_XNACK=1

mpirun -np "${SLURM_NTASKS}" --map-by ppr:4:node \
       -x HSA_XNACK \
       ./hipfftmp_3d "${1:-512}"
