#!/bin/bash
#SBATCH --job-name=heffteComm
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:20:00

module load rocm/7.2.4 openmpi
export HSA_XNACK=1
export LD_LIBRARY_PATH=$HOME/heffte-rocm/lib:$LD_LIBRARY_PATH

N="${1:-512}"

echo "=== 8-rank 2-node all-to-all: GPU-aware MPI ON ==="
HEFFTE_GPU_AWARE=1 mpirun -np 8 --map-by ppr:4:node \
   -x HSA_XNACK -x LD_LIBRARY_PATH -x HEFFTE_GPU_AWARE ./heffte_3d "$N"

echo "=== 8-rank 2-node all-to-all: host-staged (GPU-aware OFF) ==="
HEFFTE_GPU_AWARE=0 mpirun -np 8 --map-by ppr:4:node \
   -x HSA_XNACK -x LD_LIBRARY_PATH -x HEFFTE_GPU_AWARE ./heffte_3d "$N"
