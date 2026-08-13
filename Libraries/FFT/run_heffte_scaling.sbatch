#!/bin/bash
#SBATCH --job-name=heffteWS
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#
# heFFTe weak-scaling study: ~256^3 grid points per rank held fixed while the
# rank count grows 1 -> 2 -> 4 -> 8 (global cube grows to match).
#   ranks : global N   (N^3 / ranks ~= 256^3 = 16.78 M points per rank)
#     1   : 256   (16.78 M/rank)
#     2   : 324   (17.0  M/rank)
#     4   : 400   (16.0  M/rank)
#     8   : 512   (16.78 M/rank)

module load rocm/7.2.4 openmpi
export HSA_XNACK=1
export LD_LIBRARY_PATH=$HOME/heffte-rocm/lib:$LD_LIBRARY_PATH

echo "=== heFFTe weak scaling (fixed ~256^3 points/rank) ==="
for cfg in "1 256" "2 324" "4 400" "8 512"; do
  set -- $cfg; NP=$1; N=$2
  mpirun -np "$NP" --map-by ppr:4:node \
         -x HSA_XNACK -x LD_LIBRARY_PATH \
         ./heffte_3d "$N"
done
