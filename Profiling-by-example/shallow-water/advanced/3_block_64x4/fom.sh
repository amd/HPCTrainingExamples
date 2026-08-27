#!/bin/bash
#SBATCH --job-name=sw-adv-3-fom
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=fom_%j.out

set -e
STAGE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SW_ROOT="$(cd "${STAGE_DIR}/../.." && pwd)"
source "${SW_ROOT}/env.sh"

make clean && make
for n in 1 2 4; do
    mpirun -n $n ${MPI_BIND} ${GPU_BIND} ./shallow_mpi
done
