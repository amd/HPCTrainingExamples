#!/bin/bash
#SBATCH --job-name=sw-adv-6-fom
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
make -C ../5_halo_pipeline
for n in 1 2 4; do
    echo "=== slab (stage 5), $n ranks ==="
    mpirun -n $n ${MPI_BIND} ${GPU_BIND} ../5_halo_pipeline/shallow_mpi
    echo "=== tile (stage 6), $n ranks ==="
    mpirun -n $n ${MPI_BIND} ${GPU_BIND} ./shallow_mpi
done
