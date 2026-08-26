#!/bin/bash
#SBATCH --job-name=sw-adv-0-profile
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=profile_%j.out

set -e
STAGE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SW_ROOT="$(cd "${STAGE_DIR}/../.." && pwd)"
source "${SW_ROOT}/env.sh"

make clean && make

mpirun -n 2 ${MPI_BIND} ${GPU_BIND} \
    rocprofv3 --kernel-trace --stats -S -T -f csv \
    -o results_%env{OMPI_COMM_WORLD_RANK}% -- ./shallow_mpi

mpirun -n 2 ${MPI_BIND} ${GPU_BIND} \
    rocprof-sys-run --preset=trace-hpc --flat-profile -o trace_full -- ./shallow_mpi

mpirun -n 2 ${MPI_BIND} ${GPU_BIND} \
    rocprof-sys-run --preset=trace-hpc --flat-profile \
    --selected-regions step_3,step_4,step_5 -o trace -- ./shallow_mpi
