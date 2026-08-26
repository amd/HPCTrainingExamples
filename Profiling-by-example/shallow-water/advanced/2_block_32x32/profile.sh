#!/bin/bash
#SBATCH --job-name=sw-adv-2-profile
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

rocprofv3 --att --att-activity 8 --kernel-include-regex compute_rhs \
    -o att -- ./shallow_mpi

mpirun -n 2 ${MPI_BIND} ${GPU_BIND} \
    rocprofv3 --pmc VALUBusy -T -f csv \
    -o results_%env{OMPI_COMM_WORLD_RANK}% -- ./shallow_mpi

PREV_WL="$(cd "${STAGE_DIR}/../1_larger_domain" && pwd)/workloads/1_larger_domain/0"
CUR_WL="${STAGE_DIR}/workloads/2_block_32x32/0"

rocprof-compute profile -n 2_block_32x32 --overwrite --no-roof -k compute_rhs \
    --iteration-multiplexing -- ./shallow_mpi
rocprof-compute analyze -p "$PREV_WL" -p "$CUR_WL" \
        -b 2.1.0 2.1.9 2.1.14 2.1.15 2.1.18 2.1.19 2.1.20 2.1.21 \
        > "${STAGE_DIR}/rocprof_compute_compare.txt"
