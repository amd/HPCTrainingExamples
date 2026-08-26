#!/bin/bash
#SBATCH --job-name=sw-adv-4-profile
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

PREV_WL="$(cd "${STAGE_DIR}/../3_block_64x4" && pwd)/workloads/3_block_64x4/0"
CUR_WL="${STAGE_DIR}/workloads/4_vectorized_loads/0"

rocprof-compute profile -n 4_vectorized_loads --overwrite --no-roof -k compute_rhs \
    --iteration-multiplexing -- ./shallow_mpi
rocprof-compute analyze -p "$PREV_WL" -p "$CUR_WL" \
        -b 2.1.0 2.1.9 2.1.14 2.1.15 2.1.18 2.1.19 2.1.20 2.1.21 15.1.1 \
        > "${STAGE_DIR}/rocprof_compute_compare.txt"

for n in 2 4; do
    mpirun -n $n ${MPI_BIND} ${GPU_BIND} \
        rocprof-sys-run --preset=trace-hpc --flat-profile \
        --selected-regions step_3,step_4,step_5 -o trace_n$n -- ./shallow_mpi
done

mpirun -n 4 ${MPI_BIND} ${GPU_BIND} \
    rocprofv3 --kernel-trace --stats -S -T -f csv \
    -o results_%env{OMPI_COMM_WORLD_RANK}% -- ./shallow_mpi

# rocprof-compute profile -n 4_vectorized_loads --overwrite --roof-only --device 0 -k compute_rhs \
#     --iteration-multiplexing -- ./shallow_mpi
# rocprof-compute analyze -p "${STAGE_DIR}/workloads/4_vectorized_loads/0"
