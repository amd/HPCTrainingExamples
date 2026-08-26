#!/bin/bash
#SBATCH --job-name=sw-adv-3-profile
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

PREV_WL="$(cd "${STAGE_DIR}/../2_block_32x32" && pwd)/workloads/2_block_32x32/0"
CUR_WL="${STAGE_DIR}/workloads/3_block_64x4/0"

rocprof-compute profile -n 3_block_64x4 --overwrite --no-roof -k compute_rhs \
    --iteration-multiplexing -- ./shallow_mpi
rocprof-compute analyze -p "$PREV_WL" -p "$CUR_WL" \
        -b 2.1.0 2.1.9 2.1.14 2.1.15 2.1.18 2.1.19 2.1.20 2.1.21 15.1.1 \
        > "${STAGE_DIR}/rocprof_compute_compare.txt"
