#!/bin/bash
#SBATCH --job-name=sw-adv-5-profile
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=profile_%j.out

set -e
STAGE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SW_ROOT="$(cd "${STAGE_DIR}/../.." && pwd)"
source "${SW_ROOT}/env.sh"
# source "${SW_ROOT}/advanced/nic_trace.sh"

make clean && make

mpirun -n 4 ${MPI_BIND} ${GPU_BIND} \
    rocprof-sys-run --preset=trace-hpc --flat-profile \
    --selected-regions step_3,step_4,step_5 -o trace -- ./shallow_mpi

# NIC trace: needs -N 2 and nic_trace.sh; see stage 6 README.
# if [ -n "${ROCPROFSYS_NETWORK_INTERFACE:-}" ]; then
#     run_nic_trace 8 nic ./shallow_mpi
# fi
