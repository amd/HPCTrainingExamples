#!/bin/bash
#SBATCH --job-name=sw-nov-5-profile
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=profile_%j.out

set -e
STAGE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SW_ROOT="$(cd "${STAGE_DIR}/../.." && pwd)"
source "${SW_ROOT}/env.sh"

make clean && make

rm -rf att
rocprofv3 --att --att-activity 8 --kernel-include-regex compute_rhs \
    -d att -o att -- ./shallow

rocprof-compute profile -n 5_vectorized_loads --overwrite --roof-only --device 0 -k compute_rhs \
    --iteration-multiplexing -- ./shallow
rocprof-compute analyze -p "${STAGE_DIR}/workloads/5_vectorized_loads/0"
