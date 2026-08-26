#!/bin/bash
#SBATCH --job-name=sw-nov-3-fom
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=fom_%j.out

set -e
STAGE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SW_ROOT="$(cd "${STAGE_DIR}/../.." && pwd)"
source "${SW_ROOT}/env.sh"

make clean && make
./shallow
