#!/bin/bash
#SBATCH --job-name=cg_sdma_sweep
#SBATCH --partition=1CN192C4G1H_MI300A_Ubuntu22
#SBATCH --exclusive
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --output=cg_sdma_sweep_%j.out
#SBATCH --error=cg_sdma_sweep_%j.out

set -u
date
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
chmod +x sweep_sdma.sh gpu_bind.sh
./sweep_sdma.sh
date
