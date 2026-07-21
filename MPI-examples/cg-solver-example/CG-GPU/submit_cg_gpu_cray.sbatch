#!/bin/bash
# =============================================================================
# SLURM submission wrapper for the CG-GPU test on an HPE Cray EX MI300A node
# using the PrgEnv-amd programming environment and cray-mpich.
#
# Submit from CG-GPU/:
#   sbatch submit_cg_gpu_cray.sbatch
#   sbatch --export=ALL,CG_SEED=12345 submit_cg_gpu_cray.sbatch
#
# Override the PrgEnv-amd variant (default = the base PrgEnv-amd/8.7.0, ROCm
# 7.0.3) for a version comparison, e.g.:
#   sbatch --export=ALL,PRGENV_AMD=PrgEnv-amd-new/8.7.0-7.2.4 submit_cg_gpu_cray.sbatch
# =============================================================================
#SBATCH --job-name=cg_gpu_cray
#SBATCH --partition=192C4G1H_MI300A_RHEL9_A1
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --output=cg_gpu_cray_%j.out
#SBATCH --error=cg_gpu_cray_%j.out

set -u
date
echo "Node(s): $SLURM_JOB_NODELIST"

# ---------------------------------------------------------------------------
# Toolchain: PrgEnv-amd (AMD compilers + ROCm) + cray-mpich, with the GPU-aware
# GTL library enabled via craype-accel-amd-gfx942.
# ---------------------------------------------------------------------------
module swap PrgEnv-cray "${PRGENV_AMD:-PrgEnv-amd/8.7.0}" 2>/dev/null \
   || module load "${PRGENV_AMD:-PrgEnv-amd/8.7.0}"
module load craype-accel-amd-gfx942 2>/dev/null
module list 2>&1

# GPU-aware cray-mpich (needed for the isend/alltoallv GPU-pointer variants).
export MPICH_GPU_SUPPORT_ENABLED=1

TEST_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$TEST_DIR" || { echo "FAIL: cannot cd to $TEST_DIR"; exit 1; }
[ -x ./test_cg_gpu.sh ] || { echo "FAIL: test_cg_gpu.sh not found in $TEST_DIR (submit from CG-GPU/)"; exit 1; }

export NUM_RANKS=${SLURM_NTASKS:-4}

# Pin each rank to one GPU + its NUMA-local CPUs/memory (MI300A: GPU i <-> NUMA i).
# test_cg_gpu.sh launches srun with --cpu-bind=none so numactl can bind freely.
export GPU_BIND=./gpu_bind.sh

./test_cg_gpu.sh
STATUS=$?

date
exit $STATUS
