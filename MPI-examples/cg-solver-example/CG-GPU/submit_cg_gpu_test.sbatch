#!/bin/bash
# =============================================================================
# SLURM submission wrapper for the CG-GPU test.
#
# The login/build node has no GPU, so both the build and the run happen inside
# this allocation on a GPU compute node.
#
# Submit:
#   sbatch submit_cg_gpu_test.sbatch
#
# Override the partition / GPU count on the command line, e.g.:
#   sbatch -p 1CN192C24G1H_MI300A_Ubuntu22 --gpus=4 --ntasks=4 submit_cg_gpu_test.sbatch
#
# Defaults target the 4-GPU MI300A partition, which gives the 4-rank run shown
# in README.md.
# =============================================================================
#SBATCH --job-name=cg_gpu_test
#SBATCH --partition=1CN192C4G1H_MI300A_Ubuntu22
#SBATCH --exclusive
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --time=00:20:00
#SBATCH --output=cg_gpu_test_%j.out
#SBATCH --error=cg_gpu_test_%j.out

set -u
date
echo "Node(s): $SLURM_JOB_NODELIST"

# Modules (ROCm + GPU-Aware OpenMPI).  Override the ROCm version for comparison
# studies, e.g.:  sbatch --export=ALL,ROCM_MODULE=rocm/7.2.4 submit_cg_gpu_test.sbatch
module purge 2>/dev/null
module load "${ROCM_MODULE:-rocm/6.4.1}"
module load "${OPENMPI_MODULE:-openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4}" 2>/dev/null || module load openmpi
module list 2>&1

# SLURM copies the batch script to a spool dir, so resolve the test directory
# from the submission directory (sbatch is launched from CG-GPU/).
TEST_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$TEST_DIR" || { echo "FAIL: cannot cd to $TEST_DIR"; exit 1; }
[ -x ./test_cg_gpu.sh ] || { echo "FAIL: test_cg_gpu.sh not found in $TEST_DIR (submit from CG-GPU/)"; exit 1; }

# Let the number of MPI ranks follow the SLURM allocation.
export NUM_RANKS=${SLURM_NTASKS:-4}

# Pin each rank to one GPU + its NUMA-local CPUs/memory (MI300A: GPU i <-> NUMA i).
# Requires --exclusive so numactl can bind to whole NUMA nodes.
export GPU_BIND=./gpu_bind.sh

./test_cg_gpu.sh
STATUS=$?

date
exit $STATUS
