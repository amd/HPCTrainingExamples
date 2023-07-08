#!/bin/bash
#SBATCH -p small-g
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 10:00

module reset
module load craype-accel-amd-gfx90a
module load rocm
cd $HOME/HPCTrainingExamples/HIP/vectorAdd

# If only getting some of the GPUs on a node, the GPU detection will fail
#   in some cases in rocm_agent_enumerator utility. Set HCC_AMDGPU_TARGET to
#   bypass GPU detection
# Setting explicit GPU target
#export HCC_AMDGPU_TARGET=gfx90a
# Using rocminfo to determine which GPU to build code for

export HCC_AMDGPU_TARGET=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *\(gfx[0-9,a-f]*\) *$/\1/'`

make vectoradd
srun ./vectoradd
