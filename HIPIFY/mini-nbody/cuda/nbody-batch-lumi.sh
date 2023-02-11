#!/bin/bash 
#SBATCH -p small-g 
#SBATCH -N 1 
#SBATCH --gpus=1 
#SBATCH -t 10:00 
#SBATCH -A project_xxxxxxxxx 

module reset 
module load craype-accel-amd-gfx90a 
module load rocm 

# Setting explicit GPU target 
#export ROCM_GPU=gfx90a 

# Using rocminfo to determine which GPU to build code for 
export ROCM_GPU=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *\(gfx[0-9,a-f]*\) *$/\1/'` 

cd HPCTrainingExamples/HIPIFY/mini-nbody/cuda  
hipify-perl -inplace -print-stats nbody-orig.cu 
hipcc --offload-arch=${ROCM_GPU} -DSHMOO -I ../ nbody-orig.cu -o nbody-orig 
srun ./nbody-orig 
