#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 10:00
#SBATCH -A <your account id>

module load PrgEnv-amd
module load amd
# needed for cmake builds
module load cmake
cd $HOME/HPCTrainingExamples/HIP/vectorAdd

# portable Makefile system

make vectoradd
# Run the vectoradd application
srun ./vectoradd
# cleanup
make clean

# portable cmake system

mkdir build && cd build
cmake ..
make

# Run the vectoradd application
srun ./vectoradd
