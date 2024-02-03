#!/bin/bash
#SBATCH -N 1
#SBATCH -p LocalQ
#SBATCH --gpus=1
#SBATCH -t 10:00

module load rocm
cd $HOME/HPCTrainingExamples/HIP/vectorAdd

make vectoradd
./vectoradd
