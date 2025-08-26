#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 10:00

module load rocm
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd $REPO_DIR/../HIP/vectorAdd

mkdir build && cd build
cmake ..
make vectoradd
./vectoradd
