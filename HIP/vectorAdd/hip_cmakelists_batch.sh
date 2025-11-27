#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 10:00

if ! module is-loaded "rocm"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd $REPO_DIR/../HIP/vectorAdd

mkdir build && cd build
cmake ..
make vectoradd
./vectoradd
