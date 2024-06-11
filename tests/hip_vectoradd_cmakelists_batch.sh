#!/bin/bash
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
sbatch --wait ${REPO_DIR}/HIP/vectorAdd/hip_cmakelists_batch.sh

grep PASSED! slurm-*.out
ls
rm  slurm-*.out
