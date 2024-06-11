#!/bin/bash
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
sbatch --wait ${REPO_DIR}/HIP/vectorAdd/hip_makefile_batch.sh

grep PASSED! slurm-*.out
rm  slurm-*.out
