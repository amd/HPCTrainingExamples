#!/bin/bash
sbatch --wait ${REPO_DIR}/HIP/vectorAdd/hip_makefile_batch.sh

grep PASSED! slurm-*.out
rm  slurm-*.out
