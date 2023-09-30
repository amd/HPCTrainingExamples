#!/bin/bash
sbatch --wait ~/HPCTrainingExamples/HIP/vectorAdd/hip_makefile_batch.sh

grep PASSED! slurm-*.out
rm  slurm-*.out
