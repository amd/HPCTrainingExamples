#!/bin/bash
sbatch --wait ~/HPCTrainingExamples/HIP/vectorAdd/hip_cmakelists_batch.sh

grep PASSED! slurm-*.out
ls
rm  slurm-*.out
