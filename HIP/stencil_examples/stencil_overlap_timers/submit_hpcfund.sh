#!/bin/bash

#SBATCH -J stencil_overlap_timers
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./stencil
