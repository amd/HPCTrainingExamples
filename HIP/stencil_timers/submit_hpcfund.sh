#!/bin/bash

#SBATCH -J 06_stencil_timers
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./stencil
