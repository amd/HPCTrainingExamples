#!/bin/bash

#SBATCH -J 05_stencil_overlap
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./stencil
