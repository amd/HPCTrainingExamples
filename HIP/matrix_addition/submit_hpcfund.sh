#!/bin/bash

#SBATCH -J matrix_addition
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./matrix_addition
