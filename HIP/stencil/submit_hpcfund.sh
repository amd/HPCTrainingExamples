#!/bin/bash

#SBATCH -J stencil
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./stencil
