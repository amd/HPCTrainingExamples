#!/bin/bash

#SBATCH -J 06_hipify_pingpong 
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./pingpong
