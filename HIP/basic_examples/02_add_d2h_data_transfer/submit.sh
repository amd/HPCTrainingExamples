#!/bin/bash

#SBATCH -J 02_add_d2h_data_transfer
#SBATCH -N 1
#SBATCH -t 5

srun -N1 -n1 ./add_one
