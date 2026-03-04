#!/bin/bash



module load miniconda3

conda create -y -n env_for_conda_create_test numpy pandas

conda activate env_for_conda_create_test

conda info --envs

conda deactivate

conda remove -y -n env_for_conda_create_test --all

module unload miniconda3
