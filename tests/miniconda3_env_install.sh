#!/bin/bash



module load miniconda3

conda create -y -n env_for_conda_install_test numpy pandas

conda activate env_for_conda_install_test

conda install -y python=3.10
conda install -y jax

conda list -n env_for_conda_install_test

conda deactivate

conda remove -y -n env_for_conda_install_test --all

module unload miniconda3
