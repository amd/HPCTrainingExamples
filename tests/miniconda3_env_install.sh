#!/bin/bash

export CONDA_TMPDIR=$(mktemp -d)
export CONDA_PKGS_DIRS="${CONDA_TMPDIR}/pkgs"
export CONDA_ENVS_DIRS="${CONDA_TMPDIR}/envs"

module load miniconda3

conda create -y -n env_for_conda_install_test numpy pandas

conda activate env_for_conda_install_test

conda install -y python=3.10
conda install -y jax

conda list -n env_for_conda_install_test

conda deactivate

conda remove -y -n env_for_conda_install_test --all

module unload miniconda3

rm -rf ${CONDA_TMPDIR}
