#!/bin/bash

module load miniforge3

export CONDA_TMPDIR=$(mktemp -d)
export CONDA_PKGS_DIRS="${CONDA_TMPDIR}/pkgs"
export CONDA_ENVS_PATH="${CONDA_TMPDIR}/envs"

mamba create -y -n env_for_install_test numpy pandas

mamba activate env_for_install_test

mamba install -y jax

mamba list -n env_for_install_test

mamba deactivate

mamba remove -y -n env_for_install_test --all

module unload miniforge3

rm -rf ${CONDA_TMPDIR}
