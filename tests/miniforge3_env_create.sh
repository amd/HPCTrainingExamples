#!/bin/bash

module load miniforge3

export CONDA_TMPDIR=$(mktemp -d)
export CONDA_PKGS_DIRS="${CONDA_TMPDIR}/pkgs"
export CONDA_ENVS_PATH="${CONDA_TMPDIR}/envs"

mamba create -y -n env_for_create_test numpy pandas

mamba activate env_for_create_test

mamba info --envs

mamba deactivate

mamba remove -y -n env_for_create_test --all

module unload miniforge3

rm -rf ${CONDA_TMPDIR}
