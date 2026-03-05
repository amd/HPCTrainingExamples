#!/bin/bash

export CONDA_TMPDIR=$(mktemp -d)
export CONDA_PKGS_DIRS="${CONDA_TMPDIR}/pkgs"
export CONDA_ENVS_DIRS="${CONDA_TMPDIR}/envs"

module load miniforge3

mamba create -y -n env_for_create_test numpy pandas

mamba activate env_for_create_test

mamba info --envs

mamba deactivate

mamba remove -y -n env_for_create_test --all

module unload miniforge3

rm -rf ${CONDA_TMPDIR}
