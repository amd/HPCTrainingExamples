#!/bin/bash



module load miniforge3

mamba create -y -n env_for_create_test numpy pandas

mamba activate env_for_create_test

mamba info --envs

mamba deactivate

mamba remove -y -n env_for_create_test --all

module unload miniforge3
