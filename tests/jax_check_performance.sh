#!/bin/bash

# Test contributed by Corey Adams (corey.adams@amd.com)

# This test checks that JAX
# shows the expected performance
# for different data types

# NOTE: this test assumes JAX has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/jax_setup.sh


REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

module load jax

python3 ${REPO_DIR}/tests/jax_check_performance.py
