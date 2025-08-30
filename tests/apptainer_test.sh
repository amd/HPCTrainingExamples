#!/bin/bash

module load rocm apptainer
apptainer exec --rocm docker://rocm/dev-ubuntu-22.04:6.4.1 rocminfo

# to launch a shell session inside a container
# apptainer shell --rocm  docker://rocm/dev-ubuntu-22.04:6.4.1
