#!/bin/bash

module load rocm
module load openmpi

ompi_info | grep "MPI extensions"
