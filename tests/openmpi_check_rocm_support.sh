#!/bin/bash

module load openmpi

ompi_info | grep "MPI extensions"
