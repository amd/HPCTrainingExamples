#!/bin/bash

# Credits: Samuel Antao AMD

# This test checks basic functionalities
# of mpi4py using cupy

# NOTE: this test assumes openmpi has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/rocm/scripts/openmpi_setup.sh

MODULE_NAME=""

usage()
{
    echo ""
    echo "--help : prints this message"
    echo "--module-name : specifies the module to load to get mpi4py"
    echo ""
    exit
}

send-error()
{
    usage
    echo -e "\nError: ${@}"
    exit 1
}

reset-last()
{
   last() { send-error "Unsupported argument :: ${1}"; }
}

n=0
while [[ $# -gt 0 ]]
do
   case "${1}" in
      "--module-name")
          shift
          MODULE_NAME=${1}
          reset-last
          ;;
     "--help")
          usage
          ;;
      "--*")
          send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
          ;;
      *)
         last ${1}
         ;;
   esac
   n=$((${n} + 1))
   shift
done

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"



module load ${MODULE_NAME}
module load cupy

python3 ${REPO_DIR}/Python/mpi4py/mpi4py_cupy.py

