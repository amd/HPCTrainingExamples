#!/bin/bash

# This test imports the mpi4py package in Python to test
# if Python MPI  is installed and accessible

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



module load ${MODULE_NAME}

python3 -c 'import mpi4py' 2> /dev/null && echo 'Success' || echo 'Failure'


