#!/bin/bash

pushd $(dirname $0)

AI=0
CUPY=0
PYTORCH=0
OMNITRACE=0
OMNIPERF=0
HIP=0
KOKKOS=0
OPENMPI=0
OPENMP=0
OPENACC=0
MVAPICH2=0
GPU_AWARE_MPI=0
STD_PAR=0
NODE_MEM_MODEL=0
PROG_MODEL=0
ROCPROF=0
USM=0


usage()
{
    echo ""
    echo "By default, this script will run ALL tests in the suite"
    echo " "
    echo "--help : prints this message"
    echo "--ai : runs the ai/ml tests"
    echo "--cupy : runs the cupy tests"
    echo "--pytorch : runs the pytorch tests"
    echo "--omnitrace: runs omnitrace tests"
    echo "--omniperf: runs omniperf tests"
    echo "--hip: runs the hip tests"
    echo "--kokkos: runs the kokkos tests"
    echo "--mpi : runs the mpi tests"
    echo "--openmpi : runs the openmpi tests"
    echo "--openmp : runs the openmp tests"
    echo "--openacc : runs the openacc tests"
    echo "--mvapich2 : runs the mvapich2 tests"
    echo "--gpu-aware-mpi : runts the gpu aware mpi tests"
    echo "--std-par : runs the hip std par tests"
    echo "--node-mem-model : runs the node mem model tests"
    echo "--prog-model : runs the programming  model tests"
    echo "--rocprof : runs the rocprof tests"
    echo "--usm : runs the usm tests"
    echo ""
    exit 1
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
      "--ai")
          shift
          AI=1
          reset-last
          ;;
      "--cupy")
          shift
          CUPY=1
          reset-last
          ;;
      "--pytorch")
          shift
          PYTORCH=1
          reset-last
          ;;
      "--omnitrace")
          shift	   
          OMNITRACE=1
          reset-last
          ;;	  
      "--omniperf")
          shift	   
          OMNIPERF=1
          reset-last
          ;;	  
      "--hip")
          shift
          HIP=1
          reset-last
          ;;
      "--kokkos")
          shift
          KOKKOS=1
          reset-last
          ;;
      "--openmpi")
          shift
          OPENMPI=1
          reset-last
          ;;
      "--openmp")
          shift
          OPENMP=1
          reset-last
          ;;
      "--openacc")
          shift
          OPENACC=1
          reset-last
          ;;
      "--mvapich2")
          shift
          MVAPICH2=1
          reset-last
          ;;
      "--gpu-aware-mpi")
          shift
          GPU_AWARE_MPI=1
          reset-last
          ;;
      "--std-par")
          shift
          STD_PAR=1
          reset-last
          ;;
      "--node-mem-model")
          shift
          NODE_MEM_MODEL=1
          reset-last
          ;;
      "--prog-model")
          shift
          PROG_MODEL=1
          reset-last
          ;;
      "--rocprof")
          shift
          ROCPROF=1
          reset-last
          ;;
      "--usm")
          shift
          USM=1
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


rm -rf build
mkdir build 
cd build
cmake ..

if [ ${AI} -eq 1 ]; then
   ctest -R Cupy
   ctest -R Pytorch
elif [ ${CUPY} -eq 1 ]; then
   ctest -R Cupy
elif [ ${PYTORCH} -eq 1 ]; then
   ctest -R Pytorch
elif [ ${OMNITRACE} -eq 1 ]; then
   ctest -R Omnitrace
elif [ ${OMNIPERF} -eq 1 ]; then
   ctest -R Omniperf
elif [ ${HIP} -eq 1 ]; then
   ctest -R HIP
   ctest -R Hipify
elif [ ${KOKKOS} -eq 1 ]; then
   ctest -R Kokkos
elif [ ${OPENMPI} -eq 1 ]; then
   ctest -R OpenMPI
elif [ ${OPENMP} -eq 1 ]; then
   ctest -R OpenMP
elif [ ${OPENACC} -eq 1 ]; then
   ctest -R OpenACC
elif [ ${MVAPICH2} -eq 1 ]; then
   ctest -R Mvapich2
elif [ ${GPU_AWARE_MPI} -eq 1 ]; then
   ctest -R GPUAware
elif [ ${STD_PAR} -eq 1 ]; then
   ctest -R StdPar
elif [ ${NODE_MEM_MODEL} -eq 1 ]; then
   ctest -R NodeMemModel
elif [ ${PROG_MODEL} -eq 1 ]; then
   ctest -R Programming_Model
elif [ ${ROCPROF} -eq 1 ]; then
   ctest -R Rocprof
elif [ ${USM} -eq 1 ]; then
   ctest -R USM
else
   ctest
fi

popd
