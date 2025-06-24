#!/bin/bash

pushd $(dirname $0)

AI=0
CUPY=0
TENSORFLOW=0
ROCPROFV3=0
PYTORCH=0
FTORCH=0
JULIA=0
JAX=0
PETSC=0
HYPRE=0
TAU=0
ROCPROF_SYS=0
ROCPROF_COMPUTE=0
HIP=0
HIPIFY=0
HIPIFLY=0
KOKKOS=0
HPCTOOLKIT=0
OPENMPI=0
MPI=0
MPI4PY=0
FFTW=0
OPENMP=0
OPENACC=0
MVAPICH2=0
MINICONDA3=0
MINIFORGE3=0
HDF5=0
HIPSTDPAR=0
NETCDF=0
HIPFORT=0
GPU_AWARE_MPI=0
STD_PAR=0
NODE_MEM_MODEL=0
PROG_MODEL=0
ROCPROF=0
SCOREP=0
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
    echo "--petsc : runs the petsc tests"
    echo "--hypre : runs the hypre tests"
    echo "--rocprof-sys: runs ROCm rocprof-sys tests depending on the ROCm version"
    echo "--rocprof-compute: runs ROCm rocprof-compute tests depending on the ROCm version"
    echo "--hip: runs the hip tests"
    echo "--hipify: runs the hipify tests"
    echo "--hipstdpar: runs the hipstdpar tests"
    echo "--hipifly: runs the hipifly tests"
    echo "--kokkos: runs the kokkos tests"
    echo "--hpctoolkit: runs the hpctoolkit tests"
    echo "--tau: runs the tau tests"
    echo "--scorep: runs the score-p tests"
    echo "--mpi : runs all the mpi tests (same as including --opempi --mpi4py --mvapich2 --gpu-aware-mpi)"
    echo "--openmpi : runs the openmpi tests"
    echo "--fftw: runts the fftw tests"
    echo "--mpi4py : runs the mpi4py tests"
    echo "--jax : runs the jax tests"
    echo "--openmp : runs the openmp tests"
    echo "--miniconda3 : runs miniconda3 tests"
    echo "--miniforge3 : runs miniforge3 tests"
    echo "--hdf5 : runs hdf5 tests"
    echo "--ftorch: runs ftorch tests"
    echo "--netcdf : runs netcdf tests"
    echo "--hipfort : runs hipfort tests"
    echo "--openacc : runs the openacc tests"
    echo "--mvapich2 : runs the mvapich2 tests"
    echo "--gpu-aware-mpi : runts the gpu aware mpi tests"
    echo "--std-par : runs the hip std par tests"
    echo "--node-mem-model : runs the node mem model tests"
    echo "--prog-model : runs the programming  model tests"
    echo "--rocprofv3 : runs the rocprofv3 tests"
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
      "--tensorflow")
          shift
          TENSORFLOW=1
          reset-last
          ;;
      "--pytorch")
          shift
          PYTORCH=1
          reset-last
          ;;
      "--petsc")
          shift
          PETSC=1
          reset-last
          ;;
      "--hypre")
          shift
          HYPRE=1
          reset-last
          ;;
      "--rocprof-sys")
          shift
          ROCPROF_SYS=1
          reset-last
          ;;
      "--rocprof-compute")
          shift
          ROCPROF_COMPUTE=1
          reset-last
          ;;
      "--hip")
          shift
          HIP=1
          reset-last
          ;;
      "--hipify")
          shift
          HIPIFY=1
          reset-last
          ;;
      "--ftorch")
          shift
          FTORCH=1
          reset-last
          ;;
      "--julia")
          shift
          JULIA=1
          reset-last
          ;;
      "--hipstdpar")
          shift
          HIPSTDPAR=1
          reset-last
          ;;
      "--hipifly")
          shift
          HIPIFLY=1
          reset-last
          ;;
      "--kokkos")
          shift
          KOKKOS=1
          reset-last
          ;;
      "--hpctoolkit")
          shift
          HPCTOOLKIT=1
          reset-last
          ;;
      "--tau")
          shift
          TAU=1
          reset-last
          ;;
      "--scorep")
          shift
          SCOREP=1
          reset-last
          ;;
      "--miniconda3")
          shift
          MINICONDA3=1
          reset-last
          ;;
      "--miniforge3")
          shift
          MINIFORGE3=1
          reset-last
          ;;
      "--hdf5")
          shift
          HDF5=1
          reset-last
          ;;
      "--fftw")
          shift
          FFTW=1
          reset-last
          ;;
      "--netcdf")
          shift
          NETCDF=1
          reset-last
          ;;
      "--jax")
          shift
          JAX=1
          reset-last
          ;;
      "--openmpi")
          shift
          OPENMPI=1
          reset-last
          ;;
      "--mpi")
          shift
          MPI=1
          reset-last
          ;;
      "--mpi4py")
          shift
          MPI4PY=1
          reset-last
          ;;
      "--hipfort")
          shift
          HIPFORT=1
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
      "--rocprofv3")
          shift
          ROCPROFV3=1
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
   ctest -R JAX
elif [ ${CUPY} -eq 1 ]; then
   ctest -R Cupy
elif [ ${JAX} -eq 1 ]; then
   ctest -R JAX
elif [ ${PYTORCH} -eq 1 ]; then
   ctest -R Pytorch
elif [ ${ROCPROF_SYS} -eq 1 ]; then
   ctest -R Rocprof-sys_ROCm
elif [ ${ROCPROF_COMPUTE} -eq 1 ]; then
   ctest -R Rocprof-compute_ROCm
elif [ ${HIP} -eq 1 ]; then
   ctest -R HIP
   ctest -R Hipify
   ctest -R Hipifly
elif [ ${HIPIFY} -eq 1 ]; then
   ctest -R Hipify
elif [ ${HIPSTDPAR} -eq 1 ]; then
   ctest -R StdPar
elif [ ${HIPIFLY} -eq 1 ]; then
   ctest -R Hipifly
elif [ ${TENSORFLOW} -eq 1 ]; then
   ctest -R TensorFlow
elif [ ${KOKKOS} -eq 1 ]; then
   ctest -R Kokkos
elif [ ${HIPFORT} -eq 1 ]; then
   ctest -R HIPFort
elif [ ${HDF5} -eq 1 ]; then
   ctest -R HDF5
elif [ ${NETCDF} -eq 1 ]; then
   ctest -R Netcdf
elif [ ${HPCTOOLKIT} -eq 1 ]; then
   ctest -R HPCToolkit
elif [ ${MINICONDA3} -eq 1 ]; then
   ctest -R Miniconda3
elif [ ${MINIFORGE3} -eq 1 ]; then
   ctest -R Miniforge3
elif [ ${TAU} -eq 1 ]; then
   ctest -R TAU
elif [ ${FTORCH} -eq 1 ]; then
   ctest -R FTorch
elif [ ${JULIA} -eq 1 ]; then
   ctest -R Julia
elif [ ${PETSC} -eq 1 ]; then
   ctest -R PETSc
elif [ ${HYPRE} -eq 1 ]; then
   ctest -R HYPRE
elif [ ${FFTW} -eq 1 ]; then
   ctest -R FFTW
elif [ ${SCOREP} -eq 1 ]; then
   ctest -R Score-P
elif [ ${MPI} -eq 1 ]; then
   ctest -R OpenMPI
   ctest -R Mvapich2
   ctest -R GPUAware
   ctest -R MPI4PY
elif [ ${OPENMPI} -eq 1 ]; then
   ctest -R OpenMPI
elif [ ${MPI4PY} -eq 1 ]; then
   ctest -R MPI4PY
elif [ ${OPENMP} -eq 1 ]; then
   ctest -R OpenMP_
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
elif [ ${ROCPROFV3} -eq 1 ]; then
   ctest -R RocprofV3
elif [ ${USM} -eq 1 ]; then
   ctest -R USM
else
   ctest
fi

rm -rf build

popd
