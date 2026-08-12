#!/bin/bash

# GPU-aware MPI device/managed correctness probe.
#
# Fast, self-contained check that MPI point-to-point (ring MPI_Sendrecv) and
# collective (MPI_Allreduce) work on explicit ROCm *device* (hipMalloc) and
# *managed* (hipMallocManaged) buffers through the openmpi / ucx / ucc stack.
#
# Scope note: this exercises buffers UCX detects as ROCm memory (device/
# managed), so it validates the GPU-aware transport path in general. The
# coherent host-range buffer / CMA transport regression (ROCm 10.1 on MI300A)
# is guarded separately by MPI_Ghost_Exchange_Ver3/Ver6 and the UCC AllReduce
# test, whose buffers UCX misclassifies as host memory.
#
# Source: ../MPI-examples/mpi_gpu_probe.cpp

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
fi

GPU_COUNT=`rocminfo | grep "Device Type:             GPU" | wc -l`
if [ ${GPU_COUNT} -lt 2 ]; then
   echo "Skip"
else
   MPI_OPTS=""
   if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
      MPIRUN=srun
      MPICXX=CC
   else
      module load openmpi
      MPIRUN=mpirun
      MPI_OPTS="--mca pml ucx --mca coll ^hcoll --bind-to core"
      export OMPI_CXX=hipcc
      MPICXX=mpicxx
   fi

   REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
   cd ${REPO_DIR}/MPI-examples
   ${MPICXX} -O2 -o ./mpi_gpu_probe ./mpi_gpu_probe.cpp -I${ROCM_PATH}/include

   probe_rc=0
   for mode in 0 1; do
      out=`${MPIRUN} -n 4 ${MPI_OPTS} ./mpi_gpu_probe ${mode} 16384 2>&1`
      echo "${out}"
      echo "${out}" | grep -q "p2p=PASS allreduce=PASS"
      if [ $? -ne 0 ]; then
         probe_rc=1
      fi
   done

   rm -f ./mpi_gpu_probe

   if [ ${probe_rc} -eq 0 ]; then
      echo "GPUAWARE_PROBE: PASS"
   else
      echo "GPUAWARE_PROBE: FAIL"
   fi
fi
