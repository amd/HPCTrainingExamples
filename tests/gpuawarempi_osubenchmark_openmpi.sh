#!/bin/bash

mkdir OMB && cd OMB

wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.3.tar.gz

tar -xvf osu-micro-benchmarks-7.3.tar.gz

cd osu-micro-benchmarks-7.3

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
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
   module load openmpi
fi

rm -rf build

mkdir build

./configure --prefix=`pwd`/../build/ \
	CC=`which mpicc` \
	CPPFLAGS=-D__HIP_PLATFORM_AMD__=1 \
        CXX=`which mpicxx` \
	--enable-rocm \
	--with-rocm=${ROCM_PATH}

make -j

make install

ls -l ../build/libexec/osu-micro-benchmarks/mpi

export HIP_VISIBLE_DEVICES=0,1

# Launch with the launcher that MATCHES the MPI the benchmark was built with
# (CC=`which mpicc`). The project's from-source GPU-aware MPICH (mpich-wrappers)
# and OpenMPI both ship their own mpirun/mpiexec next to mpicc. Launching that
# MPICH with srun gives each rank a singleton MPI_COMM_WORLD ("This test
# requires exactly two processes") because srun's Cray PMI does not wire it up.
# Only a bare Cray MPICH (mpicc under /opt/cray, with no co-located launcher)
# is launched with srun.
MPI_BINDIR=$(dirname "$(which mpicc)")
if [ -x "${MPI_BINDIR}/mpirun" ]; then
   LAUNCHER="${MPI_BINDIR}/mpirun"
elif [ -x "${MPI_BINDIR}/mpiexec" ]; then
   LAUNCHER="${MPI_BINDIR}/mpiexec"
else
   LAUNCHER=""
fi

OSU_BW=../build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw
if [ -n "${LAUNCHER}" ]; then
   if "${LAUNCHER}" --version 2>&1 | grep -qiE "open[ -]?mpi|openrte"; then
      # OpenMPI: select the UCX point-to-point/collective stack.
      "${LAUNCHER}" -np 2 -mca pml ucx -mca coll_ucc_enable 1 ${OSU_BW} -m 10240000
   else
      # MPICH/Hydra: -mca is OpenMPI-only and would be rejected.
      "${LAUNCHER}" -np 2 ${OSU_BW} -m 10240000
   fi
else
   srun -N 1 -n 2 ${OSU_BW} -m 10240000
fi

cd ../..

rm -rf OMB
