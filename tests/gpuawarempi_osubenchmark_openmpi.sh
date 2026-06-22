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

if [ -n "${CRAY_MPICH_VERSION:-}" ]; then
   MPIRUN=srun
   MPI_RUN_OPTIONS=""
else
   MPIRUN=mpirun
   MPI_RUN_OPTIONS="-mca pml ucx -mca coll_ucc_enable 1  -mca coll_ucc_enable 1"
fi

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

#${MPIRUN} -N 2 -n 2 ${MPI_RUN_OPTIONS} ../build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw -m 10240000
${MPIRUN} -N 1 -n 2 ${MPI_RUN_OPTIONS} ../build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw -m 10240000

cd ../..

rm -rf OMB
