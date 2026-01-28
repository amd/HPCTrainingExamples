#!/bin/bash

mkdir OMB && cd OMB

wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.3.tar.gz

tar -xvf osu-micro-benchmarks-7.3.tar.gz

cd osu-micro-benchmarks-7.3


module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
   module load mpi/openmpi-x86_64
else
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

mpirun -N 2 -n 2 -mca pml ucx -mca coll_ucc_enable 1  -mca coll_ucc_enable 1  ../build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw -m 10240000

cd ../..

rm -rf OMB
