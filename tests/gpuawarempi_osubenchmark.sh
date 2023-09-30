#!/bin/bash
mkdir OMB && cd OMB
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.1-1.tar.gz
tar -xf osu-micro-benchmarks-7.1-1.tar.gz

module load gcc/12 rocm openmpi
mkdir build
cd osu-micro-benchmarks-7.1-1

./configure --prefix=`pwd`/../build/ \
	CC=`which mpicc` \
        CXX=`which mpicxx` \
	--enable-rocm \
	--with-rocm=${ROCM_PATH}

make -j 12
make install

ls -l ../build/libexec/osu-micro-benchmarks/mpi

export HIP_VISIBLE_DEVICES=0,1

mpirun -N 2 -n 2 ../build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw -m 10240000

rm -rf ../../OMB
