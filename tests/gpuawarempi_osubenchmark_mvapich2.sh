#!/bin/bash

mkdir OMB && cd OMB

wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.3.tar.gz

tar -xvf osu-micro-benchmarks-7.3.tar.gz

cd osu-micro-benchmarks-7.3

module purge

#module load rocm mvapich2/2.3.7 

module load rocm

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

mpiexec -np 2  ../build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw -m 10240000

cd ../..

rm -rf OMB
