# MPI-examples


Follow these instructions to run MPI examples:

Get the node with at least two GPUs/tasks:

On AAC Training System, no salloc is necessary. But if you
use it, the command is
$salloc -N 1 -p LocalQ --gpus=2 --ntasks 2

On HPE systems -- may need to be modified for some sites
$salloc -N 1 -p MI250 --gpus=2 --ntasks 2

Load OpenMPI module

On AAC Training System
$ module load openmpi rocm
On HPE systems 
$module load openmpi/4.1.4-gcc

All Systems: Change mpicxx wrapper compiler to use hipcc

$export OMPI_CXX=hipcc

Compile and run the code

$mpicxx -o ./pt2pt ./pt2pt.cpp

On AAC system
$mpirun -n 2 -mca pml ucx ./pt2pt
On HPE system
$mpirun -n 2 ./pt2pt

You can get around the message "WARNING: There was an error initializing an OpenFabrics device" by telling OpenMPI to exclude openib:

$mpirun -n 2 --mca btl ^'openib' ./pt2pt

## OSU Benchmark

On AAC Training System

mkdir OMB && cd OMB
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.1-1.tar.gz
tar -xf osu-micro-benchmarks-7.1-1.tar.gz

module load gcc/12 rocm openmpi
mkdir build
cd osu-micro-benchmarks-7.1-1

./configure --prefix=`pwd`/../build/ \
        CC=`which mpicc` \
        CPPFLAGS=-D__HIP_PLATFORM_AMD__=1 \
        CXX=`which mpicxx` \
        --enable-rocm \
        --with-rocm=${ROCM_PATH}

make -j 12
make install

ls -l ../build/libexec/osu-micro-benchmarks/mpi

export HIP_VISIBLE_DEVICES=0,1

mpirun -N 2 -n 2 ../build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw -m 10240000

rm -rf ../../OMB
