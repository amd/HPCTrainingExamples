# MPI-examples

Follow these instructions to run MPI examples:

Get the node with at least two GPUs/tasks:

$salloc -N 1 -p MI250 --gpus=2 --ntasks 2

Load OpenMPI module

$module load openmpi/4.1.4-gcc

Change mpicxx wrapper compiler to use hipcc

$export OMPI_CXX=hipcc

Compile and run the code

$mpicxx -o ./pt2pt ./pt2pt.cpp
$mpirun -n 2 ./pt2pt

You can get around the message "WARNING: There was an error initializing an OpenFabrics device" by telling OpenMPI to exclude openib:

$mpirun -n 2 --mca btl ^'openib' ./pt2pt
