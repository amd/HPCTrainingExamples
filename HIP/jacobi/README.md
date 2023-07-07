Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.

This training example is released under the MIT license as listed
in the top-level directory. If this example is separated from the
main directory, include the LICENSE file with it.

Contributions from Suyash Tandon, Noel Chalmers, Nick Curtis,
Justin Chang, and Gina Sitaraman.

# Description document for the GPU-based Jacobi solver

## Contents:
---------
1.	[Application overview](#application-overview)
2.  [Prerequisites](#prerequisites)
3.	[Build instructions](#build-instructions)
4.	[Run instructions](#run-instructions)
---
## Application overview

This is a distributed Jacobi solver, using GPUs to perform the computation and MPI for halo exchanges.
It uses a 2D domain decomposition scheme to allow for a better computation-to-communication ratio than just 1D domain decomposition.

The flow of the application is as follows:
*	The MPI environment is initialized
*	The command-line arguments are parsed, and a MPI grid and mesh are created
*	Resources (including host and device memory blocks, streams etc.) are initialized
*	The Jacobi loop is executed; in every iteration, the local block is updated and then the halo values are exchanged; the algorithm
	converges when the global residue for an iteration falls below a threshold, but it is also limited by a maximum number of
	iterations (irrespective if convergence has been achieved or not)
*	Run measurements are displayed and resources are disposed

The application uses the following command-line arguments:
*	`-g x y`		-	mandatory argument for the process topology, `x` denotes the number of processes on the X direction (i.e. per row) and `y` denotes the number of processes on the Y direction (i.e. per column); the topology size must always match the number of available processes (i.e. the number of launched MPI processes must be equal to x * y)
*	`-m dx dy` 	-	optional argument indicating the size of the local (per-process) domain size; if it is omitted, the size will default to `DEFAULT_DOMAIN_SIZE` as defined in `defines.h`
* `-h | --help`	-	optional argument for printing help information; this overrides all other arguments

## Prerequisites

To build and run the jacobi application on A+A hardware, the following dependencies must be installed first:

* an MPI implementation (openMPI, MPICH, etc.)
* ROCm 2.1 or later.

## Build Instructions

A `Makefile` is included along with the source files that configures and builds multiple objects and then stitches them together to build the binary for the application `Jacobi_hip`. To build, simply run:
```
make
```
An alternative cmake build system is also include
```
mkdir build && cd build
cmake ..
make
```

## Run instructions

To run use:
```
mpirun -np 2 ./Jacobi_hip -g 2 1
```
