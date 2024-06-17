<!---
Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
--->

# Jacobi Solver

These are the instructions for building and executing both the **HIP** and **OpenMP** offloaded versions 
of the Jacobi Solver. The solver executes is offloaded onto a single GPU or GCD. Optionally, the solver can
execute on the CPUs using **OpenMP** threads. ROCm 5.0+ is needed to execute either the 
**HIP** or **OpenMP** offloaded versions.

## Code structure

Below is a brief description of each source file

- `BoundaryConditions.cpp`: **HIP** and **OpenMP** implementations of enforcing the boundary conditions
- `Input.cpp`: Command-line argument parser and support functions
- `Jacobi.cpp`: Initializes the mesh and executes the Jacobi solver
- `Laplacian.cpp`: **HIP** and **OpenMP** implementations of the finite-difference stencil
- `Main.cpp`: Application entry point
- `Norm.cpp`: **HIP** and **OpenMP** implementations of the residual computation
- `Update.cpp`: **HIP** and **OpenMP** implementations of updating the solution and residual vectors

## Building the code

- Modify the `Makefile` to point to the appropriate compilers.
- Build the **OpenMP** version of the code only by typing `make omp`
- Build the **HIP** version of the code only by typing `make hip`
- Build both versions by typing `make`

The executing binaries will be shown as `Jacobi_omp` and `Jacobi_hip`. To clean the builds, type
`make clean_omp` or `make clean_hip` for the **OpenMP** and **HIP** versions, respectively. Typing
`make clean` cleans both versions.


## Running the code

To execute with default setting on a 4096 x 4096 domain size:

```
$ ./Jacobi_omp
```

To execite with a specified domain size e.g., 8192 nodes along both x- and y-directions:

```
$ ./Jacobi_omp -m 8192 8192
```

To execute the CPU version of **OpenMP** with `X` threads, run with the following commands:

```
$ OMP_TARGET_OFFLOAD=DISABLED OMP_NUM_THREADS=X ./Jacobi_omp
```
