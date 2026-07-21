# MPI Example: Ghost Exchange with HIP

In this version of the Ghost Exchange example we use HIP to perform the necessary computations in parallel on GPUs.These computations are for instance data initialization and solution advancement. When running in parallel, each MPI process will execute the prescribed kernels in parallel, and these will execute in parallel on the GPU, thanks to HIP.
The original CPU only version is omitted here and can be found in the OpenMP [directory](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig). Several improved versions are provided which are outlined in the next paragraph.

## Features of the various versions
The Ghost Exchange example with HIP contains several implementations at various stages
of performance optimization. Generally speaking, however, the various versions follow the same basic algorithm, what changes is where the computation happens, or the data movement and location. See below a breakdown of the features of the various versions:

- **Ver1**: this version is a variation of Orig  from the OpenMP directory, that uses HIP and unified shared memory to offload the computations to the GPUs. Arrays allocated on the CPU are passed to the HIP kernels, thanks to the unified shared memory framework. Note that arrays allocated on the CPU are also used for MPI communication, henche GPU aware MPI is not used in this version. To enable unified shared memory, `export HSA_XNACK=1` before running the example.
- **Ver1Cuda**: this version is a variation of Ver1, where HIP is replaced with CUDa. This version is present to test hipify tools.
- **Ver1WithBug**: this version is the same as Ver1 except for a bug that has been introduced in the thread grid launch parameters to have users debug it on their own.
- **Ver2**: this is a variation of Ver1, adding `roctx` ranges to get more easily readable profiling output. This change does not affect performance.
- **Ver3**: this is a variation of Ver2, allocating the communication buffers on GPU using `hipMalloc`, hence using GPU aware MPI.
- **Ver4**: this is a variation of Ver2, exploring dynamically allocating communication buffers on the CPU using malloc.
- **Ver5**: this is a variation of Ver4, where the solution array is unrolled from a 2D array into a 1D array.
- **Ver6**: this is a variation of Ver5, where the solution array and the communication buffers are allocated on the GPU with `hipMalloc`. In this version, GPU aware MPI is leveraged, thanks to the allocation of the communication buffers with `hipMalloc`. For this version, unified shared memory is not required and therefore one could `unset HSA_XNACK`.
- **Ver8**: where the computation advancing the solution happens on the GPU and overlaps with the MPI exchanges happening on the CPU. This feature is particularly valuable for the MI300A architecture since no copy and transfer of data has to be performed. Note that, unlike in Ver6, the communication buffers are allocated on the CPU. Unified shared memory is needed for this example, hence we need to `export HSA_XNACK=1` (variation of [Orig8](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig8) from the OpenMP dir).


## Overview of the implementation

The code is controlled with the following arguments:
- `-i imax -j jmax`: set the total problem size to `imax*jmax` cells.
- `-x nprocx -y nprocy`: set the number of MPI processes in the x and y direction respectively, with `nprocx*nprocy` total processes.
- `-h nhalo`: number of halo layers, the minimum value dictated by the mathematical operator in this case is one, but it can be made bigger to increase the communication work, for experimentation.
- `-t`: legacy argument used to include MPI barriers before the ghost exchange. Currently has no impact.
- `-c`: include as input argument to include the ghost cells in the corners of the MPI subdomains during the ghost exchange.
- `-p`: include as input argument to print the solution field (including values on the halo). Printing is limited above by the size of the problem.

The kernel used to advance the solution is a blur kernel, that modifies the value of a
given element by averaging the values at a 5-point stencil location centered at the given element:

`xnew[j][i] = (x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i])/5.0`

The halo exchange happens in a two-step fashion as shown in the image below, from the book [Parallel and high performance computing, by Robey and Zamora](https://www.manning.com/books/parallel-and-high-performance-computing):
<p>
<img src="ghost_exchange2.png" \>
</p>
Above, a ghost cell on a process is delimited by a dashed outline, while cells owned by a process are marked with a solid line. Communication is represented with arrows and colors representing the original data, and the location that data is being communicated and copied to. We see that each process communicates based on the part of the problem it owns: the process that owns the central portion of data must communicate in all four directions, while processes on the corner only have to communicate in two directions only.

We now describe how to compile and run some of the above versions. Note that the modules that will be loaded next rely on the model installation described in the HPCTrainingDock [repo](https://github.com/amd/HPCTrainingDock).

Note: to enable writing the solution to a netCDF file called `solution.nc`, load the `netcdf-c` module and configure with:
```
cmake -DUSE_PNETCDF=ON ..
```
Then you can install the Python requirements using the `requirements.txt` file and then execute the `print_solution.py` file to print the initial solution and final solution.
As part of the function to write the netCDF file, there is this call:
```
ncmpi_def_dim(ncid, "time", NC_UNLIMITED, &dimid_t);
```
that defines the "time" dimension as "unlimited", meaning that the last value defines the size. The last value is `maxIter-1` so space for the previous solutions is still allocated
but the array is filled with zeros.

## Version 1 -- HIP kernels to offload to GPU

Setting `HSA_XNACK=1` now for all of the following runs, except for Ver6, for which it is not needed.

```
export HSA_XNACK=1
export MAX_ITER=1000
```

Build the example

```
cd Ver1
mkdir build && cd build
cmake ..
make -j
```

Now run the example

```
echo "Ver 1: Timing for GPU version with HIP kernels and 4 ranks"
mpirun -n 4  --bind-to core     -map-by ppr:1:numa  --report-bindings ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 1 -t -c -I ${MAX_ITER}
```

Adding affinity script

```
echo "Ver 1: Timing for GPU version with HIP kernels, 4 ranks, and affinity"
mpirun -n 4  --bind-to core     -map-by ppr:1:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 1 -t -c -I ${MAX_ITER}
```

You can export the environment variable below to check that the kernels are indeed executing on the GPU:

```
export AMD_LOG_LEVEL=4
```

Version 2 through 6 can be run similarly. For version 6, we recommend to `unset HSA_XNACK` since all the arrays are allocated on the GPU with `hipMalloc`. For Ver8, we need to `export HSA_XNACK=1` again.  

