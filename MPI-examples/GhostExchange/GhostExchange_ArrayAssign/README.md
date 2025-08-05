# MPI Example: Ghost Exchange with OpenMP

In this version of the Ghost Exchange example with use OpenMP to perform the necessary computations in parallel on GPUs.These computations are for instance data initialization and solution advancement. When running in parallel, each MPI process will execute the prescribed kernels in parallel, and these will execute in parallel on the GPU, thanks to OpenMP. 
We begin with an original implementation that can run in parallel thanks to MPI but is CPU only, meaning that the computations will run in serial on the CPU on a per process basis. Several improved versions are provided which are outlined in the next paragraph.

## Features of the various versions
The Ghost Exchange example with OpenMP contains several implementations at varying stages 
of performance optimization. Generally speaking, however, the various versions follow the same basic algorithm, what changes is where the computation happens, or the data movement and location. See below a breakdown of the features of the various versions:

- **Orig**: this is a CPU-only implementation that runs in parallel with MPI, and serves as the starting point for further optimizations. It is recommended to start here.
- **Ver1**: this version is a variation of Orig that uses OpenMP and unified shared memory to offload the computations to the GPUs. Memory can be moved to the GPU using map clauses with OpenMP, however it is much easier to not have to worry about explicit memory management for an initial port, which is what the unified shared memory allows. Note that arrays allocated on the CPU are used for MPI communication, henche GPU aware MPI is not used in this version. To enable unified shared memory, `export HSA_XNACK=1` before running the example.
- **Ver2**: this is a variation of Ver1, adding `roctx` ranges to get more easily readable profiling output. This change does not affect performance.
- **Ver3**: this is a variation of Ver2, allocating the communication buffers on GPU using the OpenMP API.
- **Ver4**: this is a variation of Ver2, exploring dynamically allocating communication buffers on the CPU using malloc.
- **Ver5**: this version unrolls a 2D array to a 1D array.
- **Ver6**: this version use explicit memory management directives to specify when data movement should happen. In this context unified shared memory is not required and therefore one could `unset HSA_XNACK`.
- **Ver7**: currently under construction, not expected to work at this time.

## Overview of the implementation 

The code is controlled with the following arguments:
- `-i imax -j jmax`: set the total problem size to `imax*jmax` cells.
- `-x nprocx -y nprocy`: set the number of MPI processes in the x and y direction respectively, with `nprocx*nprocy` total processes.
- `-h nhalo`: number of halo layers, the minimum value dictated by the mathematical operator in this case is one, but it can be made bigger to increase the communication work to experiment. 
- `-t (0|1)`: whether time synchronization should be performed.
- `-c (0|1)`: whether corners of the ghost halos should also be communicated.
- `-p (0|1)`: whether matrix, if small enough, should be printed. Used only for debugging.

The computation done on each data element after setup is a blur kernel, that modifies the value of a
given element by averaging the values at a 5-point stencil location centered at the given element:

`xnew[j][i] = (x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i])/5.0`

The communication pattern used is best shown in a diagram that appears in [Parallel and high performance computing, by Robey and Zamora](https://www.manning.com/books/parallel-and-high-performance-computing):
<p>
<img src="ghost_exchange2.png" \>
</p>
In this diagram, a ghost on a process is represented with a dashed outline, while owned data on a process is represented with a solid line. Communication is represented with arrows and colors representing the original data, and the location that data is being communicated and copied to. We see that each process communicates based on the part of the problem it owns: the process that owns the central portion of data must communicate in all four directions, while processes on the corner only have to communicate in two directions.


## Original version of Ghost Exchange

```
module load openmpi amdclang
```

Setting `HSA_XNACK` now for all of the following runs

```
export HSA_XNACK=1
MAX_ITER=1000
```

Build the code

```
cd Orig
mkdir build && cd build
cmake ..
make
```

Run the example

```
echo "Orig Ver: Timing for CPU version with 4 ranks"
mpirun -n 4  ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```

Note the time that it took to run and the time for each part of the application.

Now we will try and run it with some simple affinity settings. These map the 4 process to separate NUMA regions and binds the process to the core

```
echo "Orig Ver: Timing for CPU version with 4 ranks with affinity"
mpirun  -n 4  --bind-to core     -map-by ppr:1:numa  --report-bindings ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```

Other affinity settings that you can try. These are for larger number of ranks and GPUs. Note that the number of processes per resource (ppr)
increases 

```
echo "Orig Ver: Timing for CPU version with 16 ranks"
mpirun -n 16  ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
echo "Orig Ver: Timing for CPU version with 16 ranks with affinity"
mpirun -n 16  --bind-to core     -map-by ppr:2:numa  --report-bindings ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
mpirun -n 64  --bind-to core     -map-by ppr:8:numa  --report-bindings ./GhostExchange -x 8  -y 8  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
mpirun -n 256 ./GhostExchange -x 16 -y 16 -i 20000 -j 20000 -h 2 -t -c
mpirun -n 16  --bind-to core     -map-by ppr:2:numa  ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c
mpirun -n 64  --bind-to core     -map-by ppr:8:numa  ./GhostExchange -x 8  -y 8  -i 20000 -j 20000 -h 2 -t -c
mpirun -n 256 --bind-to hwthread -map-by ppr:32:numa ./GhostExchange -x 16 -y 16 -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```

## Version 1 -- Adding OpenMP target offload to original CPU code

cd ../../Ver1

Build the example

```
mkdir build && cd build
cmake ..
make
```

Now run the example

```
echo "Ver 1: Timing for GPU version with 4 ranks with compute pragmas"
mpirun -n 4  --bind-to core     -map-by ppr:1:numa  --report-bindings ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```

Adding affinity script

```
echo "Ver 1: Timing for GPU version with 4 ranks with compute pragmas"
mpirun -n 4  --bind-to core     -map-by ppr:1:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```

Uncomment these to confirm it is running on the GPU

```
export LIBOMPTARGET_INFO=-1
export LIBOMPTARGET_KERNEL_TRACE=1
```

## Version 4

We'll skip a version that is under construction and another that is for adding markers for the profiler. So on to version 4 where we allocate the 
MPI communication buffer in the heap just once instead of every iteration.

```
cd ../../Ver4
```

Building the code like the previous versions.

```
mkdir build && cd build
cmake ..
make
```

And running this version

```
echo "Ver 4: Timing for GPU version with 4 ranks with memory allocation once in main"
mpirun -n 4  --bind-to core     -map-by ppr:1:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```

## Version 5 Converting 2D indexing to 1D

```
cd ../../Ver5
```

Build the code
```
rm -rf build
mkdir build && cd build
cmake ..
make
```

```
echo "Ver 5: Timing for GPU version with 4 ranks with indexing converted from 2d to 1D"
mpirun -n 4  --bind-to core     -map-by ppr:1:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```

Running with more MPI ranks

```
echo "Ver 5: Timing for GPU version with 16 ranks with indexing converted from 2d to 1D"
mpirun -n 16  --bind-to core     -map-by ppr:4:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```

## Version 6 with explicit memory management

cd ../../Ver6

Setting the environment and building the code

```
unset HSA_XNACK
rm -rf build
mkdir build && cd build
cmake ..
make
```

Running the code

```
echo "Ver 6: Timing for GPU version with 4 ranks with explicit memory management"
mpirun -n 4  --bind-to core     -map-by ppr:1:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```

Running with more MPI ranks

```
echo "Ver 6: Timing for GPU version with 16 ranks with explicit memory management"
mpirun -n 16  --bind-to core     -map-by ppr:4:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 4  -y 4  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```
