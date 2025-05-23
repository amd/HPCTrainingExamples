# MPI Ghost Exchange Optimization Examples

## Changes Between Example Versions
This code contains several implementations of the same ghost exchange algorithm at varying stages 
of optimization:
- **Orig**: Shows a CPU-only implementation that uses MPI, and serves as the starting point for further optimizations. It is recommended to start here!
- **Ver1**: Shows an OpenMP target offload implementation that uses the Managed memory model to port the code to GPUs using host allocated memory for MPI communication.
- **Ver2**: Shows the usage and advantages of using `roctx` ranges to get more easily readable profiling output from Omnitrace.
- **Ver3**: Under Construction, not expected to work at the moment
- **Ver4**: Explores heap-allocating communication buffers once on host.
- **Ver5**: Explores unrolling a 2D array to a 1D array.
- **Ver6**: Explores using explicit memory management directives to specify when data movement should happen.
- **Ver7**: Under Construction, not expected to work at this time.

<details>
<summary><h3>Background Terminology: We're Exchanging <i>Ghosts?</i></h3></summary>
<h4>Problem Decomposition</h4>
In a context where the problem we're trying to solve is spread across many compute resources, 
it is usually inefficient to store the entire data set on every compute node working to solve our problem.
Thus, we "chop up" the problem into small pieces we assign to each node working on our problem.
Typically, this is referred to as a <b>problem decomposition</b>.<br/>
<h4>Ghosts, and Their Halos</h4>
In problem decompositions, we may still need compute nodes to be aware of the work that other nodes 
are currently doing, so we add an extra layer of data, referred to as a <b>halo</b> of <b>ghosts</b>.
This region of extra data can also be referred to as a <b>domain boundary</b>, as it is the <b>boundary</b> 
of the compute node's owned <b>domain</b> of data.
We call it a <b>halo</b> because typically we need to know all the updates happening in the region surrounding a single compute node's data. 
These values are called <b>ghosts</b> because they aren't really there: ghosts represent data another
 compute node controls, and the ghost values are usually set unilaterally through communication 
between compute nodes. 
This ensures each compute node has up-to-date values from the node that owns the underlying data.
These updates can also be called <b>ghost exchanges</b>.
</details>

## Overview of the Ghost Exchange Implementation
The implementations presented in these examples follow the same basic algorithm.
They each implement the same computation, and set up the same ghost exchange, we just change where computation happens, or specifics with data movement or location. 

The code is controlled with the following arguments:
- `-i imax -j jmax`: set the total problem size to `imax*jmax` elements.
- `-x nprocx -y nprocy`: set the number of MPI ranks in the x and y direction, with `nprocx*nprocy` total processes.
- `-h nhalo`: number of halo layers, typically assumed to be 1 for our diagrams.
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

Setting HSA_XNACK now for all of the following runs

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
mpirun -n 16  --bind-to core     -map-by ppr:4:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
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
mpirun -n 16  --bind-to core     -map-by ppr:4:numa  --report-bindings ../../affinity_script.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I ${MAX_ITER}
```
