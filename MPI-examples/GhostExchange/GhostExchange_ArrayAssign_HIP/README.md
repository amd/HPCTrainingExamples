# MPI Ghost Exchange Optimization HIP Examples
This series of examples walk you through several implementations of the same Ghost Exchange
algorithm at varying stages of optimization. Our starting point is the original CPU-only 
implementation found at 
[HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig).
In this series, we use the HIP programming model for offloading compute to the GPU. For 
a good overview of the Ghost Exchange implementation, refer to 
[the documentation here](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign).

## Changes Between Example Versions
Brief descriptions of the various versions of the Ghost Exchange HIP implementation can be found below:

- **Ver1**: Shows a HIP implementation that uses the Managed memory model to port the code to GPUs using 
host allocated memory for work buffers and MPI communication
- **Ver2**: Shows the usage and advantages of using `roctx` ranges to get more easily readable profiling output from Omnitrace
- **Ver4**: Explores heap-allocating communication buffers once on host
- **Ver5**: Explores unrolling a 2D array to a 1D array
- **Ver6**: Explores using explicit memory management directives to reduce data access latency
- **Ver8**: Under Construction, showcases overlap of compute and communication


Setting up the HIP example exercises

```
cd HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP
```

```
module load openmpi amdclang
```

Setting `HSA_XNACK` now for all of the following runs

```
export HSA_XNACK=1
MAX_ITER=1000
```

Build the code

## Version 1 -- Adding HIP kernel to original CPU code

cd Ver1

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
