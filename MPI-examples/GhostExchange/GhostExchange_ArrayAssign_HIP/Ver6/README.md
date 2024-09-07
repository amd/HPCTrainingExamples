# Ghost Exchange: Explicit Memory Management

In this example we explicitly manage the memory movement onto the device by using
hipMalloc, the device memory allocator, for all data arrays instead of using
OS-managed page migrations. We no longer need the `HSA_XNACK=1` setting.

Typically, startup costs of an application are not as important as the kernel runtimes. 
In this case, by explicitly moving memory at the beginning of our run, 
we're able to remove the overhead of memory movement from kernels. 
However our startup is slightly slower since we need to allocate a copy
of all buffers on the device up-front.

## Environment Setup

We recommend installing OpenMPI 5.0.3 with UCX 1.16.x. Instructions
[here](https://github.com/amd/HPCTrainingDock/blob/main/comm/sources/scripts/openmpi_setup.sh)
may be useful reference for this OpenMPI install. We also recommend
using cmake version 3.23.2 or greater.

```
module load rocm/6.2.0
module load cmake/3.23.2
module load openmpi/5.0.3_ucx1.16.x
```

## Build and Run

```
cd Ver6
mkdir build; cd build;
cmake -D CMAKE_CXX_COMPILER=${ROCM_PATH}/bin/amdclang++ -D CMAKE_C_COMPILER=${ROCM_PATH}/bin/amdclang ..
make -j8
mpirun -np 4 --mca pml ucx --mca coll ^hcoll --map-by NUMA ../../set_gpu_device.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

The output for this run should look like:

```
GhostExchange_ArrayAssign_HIP Timing is stencil 1.162936 boundary condition 0.004969 ghost cell 0.045186 total 1.522770
```

Now we see an improvement in our runtime which can be attributed to the lack of 
page migration.

## Get a Trace

```
unset HSA_XNACK
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
omnitrace-instrument -o ./GhostExchange.inst -- ./GhostExchange
mpirun -np 4 --mca pml ucx --mca coll ^hcoll --map-by NUMA ../../set_gpu_device.sh omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

Here's what the trace looks like for this run:

<p><img src="initial_trace.png"/></p>

The biggest difference we see is that the first invocation of the `blur` kernel is now
just as fast as the subsequent invocations at 11ms. This indicates that we don't spend
time in page migration anymore. The implicit data movement was a large portion of our
kernel overhead.

## Look at Timemory output

The `wall_clock-0.txt` file shows our overall run got faster:

<p><img src="timemory_output.png"/></p>

Previously we ran in 1.9 seconds, and now the uninstrumented runtime is 1.5 seconds
(from above), while `wall_clock-0.txt` shows our runtime is 2.23 seconds. 

However, we see that the location of our data on CPU+GPU system matters quite a lot
to performance. Implicit memory movement may not get the best performance, and it is
usually worth it to pay the memory movement cost up front once than repeatedly for
each kernel.
