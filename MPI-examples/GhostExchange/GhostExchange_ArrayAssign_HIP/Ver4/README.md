# Ghost Exchange: Reduce Allocations

In the first `roctx` range example we saw that BufAlloc was being called 101 times,
indicating we were allocating our buffers several times. In this example, we move
the allocations so that we only need to allocate the buffers one time and explore
how that impacts performance through Omnitrace.

## Environment Setup

We recommend installing OpenMPI 5.0.3 with UCX 1.16.x. Instructions [here](https://github.com/amd/HPCTrainingDock/blob/main/comm/sources/scripts/openmpi_setup.sh) may be useful reference for this OpenMPI install. We also recommend using cmake version 3.23.2 or greater.

```
module load rocm/6.2.0
module load cmake/3.23.2
module load openmpi/5.0.3_ucx1.16.x
```

## Build and Run

```
cd Ver4
mkdir build; cd build;
cmake -D CMAKE_CXX_COMPILER=${ROCM_PATH}/bin/amdclang++ -D CMAKE_C_COMPILER=${ROCM_PATH}/bin/amdclang ..
make -j8
mpirun -np 4 --mca pml ucx --mca coll ^hcoll --map-by NUMA ../../set_gpu_device.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

The output for this run should look like:

```
GhostExchange_ArrayAssign_HIP Timing is stencil 1.420971 boundary condition 0.005626 ghost cell 0.196968 total 2.066070
```

Note we see similar runtimes to previous examples, so these changes do not fix the issue.

## Get an Initial Trace

```
export HSA_XNACK=1
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
omnitrace-instrument -o ./GhostExchange.inst -- ./GhostExchange
mpirun -np 4 --mca pml ucx --mca coll ^hcoll --map-by NUMA ../../set_gpu_device.sh omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

This trace should look largely like the previous roctx trace. Searching for the roctx region named "BufAlloc" yields only one result in the trace.

<p><img src="bufalloc_once.png"/></p>

An easier way to see how this code has changed is to look at `wall_clock-0.txt`, by adding 
`OMNITRACE_PROFILE=true` and `OMNITRACE_FLAT_PROFILE=true` to `~/.omnitrace.cfg`:

<p><img src="timemory_output.png"/></p>

Here we see that the change has the intended effect of reducing the number of calls
to `BufAlloc` to one, rather than 101.
