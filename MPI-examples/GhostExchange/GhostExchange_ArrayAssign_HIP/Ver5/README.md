# Ghost Exchange: Changing Data Layout

In this example we explore changing our 2D array layout to 1D and use Omnitrace
to investigate the performance impact.

This sort of change typically requires significant development overhead, as the
indexing of the data must change everywhere in the application.

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
cd Ver5
mkdir build; cd build;
cmake -D CMAKE_CXX_COMPILER=${ROCM_PATH}/bin/amdclang++ -D CMAKE_C_COMPILER=${ROCM_PATH}/bin/amdclang ..
make -j8
mpirun -np 4 --mca pml ucx --mca coll ^hcoll --map-by NUMA ../../set_gpu_device.sh ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

The output from this run should look like:

```
GhostExchange_ArrayAssign_HIP Timing is stencil 1.364763 boundary condition 0.005312 ghost cell 0.156599 total 1.973974
```

We still see a similar runtime indicating this code change did not fix our issue.

## Get a Trace

```
export HSA_XNACK=1
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
omnitrace-instrument -o ./GhostExchange.inst -- ./GhostExchange
mpirun -np 4 --mca pml ucx --mca coll ^hcoll --map-by NUMA ../../set_gpu_device.sh omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

The initial trace for this example should look very similar to previous traces we have seen:

<p><img src="initial_trace.png"/></p>

That is, we cannot see an obvious change in performance from just looking at this trace. We will make a note of the runtime of the first `blur` kernel invocation as taking 128ms, and a subsequent `blur` kernel invocation as taking 12ms. In the next version we will compare these numbers to another modification.

## Look at Timemory output

We also see that our `wall_clock-0.txt` file looks pretty similar to our previous example:

<p><img src="timemory_output.png"/></p>

To enable the output of this file, add `OMNITRACE_PROFILE=true` and `OMNITRACE_FLAT_PROFILE=true` to your `~/.omnitrace.cfg` file.
