
# Stage 0: Baseline

This is the starting point, before any optimization. The global domain is 512x512, split as a 1D
slab in Y across the ranks. Every RK4 stage applies the boundary conditions, exchanges halos, and
evaluates the right-hand side, and the compute kernels are launched with 16x16 thread blocks.

The halo exchange is the part to keep an eye on. Look at `exchange_y_halos` in `shallow_mpi.hip`:

```c++
    // Ensure device writes are visible
    CHECK_CUDA(hipDeviceSynchronize());
    ...
    if (rank_down != MPI_PROC_NULL) {
        MPI_Sendrecv(send_bot_h, nx, MPI_FLOAT, rank_down, 100,
                     recv_bot_h, nx, MPI_FLOAT, rank_down, 200, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bot_hu, nx, MPI_FLOAT, rank_down, 101,
                     recv_bot_hu, nx, MPI_FLOAT, rank_down, 201, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bot_hv, nx, MPI_FLOAT, rank_down, 102,
                     recv_bot_hv, nx, MPI_FLOAT, rank_down, 202, comm, MPI_STATUS_IGNORE);
    }
```

That is a device-wide synchronization followed by up to six separate blocking exchanges, three
arrays in each of two directions, and it happens four times per time step. Nothing here is
deliberately broken; it is the shortest correct way to write a halo exchange, and it is what most
codes start out with. Our job over the next six stages is to let the profiler tell us what to change.

## Build and run

```bash
module load rocm openmpi
make
mpirun -n 2 --bind-to none ../gpu_bind.sh ./shallow_mpi
```

or with CMake:

```bash
mkdir build && cd build
cmake ..
make
mpirun -n 2 --bind-to none ../gpu_bind.sh ./shallow_mpi
```

`gpu_bind.sh` gives each rank one GPU and pins it to that GPU's NUMA node, inside an `--exclusive`
allocation. Every timing in this track depends on it, to the tune of about 100x, so if you skip one
thing here do not let it be this one; see
[affinity](../README.md#affinity-which-decides-everything-else) in the track README.

## Expected output

```
MPI ranks: 2  |  GPUs detected: 1
Domain: 512x512 (global), steps=500, dt=0.0728643
Elapsed (max over ranks): 0.106 s  |  Throughput: 4950.31 MCUPS
Mass: initial=2.626466547e+05, final=2.626464582e+05, rel.err=7.483e-07
Min(h) after run: 0.981776
```

Keep the last two lines in view for the rest of the tutorial. Mass conservation and a positive
minimum depth are how we confirm that an optimization sped the code up without changing the answer,
and in a multi-rank code they are also how we confirm that the halo exchange is still delivering the
data it is supposed to.

## Step 1: Does it scale at all?

Before profiling anything, run the same binary on one, two and four GPUs. This costs three commands
and it decides what to do next:

```bash
for n in 1 2 4; do mpirun -n $n --bind-to none ../gpu_bind.sh ./shallow_mpi; done
```

| Ranks | Elapsed | MCUPS | Efficiency |
|---|---|---|---|
| 1 | 0.063 s | 8309.26 | -- |
| 2 | 0.106 s | 4950.31 | 30 percent |
| 4 | 0.174 s | 3006.97 | 9 percent |

**Adding GPUs makes this code slower in absolute terms.** Four GPUs take nearly three times as long as one
to do exactly the same work. This is not a subtle scaling inefficiency to be shaved down later, it is
a signal that the configuration being measured is the wrong one, and it invalidates any conclusion
we might draw about the kernels while it holds.

It is worth being clear about why, because the diagnosis determines the fix. A 512x512 global domain
divided across four ranks leaves each rank a 512x128 band. That band has 65536 interior cells, and
its two boundary rows are 512 cells each. So roughly one interior cell in 64 has to be shipped to a
neighbour every RK4 stage, and there is not remotely enough arithmetic per cell to hide the latency
of doing so. Worse, 65536 cells in 16x16 blocks is 256 workgroups, spread across a GPU with 228
compute units: barely one workgroup per compute unit, so most of the machine is idle even before
communication starts.

## Step 2: Where does the time go, per rank?

`rocprofv3` profiles a single process, so under `mpirun` it has to be applied to each rank and told
to write somewhere different for each one. The `%rank%` substitution in the output name does that:

```bash
mpirun -n 2 --bind-to none ../gpu_bind.sh \
    rocprofv3 --kernel-trace --stats -S -T -d prof -o shallow_%rank% -- ./shallow_mpi
```

Note the position of `rocprofv3`: it wraps the application, and `mpirun` launches the profiler rather
than the other way around. Reversing the two profiles `mpirun` itself, which is not what we want.
`gpu_bind.sh` stays outermost of the three, so each rank is already bound to its GPU and NUMA node by
the time the profiler starts the process it measures.
The remaining options are as in the novice example: `--kernel-trace` records every kernel dispatch,
`--stats` computes per-kernel statistics, `-S` prints the summary to the console, `-T` truncates the
demangled kernel names so the table stays readable, and `-d`/`-o` set the output directory and file
prefix.

You now get one summary per rank. On a symmetric decomposition like this they should look nearly
identical, and confirming that they do is the point: a rank that is doing noticeably more or less
work than its neighbours is a load-balance problem, and no amount of kernel tuning will fix it.

<!-- MEASUREMENT TODO: per-rank ROCPROFV3 kernel summary tables for 2 ranks -->

The call counts confirm that the trace matches the algorithm: 500 time steps with four RK4 stages
each, so 2000 `compute_rhs` calls, 1500 `update_stage` calls, 500 `final_update` calls, and 2001
boundary-condition calls including the one during initialization.

## Step 3: What the kernel trace cannot tell you

Add up the kernel durations in those tables and compare the total against the wall-clock time the
application reported. The gap is large, and it is the whole reason this example exists: a kernel
trace accounts for time spent *on* the GPU, and says nothing about time spent waiting for MPI. On
this configuration most of the run is in the gap.

To see it, we need a tool that instruments the host side as well. `rocprof-sys` gathers
function-level performance data by rewriting the binary, which lets it time MPI calls and kernel
launches on the same timeline:

```bash
rocprof-sys-instrument -o shallow_mpi.inst -- ./shallow_mpi
mpirun -n 2 --bind-to none ../gpu_bind.sh rocprof-sys-run -- ./shallow_mpi.inst
```

The first command produces an instrumented copy of the executable, the second runs it. Binary
instrumentation is the recommended path of the several that `rocprof-sys` offers, because it needs no
source changes and still resolves individual functions.

<!-- MEASUREMENT TODO: rocprof-sys timeline screenshot, 2 ranks at 512x512 -->

The timeline shows the MPI calls taking substantially longer than the kernels they surround. The
`hipDeviceSynchronize()` at the top of `exchange_y_halos` makes this worse than it needs to be, since
it drains the whole device before the first byte moves, but the dominant term is simply that there is
too little work per rank.

## What we learned, and what to do about it

Two independent measurements point the same way. The scaling study says the problem is too small to
divide, and the timeline says communication costs more than the compute it feeds. Both have the same
cheapest possible experiment behind them: give each rank more work to do, and see whether the
communication cost becomes a tolerable fraction of it.

So grow the global domain from 512x512 to 8192x8192, a 256x increase in cells. If the diagnosis is
right, throughput per cell should rise sharply on a single GPU, and the scaling efficiency should
recover at the same time.

Continue to [`1_larger_domain`](../1_larger_domain), and note that the entire change is two lines:

```bash
diff ../0_baseline/shallow_mpi.hip ../1_larger_domain/shallow_mpi.hip
```
