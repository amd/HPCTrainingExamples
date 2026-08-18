
# Advanced-level Profiling: Multi-GPU 2D Shallow-Water Solver

README.md from `HPCTrainingExamples/Profiling-by-example/shallow-water/advanced` from the Training Examples repository.

This example is a guided, hands-on walkthrough of profiling and optimizing an MPI application that
runs across several AMD GPUs. Like its
[novice-level counterpart](../novice), it starts from a straightforward implementation and improves
it one step at a time, where every step is motivated by something a profiling tool told us. The
difference is that time can now be lost in two places rather than one, on the GPU and on the
network, and a large part of the work is deciding which of the two to attack.

"Advanced" describes the starting assumptions, not the difficulty:

- The application already runs as several MPI processes, each driving a GPU, and you understand the
  basic purpose of each kernel.
- Terms like "latency-bound" and "memory-bound" mean something to you, and you have used a profiler
  on a single-process application before. If not, work through the [novice](../novice) example first.

By the end you should be able to profile every rank of a multi-process run, measure how much of the
time goes into communication rather than computation, read a thread trace down to the level of
individual instructions, and recognize when a performance problem can only be fixed by changing the
decomposition rather than the kernels.

## The application

The solver is the same one the novice example uses: the 2D shallow-water equations over a flat bed,
central finite differences in space, a classical four-stage Runge-Kutta scheme in time, a light
Laplacian viscosity term for robustness, and single precision throughout. What is new is that the
global domain is now split across MPI ranks, so each rank owns a subdomain surrounded by a one-cell
ghost ring that has to be refreshed from its neighbours before every RK4 stage.

The decomposition is a **1D slab in the Y direction** for stages 0 through 5: every rank keeps all
`NXG` columns and a contiguous band of rows. Ghost cells on the left and right are still filled
locally by the reflective boundary condition, while the bottom and top ghost rows come from the
neighbouring ranks. Stage 6 replaces this with a 2D tiling, for reasons that only become visible
once enough ranks are involved.

The solver is made of the same five kernels, plus the halo machinery that appears from stage 5:

| Kernel | Role |
|---|---|
| `init_gaussian` | Sets the initial state: flat water plus a Gaussian bump in `h` |
| `apply_reflect_bc_x_yphys` | Fills the ghost cells that lie on a physical wall, leaving the rest to the halo exchange |
| `compute_rhs` | Evaluates the right-hand side, the flux divergence plus viscosity. This is the hot kernel |
| `update_stage` | Forms an intermediate RK4 state, `y + a*dt*k` |
| `final_update` | Combines the four RK4 slopes into the new solution |
| `pack_y_halos`, `unpack_y_halos` | From stage 5, gather the outgoing ghost data into one contiguous message per neighbour and scatter the incoming data back |

The halo exchange itself is GPU-aware: MPI is handed device pointers directly, so the ghost data
never travels through host memory. Set `GPU_AWARE_MPI` to 0 at the top of the source to fall back on
pinned host bounce buffers, which is a useful comparison to make if your MPI is not ROCm-aware.

Each run prints its own timing and two correctness checks, so you can confirm that an optimization
did not change the answer:

```
MPI ranks: 4  |  GPUs detected: 1
Domain: 8192x8192 (global), steps=500, dt=0.0728643
Elapsed (max over ranks): 1.075 s  |  Throughput: 124892.14 MCUPS
Mass: initial=6.710936665e+07, final=6.710936646e+07, rel.err=2.929e-09
Min(h) after run: 0.981776
```

One detected GPU on a four-GPU run is correct, not a misconfiguration: `gpu_bind.sh` gives each rank
a single visible device, so the count each rank reports is its own and not the node's. A run that
reports 4 here has its ranks sharing devices, and the affinity section below applies.

The performance metric is MCUPS, millions of cell updates per second over the whole global domain,
counting all four RK4 stages. Higher is better. Because the global domain is fixed, MCUPS can be
compared directly across rank counts, which is what makes the scaling efficiencies below meaningful.
The accuracy checks are global: mass is summed over all ranks and the minimum water depth is reduced
across them. Both matter more here than in the single-GPU case, because the most common way to break
a halo exchange is to send data that is subtly stale, and mass conservation is what catches it.

## The optimization stages

Each sub-directory is one iteration of the profile, analyze, optimize, validate loop. Every stage
builds and runs on its own, so you can reproduce each number without editing any source. Work
through them in order.

| Stage | Tool that motivated it | Finding | Change |
|---|---|---|---|
| [`0_baseline`](0_baseline) | `rocprofv3` per-rank kernel trace, `rocprof-sys` | Adding GPUs makes the code *slower*; the halo exchange costs more than the compute it feeds | none, this is the starting point |
| [`1_larger_domain`](1_larger_domain) | same | Each subdomain is far too small to fill a GPU or to amortize its own halo | Domain 512x512 to 8192x8192 |
| [`2_block_32x32`](2_block_32x32) | `rocprofv3` `VALUBusy` | Vector ALUs busy only half the time; a larger tile caches the stencil better | Block size 16x16 to 32x32 |
| [`3_block_64x4`](3_block_64x4) | `rocprofv3` `VALUBusy`, `OccupancyPercent` | 32x32 raised `VALUBusy` but cost occupancy; a wide, short tile recovers both | Block size 32x32 to 64x4 |
| [`4_vectorized_loads`](4_vectorized_loads) | thread trace, roofline | Three separate global loads per cell for `h`, `hu` and `hv` dominate the instruction timeline | One `float4` load per array, cache-line-aligned row pitch |
| [`5_halo_pipeline`](5_halo_pipeline) | `rocprofv3` at 4 ranks, `rocprof-sys` | Communication is now the limiter: six blocking `MPI_Sendrecv` calls and a device-wide sync per stage | One fused non-blocking message per neighbour, overlapped with interior compute |
| [`6_2d_decomposition`](6_2d_decomposition) | `rocprof-sys` NIC counters, arithmetic | A slab's halo volume per rank is constant in the rank count; a tile's shrinks as its square root | 2D Cartesian process grid |

Throughput in MCUPS, and scaling efficiency against the same stage's own single-GPU run:

| Stage | 1 GPU | 2 GPUs | 4 GPUs | Efficiency at 4 |
|---|---|---|---|---|
| `0_baseline` | 8309 | 4950 | 3007 | 9 percent |
| `1_larger_domain` | 22229 | 43365 | 79749 | 90 percent |
| `2_block_32x32` | 28919 | 58032 | 105163 | 91 percent |
| `3_block_64x4` | 33816 | 67153 | 125015 | 92 percent |
| `4_vectorized_loads` | 36089 | 70413 | 124868 | 87 percent |
| `5_halo_pipeline` | 36127 | 69515 | 136294 | 94 percent |
| `6_2d_decomposition` | 36129 | 69001 | 131756 | 91 percent |

Read the table in two directions, because that is the whole point of the exercise. Reading down a
column shows what the optimizations did to raw speed: on one GPU the solver goes from 22229 to 36127
MCUPS between stages 1 and 5, a **1.63x** gain that comes entirely from kernel work. Reading across a
row shows what they did to scalability, which is a separate question with separate answers. Stages 2
and 3 speed up the kernels without disturbing efficiency at all, which holds at 90 to 92 percent.
Stage 4 is where the trade becomes visible: it makes the compute fast enough that communication is a
significant share of the step, so its 1.07x single-GPU gain arrives as 1.00x at four GPUs and
efficiency falls to 87 percent. Stage 5 attacks that share directly, recovering 7 points of
efficiency and 1.09x of four-GPU throughput without making a single kernel faster.

Stage 0 deserves its own warning. It is the only stage where adding GPUs actively hurts, and it does
so dramatically: 8309 MCUPS on one GPU becomes 3007 on four. A 512x512 domain split four ways gives
each rank a 512x128 band, which is both too little work to fill an MI300A and too little interior to
justify exchanging its own boundary. Any scaling study that starts from a problem this small will
measure the decomposition rather than the solver.

Stage 6 is the one stage that does not pay off at this scale, and it is kept for that reason. A 2x2
tiling of four ranks trades a contiguous row halo for a strided column halo, and with only four GPUs
inside a single node there is not enough network cost for the smaller halo volume to repay that. The
argument for tiling is asymptotic, and the crossover lies beyond the rank counts reachable here.

All numbers quoted in this tutorial were measured on **MI300A** nodes in SPX mode, four GPUs per
node, with ROCm 7.14 and Open MPI 5.0.10, each figure the median of three runs on an exclusive node
through `gpu_bind.sh`. Run-to-run spread was under 2 percent, so differences of a few percent between
stages are real, but do not read anything into a difference smaller than that. Your absolute numbers
will differ on other hardware, and
part of the point of the exercise is that the best block size, the value of overlapping
communication, and the rank count at which tiling starts to win are all machine-dependent. Treat
every figure as something to re-measure rather than to expect.

## Setup

Get the examples:

```bash
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/Profiling-by-example/shallow-water/advanced
```

The stages are built with `hipcc` against an MPI installation and profiled with `rocprofv3`,
`rocprof-compute` and `rocprof-sys`, all of which ship with ROCm. The modules referenced below rely
on the model installation described in the HPCTrainingDock
[repo](https://github.com/amd/HPCTrainingDock).

```bash
module load rocm openmpi
```

The MPI you use must be built with ROCm support if you want to leave `GPU_AWARE_MPI` at 1, which is
the default and what every number here was measured with. For Open MPI that means UCX with ROCm
enabled.

### A Python environment for `rocprof-compute analyze`

`rocprof-compute` has two modes with different requirements. Its `profile` mode collects counters
and needs nothing beyond ROCm and a base Python installation, but its `analyze` mode, which turns
those counters into tables and roofline plots, is a Python application with pinned dependencies that
ROCm does not install for you. Skip this step and every `analyze` invocation stops with a list like
this instead of a report:

```
ERROR The 'pandas==2.2.3' package was not found in the current execution environment.
ERROR The 'tabulate==0.9.0' package was not found in the current execution environment.
```

The pinned list ships with ROCm, so build a virtual environment from it once, on the login node
where `pip` can reach the package index:

```bash
python3 -m venv ~/rocprof-compute-venv
source ~/rocprof-compute-venv/bin/activate
pip install -r $ROCM_PATH/libexec/rocprofiler-compute/requirements.txt
```

Activate it in any shell where you intend to run `rocprof-compute analyze`.

### Getting GPUs

The stage numbers above use one, two and four GPUs, so ask for a whole node. `--exclusive` is not
optional here, for reasons the next section explains, and it does not by itself put GPUs in the
job's cgroup, so ask for those as well:

```bash
salloc -N 1 -p LocalQ --exclusive --gres=gpu:4 -t 2:00:00
```

Every stage is built the same way, either with the portable Makefile:

```bash
cd 0_baseline
make
mpirun -n 4 --bind-to none ../gpu_bind.sh ./shallow_mpi
```

or with CMake:

```bash
cd 0_baseline
mkdir build && cd build
cmake ..
make
mpirun -n 4 --bind-to none ../gpu_bind.sh ./shallow_mpi
```

`make test` runs the executable on two ranks, and `make test NRANKS=4` on four. Run `make clean`
before `make` to be sure nothing is left over from a previous build.

### Affinity, which decides everything else

Every command in this track launches the solver through `gpu_bind.sh`, a three-line wrapper in this
directory. Get this wrong and no other measurement in the tutorial means anything.

On MI300A each GPU is local to one NUMA node, GPU *i* to node *i*. The wrapper gives each rank one
visible GPU with `ROCR_VISIBLE_DEVICES` and pins the rank's CPUs and memory to that GPU's node with
`numactl`:

```bash
mpirun -n 4 --bind-to none ../gpu_bind.sh ./shallow_mpi
```

`--bind-to none` tells Open MPI to keep its hands off placement so the wrapper owns it. Each rank
still selects its GPU with `hipSetDevice(rank % device_count)`, but with one device visible that
reduces to device 0, which is the GPU the rank was bound to.

Two things go wrong without this. A non-exclusive allocation hands the whole job only a handful of
hardware threads, which every rank then shares, and those threads are remote from most of the GPUs;
every kernel launch and MPI progress call pays for it. The
[CG solver example](../../../MPI-examples/cg-solver-example/CG-GPU/README.md#cpu--numa-affinity-important-for-performance)
measures **~100x** between an unbound, non-exclusive launch and this one, on the same binary. That is
larger than every effect this tutorial goes on to study, so it has to be settled before the first
measurement, not after.

## Tracing three steps instead of five hundred

A full `rocprof-sys` trace of this solver records 500 time steps, each with four RK4 stages and a
halo exchange, on every rank. That is a lot of trace for a run whose steps are all alike, and the
size makes the Perfetto UI slow to load precisely when we want to zoom in on one exchange.

Every stage is therefore annotated with ROCTx ranges, from `rocprofiler-sdk-roctx`, which name the
phases of the run as the source sees them. The nesting mirrors the algorithm:

```
timeloop
└── step_0, step_1, ... step_499       one per time step, named with its index
    ├── rk4_k1 ... rk4_k4             the four Runge-Kutta stages
    │   ├── rhs_interior              stage 5 onward: compute that needs no ghost data
    │   ├── halo_pack                 gather outgoing ghosts into one message per neighbour
    │   ├── halo_mpi_post             MPI_Isend and MPI_Irecv
    │   ├── halo_mpi_wait             MPI_Waitall, where imbalance shows up
    │   ├── halo_unpack
    │   └── rhs_edges                 compute that had to wait for ghost data
    └── final_update
```

Stages 0 through 4 have the simpler shape, with a single `halo_exchange` containing `halo_mpi`
instead of the pack, post, wait and unpack breakdown, because the blocking exchange has no phases to
separate. Stage 6 subdivides `rhs_edges` into `rhs_edge_rows` and `rhs_edge_cols`, matching the two
directions a tile has to exchange in.

The step ranges carry their index because that is what makes them selectable:

```c
char step_name[24];
roctxRangePushA("timeloop");
for (int n = 0; n < NSTEPS; ++n) {
    sprintf(step_name, "step_%d", n);
    roctxRangePushA(step_name);
```

`rocprof-sys-run` can then be told to record only while named regions are open, either with
`--selected-regions` or with `ROCPROFSYS_SELECTED_REGIONS`:

```bash
cd 5_halo_pipeline
mpirun -n 2 --bind-to none ../gpu_bind.sh \
    rocprof-sys-run --selected-regions step_10,step_11,step_12 -- ./shallow_mpi
```

It confirms the filter on startup, which is worth checking before waiting on a long run:

```
[trace_control.cpp:38 trace_control][info] Trace controller: region filter active for regions: [step_10, step_11, step_12]
```

Skipping the first few steps matters: step 0 pays for lazy kernel-code loading and first-touch page
faults, so it is the least representative step in the run. The effect on the trace is large. Two
ranks of `5_halo_pipeline`, measured three ways:

| What was recorded | Filter | Trace per rank |
|---|---|---|
| The whole run | none | 17.1 MB |
| Every halo wait and interior compute, all 500 steps | `rhs_interior,halo_mpi_wait` | 6.85 MB |
| Three time steps, complete | `step_10,step_11,step_12` | 103 KB |

The last trace is **166x** smaller and contains exactly what was asked for: the ranges for steps 10,
11 and 12 and nothing from step 13, with 12 `rhs_interior` and 12 `halo_mpi_wait` regions, being
three steps of four RK4 stages each, against 2000 of them in the unfiltered trace.

Choose the filter to match the question. Naming a phase, as in the middle row, answers "how does
this phase behave over the whole run" and still keeps every step's worth of it. Naming a few steps
answers "what exactly happens inside one step", which is the more common need when reading a
timeline, and it is the cheap one.

## How to read the diffs

The change made at each stage is small and deliberate. To see exactly what moved between two
consecutive stages, diff them:

```bash
diff 2_block_32x32/shallow_mpi.hip 3_block_64x4/shallow_mpi.hip
```

For stages 1, 2 and 3 that diff is one or two lines. That is the point: profiling tells you which
line to change. Stages 4 through 6 are progressively larger rewrites, which is also the point,
because the deeper a problem sits in the structure of the code the more code it takes to fix.

## Start here

Go to [`0_baseline`](0_baseline) and take the first measurement.
