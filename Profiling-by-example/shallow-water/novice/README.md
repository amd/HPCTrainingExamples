
# Novice-level Profiling: 2D Shallow-Water Solver

README.md from `HPCTrainingExamples/Profiling-by-example/shallow-water/novice` from the Training Examples repository.

This example is a guided, hands-on walkthrough of profiling and optimizing a HIP application on
AMD GPUs. Rather than presenting a fast code and explaining why it is fast, it starts from a
straightforward implementation and improves it one step at a time, where every step is motivated by
something a profiling tool told us. The companion material is the ROCm blog article on
[novice-level profiling](https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/novice/README.html).

"Novice" here describes the starting assumptions, not the difficulty:

- The application already uses the GPU, and you understand the basic purpose of each kernel.
- You know that data transfers between CPU and GPU cost time, even if terms like "latency-bound"
  and "memory-bound" are not yet familiar.

By the end you should be able to measure where time is spent on the GPU, pinpoint basic limits such
as poor occupancy or heavy memory traffic, and explain why performance can differ between GPU
architectures.

## The application

The code solves the 2D shallow-water equations over a flat bed. These describe the depth `h` and the
momentum components `hu` and `hv` of a thin layer of fluid, and they are the standard model for
phenomena like tsunami propagation and dam breaks. Space is discretized with central finite
differences on a uniform grid, time is advanced with a classical four-stage Runge-Kutta (RK4)
scheme, and a light Laplacian viscosity term is added for robustness. Everything is single
precision.

The domain carries a one-cell ghost ring, and reflective boundary conditions are imposed on it by
mirroring the interior values and flipping the sign of the momentum component normal to the wall.
The initial condition is still water with a Gaussian bump in the middle, which then spreads outward
and reflects off the walls.

The solver is made of five kernels:

| Kernel | Role |
|---|---|
| `init_gaussian` | Sets the initial state: flat water plus a Gaussian bump in `h` |
| `apply_reflect_bc` | Fills the ghost ring to impose reflective walls |
| `compute_rhs` | Evaluates the right-hand side, the flux divergence plus viscosity. This is the hot kernel |
| `update_stage` | Forms an intermediate RK4 state, `y + a*dt*k` |
| `final_update` | Combines the four RK4 slopes into the new solution |

Each run prints its own timing and two correctness checks, so you can confirm that an optimization
did not change the answer:

```
Domain: 512x512, steps=500, dt=0.0728643
Elapsed: 0.083 s  |  Throughput (including RK4 stages): 6282.26 MCUPS
Mass: initial=2.626466547e+05, final=2.626464583e+05, rel.err=7.477e-07
Min(h) after run: 0.981776
```

The performance metric is MCUPS, millions of cell updates per second, counting all four RK4 stages.
Higher is better. The accuracy checks are mass conservation, which should stay near zero relative
error, and the minimum water depth, which must never go negative.

This is deliberately a playground rather than a production application. The kernels are simple
enough to reason about in one sitting, which is what makes them useful for learning to read profiler
output.

## The optimization stages

Each sub-directory is one iteration of the profile, analyze, optimize, validate loop. Every stage
builds and runs on its own, so you can reproduce each number without editing any source. Work
through them in order.

| Stage | Tool that motivated it | Finding | Change | MCUPS | Speedup |
|---|---|---|---|---|---|
| [`0_baseline`](0_baseline) | `rocprofv3` kernel trace, `OccupancyPercent`, `rocprof-compute` roofline | `compute_rhs` dominates; occupancy only 24 percent; far below the roofline | none, this is the starting point | 6282.26 | -- |
| [`1_larger_domain`](1_larger_domain) | same | Not enough parallelism to fill the GPU or hide latency | Domain 512x512 to 2048x2048 | 19638.63 | 3.13x |
| [`2_no_device_sync`](2_no_device_sync) | `rocprofv3` HIP API trace | Gaps between kernels caused by `hipDeviceSynchronize()` that a single stream already guarantees | Remove the redundant synchronizations | 21204.86 | 1.08x |
| [`3_block_32x32`](3_block_32x32) | `rocprofv3` `VALUBusy` | Vector ALUs busy only 47 percent of the time; a larger tile caches the stencil better | Block size 16x16 to 32x32 | 29400.84 | 1.39x |
| [`4_block_64x4`](4_block_64x4) | `rocprofv3` `VALUBusy`, `OccupancyPercent` | 32x32 raised `VALUBusy` but cost occupancy; a wide, short tile recovers both | Block size 32x32 to 64x4 | 34551.31 | 1.18x |

Together the five stages give a cumulative **5.50x** speedup, from 6282 to 34551 MCUPS.

All numbers quoted in this tutorial, both timings and counters, were measured on a single
**MI300A** in SPX mode, taking the median of three runs. Your
absolute numbers will differ on other hardware, and part of the point of the exercise is that the
best block size and the size of each speedup are architecture-dependent. The relative trends should
mostly hold, but treat every figure as something to re-measure rather than to expect.

## Setup

Get the examples:

```bash
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/Profiling-by-example/shallow-water/novice
```

The stages are built with `hipcc` and profiled with `rocprofv3` and `rocprof-compute`, all of which
ship with ROCm. The modules referenced below rely on the model installation described in the
HPCTrainingDock [repo](https://github.com/amd/HPCTrainingDock).

```bash
module load rocm
```

### Getting a GPU

Get an interactive session with one GPU before building and running:

```bash
salloc -N 1 -p LocalQ --gpus=1 -t 30:00
```

Every stage is built the same way, either with the portable Makefile:

```bash
cd 0_baseline
make
./shallow
```

or with CMake:

```bash
cd 0_baseline
mkdir build && cd build
cmake ..
make
./shallow
```

Run `make clean` before `make` to be sure nothing is left over from a previous build.

## How to read the diffs

The change made at each stage is small and deliberate. To see exactly what moved between two
consecutive stages, diff them:

```bash
diff 2_no_device_sync/shallow.hip 3_block_32x32/shallow.hip
```

For stages 1, 3 and 4 that diff is a single line. That is the point: profiling tells you which one
line to change.

## Profiling tool setup

### Roofline plots

Every stage shows its roofline twice, once with the Roofline Extractor:

```bash
module load rocm roofline-extractor
roofline-extractor-profile -o roofline_out --arch MI300A -- ./shallow
```

and once with `rocprof-compute`, whose `analyze` step needs the Python environment below:

```bash
rocprof-compute profile -n 0_baseline --roof-only --device 0 -k compute_rhs --iteration-multiplexing -- ./shallow
rocprof-compute analyze -p workloads/0_baseline/0
```

The extractor writes the plot itself, as `roofline_out/counters.html`; its options and outputs are
documented in the [Roofline Extractor repo](https://github.com/AMD-HPC/rooflineExtractor). It is an
AMD research project whose capabilities are being integrated into `rocprof-compute` for a future
release, so it produces the plots in this tutorial for now and the `rocprof-compute` commands are
given alongside for when that lands.

### A Python environment for `rocprof-compute analyze`

`rocprof-compute analyze` is a Python application with pinned dependencies that ROCm does not
install for you, so without them it stops with a list of missing packages instead of a report. The
pinned list ships with ROCm, so build a virtual environment from it once, on the login node where
`pip` can reach the package index:

```bash
python3 -m venv ~/rocprof-compute-venv
source ~/rocprof-compute-venv/bin/activate
pip install -r $ROCM_PATH/libexec/rocprofiler-compute/requirements.txt
```

Activate it in any shell where you intend to run `analyze`. Leaving it active for the whole session
is simplest, since it does not interfere with `profile` mode, `rocprofv3`, or building the code.

## Start here

Go to [`0_baseline`](0_baseline) and take the first measurement.
