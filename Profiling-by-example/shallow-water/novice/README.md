
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

Together the five stages give a cumulative 5.50x speedup, from 6282 to 34551 MCUPS.

All numbers quoted in this tutorial, both timings and counters, were measured on a single
MI300A in SPX mode, taking the median of three runs. Your
absolute numbers will differ on other hardware, and part of the point of the exercise is that the
best block size and the size of each speedup are architecture-dependent. The relative trends should
mostly hold, but treat every figure as something to re-measure rather than to expect.

## Setup

Get the examples:

```bash
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/Profiling-by-example/shallow-water/novice
```

The modules referenced below rely on the model installation described in the HPCTrainingDock
[repo](https://github.com/amd/HPCTrainingDock).

```bash
module load rocm
```

That one module covers almost everything the tutorial uses:

| Tool | Used for | Extra setup |
|---|---|---|
| `hipcc` | Building every stage | none |
| `rocprofv3` | Kernel and HIP API traces, and the `OccupancyPercent` and `VALUBusy` counters | none |
| `rocpd2pftrace` | Turning a run's rocpd database into a Perfetto trace | none |
| `rocpd2csv`, `rocpd2summary` | Turning the same database into a CSV or a summary table | pandas |
| `rocprof-compute profile` | Collecting the counters behind the roofline | none |
| `rocprof-compute analyze` | Reporting those counters as tables and plots | its own pinned Python packages |
| Roofline Extractor `profile_app.py` | The roofline plot shown at every stage | the code, plus its own Python packages |

Every row marked "none" is ready the moment `module load rocm` succeeds. That includes
`rocprof-compute profile`, so it is only the reporting half of `rocprof-compute` that needs
anything more. `rocpd2csv` and `rocpd2summary` want any reasonably recent pandas, which many
systems already provide; without it they print `Error: No module named 'pandas'` and write nothing.
The last two rows are the ones that need environments of their own, which the next section sets up.
The viewers used later need nothing installed on the cluster either, since Perfetto runs in a
browser and ROCm Optiq is a desktop application.

## Tools with Python dependencies

Exactly two things in this tutorial need a virtual environment of their own: the Roofline Extractor
and `rocprof-compute analyze`. Neither is needed to build the code, run `rocprofv3`, run
`rocprof-compute profile`, or export a trace with `rocpd2pftrace`.

### Roofline Extractor

The Roofline Extractor is not part of ROCm, so both the code and its Python dependencies have to be
installed once, on a login node:

```bash
git clone https://github.com/AMD-HPC/rooflineExtractor.git ~/rooflineExtractor
export ROOFLINE_EXTRACTOR=$HOME/rooflineExtractor
../setup_roofline_extractor_venv.sh
source ~/roofline-venv/bin/activate
```

Activate it in any shell where you intend to run `profile_app.py`. On AAC6, point
`ROOFLINE_EXTRACTOR` at the site install and let `env.sh` activate `~/roofline-venv`; see
[AAC6.md](../AAC6.md). Some sites ship a pre-built install or a module wrapper
(`roofline-extractor-profile`); those call the same `profile_app.py` with Python already configured.

### `rocprof-compute analyze`

`rocprof-compute` is a Python application throughout, but only its `analyze` mode has pinned
dependencies that ROCm does not install for you, so without them `analyze` stops with a list of
missing packages instead of a report while `profile` carries on working. The pinned list itself
ships with ROCm, headed `# Analyze mode only.`, so build a virtual environment from it once, on the
login node where `pip` can reach the package index:

```bash
python3 -m venv ~/rocprof-compute-venv
source ~/rocprof-compute-venv/bin/activate
pip install -r $ROCM_PATH/libexec/rocprofiler-compute/requirements.txt
```

Activate it in any shell where you intend to run `analyze`. Leaving it active for the whole session
is simplest, since it does not interfere with `profile` mode, `rocprofv3`, or building the code.

## Roofline plots

Every stage shows its roofline twice, once with the Roofline Extractor and once with
`rocprof-compute`. Novice `profile.sh` runs one backend per job; set
`ROOFLINE_TOOL` in `env.sh` to `extractor` (default) or `rocprof-compute`.

The extractor is invoked as `profile_app.py` (Option 1 in the upstream README), with its
[environment](#roofline-extractor) active:

```bash
python3 "$ROOFLINE_EXTRACTOR/profile_app.py" -o roofline_out --arch MI300A -- ./shallow
```

Set `--arch` to match your GPU (`MI300A`, `MI300X`, `MI250X`, and others listed in the extractor
README). The extractor writes the plot itself, as `roofline_out/counters.html`; its options and
outputs are documented in the
[Roofline Extractor repo](https://github.com/AMD-HPC/rooflineExtractor).

The `rocprof-compute` roofline is collected and then reported, and only the second command needs
the [`rocprof-compute analyze` environment](#rocprof-compute-analyze):

```bash
rocprof-compute profile -n 0_baseline --roof-only --device 0 -k compute_rhs --iteration-multiplexing -- ./shallow
rocprof-compute analyze -p workloads/0_baseline/0
```

The extractor is an AMD research project whose capabilities are being integrated into
`rocprof-compute` for a future release, so it produces the plots in this tutorial for now and the
`rocprof-compute` commands are given alongside for when that lands.

## Getting a GPU and building

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

## Start here

Go to [`0_baseline`](0_baseline) and take the first measurement.
