# Rocprofv3 Exercises for HIPStdPar

This exercise shows how to profile a **C++ standard parallelism (HIPStdPar)** application
with `rocprofv3`: timeline tracing, kernel renaming, and hardware counter collection. We use
the [ShallowWater_StdPar](https://github.com/amd/HPCTrainingExamples/tree/main/HIPStdPar/CXX/ShallowWater_StdPar)
example, a 2D shallow water solver that offloads to the GPU through `std::for_each` and
`std::transform_reduce` with `std::execution::par_unseq`.

Why HIPStdPar is particularly interesting to profile: the standard algorithms are lowered to `rocprim` and
`thrust` library kernels, so the kernel names you see in a profile are extremely long, mangled
C++ template names that give no hint about which phase of *your* code they belong to. A single
`std::transform_reduce` even expands to **three** GPU kernels. This exercise teaches you to make
the profile readable with roctx markers and `--kernel-rename`.

This exercise focuses on GPU-only profiling which is more lightweight.
A related example profiles the same app with the ROCm Systems Profiler: [rocprofiler-systems/ShallowWater/HIPStdPar](https://github.com/amd/HPCTrainingExamples/tree/main/rocprofiler-systems/ShallowWater/HIPStdPar) which gives more insights in CPU and GPU interaction.

## Setup environment

This exercise runs on a single MI300A GPU on a compute node.

Allocate a single GPU on one node on a non-default partition (replace the partition name with
one from your system):

```
salloc -N 1 --gpus=1 -p=<your partition>
```

Load a ROCm module and enable unified memory (`XNACK`), which is required for HIPStdPar managed
memory on MI300A:

```
module load module load rocm/therock-dist-linux-gfx94X-dcgpu-7.14.0a20260608  # this exercise was tested with a pre-release version of therock 7.14
export HSA_XNACK=1
```

> **Version note:** HIPStdPar needs a recent ROCm/TheRock build that contains the HIPStdPar
> fixes in TheRock 7.14 pre-release or newer needed to run the profilers as explained here.

## Build and run the unmodified application

Start from the clean, unmodified source:

```bash
cd HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_StdPar
export CXX=amdclang++
make clean && make
./ShallowWater
```

The run should finish in well under a second and conserve the total mass:

```
Iteration:00000, Time:0.000000, Timestep:0.048059 Total mass:399200.000000
...
Iteration:02000, Time:0.441816, Timestep:0.022692 Total mass:399200.000000
 Flow finished in 0.211508 seconds
```

## Step 1: Profile without markers (the problem)

Collect kernel statistics for the unmodified app:

```bash
rocprofv3 --stats --kernel-trace --output-format csv -- ./ShallowWater
```

Open the generated `*_kernel_stats.csv`. Every row is a `rocprim`/`thrust` template name that is
hundreds of characters long, for example:

```bash
"Name","Calls","TotalDurationNs",...
"void thrust::THRUST_200805_400400_NS::hip_rocprim::__parallel_for::kernel<256u, thrust::...for_each_f<range::iterator, ...main::{lambda(int)#9}...> >(...)",...
"void rocprim::ROCPRIM_400400_NS::detail::trampoline_kernel<...reduce_impl<...transform_iterator<main::{lambda(int)#10}...> >(...)",...
"__amd_rocclr_copyBuffer",...
```

Observations:

- You cannot tell which simulation phase (boundary conditions, flux, state update, ...) a kernel
  belongs to.
- One `std::transform_reduce` shows up as **three** kernels (two `trampoline_kernel` reductions
  plus a `__amd_rocclr_copyBuffer`), so per-kernel counts are confusing.

## Step 2: Add roctx markers (the learning step)

We annotate the source with [roctx](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
range markers so each computation phase has a readable name. (Tip: an AI assistant can do the repetitive marker insertion for you in seconds.)

**1. Link the roctx library.** In the `Makefile`, change:

```makefile
CXXFLAGS = -g -O3 -fstrict-aliasing ${STDPAR_FLAGS}
LDFLAGS = -fno-lto -lm
```

to:

```makefile
CXXFLAGS = -g -O3 -fstrict-aliasing ${STDPAR_FLAGS} -I${ROCM_PATH}/include
LDFLAGS = -fno-lto -lm -L$(ROCM_PATH)/lib -lrocprofiler-sdk-roctx
```

**2. Include the roctx header** near the top of `ShallowWater.cpp`:

```cpp
#include <rocprofiler-sdk-roctx/roctx.h>
```

**3. Wrap each phase** with `roctxRangePush("name")` before the region and `roctxRangePop()`
after it. Use these names so the output matches this guide:

| Region name | Code section |
|---|---|
| `init_state` | Initial H, U, V setup (`std::for_each`) |
| `init_total_mass` | Original total mass (`std::transform_reduce`) |
| `init_timestep` | Initial timestep (`std::transform_reduce`) |
| `bc_y` | Y-direction boundary conditions (in burst loop) |
| `bc_x` | X-direction boundary conditions (in burst loop) |
| `calc_timestep` | Timestep recalculation (in burst loop) |
| `flux_x` | X-direction flux (in burst loop) |
| `flux_y` | Y-direction flux (in burst loop) |
| `state_update` | State variable update (in burst loop) |
| `check_mass` | Mass conservation check (after burst loop) |
| `memory_cleanup` | Freeing allocated memory |

For example, the timestep calculation in the burst loop becomes:

```cpp
roctxRangePush("calc_timestep");
deltaT = std::transform_reduce(
  std::execution::par_unseq,
  flatindexrange.begin(), flatindexrange.end(),
  1.0e20,
  [](auto l, auto r){ return std::fmin(l,r); },
  [=](int flatindex) { /* ... timestep calculation ... */ }
);
roctxRangePop();
```

Rebuild and confirm the output is unchanged (markers do not change results):

```
make clean && make
./ShallowWater
```

## Step 3: Timeline trace

Collect a kernel trace including the markers and write a Perfetto trace file:

```bash
rocprofv3 --kernel-trace --marker-trace --output-format pftrace --output-file timeline -- ./ShallowWater
```

This produces `timeline_results.pftrace`. Copy it to your laptop and open it in
[Perfetto UI](https://ui.perfetto.dev/). Use `w`/`s` to zoom and `a`/`d` to
pan. The named roctx regions appear on the CPU timeline and group the GPU kernel dispatches, so
you can finally see which kernels belong to `flux_x`, `state_update`, etc.

```
scp <user>@<host>:<path>/timeline_results.pftrace .
```

<!-- IMAGE: Perfetto timeline with roctx regions grouping the HIPStdPar kernels -->

## Step 4: Rename kernels with the marker names

Now use `--kernel-rename`, which replaces each kernel name with the enclosing roctx range name:

```bash
rocprofv3 --stats --kernel-trace --marker-trace --output-format csv --kernel-rename -- ./ShallowWater
```

The kernel names in the `*_kernel_stats.csv` file are now readable:

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"calc_timestep",6000,29349493,4891.582167,31.75,800,9160,2415.840931
"bc_y",2000,16726779,8363.389500,18.10,6920,9960,685.548138
"state_update",2000,14978512,7489.256000,16.21,6680,541203,11941.465077
"flux_y",2000,12209094,6104.547000,13.21,5160,350282,7706.673181
"flux_x",2000,10748326,5374.163000,11.63,4480,737324,16376.742939
"bc_x",2000,7222356,3611.178000,7.81,2640,4640,339.870308
"init_state",1,961605,961605.000000,1.04,961605,961605,0.00000000e+00
"check_mass",60,206720,3445.333333,0.2237,1520,6200,1561.719464
"init_timestep",3,15000,5000.000000,0.0162,2600,8840,3360.000000
"init_total_mass",3,11120,3706.666667,0.0120,2120,5400,1642.599566
```

> **CAREFUL:** `--kernel-rename` renames **every** kernel inside a roctx range to that range's
> name. All kernels in one range are merged into a single entry. That is exactly why
> `calc_timestep` reports **6000** calls here: the range runs 100 times and each
> `std::transform_reduce` launches 3 individual kernels (2000 x 3 = 6000). This is convenient for grouping by phase,
> but you lose the per-kernel breakdown. Use Step 1 (no rename) when you need the individual
> kernels, and choose your marker granularity accordingly: one marker per kernel keeps the rename
> one-to-one, currently there is no feature (yet) to rename kernels within a region with
> individual names, thus be mindful about the 3 reduction kernels we saw earlier.

### Limiting collection to a range of iterations (optional)

For long runs, you can restrict data collection to a representative window of iterations or a
certain part of the code instead of tracing the whole run.
Add `roctxProfilerPause(0)` at the start of `main()` and toggle around a target
iteration range, e.g. resume at iteration 500 and pause again at 600:

```cpp
roctxProfilerPause(0);                       // pause at the start of main()
...
if (n == 500) roctxProfilerResume(0);        // inside the iteration loop
...
if (n == 600) roctxProfilerPause(0);
```

`rocprofv3` honors these calls, so only the kernels dispatched between resume and pause are
recorded (the counts above reflect such a window). This keeps trace files small and focused.

## Step 5: Hardware counters

Read the counters available for this GPU (look for the `gfx942` section):

```bash
less $ROCM_PATH/lib/rocprofiler/gfx_metrics.xml
```

Create an `input_counters.txt`. Each `pmc:` line is collected in a separate pass. Keep only a few
counters per line: many derived metrics (e.g. `VALUUtilization`, `MeanOccupancyPerCU`) expand to
several base counters and a single line can exceed the hardware counter slots, which fails with
`error code 38: Request exceeds the capabilities of the hardware to collect`.
A working set on MI300A (gfx942) is for example:

```
pmc: VALUBusy FetchSize
pmc: WriteSize MemUnitStalled
pmc: CU_OCCUPANCY GPU_UTIL
```
For instance, combining `FetchSize` and `WriteSize` on the same line will not work.
Collect the counters. Combine with `--kernel-rename` to get per-phase counter values:

```bash
rocprofv3 -i input_counters.txt --kernel-trace --output-format csv --kernel-rename -- ./ShallowWater
```

`rocprofv3` runs one pass per `pmc:` line and writes the results into `pmc_1`, `pmc_2`, `pmc_3`
subdirectories. Each `*_counter_collection.csv` has one row per (kernel, counter); with the
markers the `Kernel_Name` column shows the readable region names:

```
...,"Kernel_Name",...,"Counter_Name","Counter_Value",...
...,"bc_y",...,"FetchSize",85.687500,...
...,"calc_timestep",...,"FetchSize",2417.250000,...
...,"calc_timestep",...,"VALUBusy",1.006513,...
```


## Known issues

- **`implib-gen: libamd_comgr.so.3` abort / hang:** seen with older versions  (e.g. afar 23.1.0). The
  profiler aborts with `SIGABRT` and hangs at the end of the run. Use a newer ROCm/TheRock build
  with the HIPStdPar fixes (TheRock 7.14).

## Interesting References

- [rocprofv3 documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [Perfetto UI](https://ui.perfetto.dev/)
- [ShallowWater_StdPar example](https://github.com/amd/HPCTrainingExamples/tree/main/HIPStdPar/CXX/ShallowWater_StdPar)
- [rocprof-sys ShallowWater HIPStdPar example](https://github.com/amd/HPCTrainingExamples/tree/main/rocprofiler-systems/ShallowWater/HIPStdPar)
- [Rocprofv3 HIP example](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprofv3/HIP)
- [Rocprofv3 OpenMP example](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprofv3/OpenMP)
