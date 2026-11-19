Note: The following was tested with a therock 7.14 preview build from 6th June 2026 on MI300A (https://rocm.nightlies.amd.com/tarball/) with a module analogous to therock preview builds on aac6.

# Profiling the ShallowWater HIPStdPar Example with `rocprof-sys`

This exercise demonstrates how to use `rocprof-sys` (ROCm Systems Profiler) to profile a C++ standard parallelism (stdpar) application targeting AMD GPUs. We use the [ShallowWater_StdPar](https://github.com/amd/HPCTrainingExamples/tree/main/HIPStdPar/CXX/ShallowWater_StdPar) example, a 2D shallow water equation solver that uses `std::for_each` and `std::transform_reduce` with `std::execution::par_unseq` to offload computation to the GPU.

The exercise progressively introduces:
1. Runtime-only profiling (no code changes)
2. Binary instrumentation for host function visibility
3. roctx markers for labeling application phases
4. Pause/resume to limit data collection
5. Custom trace flags for detailed GPU activity analysis

## Environment setup

Load a ROCm module that contains `rocprof-sys`. The exact module name depends on your system.

```
module load rocm/therock-dist-linux-gfx94X-dcgpu-7.14.0a20260608
```

Verify that the tools are available:

```
rocprof-sys-run --version
rocprof-sys-instrument --version
```

You can inspect available presets with:

```
rocprof-sys-run --list-presets
```

## Build and run the application

Navigate to the ShallowWater stdpar example and build the original code:

```
cd HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_StdPar
export CXX=amdclang++
make clean && make
```

Run the application to verify it works correctly:

```
HSA_XNACK=1 ./ShallowWater
```

The output should show iterations progressing with conservation of mass:

```
Iteration:00000, Time:0.000000, Timestep:0.048059 Total mass:399200.000000
...
Iteration:02000, Time:0.441816, Timestep:0.022692 Total mass:399200.000000
 Flow finished in 0.211508 seconds
```

Note: `HSA_XNACK=1` is required on MI300A for stdpar managed memory. If your GPU does not support XNACK, the Makefile will add `--hipstdpar-interpose-alloc` automatically.

## Runtime-only profiling (no instrumentation, no markers)

The simplest way to profile is to run the application through `rocprof-sys-run` without any binary instrumentation or source code changes. This traces ROCm API calls, GPU kernel dispatches, memory copies, and GPU device metrics.

```
HSA_XNACK=1 rocprof-sys-run --preset=trace-gpu \
  -o rocprofsys-nomarkers \
  -- ./ShallowWater
```

You can inspect what the `trace-gpu` preset enables:

```
rocprof-sys-run --explain=trace-gpu
```

It sets: tracing enabled, HIP runtime API, kernel dispatch, marker API, memory copy, scratch memory, AMD SMI device metrics (busy, temp, power, mem_usage), and process sampling.

### View the trace

The output directory contains a Perfetto trace file:

```
ls rocprofsys-nomarkers/
```

Look for `perfetto-trace-<pid>.proto`. Copy this file to your local machine and open it in the [Perfetto UI](https://ui.perfetto.dev) (use Chrome browser). Navigate with WASD keys to zoom and move.

<img width="1550" height="808" alt="image" src="https://github.com/user-attachments/assets/0d4c6f9a-1cc6-45bb-928d-f21154e3fe79" />

**What you see:** GPU kernel dispatch events, HIP runtime API calls, and GPU device metrics. However, all GPU kernels have auto-generated names and there is no indication of which application phase (boundary conditions, flux calculation, state update, etc.) each kernel belongs to.

<img width="1555" height="868" alt="image" src="https://github.com/user-attachments/assets/9b2af8a7-0774-4661-ac06-ac371c19708a" />

## Binary instrumentation (no markers yet)

To also see host-side function entry and exit in the trace, use `rocprof-sys-instrument` to create an instrumented binary:

```
rocprof-sys-instrument -o ShallowWater.inst -i 16 -- ./ShallowWater
```

This uses dynamic binary rewriting to insert profiling hooks into host functions. By default, only functions with more than 1024 instructions are instrumented.
In this easy example, only a single function would be instrumented with this default, so we lower this threshold with the `-i 16` argument to 16 instructions.
You can also select specific functions explicitly with `--function-include`.

Now run the instrumented binary:

```
HSA_XNACK=1 rocprof-sys-run --preset=trace-gpu \
  -o rocprofsys-inst-nomarkers \
  -- ./ShallowWater.inst
```

### View the trace

Open the generated `perfetto-trace-<pid>.proto` in [Perfetto UI](https://ui.perfetto.dev).

**What changed:** In addition to GPU kernel dispatches and HIP API calls, you now see host function entry/exit spans on the CPU thread timeline. This helps understand the host-side call structure.

However, the host functions have compiler-generated names that may not clearly convey the application logic. This is where roctx markers become useful -- they let you label high-level computation phases with meaningful names.

## Adding roctx markers

roctx (ROCm Tooling Extension) markers let you annotate your source code with named regions that appear in the trace. This makes it easy to see which computation phase each GPU kernel belongs to.

### Step 1: Update the Makefile

Add the roctx include path and link library. Change these two lines:

```makefile
CXXFLAGS = -g -O3 -fstrict-aliasing ${STDPAR_FLAGS}
LDFLAGS = -fno-lto -lm
```

to:

```makefile
CXXFLAGS = -g -O3 -fstrict-aliasing ${STDPAR_FLAGS} -I${ROCM_PATH}/include
LDFLAGS = -fno-lto -lm -L$(ROCM_PATH)/lib -lrocprofiler-sdk-roctx
```

### Step 2: Add the roctx include

At the top of `ShallowWater.cpp`, add the roctx header:

```cpp
#include <rocprofiler-sdk-roctx/roctx.h>
```

### Step 3: Wrap computation phases with roctx markers

Use `roctxRangePush("name")` before a region and `roctxRangePop()` after it. For example, to mark the timestep calculation inside the burst loop:

```cpp
roctxRangePush("calc_timestep");
deltaT = std::transform_reduce(
  std::execution::par_unseq,
  flatindexrange.begin(),
  flatindexrange.end(),
  1.0e20,
  [](auto l, auto r){ return std::fmin(l,r); },
  [=](int flatindex) {
    // ... timestep calculation ...
  }
);
roctxRangePop();
```

Apply this pattern to all computation phases. The recommended regions to mark are:

| Region name | Code section |
|---|---|
| `init_state` | Initial H, U, V setup |
| `init_total_mass` | Original total mass calculation |
| `init_timestep` | Initial timestep calculation |
| `bc_y` | Y-direction boundary conditions (inside burst loop) |
| `bc_x` | X-direction boundary conditions (inside burst loop) |
| `calc_timestep` | Timestep recalculation (inside burst loop) |
| `flux_x` | X-direction flux computation (inside burst loop) |
| `flux_y` | Y-direction flux computation (inside burst loop) |
| `state_update` | State variable update (inside burst loop) |
| `check_mass` | Mass conservation check (after burst loop) |
| `memory_cleanup` | Freeing allocated memory |

Don't forget: your favorite AI assistent is likely happy to help you with such annoying tasks. It does not need frontier large language models to insert roctx markers in a codebase in a few seconds.

### Step 4: Rebuild

```
make clean && make
```

Run the application to verify markers do not change behavior:

```
HSA_XNACK=1 ./ShallowWater
```

The output should be identical to the baseline run.

## Profiling with instrumentation and markers

Re-instrument the updated binary and run with profiling:

```
rocprof-sys-instrument -o ShallowWater.inst -- ./ShallowWater
HSA_XNACK=1 rocprof-sys-run --preset=trace-gpu \
  -o rocprofsys-markers \
  -- ./ShallowWater.inst
```

### View the trace

Open the generated `perfetto-trace-<pid>.proto` in [Perfetto UI](https://ui.perfetto.dev).
<img width="887" height="537" alt="image" src="https://github.com/user-attachments/assets/803105a2-2e3e-4c7d-8a7f-8296e1c8688b" />

**What changed:** The trace now shows named roctx regions (`bc_y`, `calc_timestep`, `flux_x`, etc.) on the CPU thread timeline. GPU kernel dispatches are visually grouped under these regions, making it clear which kernel belongs to which computation phase. This is especially valuable for identifying performance bottlenecks in specific phases.

## Limiting data collection with `roctxProfilerPause`/`roctxProfilerResume`

For a long-running application with many iterations, collecting a full trace can produce very large files. You can use `roctxProfilerPause(0)` and `roctxProfilerResume(0)` to only collect data for specific iterations of interest.

### Step 1: Add pause/resume to the source

At the beginning of `main()`, pause profiling so data collection is disabled at the start:

```cpp
int main(int argc, char *argv[])
{
  roctxProfilerPause(0);
  // ... existing code ...
```

Inside the outer iteration loop (`for (int n = 0; n < ntimes; )`), add conditions to resume and pause around a targeted iteration range. For example, to profile only iterations 500-600, add just before the burst loop:

```cpp
for (int n = 0; n < ntimes; ) {

    if (n == 500) roctxProfilerResume(0);

    for (int ib=0; ib<nburst; ib++){
      // ... burst loop body ...
    } // burst loop

    // ... check_mass, print, etc. ...

    if (n == 600) roctxProfilerPause(0);

  } // End of iteration loop
```

### Step 2: Rebuild and re-instrument

```
make clean && make
rocprof-sys-instrument -o ShallowWater.inst -- ./ShallowWater
```

### Step 3: Run with pause and resume markers

```
HSA_XNACK=1 rocprof-sys-run --preset=trace-gpu \
  -o rocprofsys-limited \
  -- ./ShallowWater.inst
```


### View the trace

Open the generated trace in [Perfetto UI](https://ui.perfetto.dev). Only the iterations between the resume and pause calls are captured, resulting in a much smaller and more focused trace file.

## Customizing the trace for GPU activity

The `trace-gpu` preset is a good starting point, but you can go further by explicitly specifying trace options. To see HSA-level synchronization and unified memory events:

```
HSA_XNACK=1 rocprof-sys-run --trace \
  --rocm=hip,kernel,memory,hsa,marker \
  --gpu \
  --use-unified-memory-profiling \
  -o rocprofsys-custom \
  -- ./ShallowWater.inst
```

What each flag does:

| Flag | Purpose |
|---|---|
| `--trace` | Enable trace output |
| `--rocm=hip,kernel,memory,hsa,marker` | Trace HIP runtime API, kernel dispatches, memory copies, HSA core API, and roctx markers |
| `--gpu` | Enable AMD SMI device metrics (GPU temperature, power, utilization, memory usage) |
| `--use-unified-memory-profiling` | Enable KFD page fault and page migration event reporting (requires `HSA_XNACK=1`) |

The `--use-unified-memory-profiling` flag shows KFD page fault and migration events, which can explain unexpected latency during memory allocation, deallocation, or first-touch access patterns.


### View the trace

Open the generated trace in [Perfetto UI](https://ui.perfetto.dev).

**What to look for:**

- **roctx marker rows** (visible before): Named regions showing application phases.
- **Kernel dispatch row** (visible before): GPU kernel execution timeline.
- **HIP API row** (visible before): HIP runtime API calls on the host.
- **HSA API row** (new): HSA-level calls including `hsa_signal_wait_*` that show synchronization points.
- **GPU metrics rows** (new): Device temperature, power, utilization, and memory usage over time.
- **KFD events** (new, if present): Page fault and migration events

<img width="1902" height="1001" alt="image" src="https://github.com/user-attachments/assets/5271c59f-6e95-4dbe-8a15-ff21faf83812" />


