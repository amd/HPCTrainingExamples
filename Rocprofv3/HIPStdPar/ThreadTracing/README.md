# Thread Tracing with rocprofv3 — HIPStdPar ShallowWater

This example demonstrates thread tracing of a C++ standard parallelism application using
`rocprofv3 --att`. It uses the
[ShallowWater_StdPar](https://github.com/amd/HPCTrainingExamples/tree/main/HIPStdPar/CXX/ShallowWater_StdPar)
2D shallow-water solver as the target workload.


## Setup

Allocate a compute node and load the ROCm module:

```bash
salloc -N 1 --gpus=1
module load therock/therock-dist-linux-gfx94X-dcgpu-7.14.0a20260608
export HSA_XNACK=1
```

Build ShallowWater example:

```bash
cd HPCTrainingExamples/HIPStdPar/CXX/ShallowWater_StdPar
export CXX=amdclang++
make clean && make
```
## collect a trace for a certain kernel iteration
Run the profiler with the --kernel-iteration-range option
```bash
rocprofv3 --att --att-activity 8 --output-directory=ShallowWater_test_trace --kernel-iteration-range 10  -- ./ShallowWater

```
Filtering is required to keep output files small and focused: without it the tracer captures
every kernel dispatch in the entire application, which produces enormous amount 
of files which may bust quota and file system limitations.
There are several filtering options, see:
https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html#rocprofv3-parameters-for-thread-tracing

## selected-regions flag

`--selected-regions` tells rocprofv3 to start and stop thread trace collection
only around `roctxProfilerResume(0)` / `roctxProfilerPause(0)` calls. 


Extend `ShallowWater.cpp` with a pause at the start of `main()` and a
resume/pause window around a representative section:

```cpp
#include <rocprofiler-sdk-roctx/roctx.h>
...
roctxProfilerPause(0); //selected regions does this also per default, but timeline tracing for example not 


roctxProfilerResume(0);
...potential warmup kernel...
...kernel of interest... 
roctxProfilerPause(0);

```


Each resume–pause cycle produces its own set of output files (ATT data,
`stats_*.csv`, and `ui_output_agent_*` directory). 

Link the roctx library in the `Makefile`, change

```makefile
CXXFLAGS = -g -O3 -fstrict-aliasing ${STDPAR_FLAGS}
LDFLAGS = -fno-lto -lm
```
to:
```makefile
CXXFLAGS = -g -O3 -fstrict-aliasing ${STDPAR_FLAGS} -I${ROCM_PATH}/include
LDFLAGS = -fno-lto -lm -L$(ROCM_PATH)/lib -lrocprofiler-sdk-roctx
```

Note: The first kernel with use of the profiler may create page faults due to profiler issued allocations with HSA_XNACK=1 active on MI300A! Include a dummy kernel you are not interested in profiling the performance to avoid this issue in the kernel of interest.


Rebuild after the edit:

```bash
make clean && make
```

## Collect thread trace

Run under rocprofv3 (options see https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html#rocprofv3-parameters-for-thread-tracing):

```bash
rocprofv3 --att --att-activity 8 --selected-regions \
          --output-directory out_threadtrace_dir \
          -- ./ShallowWater

> `--selected-regions` is required when using pause/resume markers.
> Without it rocprofv3 ignores `roctxProfilerPause/Resume` calls and traces
> the whole application. CAREFUL: the selected-regions flag also exists in rocprof-sys but with a different meaning!

## Output files

After the run, `out_threadtrace_dir/` contains:

| File / directory | Description |
|---|---|
| `stats_*.csv` | Per-instruction summary: hitcount, latency, stall, idle, source line |
| `ui_output_agent_{id}_dispatch_{id}/` | JSON directory for ROCprof Compute Viewer |
| `*.att` | Raw SQTT binary data |
| `*.out` | Code-object binaries (for ISA analysis) |

The `ui_output_agent_*` directories are the primary input to the viewer.

### Reading stats_*.csv

The CSV maps instructions back to source lines when the binary is compiled with
`-g` (ShallowWater's Makefile adds this flag):

```
Codeobj,Vaddr,Instruction,Hitcount,Latency,Stall,Idle,Source
11,5888,s_load_dwordx4 s[40:43]...,48,276,96,48,ShallowWater.cpp:142
```

- **Latency** — stall + issue time (on gfx9)
- **Stall** — cycles the hardware pipe could not issue (hardware unit is busy)
- **Idle** — gap between previous and current instruction (e.g. register dependency, instruction cache miss)

## Visualize with ROCprof Compute Viewer

The `ui_output_agent_*` directories are opened with
**ROCprof Compute Viewer (RCV)**, which runs on your laptop — no cluster
connection needed once the files are copied.

**1. Download the viewer**

Windows installer (and Linux packages) are available at:

<https://github.com/ROCm/rocprof-compute-viewer/releases>

**2. Copy output to your laptop**

```bash
scp -i <ssh-key> -r <user>@<host>:<path>/out_threadtrace_dir .
```

**3. Open in RCV**

Launch the installed ROCprof Compute Viewer, click **Open**, and select one of
the `ui_output_agent_*` directories. The viewer shows:

- Instruction timeline per wavefront
- Stall and idle breakdown per instruction
- Source-line annotations (when `-g` is present)
- SQ performance counter overlays (when `--att-activity` was used)

Each resume–pause cycle gets its own `ui_output_agent_*` directory; open them
in separate tabs to compare different iteration windows.

## Troubleshooting
**Trace buffer full / data loss warnings**

Increase the buffer size (default 96 MB):

```bash
rocprofv3 --att --att-activity 8 --selected-regions \
          --att-buffer-size 268435456 \
          --output-directory out_threadtrace_dir \
          -- ./ShallowWater
```

Alternatively, narrow the traced window (fewer iterations between resume and
pause) or add `--kernel-include-regex` to trace one kernel at a time or limit the iterations traced.

## References

- [rocprofv3 thread trace documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html)
- [ROCprof Compute Viewer documentation](https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/latest/)
- [ROCprof Compute Viewer releases (Windows / Linux)](https://github.com/ROCm/rocprof-compute-viewer/releases)
- [Rocprofv3 HIPStdPar exercise](../HIPStdPar/README.md) — prerequisites: build, roctx, pause/resume
- [Rocprofv3 ThreadTrace example (HIP GEMM)](../ThreadTrace/README.md) — pure HIP reference with three kernels
- [ShallowWater_StdPar source](https://github.com/amd/HPCTrainingExamples/tree/main/HIPStdPar/CXX/ShallowWater_StdPar)
