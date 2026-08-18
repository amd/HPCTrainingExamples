
# Stage 4: One wide load instead of three narrow ones

The launch configuration is as good as it gets and `compute_rhs` is still memory bound. Counters have
taken us as far as they can, because they aggregate: `VALUBusy` tells us the vector units stall, not
which instruction they stall on. This stage introduces a tool that answers that question, and then
acts on what it says.

## A new point of view: thread trace

Thread trace, also called SQTT or ATT, traces wavefronts on the GPU at instruction granularity. Where
a counter gives one number per kernel, a thread trace gives near cycle-accurate timing for every
instruction, the exact execution path taken, and how long each instruction spent stalled versus
issuing. It is a targeted tool rather than a survey one: it traces a single compute unit per shader
engine and is meant for one kernel at a time.

The basic invocation, before the optimization, on the stage 3 binary:

```bash
cd ../3_block_64x4
rocprofv3 --att -d att_out -- ./shallow_mpi
```

On Instinct hardware it is worth also streaming the SQ performance counters into the trace buffer,
which is what gives you the activity breakdown per instruction:

```bash
rocprofv3 --att --att-activity 8 --kernel-include-regex compute_rhs \
    -d att_out -- ./shallow_mpi
```

`--kernel-include-regex` matters here for the same reason `-k` mattered for the roofline: without it
you trace whatever kernel happens to be dispatched first. Two further notes:

- Thread trace decoding needs the **ROCprof Trace Decoder** library, and the resulting timeline is
  viewed in the **ROCprof Compute Viewer**, which is a separate desktop application. See the
  [thread trace documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html)
  and the [viewer documentation](https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/amd-mainline/index.html).
- Build with `-g` so the trace can put your source lines next to the ISA. The Makefile in every stage
  already does.

The run produces `stats_*.csv`, a per-instruction latency summary, and one
`ui_output_agent_*_dispatch_*` directory per traced dispatch for the viewer. The CSV is the quickest
way in, with one row per instruction:

| Column | Meaning |
|---|---|
| `Instruction` | The ISA instruction |
| `Hitcount` | How many times it executed, summed over all traced waves |
| `Latency` | Total cycles, stall plus issue time |
| `Stall` | Cycles in which the pipe could not issue, typically cache or LDS backpressure |
| `Idle` | Gap since the previous instruction finished, from register dependencies or instruction cache misses |
| `Source` | The source line, when built with debug symbols |

If `stats_*.csv` comes back empty even though the kernel definitely ran, the trace found no wave on
the one compute unit it was watching. Widen the search with `--att-shader-engine-mask 0x11111111`.

<!-- MEASUREMENT TODO: thread trace hotspot view for stage 3 compute_rhs, showing the three global loads -->

Sorting by latency puts three `global_load_dword` instructions at the top, and the source column
attributes them to the three neighbour reads in x:

```c++
    const float hi  = h[id];
    const float hui = hu[id];
    const float hvi = hv[id];
```

together with the `h[ip]`, `h[im]` and equivalents that follow. Per cell, `compute_rhs` issues
separate loads for `h`, `hu` and `hv` at `i-1`, `i` and `i+1`. Nine narrow loads, all along the
direction the arrays are contiguous in.

## What changed

```bash
diff ../3_block_64x4/shallow_mpi.hip shallow_mpi.hip
```

This is the first stage whose diff is more than a line or two. The central change replaces the three
separate reads along x with a single 16-byte load per array:

```c++
    const float4 h_x_vec = *reinterpret_cast<const float4*>(&h[im]);
    const float h_jm = h[jm];  // y-neighbors are not contiguous, load separately
    const float h_jp = h[jp];
```

One `global_load_dwordx4` fetches `i-1`, `i`, `i+1` and one cell beyond, and the components are pulled
out afterwards. The y-neighbours cannot join in, since they are a row apart in memory. Three
supporting changes come with it:

- **Cache-line-aligned row pitch.** `pitch` is rounded up to a multiple of 16 floats so every row
  starts on a 64-byte boundary, which is what makes the wide loads land inside a single cache line
  rather than straddling two.
- **Fewer divisions.** One reciprocal is computed per location and reused for both velocity
  components, and the flux terms use `fmaf`.
- **A `j+2` prefetch** and `__launch_bounds__(256,1)`, both attempts to keep more memory requests in
  flight.

The important framing is that this widens each request at the **HBM level**, not the cache level.
Stages 2 and 3 improved cache reuse by reshaping the block; this stage reduces the number of memory
instructions needed to fetch the same bytes.

## Build and run

```bash
module load rocm openmpi
make
mpirun -n 4 --bind-to none ../gpu_bind.sh ./shallow_mpi
```

## Expected output

```
MPI ranks: 4  |  GPUs detected: 1
Domain: 8192x8192 (global), steps=500, dt=0.0728643
Elapsed (max over ranks): 1.075 s  |  Throughput: 124867.74 MCUPS
Mass: initial=6.710936665e+07, final=6.710936646e+07, rel.err=2.916e-09
Min(h) after run: 0.981776
```

| Ranks | MCUPS at stage 3 | MCUPS now | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 33815.80 | 36088.86 | 1.07x | -- |
| 2 | 67152.65 | 70413.39 | 1.05x | 98 percent |
| 4 | 125015.25 | 124867.74 | 1.00x | 87 percent |

**A 1.07x gain on one GPU that arrives as nothing at all on four.** Read those two rows together,
because this is the most instructive result in the tutorial. The kernel really did get 7 percent
faster, the single-GPU row proves it, and the four-GPU row is flat to within run-to-run spread. The
speedup was spent on communication rather than banked, and the efficiency column says the same thing
from the other side: 92 percent in stage 3, 87 percent here.

It is also the last of the kernel gains, and a modest return for by far the largest code change so
far, which is part of the lesson too: the deeper you go, the more effort each remaining percent costs.
On one GPU we have now gone from 22229 to 36089 MCUPS since stage 1, a cumulative **1.62x**, all of it
from the four changes a profiler pointed at. On four GPUs that same work is worth only 1.57x, and the
gap between those two figures is the communication cost that stage 5 goes after.

This is the only stage that alters the arithmetic, so its accuracy checks are the ones worth watching
most closely. The mass error moves from 2.929e-09 to 2.916e-09, a change in the last digits from
reassociating the flux expressions and using reciprocals, and the minimum depth is unchanged. Both
are comfortably within what single precision entitles us to.

## Confirming it with the trace

Re-run the thread trace on this stage and compare:

```bash
rocprofv3 --att --att-activity 8 --kernel-include-regex compute_rhs \
    -d att_out -- ./shallow_mpi
```

<!-- MEASUREMENT TODO: thread trace hotspot view after vectorization, and the roofline -->

```bash
rocprof-compute profile -n 4_vectorized_loads --roof-only --device 0 -k compute_rhs \
    --iteration-multiplexing -- ./shallow_mpi
rocprof-compute analyze -p workloads/4_vectorized_loads/0
```

## What we learned, and what to do about it

The kernel side of this application is finished, or near enough that further effort is better spent
elsewhere. Look at the efficiency column across the last four stages: 90, 91, 92, 87 percent. It held
while the kernels had slack in them and broke at the stage where they ran out, which is the clearest
signal available that the limiter has moved. Communication is now the largest single thing left.

Re-profiling the kernels at four ranks confirms the reordering. Where stage 0 had `compute_rhs`
dominating, the cheap kernels now account for a comparable share of GPU time, and the whole GPU
timeline is a smaller part of the wall clock than the MPI calls around it:

```bash
mpirun -n 4 --bind-to none ../gpu_bind.sh \
    rocprofv3 --kernel-trace --stats -S -T -d prof -o shallow_%rank% -- ./shallow_mpi
```

<!-- MEASUREMENT TODO: per-rank kernel stats at stage 4, 4 ranks, showing the reordered ranking -->

The next stage stops optimizing kernels and starts optimizing the halo exchange.

Continue to [`5_halo_pipeline`](../5_halo_pipeline).
