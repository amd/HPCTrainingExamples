
# Stage 0: Baseline

This is the starting point, before any optimization. The domain is 512x512, the RK4 loop calls
`hipDeviceSynchronize()` after every kernel launch, and the compute kernels are launched with 16x16
thread blocks.

Nothing here is deliberately broken. It is the kind of first working version you would write after
porting a solver to HIP, and our job over the next four stages is to let the profiler tell us what to
change.

## Build and run

```bash
module load rocm
make
./shallow
```

or with CMake:

```bash
mkdir build && cd build
cmake ..
make
./shallow
```

## Expected output

```
Domain: 512x512, steps=500, dt=0.0728643
Elapsed: 0.083 s  |  Throughput (including RK4 stages): 6282.26 MCUPS
Mass: initial=2.626466547e+05, final=2.626464583e+05, rel.err=7.477e-07
Min(h) after run: 0.981776
```

Keep the last two lines in view for the rest of the tutorial. Mass conservation and a positive
minimum depth are how we confirm that an optimization sped the code up without changing the answer.

6282.26 MCUPS is the number to beat.

## Step 1: Which kernel should we look at?

Always start by finding out where the time actually goes. Collect a kernel dispatch trace and ask
`rocprofv3` to summarize it:

```bash
rocprofv3 --kernel-trace --stats -S -T -d outdir -o shallow -- ./shallow
```

The options are: `--kernel-trace` records every kernel dispatch, `--stats` computes per-kernel
statistics, `-S` prints the summary to the console, `-T` truncates the demangled kernel names so the
table stays readable, and `-d`/`-o` set the output directory and file prefix.

```
|                   NAME                   |     DOMAIN      |      CALLS      | DURATION (nsec) | AVERAGE (nsec)  | PERCENT (INC) |   MIN (nsec)    |   MAX (nsec)    |     STDDEV      |
|------------------------------------------|-----------------|-----------------|-----------------|-----------------|---------------|-----------------|-----------------|-----------------|
| compute_rhs                              | KERNEL_DISPATCH |            2000 |        15955440 |       7.978e+03 |     31.263037 |            7171 |            9935 |       3.800e+02 |
| update_stage                             | KERNEL_DISPATCH |            1500 |        14241259 |       9.494e+03 |     27.904277 |            5929 |           12740 |       2.189e+03 |
| final_update                             | KERNEL_DISPATCH |             500 |        10434479 |       2.087e+04 |     20.445284 |           19230 |           22595 |       6.712e+02 |
| apply_reflect_bc                         | KERNEL_DISPATCH |            2001 |        10397246 |       5.196e+03 |     20.372330 |            3766 |            7532 |       5.639e+02 |
| init_gaussian                            | KERNEL_DISPATCH |               1 |            7692 |       7.692e+03 |      0.015072 |            7692 |            7692 |       0.000e+00 |
```

The call counts confirm that the trace matches the algorithm: 500 time steps, four RK4 stages each,
so 2000 `compute_rhs` calls and 2001 boundary-condition calls including the one during
initialization.

`compute_rhs` is the largest single consumer at 31.3 percent, which makes it the kernel to
investigate first. Note also that no single kernel dominates overwhelmingly: the top four kernels
are all between 20 and 32 percent, so we should expect to need improvements that help the whole
solver rather than one hot spot.

## Step 2: Is the GPU actually full?

The next question for any kernel is whether there is enough work in flight to keep the hardware
busy. `OccupancyPercent` measures the fraction of the maximum possible wavefront slots that are
occupied.

```bash
rocprofv3 --pmc OccupancyPercent -T --output-format csv -d outdir -o shallow -- ./shallow
```

The first rows of the resulting CSV:

```
"Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id","Thread_Id","Grid_Size","Kernel_Id","Kernel_Name","Workgroup_Size","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"
1,1,"Agent 1",2,3260601,3260601,262144,5,"init_gaussian",256,0,0,12,4,32,"OccupancyPercent",13.271172,1569078135380312,1569078135386762
2,2,"Agent 1",2,3260601,3260601,512,3,"apply_reflect_bc",256,0,0,20,4,32,"OccupancyPercent",4.34259040e-02,1569078135498332,1569078135503821
3,3,"Agent 1",2,3260601,3260601,512,3,"apply_reflect_bc",256,0,0,20,4,32,"OccupancyPercent",4.57287245e-02,1569078164965801,1569078164973132
4,4,"Agent 1",2,3260601,3260601,262144,2,"compute_rhs",256,0,0,52,4,32,"OccupancyPercent",24.333218,1569078165036469,1569078165045964
5,5,"Agent 1",2,3260601,3260601,262144,1,"update_stage",256,0,0,16,0,32,"OccupancyPercent",22.051387,1569078165097603,1569078165106777
6,6,"Agent 1",2,3260601,3260601,512,3,"apply_reflect_bc",256,0,0,20,4,32,"OccupancyPercent",4.74045895e-02,1569078165150084,1569078165155652
7,7,"Agent 1",2,3260601,3260601,262144,2,"compute_rhs",256,0,0,52,4,32,"OccupancyPercent",22.353355,1569078165200962,1569078165210296
8,8,"Agent 1",2,3260601,3260601,262144,1,"update_stage",256,0,0,16,0,32,"OccupancyPercent",21.618926,1569078165257288,1569078165266422
```

Averaging each kernel over all of its dispatches:

| Kernel | Grid size | Occupancy |
|---|---|---|
| `init_gaussian` | 262144 | 13.3 percent |
| `compute_rhs` | 262144 | 24.5 percent |
| `update_stage` | 262144 | 23.6 percent |
| `final_update` | 262144 | 25.0 percent |
| `apply_reflect_bc` | 512 | 0.05 percent |

The main kernels sit around a quarter, so three quarters of the machine is idle. `apply_reflect_bc`
looks alarming at 0.05 percent, but that is expected and not worth chasing: it only touches the
ghost ring, so its grid is 512 threads against 262144 for the others. It cannot fill the GPU
because there is not enough boundary to go around.

## Step 3: What is limiting `compute_rhs`?

A roofline plot places a kernel against the machine's compute and bandwidth ceilings, which tells us
whether it is limited by arithmetic or by memory traffic.

`profile_app.py` in Roofline Extractor needs its
[Python environment](../README.md#a-python-environment-for-roofline-extractor) active:

```bash
python3 "$ROOFLINE_EXTRACTOR/profile_app.py" -o roofline_out --arch MI300A -- ./shallow
```

The equivalent in `rocprof-compute`, whose `analyze` step needs its
[Python environment](../README.md#a-python-environment-for-rocprof-compute-analyze) active:

```bash
rocprof-compute profile -n 0_baseline --roof-only --device 0 -k compute_rhs --iteration-multiplexing -- ./shallow
rocprof-compute analyze -p workloads/0_baseline/0
```

Both are explained in [Roofline plots](../README.md#roofline-plots).

<p>
<img src="../../figs/roofline_512.png" alt="Roofline of compute_rhs at 512x512" />
</p>

The kernel does perform real arithmetic, but it sits well below the compute ceilings and to the left
of the ridge point, which makes it memory bound. More telling is that it is also a long way below
the memory bandwidth ceiling itself. A kernel that were simply bandwidth-limited would sit close to
that line. Being far beneath it means either the memory accesses are inefficient, or there is not
enough parallelism in flight to hide memory latency.

The extractor's per-kernel summary says the same thing in numbers: it puts `compute_rhs` at an
arithmetic intensity of 4.82 flops per byte of HBM traffic, moving 1.9 TB/s against the 3.7 TB/s
this MI300A can sustain, so it is leaving about half the available bandwidth unused.

### An aside: why `--iteration-multiplexing`

A GPU has only a small number of hardware counter registers, far fewer than a full
`rocprof-compute` report needs. The default way around this is *application replay*: the tool runs
your application once per set of counters and stitches the results together afterwards. On MI300A a
full profile takes 13 such runs, so profiling costs roughly 13 times the runtime of the application
itself. `--roof-only` narrows that to 3 sets, but the multiplier is still there.

`--iteration-multiplexing` removes the replay. Rather than collecting the same counters on every
dispatch and re-running the program, it collects a different subset of counters on *different
dispatches of the same kernel*, then combines them at the end. The application runs exactly once.
On the 2048x2048 domain used from stage 1 onward, a full counter collection for `compute_rhs`
dropped from 209 seconds to 24 seconds this way.

The requirement is that each kernel be dispatched enough times to cover every subset, around 15 on
current hardware with 50 recommended. Our solver launches `compute_rhs` 2000 times, so it has
plenty to spare. This is also why `-k compute_rhs` matters here: `init_gaussian` is dispatched
exactly once and can never fill all the subsets, so without the filter the tool warns about it and
drops it from the metrics. The subsets are tracked per unique combination of kernel name and launch
parameters, meaning grid size, workgroup size and LDS size.

The trade-off is accuracy. Replay measures every counter over the same set of dispatches, whereas
multiplexing measures different counters on different dispatches, so the values it reports are
close rather than identical. For working out what limits a kernel that is a good trade, but if you
are chasing a small difference, re-run the profile without the flag. The
[ROCm documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html#iteration-multiplexing)
covers the remaining caveats.

## What we learned, and what to do about it

The occupancy of about 24 percent points squarely at the second explanation. At 512x512 there
simply are not enough cells to keep every compute unit supplied with work: 512x512 interior cells
divided into 16x16 blocks is 1024 workgroups, spread across a GPU with 304 compute units.

The cheapest possible experiment is to give the GPU more of the same work: grow the domain from
512x512 to 2048x2048, a 16x increase in cells. If the diagnosis is right, throughput per cell should
rise sharply even though the total run takes longer.

Continue to [`1_larger_domain`](../1_larger_domain), and note that the entire change is two lines:

```bash
diff ../0_baseline/shallow.hip ../1_larger_domain/shallow.hip
```
