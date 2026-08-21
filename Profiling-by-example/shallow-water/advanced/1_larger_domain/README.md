
# Stage 1: Give every rank enough work

Stage 0 left us with two facts: the code runs slower on four GPUs than on one, and the timeline shows
communication outweighing computation. Both point at the same cause, subdomains too small to be worth
dividing, so the first thing to try is simply a bigger problem.

## What changed

```bash
diff ../0_baseline/shallow_mpi.hip shallow_mpi.hip
```

Two lines, the global domain size:

```c++
constexpr int   NXG     = 8192;     // global interior cells in x
constexpr int   NYG     = 8192;     // global interior cells in y
```

That is 256 times as many cells. Everything else, the kernels, the block size, the halo exchange, is
untouched.

## Build and run

```bash
module load rocm openmpi
make
mpirun -n 2 --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh ./shallow_mpi
```

## Expected output

```
MPI ranks: 2  |  GPUs detected: 1
Domain: 8192x8192 (global), steps=500, dt=0.0728643
Elapsed (max over ranks): 3.094 s  |  Throughput: 43377.47 MCUPS
Mass: initial=6.710936665e+07, final=6.710936646e+07, rel.err=2.929e-09
Min(h) after run: 0.981776
```

The mass error improved by two orders of magnitude, from 7.483e-07 to 2.929e-09, because the finer
grid resolves the Gaussian bump better. Correctness is intact.

## The scaling study, repeated

```bash
salloc -N 1 -p LocalQ --exclusive --gres=gpu:4 -t 2:00:00
for n in 1 2 4; do
    mpirun -n $n --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh ./shallow_mpi
done
```

| Ranks | MCUPS at 512x512 | MCUPS at 8192x8192 | Efficiency now |
|---|---|---|---|
| 1 | 8159.26 | 22238.83 | -- |
| 2 | 4899.27 | 43377.47 | 97.5 percent |
| 4 | 3084.88 | 79762.89 | 89.7 percent |

Two different things improved at once, and it is worth separating them.

The single-GPU number went from 8159 to 22239 MCUPS, 2.73x, and that has nothing to do with MPI.
It is the same effect the novice example sees at this step: 512x512 in 16x16 blocks is 1024
workgroups against 228 compute units, which is not enough work in flight to hide memory latency,
while 8192x8192 is 262144 workgroups and plenty.

The scaling behaviour went from actively negative to 90 percent at four GPUs, and that is the MPI
part. Each rank now owns a 8192x2048 band: 16.7 million interior cells against two boundary rows of
8192 cells each, so roughly one cell in a thousand crosses the network per stage rather than one in
64. The halo did not get smaller in absolute terms, it got smaller relative to the work it serves.

At two GPUs the code is now 8.9x faster than the same two GPUs managed in stage 0. Almost none
of that came from making anything faster; it came from measuring a sensible problem size.

Note also what the efficiency column does between two and four ranks, falling from 98 to 90 percent.
The trend is real and we will come back to it: the more we speed up the kernels in the next three
stages, the larger a share of the total the fixed communication cost becomes.

## Is the GPU busy now?

With enough parallelism in flight, the next question is whether the vector units are actually
working. `VALUBusy` is the percentage of GPU time during which the vector ALUs are processing
instructions:

```bash
mpirun -n 2 --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh \
    rocprofv3 --pmc VALUBusy -T -f csv \
    -o results_%env{OMPI_COMM_WORLD_RANK}% -- ./shallow_mpi
```

Over the 2000 `compute_rhs` dispatches the median is 67.3 percent on rank 0 and 67.1 percent on
rank 1. The two agreeing is part of the check: both ranks run the same kernel over their own half of
the domain, so a gap between them would point at load imbalance rather than at the kernel. Every other
kernel is far behind, `update_stage` at 3.8 and the boundary kernel at 0.06, so `compute_rhs` is where
any vector-unit work is happening.

Two thirds busy sounds healthy, and it is the wrong conclusion to stop at. The units are issuing
instructions most of the time, but the next two stages show the kernel is still leaving a third of the
issue slots idle waiting on memory, and closing that gap is what the block-shape changes do.

Counter collection serializes dispatches, so a counter run's wall-clock time is not a performance
measurement. At a single counter set the cost is small here, but do not compare it against the timings
above.

A roofline places the hot kernel against the machine's compute and bandwidth ceilings, which tells us
whether the remaining headroom is in arithmetic or in memory traffic. Profile a single rank, since all
four are doing the same thing:

```bash
rocprof-compute profile -n 1_larger_domain --no-roof -k compute_rhs \
    --iteration-multiplexing -- ./shallow_mpi
rocprof-compute analyze -p workloads/1_larger_domain/0
```

`--iteration-multiplexing` collects a different subset of counters on different dispatches of the same
kernel rather than replaying the whole application once per counter set, which keeps the profiling run
short; `-k compute_rhs` restricts collection to the kernel we care about. The novice example explains the trade-off in
[more detail](../../novice/0_baseline/README.md#an-aside-why---iteration-multiplexing). `analyze` needs
the [Python environment](../README.md#a-python-environment-for-rocprof-compute-analyze) from the setup
section.

The full report runs to hundreds of rows, so ask for the summary section only. `-b` takes the metric
ids from the leftmost column of the report and keeps just those rows:

```bash
rocprof-compute analyze -p workloads/1_larger_domain/0 \
    -b 2.1.0 2.1.9 2.1.14 2.1.15 2.1.18 2.1.19 2.1.20 2.1.21
```

| Metric | Value | Percent of peak |
|---|---|---|
| 2.1.0 VALU FLOPs | 10788 GFLOP/s | 17.60 |
| 2.1.9 VALU Utilization | 75.19 percent | -- |
| 2.1.14 IPC | 0.73 | 14.65 |
| 2.1.15 Wavefront Occupancy | 4838 wavefronts | 66.31 |
| 2.1.18 vL1D Cache Hit Rate | 55.29 percent | -- |
| 2.1.19 vL1D Cache BW | 13469 GB/s | 21.98 |
| 2.1.20 L2 Cache Hit Rate | 22.34 percent | -- |
| 2.1.21 L2 Cache BW | 6026 GB/s | 23.35 |

Occupancy is reported as an average wavefront count against the 7296 the device can hold, which is
the 66.31 percent in the last column. These eight rows are the baseline the block-shape experiments
in the next two stages are measured against, and the same eight come back as a comparison there.

The plot below is the novice track's measurement of this same kernel at the same 16x16 block shape.
A roofline describes the kernel rather than the decomposition, since every rank runs the same
`compute_rhs` over its own tile, so it is reused here rather than re-collected.

<p>
<img src="../../figs/roofline_2048.png" alt="Roofline of compute_rhs with 16x16 blocks" width="60%" />
</p>

## What we learned, and what to do about it

The problem size is no longer the limiter, so from here the story splits in two. There is kernel work
to do, because `compute_rhs` sits below the memory ceiling with vector units idle much of the time,
and there is communication work to do, because efficiency is already sliding as rank count grows.

Kernel work comes first, and for a reason worth stating: it is the cheaper of the two to try, and
until the kernels stop improving there is no way to know how much of the remaining time
communication really deserves. Optimizing the network in a code whose kernels are twice slower than
they need to be means optimizing against the wrong ratio.

So the next three stages tune the launch configuration and the memory access pattern of
`compute_rhs`, and only then do we return to the halo exchange. The first step is the block shape.

Continue to [`2_block_32x32`](../2_block_32x32), where the change is a single line:

```bash
diff ../1_larger_domain/shallow_mpi.hip ../2_block_32x32/shallow_mpi.hip
```
