
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
mpirun -n 4 --bind-to none ../gpu_bind.sh ./shallow_mpi
```

## Expected output

```
MPI ranks: 4  |  GPUs detected: 1
Domain: 8192x8192 (global), steps=500, dt=0.0728643
Elapsed (max over ranks): 1.683 s  |  Throughput: 79748.87 MCUPS
Mass: initial=6.710936665e+07, final=6.710936646e+07, rel.err=2.929e-09
Min(h) after run: 0.981776
```

The mass error improved by two orders of magnitude, from 7.483e-07 to 2.929e-09, because the finer
grid resolves the Gaussian bump better. Correctness is intact.

## The scaling study, repeated

```bash
for n in 1 2 4; do mpirun -n $n --bind-to none ../gpu_bind.sh ./shallow_mpi; done
```

| Ranks | MCUPS at 512x512 | MCUPS at 8192x8192 | Efficiency now |
|---|---|---|---|
| 1 | 8309.26 | 22228.56 | -- |
| 2 | 4950.31 | 43364.87 | 98 percent |
| 4 | 3006.97 | 79748.87 | 90 percent |

Two different things improved at once, and it is worth separating them.

The single-GPU number went from 8309 to 22229 MCUPS, **2.68x**, and that has nothing to do with MPI.
It is the same effect the novice example sees at this step: 512x512 in 16x16 blocks is 1024
workgroups against 228 compute units, which is not enough work in flight to hide memory latency,
while 8192x8192 is 262144 workgroups and plenty.

The scaling behaviour went from actively negative to 90 percent at four GPUs, and that is the MPI
part. Each rank now owns a 8192x2048 band: 16.7 million interior cells against two boundary rows of
8192 cells each, so roughly one cell in a thousand crosses the network per stage rather than one in
64. The halo did not get smaller in absolute terms, it got smaller relative to the work it serves.

At four GPUs the code is now **26.5x** faster than the same four GPUs managed in stage 0. Almost none
of that came from making anything faster; it came from measuring a sensible problem size.

Note also what the efficiency column does between two and four ranks, falling from 98 to 90 percent.
The trend is real and we will come back to it: the more we speed up the kernels in the next three
stages, the larger a share of the total the fixed communication cost becomes.

## Is the GPU busy now?

With enough parallelism in flight, the next question is whether the vector units are actually
working. `VALUBusy` is the percentage of GPU time during which the vector ALUs are processing
instructions:

```bash
mpirun -n 4 --bind-to none ../gpu_bind.sh \
    rocprofv3 --pmc VALUBusy -T --output-format csv -d prof -o valu_%rank% -- ./shallow_mpi
```

Counter collection serializes kernels, so this run takes considerably longer than an unprofiled one,
and its wall-clock time is not a performance measurement.

<!-- MEASUREMENT TODO: VALUBusy per kernel at 16x16, 4 ranks -->

A roofline places the hot kernel against the machine's compute and bandwidth ceilings, which tells us
whether the remaining headroom is in arithmetic or in memory traffic. Profile a single rank, since all
four are doing the same thing:

```bash
rocprof-compute profile -n 1_larger_domain --roof-only --device 0 -k compute_rhs \
    --iteration-multiplexing -- ./shallow_mpi
rocprof-compute analyze -p workloads/1_larger_domain/0
```

`--iteration-multiplexing` collects a different subset of counters on different dispatches of the same
kernel rather than replaying the whole application once per counter set, which keeps the profiling run
short; `-k compute_rhs` restricts collection to the kernel we care about, so that the once-dispatched
`init_gaussian` does not get dropped with a warning. The novice example explains the trade-off in
[more detail](../../novice/0_baseline/README.md#an-aside-why---iteration-multiplexing). `analyze` needs
the [Python environment](../README.md#a-python-environment-for-rocprof-compute-analyze) from the setup
section.

The plot below is the one the novice track measured for this same kernel at the same 16x16 block
shape, on a single GPU. It is reused here rather than re-collected because a roofline describes the
kernel, not the decomposition: every rank runs the same `compute_rhs` over its own tile, so the
arithmetic intensity and the fraction of peak bandwidth it reaches are the same quantities. What
changes with rank count is how many such kernels run at once, which is a scaling question rather than
a roofline one.

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
