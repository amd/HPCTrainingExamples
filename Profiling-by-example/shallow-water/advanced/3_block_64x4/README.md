
# Stage 3: A wavefront-shaped thread block

Stage 2 improved cache reuse by making the block bigger. The next idea changes the block's shape
instead of its size, so that each wavefront reads one long contiguous run of memory.

## What changed

```bash
diff ../2_block_32x32/shallow_mpi.hip shallow_mpi.hip
```

One line:

```c++
    dim3 block(64,4);
```

Still 256 threads, arranged 64 wide and 4 tall. Since a wavefront is 64 lanes on AMD GPUs, each
wavefront now maps onto exactly one row of 64 consecutive cells in x, which is the direction the
arrays are contiguous in.

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
Elapsed (max over ranks): 1.993 s  |  Throughput: 67339.87 MCUPS
Mass: initial=6.710936665e+07, final=6.710936646e+07, rel.err=2.929e-09
Min(h) after run: 0.981776
```

| Ranks | MCUPS at 32x32 | MCUPS at 64x4 | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 28981.84 | 33946.38 | 1.17x | -- |
| 2 | 58135.17 | 67339.87 | 1.16x | 99.2 percent |
| 4 | 105226.03 | 124762.34 | 1.19x | 91.9 percent |

A further 1.16x at two GPUs. Running total from stage 1, where the domain first became a
sensible size: 1.55x on two GPUs, and 1.53x on one.

Efficiency at four GPUs has held so far: 89.7 percent in stage 1, 90.8 in stage 2, 91.9 here, which is a
flat line once run-to-run spread is taken into account. That is worth stating plainly, because the
pressure that will eventually break it is already building. The halo exchange takes the same absolute
time it always did while the compute it hides behind keeps shrinking, so its share of each step keeps
growing. At this stage the share is still small enough not to show. One more kernel optimization is
enough to expose it.

## Counters

The `rocprof-compute` comparison is unambiguous:

| Block | Wavefront occupancy | Vector-L1 hit rate | L2 hit rate | VALU throughput |
|---|---:|---:|---:|---:|
| 32x32 | 74.78 percent | 71.07 percent | 20.66 percent | 14632 GFLOP/s |
| 64x4 | 97.27 percent | 73.61 percent | 21.07 percent | 16040 GFLOP/s |

The wide, short block raises occupancy substantially while preserving the cache gain from stage 2.
That is what makes it better than either 16x16 or 32x32 rather than a compromise between them.

## Roofline

```bash
rocprof-compute profile -n 3_block_64x4 --no-roof -k compute_rhs \
    --iteration-multiplexing -- ./shallow_mpi
rocprof-compute analyze -p workloads/3_block_64x4/0
```

Again the novice measurements of the same kernel, reused for
[the reason given in stage 1](../1_larger_domain/README.md#is-the-gpu-busy-now). 32x32 on the left,
64x4 on the right.

<p>
<img src="../../figs/roofline_block_32x32.png" alt="Roofline of compute_rhs with 32x32 blocks, before this stage" width="49%" />
<img src="../../figs/roofline_block_64x4.png" alt="Roofline of compute_rhs with 64x4 blocks, after this stage" width="49%" />
</p>

The two plots look the same, even though the change is worth 1.16x. A roofline summarizes arithmetic
intensity and achieved bandwidth, and a coalescing improvement can pay off without moving either far
enough to see on a log-log plot.

## What we learned, and what to do about it

We have run out of things to change in the launch configuration. The block is the right size and the
right shape, the kernel is still memory bound, and the next question has to be about the memory
accesses themselves rather than about how threads are grouped.

Counters will not answer that question, because they aggregate. `VALUBusy` says the vector units
stall; it does not say which instruction they stall on. For that we need a view of the kernel at
instruction granularity, which is what the thread trace in the next stage provides.

Continue to [`4_vectorized_loads`](../4_vectorized_loads).
