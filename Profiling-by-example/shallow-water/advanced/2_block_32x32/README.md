
# Stage 2: A larger thread block

Stage 1 measured `VALUBusy` at 67.3 percent for `compute_rhs`: the vector units have data to work on
only two thirds of the time, despite plenty of work in flight. For a stencil kernel the usual suspect
is cache reuse, and the block shape controls how much of it we get.

## What changed

```bash
diff ../1_larger_domain/shallow_mpi.hip shallow_mpi.hip
```

One line:

```c++
    dim3 block(32,32);
```

That takes the workgroup from 256 threads to 1024, the maximum. The grid dimensions adjust
automatically, since they are computed from `block`. The boundary-condition kernel is launched with
its own 256-thread configuration and is unaffected.

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
Elapsed (max over ranks): 2.309 s  |  Throughput: 58135.17 MCUPS
Mass: initial=6.710936665e+07, final=6.710936646e+07, rel.err=2.929e-09
Min(h) after run: 0.981776
```

| Ranks | MCUPS at 16x16 | MCUPS at 32x32 | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 22238.83 | 28981.84 | 1.30x | -- |
| 2 | 43377.47 | 58135.17 | 1.34x | 100.3 percent |
| 4 | 79762.89 | 105226.03 | 1.32x | 90.8 percent |

A 1.34x gain at two GPUs, and much the same 1.30x on one. That the two agree is the useful part: this
change is purely local to the kernel, so it should help every rank equally and leave the
communication cost alone, and the numbers say it did.

## Did VALUBusy improve?

```bash
mpirun -n 2 --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh \
    rocprofv3 --pmc VALUBusy -T -f csv \
    -o results_%env{OMPI_COMM_WORLD_RANK}% -- ./shallow_mpi
```

The median for `compute_rhs` rises from 67.3 to 96.4 percent on rank 0, and from 67.1 to 96.1 on
rank 1. That is a 43 percent relative gain in vector-unit utilization from a one-line change to the block
dimensions, and it tracks the 34 percent throughput gain in the table above.

Read the two figures together rather than separately. The kernel issues exactly the same instructions
in both stages, so nothing about the arithmetic changed; what changed is how much of the time those
units had data to work on. At 96 percent there is very little issue slack left, which is why stage 3
has to look somewhere other than the vector units for its next gain.

A block that covers a 32x32 patch of the grid has a smaller perimeter relative to its area than a
16x16 patch, so proportionally fewer of the stencil's neighbour reads fall outside the block and have
to come from memory instead of cache.

## Roofline

```bash
rocprof-compute profile -n 2_block_32x32 --no-roof -k compute_rhs \
    --iteration-multiplexing -- ./shallow_mpi
rocprof-compute analyze -p workloads/2_block_32x32/0
```

Both plots below are the novice measurements of the same kernel at the two block shapes, reused for
[the reason given in stage 1](../1_larger_domain/README.md#is-the-gpu-busy-now): a roofline
characterizes the kernel, not the decomposition. 16x16 on the left, 32x32 on the right.

<p>
<img src="../../figs/roofline_2048.png" alt="Roofline of compute_rhs with 16x16 blocks, before this stage" width="49%" />
<img src="../../figs/roofline_block_32x32.png" alt="Roofline of compute_rhs with 32x32 blocks, after this stage" width="49%" />
</p>

## What happened to occupancy and cache reuse?

For the numbers behind the plots, `analyze` accepts more than one workload. Giving it both puts every
metric side by side with the change between them, which is easier to read than two separate reports:

```bash
rocprof-compute analyze \
    -p ../1_larger_domain/workloads/1_larger_domain/0 \
    -p workloads/2_block_32x32/0 \
    -b 2.1.0 2.1.9 2.1.14 2.1.15 2.1.18 2.1.19 2.1.20 2.1.21
```

`-b` takes the metric ids in the leftmost column of the report and keeps only those rows, so the
eight below are all that comes back instead of every section the tool knows how to print.

Section 2.1, System Speed-of-Light, is the summary. 16x16 first, 32x32 second:

| Metric | 16x16 | 32x32 | Change |
|---|---|---|---|
| 2.1.0 VALU FLOPs | 11122 GFLOP/s | 14632 GFLOP/s | 31.6 percent |
| 2.1.9 VALU Utilization | 82.64 percent | 115.88 percent | 40.2 percent |
| 2.1.14 IPC | 0.79 | 1.06 | 32.7 percent |
| 2.1.15 Wavefront Occupancy | 70.99 percent | 74.78 percent | 5.3 percent |
| 2.1.18 vL1D Cache Hit Rate | 57.44 percent | 71.07 percent | 13.6 points |
| 2.1.19 vL1D Cache BW | 13886 GB/s | 18269 GB/s | 31.6 percent |
| 2.1.20 L2 Cache Hit Rate | 18.44 percent | 20.66 percent | 2.2 points |
| 2.1.21 L2 Cache BW | 5930 GB/s | 5270 GB/s | -11.1 percent |

The rows make one chain. The squarer block keeps more of the stencil's neighbour reads in the near
cache, so the vL1D hit rate gains 14 points, less traffic goes out to L2 and its bandwidth falls 11
percent, and the vector units, waiting less often, issue a third more instructions per cycle. The
FLOP rate rises by that same third, from 18.15 to 23.88 percent of the MI300A vector peak, with no
arithmetic changed anywhere.

Occupancy is the row that did least, and that is worth noticing. It is not the objective, only a
proxy for one particular failure mode, having too little work in flight to hide latency. Five percent
does not explain a 1.30x speedup, while the cache-hit change does.

VALU utilization above 100 percent is not an error. MI300A can co-issue VALU instructions, so more
than one can retire per cycle, while the metric is expressed against a single-issue peak.

## What we learned, and what to do about it

There is still stall time to recover and we are still memory bound. So far we have only changed how
*many* threads are in a block, not how they are arranged.

Arrangement matters because consecutive threads in a workgroup map to consecutive lanes of a
wavefront, and a wavefront is 64 lanes wide on AMD GPUs. With a 32x32 block, each row of the block is
only 32 threads, so a single wavefront straddles two rows of the grid and each memory request covers
two disjoint stretches of memory. Making the block 64 wide and only 4 tall lines each wavefront up
with one contiguous run of 64 cells, which is both a better fit for cache lines and a longer
consecutive read.

Continue to [`3_block_64x4`](../3_block_64x4). Note that this also takes the block back down from
1024 threads to 256, so it is a change of shape and size at once.
