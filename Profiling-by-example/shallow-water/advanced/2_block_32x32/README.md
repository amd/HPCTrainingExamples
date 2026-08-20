
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
mpirun -n 4 --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh ./shallow_mpi
```

## Expected output

```
MPI ranks: 4  |  GPUs detected: 1
Domain: 8192x8192 (global), steps=500, dt=0.0728643
Elapsed (max over ranks): 1.276 s  |  Throughput: 105226.03 MCUPS
Mass: initial=6.710936665e+07, final=6.710936646e+07, rel.err=2.929e-09
Min(h) after run: 0.981776
```

| Ranks | MCUPS at 16x16 | MCUPS at 32x32 | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 22238.83 | 28981.84 | 1.30x | -- |
| 2 | 43377.47 | 58135.17 | 1.34x | 100.3 percent |
| 4 | 79762.89 | 105226.03 | 1.32x | 90.8 percent |

A 1.32x gain at four GPUs, and much the same 1.30x on one. That the two agree is the useful part: this
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

## What happened to occupancy and cache reuse?

The `rocprof-compute` reports show wavefront occupancy moving only modestly, from 70.99 to 74.78
percent. The vector-L1 hit rate makes the larger change, from 57.44 to 71.07 percent, while the L2
hit rate moves from 18.44 to 20.66 percent. That is consistent with the larger 32x32 patch reusing
more neighbour data before it leaves the near cache.

Occupancy is not the objective; it is a proxy for one particular failure mode, namely having too
little work in flight to hide latency. Here it does not explain a 1.30x single-GPU speedup, while
the cache-hit change does.

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

The report measures 14632 GFLOP/s, 23.88 percent of the MI300A VALU peak, up from 11122 GFLOP/s and
18.15 percent in stage 1. Together with the cache-hit rates above, that explains why the wall clock
improves even though no arithmetic changed.

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
