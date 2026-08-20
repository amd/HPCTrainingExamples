
# Stage 5: Rebuild the halo exchange

Four stages of kernel work have made the compute 1.63x faster on a single GPU and, as a direct
consequence, made communication the dominant remaining cost. Scaling efficiency at four GPUs held at
90 to 92 percent through stage 3 and then fell to 86.5. This stage leaves the kernels alone and rebuilds
the exchange around them.

## What is wrong with the old exchange

`exchange_y_halos` in stage 4 does four things badly, all visible in the code:

```c++
    CHECK_CUDA(hipDeviceSynchronize());
    ...
    MPI_Sendrecv(send_bot_h, nx, MPI_FLOAT, rank_down, 100,
                 recv_bot_h, nx, MPI_FLOAT, rank_down, 200, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_bot_hu, nx, MPI_FLOAT, rank_down, 101,
                 recv_bot_hu, nx, MPI_FLOAT, rank_down, 201, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_bot_hv, nx, MPI_FLOAT, rank_down, 102,
                 recv_bot_hv, nx, MPI_FLOAT, rank_down, 202, comm, MPI_STATUS_IGNORE);
```

1. Six messages where two would do. Each of `h`, `hu` and `hv` is sent separately to each of two
   neighbours. Every message pays the same per-message latency, and at 8192 floats each they are small
   enough that latency, not bandwidth, is what you are paying for.
2. Blocking calls. `MPI_Sendrecv` returns only once the data has moved, so the GPU sits idle for
   the entire exchange even though most of the subdomain does not depend on the result.
3. A device-wide synchronization before every exchange, draining all queued work rather than just
   the writes the exchange actually needs.
4. All of it four times per time step, once per RK4 stage.

## What changed

```bash
diff ../4_vectorized_loads/shallow_mpi.hip shallow_mpi.hip
```

Three related changes.

Fuse the three arrays into one message. Two new kernels gather the outgoing boundary rows into a
contiguous buffer as interleaved triplets and scatter the incoming ones back:

```c++
    if (pack_bottom) {
        int id = IDX(t + 1, 1, pitch);
        send_bottom[base + 0] = h [id];
        send_bottom[base + 1] = hu[id];
        send_bottom[base + 2] = hv[id];
    }
```

Six messages per exchange become two, each three times larger.

Make the transfer non-blocking, so the exchange splits into a `begin` that posts
`MPI_Irecv`/`MPI_Isend` and a `finish` that waits and unpacks.

Overlap the transfer with compute. `compute_rhs` becomes `compute_rhs_range`, taking a row range
rather than always covering the whole subdomain, which lets the interior be computed while the halo is
in flight:

```c++
    int reqCount = begin_exchange_y_halos(...);
    // rows 2 .. ny-1 do not touch a ghost cell
    compute_rhs_range<<<grid_interior, block_rhs>>>(..., 2, ny - 1);
    finish_exchange_y_halos(...);
    // the two rows that do
    compute_rhs_range<<<grid_edge, block_rhs>>>(..., 1, 1);
    compute_rhs_range<<<grid_edge, block_rhs>>>(..., ny, ny);
```

Only the first and last interior rows read a ghost cell, so everything between them can proceed
immediately. The device-wide synchronization is gone, replaced by a dedicated non-blocking stream for
the halo work.

## An easy way to get this wrong

The pack kernel runs on `halo.stream` while the RK4 stage updates run on the default stream. Those two
streams are independent by construction, which is the entire point, and that means nothing orders the
pack behind the `update_stage` that produces the values it is supposed to pack. Get this wrong and the
pack samples cells before the update has landed in them, and a stale boundary row goes out over the
network.

The dependency has to be stated explicitly:

```c++
    // The pack reads the interior that the default stream just wrote, and the
    // halo stream is non-blocking, so the dependency has to be explicit.
    CHECK_CUDA(hipEventRecord(halo.ready, 0));
    CHECK_CUDA(hipStreamWaitEvent(halo.stream, halo.ready, 0));
```

This is worth dwelling on because of how the bug presents itself. It is a race, so it does not fail
every time, it does not fail on one rank, and it costs *nothing* in performance when it does fail. The
only thing that catches it is the accuracy check, and it catches it clearly. Without the two lines
above, this code reports:

| Ranks | Mass rel. err | Min(h) |
|---|---|---|
| 1 | 2.916e-09 | 0.981776 |
| 2 | 1.204e-05 | 0.86498 |
| 4 | 4.504e-06 | 0.956389 |

Four orders of magnitude of mass error, appearing only at more than one rank, from two missing lines.
This is the argument for carrying a cheap physical invariant in any code that overlaps communication
with computation. A timer would have called this version the fastest one in the tutorial.

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
Elapsed (max over ranks): 0.982 s  |  Throughput: 136644.72 MCUPS
Mass: initial=6.710936665e+07, final=6.710936646e+07, rel.err=2.916e-09
Min(h) after run: 0.981776
```

The mass error and minimum depth match stage 4 exactly, which is the check that matters: the exchange
was restructured without changing a single arithmetic operation, so the answer should be bit-identical
in aggregate, and it is.

| Ranks | MCUPS at stage 4 | MCUPS now | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 36113.05 | 36223.84 | 1.00x | -- |
| 2 | 70436.30 | 69789.74 | 0.99x | 96.3 percent |
| 4 | 124889.46 | 136644.72 | 1.09x | 94.3 percent |

A 1.09x gain at four GPUs, and 8 points of scaling efficiency recovered, without touching a
kernel that does any physics. Stage 4's kernel speedup, which four GPUs could not use at all, is
also recovered here: the two stages together are worth 1.09x on four GPUs where stage 4 alone was
worth nothing.

Read the other two rows carefully, because they say something the four-rank number hides. On one GPU
there is nothing to exchange and the result is unchanged, as it should be. On two GPUs the code is
very slightly *slower* than stage 4. Each rank has one neighbour, the exchange is short, and there is
not enough of it to hide behind the interior compute, so the extra pack and unpack launches and the
stream synchronization cost marginally more than the overlap saves.

That is not a defect, it is the shape of the optimization. Overlapping communication pays in
proportion to how much communication there is to overlap, so it should be expected to do nothing in
the regime where communication is already cheap. Had we measured only two ranks we would have
concluded this change was not worth making.

## Confirming it on the timeline

```bash
salloc -N 1 -p LocalQ --exclusive --gres=gpu:4 -t 1:00:00
mpirun -n 4 --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh \
    rocprof-sys-run --preset=trace-hpc --flat-profile \
    --selected-regions step_3,step_4,step_5 -o trace -- ./shallow_mpi
```

Trace stage 4 the same way and compare the two. Stage 4 has one opaque `halo_exchange` range per RK4
stage, with nothing running underneath it. Stage 5 replaces that with `rhs_interior` launched before
`halo_mpi_wait`, and in Perfetto the interior `compute_rhs_range` dispatch extends underneath
`MPI_Waitall` rather than starting after it. That overlap is what the 1.09x is made of.

The overlap is something only the timeline shows: the flat `wall_clock` report counts each range and
gives its spread, but not what was running alongside what. The open questions about filtered traces
noted in [stage 0](../0_baseline/README.md#step-3-what-the-kernel-trace-cannot-tell-you) apply here
too.

Across several nodes the NIC counters can be added to this trace, which puts the bytes leaving the
node on the timeline next to these ranges. The configuration is in
[stage 6](../6_2d_decomposition/README.md#looking-at-the-network-itself).

## What we learned, and what to do about it

Communication is cheaper but it has not gone away, and the way it grows is now the problem. In a 1D
slab, each rank's halo is two rows of `NXG` cells regardless of how many ranks there are. Doubling the
rank count halves the work per rank and leaves the communication per rank exactly where it was, so
communication's share of the run grows without bound as the job gets wider. No amount of overlapping
fixes that, because eventually there is nothing left to hide behind.

A 2D tiling changes the arithmetic. Split `N` ranks into a roughly square grid and each rank owns a
tile of side `L/sqrt(N)`, whose perimeter shrinks as `1/sqrt(N)`. The per-rank communication volume
falls as the job gets wider instead of staying flat.

That is a change of decomposition rather than a change of code inside a kernel, and it is the most
invasive thing in this tutorial.

Continue to [`6_2d_decomposition`](../6_2d_decomposition).
