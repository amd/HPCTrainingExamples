
# Stage 6: From slab to tile

Every stage so far has kept the same 1D slab decomposition and improved what happens inside it. This
one changes the decomposition itself, which is the most invasive change in the tutorial and the only
one that cannot be motivated by a profiler alone. The argument for it is arithmetic, and the profiler's
job is to tell you when the arithmetic has become the binding constraint.

## The argument

In a 1D slab, a rank owns all `NXG` columns and a band of rows. Its halo is two rows of `NXG` cells,
and that is true no matter how many ranks there are. Add ranks and the work per rank falls while the
communication per rank stays exactly where it was.

In a 2D tiling, a rank owns a tile of roughly `L/sqrt(N)` on a side, so its perimeter, and with it the
halo volume, falls as `1/sqrt(N)`.

For the 8192x8192 domain, counting the cells the busiest single rank has to send per exchange:

| Ranks | Tile grid | Local subdomain | Slab | Tile |
|---|---|---|---|---|
| 2 | 1 x 2 | 8192x4096 | 8192 | 8192 |
| 4 | 2 x 2 | 4096x4096 | 16384 | 8192 |
| 8 | 2 x 4 | 4096x2048 | 16384 | 10240 |
| 16 | 4 x 4 | 2048x2048 | 16384 | 8192 |
| 64 | 8 x 8 | 1024x1024 | 16384 | 4096 |

The slab column is flat by construction. The tile column drifts downward, and the gap widens without
limit as the job gets wider. At two ranks the two are the same thing, which is a useful sanity check
rather than a coincidence: `MPI_Dims_create` factors 2 as a 1x2 grid, which *is* a slab.

## What changed

```bash
diff ../5_halo_pipeline/shallow_mpi.hip shallow_mpi.hip
```

**The process grid comes from MPI.** Rather than deriving neighbours from `rank ± 1`, the ranks are
arranged on a Cartesian communicator and the neighbours come from it:

```c++
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm);
    ...
    MPI_Cart_shift(comm, 0, 1, &nb.down, &nb.up);
    MPI_Cart_shift(comm, 1, 1, &nb.left, &nb.right);
```

`MPI_Dims_create` picks the most nearly square factorization on its own, and the `reorder` argument
set to 1 lets the MPI implementation renumber ranks to suit the machine's topology. Letting MPI choose
is worth doing: it knows which ranks share a node and which have to cross a network, and this code
does not.

**Both grid directions are now decomposed**, so `nx` and `x_off_glob` are derived the same way `ny`
and `y_off_glob` always were, and `init_gaussian` takes the extra offset so the bump still lands in
the middle of the global domain.

**The boundary conditions became conditional in x.** In the slab version every rank owned the full
width, so the left and right ghost columns were always physical walls. Now they are walls only for
ranks on the edge of the process grid, so `apply_reflect_bc_phys` takes `isLeft` and `isRight`
alongside `isBottom` and `isTop`.

**The halo exchange gained two directions.** The pack and unpack kernels handle all four in a single
launch, with the x-direction gathering a strided column rather than a contiguous row:

```c++
    if (t < ny) {
        if (pack_left) {
            int id = IDX(1, t + 1, pitch);
            send_left[base + 0] = h [id];
```

Note what is *not* there: no diagonal exchange. The five-point stencil reads `(i±1, j)` and
`(i, j±1)` and never a ghost corner, so the corners are left undefined and the corner handling that
the slab version's boundary kernel carried has been dropped.

## Build and run

```bash
module load rocm openmpi
make
mpirun -n 4 --bind-to none ../gpu_bind.sh ./shallow_mpi
```

## Expected output

```
MPI ranks: 4  |  GPUs detected: 1
Process grid: 2 x 2 (x by y), local subdomain 4096x4096
Domain: 8192x8192 (global), steps=500, dt=0.0728643
Elapsed (max over ranks): 1.019 s  |  Throughput: 131756.46 MCUPS
Mass: initial=6.710936665e+07, final=6.710936646e+07, rel.err=2.916e-09
Min(h) after run: 0.981776
```

The extra line reports the grid MPI chose, which is worth checking on every run: an unexpected
factorization is the first thing to suspect if the numbers look wrong.

| Ranks | Process grid | MCUPS, slab | MCUPS, tile | Ratio | Efficiency |
|---|---|---|---|---|---|
| 1 | 1 x 1 | 36127.12 | 36129.05 | 1.00x | -- |
| 2 | 1 x 2 | 69515.03 | 69000.77 | 0.99x | 95 percent |
| 4 | 2 x 2 | 136293.57 | 131756.46 | 0.97x | 91 percent |

**The 2D decomposition is 3 percent slower than the slab at four GPUs.** This is a negative result and
it is kept deliberately, because the reason for it is more instructive than a win would have been.

The halo volume table says a 2x2 tiling should move half as many cells per rank as a four-way slab. It
does. It is still slower, which means halo volume is not what four GPUs on one MI300A node are limited
by. Those four GPUs talk over the on-package interconnect at a bandwidth high enough that halving a
16384-cell message saves almost nothing, while the tiling adds two costs that are real at any scale:
the x-direction halo is a strided gather with a stride of one row pitch, which is far less efficient
than the contiguous row copy the slab needs, and each rank now has up to four neighbours to post
messages to instead of two.

So the tiling trades a cheap, contiguous, high-volume exchange for a more expensive, strided,
low-volume one. That is a good trade exactly when volume is what costs you, which is to say when the
ranks are far enough apart that a real network is involved. It is a bad trade inside a single node.

The honest conclusion is that this stage's change is right and its measurement is inconclusive,
because the measurement was taken in the regime where the change cannot pay. The crossover sits beyond
the four GPUs a single node provides, and reproducing it needs a job wide enough that ranks span
several nodes.

## Where the crossover is

To find it on your own machine, run both stages across as many nodes as you can get and plot the two
efficiency curves against rank count. The slab curve should fall away steadily, since its per-rank
communication is constant, while the tile curve should hold up.

```bash
salloc -N 4 -p LocalQ --exclusive --gres=gpu:4 -t 1:00:00
for n in 8 16 32; do
    mpirun -n $n --bind-to none ../gpu_bind.sh ../5_halo_pipeline/shallow_mpi
    mpirun -n $n --bind-to none ../gpu_bind.sh ./shallow_mpi
done
```

<!-- MEASUREMENT TODO: slab vs tile scaling curve beyond 4 ranks, ideally to 16 -->

## Looking at the network itself

`rocprof-sys` can attribute traffic to individual network interfaces during MPI calls, which turns the
argument above into a measurement. It needs PAPI support and the relevant counters enabled in
`rocprofsys.cfg`; the
[ROCm documentation](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/) covers the
configuration.

<!-- MEASUREMENT TODO: per-NIC traffic comparison, slab vs tile, once multi-node runs are available -->

Comparing the per-NIC traces of stages 5 and 6 at the same rank count is the most direct evidence
there is that the decomposition, and not the code, determines what the network has to carry.

## What we learned

Pulling the whole tutorial together:

- **Multi-GPU makes the diagnosis harder, not just the code.** It was not obvious at the outset
  whether this application was communication-bound or compute-bound, and the answer changed as we
  optimized. Stage 0 was communication-bound because the problem was too small; stages 1 through 4
  were compute-bound; stage 4 ended communication-bound again, this time because the compute had got
  fast.
- **Optimize the GPU-only part first.** Not because it matters more, but because you cannot judge the
  communication's share of the runtime until the compute has stopped moving.
- **Different questions need different tools.** Counters found the block-shape problems, a kernel trace
  ranked the kernels, a thread trace found the redundant loads inside one kernel, and only a
  host-and-device timeline could see the MPI cost at all. No single tool would have found more than a
  fraction of this.
- **Network optimization means rewriting code.** Stages 5 and 6 are restructurings, not one-line
  changes, and stage 6 changes the meaning of a rank's data. This is the main reason communication
  problems are best identified early even if they are fixed late.
- **Quantify every change, at more than one scale.** Stage 5 looks worthless at two ranks and
  excellent at four. Stage 6 looks worthless at four and should look excellent at sixty-four. A single
  configuration would have given the wrong answer about both.
- **Keep a correctness check that a race can fail.** The mass conservation check is the only reason
  the stale-halo bug in stage 5 was ever noticed, and it cost nothing to carry.

Cumulatively, from stage 1 where the problem first became a sensible size, four GPUs went from 79749
to 136294 MCUPS, a **1.71x** speedup: 1.57x of it from kernel work and 1.09x from restructuring the
communication.
