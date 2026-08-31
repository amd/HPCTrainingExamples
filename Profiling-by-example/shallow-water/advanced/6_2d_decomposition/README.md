
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

The process grid comes from MPI. Rather than deriving neighbours from `rank ± 1`, the ranks are
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

Both grid directions are now decomposed, so `nx` and `x_off_glob` are derived the same way `ny`
and `y_off_glob` always were, and `init_gaussian` takes the extra offset so the bump still lands in
the middle of the global domain.

The boundary conditions became conditional in x. In the slab version every rank owned the full
width, so the left and right ghost columns were always physical walls. Now they are walls only for
ranks on the edge of the process grid, so `apply_reflect_bc_phys` takes `isLeft` and `isRight`
alongside `isBottom` and `isTop`.

The halo exchange gained two directions. The pack and unpack kernels handle all four in a single
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
mpirun -n 4 --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh ./shallow_mpi
```

## Expected output

```
MPI ranks: 4  |  GPUs detected: 1
Process grid: 2 x 2 (x by y), local subdomain 4096x4096
Domain: 8192x8192 (global), steps=500, dt=0.0728643
Elapsed (max over ranks): 1.008 s  |  Throughput: 133090.52 MCUPS
Mass: initial=6.710936665e+07, final=6.710936646e+07, rel.err=2.916e-09
Min(h) after run: 0.981776
```

The extra line reports the grid MPI chose, which is worth checking on every run: an unexpected
factorization is the first thing to suspect if the numbers look wrong.

| Ranks | Process grid | MCUPS, slab | MCUPS, tile | Ratio | Efficiency |
|---|---|---|---|---|---|
| 1 | 1 x 1 | 36223.84 | 36159.43 | 1.00x | -- |
| 2 | 1 x 2 | 69789.74 | 69260.14 | 0.99x | 95.8 percent |
| 4 | 2 x 2 | 136644.72 | 133090.52 | 0.97x | 92.0 percent |

The 2D decomposition is 3 percent slower than the slab at four GPUs, and that is the expected
result rather than a disappointment.

The halo volume table says a 2x2 tiling moves half as many cells per rank as a four-way slab, and it
does. It is still not faster, which means halo volume is not what four GPUs on one MI300A node are
limited by. They talk over the on-package interconnect, where halving a 16384-cell message saves
almost nothing, while the tiling adds two costs that are real at any scale: the x-direction halo is a
strided gather with a stride of one row pitch rather than a contiguous row copy, and each rank now has
up to four neighbours to post messages to instead of two.

So the tiling trades a cheap, contiguous, high-volume exchange for a more expensive, strided,
low-volume one. That is a good trade exactly when volume is what costs you, which is to say when
enough ranks are far enough apart that a real network is involved.

## Finding the crossover

We do not measure the crossover here. Every number in this tutorial comes from a single node, and
`1/sqrt(N)` needs 16 ranks or more before it opens a gap worth measuring, so finding it is left as an
exercise. Given a wider allocation, run both stages over as many ranks as you can get and plot the
two efficiency curves against rank count. The slab curve should fall away steadily, since its
per-rank communication is constant, while the tile curve should hold up.

```bash
for n in 4 8 16 32; do
    mpirun -n $n --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh ../5_halo_pipeline/shallow_mpi
    mpirun -n $n --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh ./shallow_mpi
done
```

Those curves say whether the trade paid. They do not say whether the halo region itself shrank, which
is what the arithmetic actually predicts, and for that we need the timeline again. Both stages carry
the same ROCTx ranges, so tracing them the same way at 16 ranks puts `halo_mpi_wait` beside
`rhs_interior` in each:

```bash
salloc -N 4 -p LocalQ --exclusive --gres=gpu:4 -t 1:00:00

mpirun -n 16 --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh \
    rocprof-sys-run --preset=trace-hpc --flat-profile \
    --selected-regions step_3,step_4,step_5 -o trace_slab -- ../5_halo_pipeline/shallow_mpi

mpirun -n 16 --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh \
    rocprof-sys-run --preset=trace-hpc --flat-profile \
    --selected-regions step_3,step_4,step_5 -o trace_tile -- ./shallow_mpi
```

Read the pair the way
[stage 5 reads its four-rank trace](../5_halo_pipeline/README.md#confirming-it-on-the-timeline),
asking how much of each RK4 stage the exchange occupies. In the slab we expect it to have grown into a
band wide enough to see, since a rank there sends two full rows of 8192 cells however many ranks
there are while the interior it hides behind keeps shrinking. In the tile we expect it to still be
covered by `rhs_interior`. The flat `wall_clock` report turns the same question into numbers, one row
per range with a count and per-occurrence statistics, so the `halo_mpi_wait` totals of the two runs
can be compared against their own `step_*` totals rather than eyeballed in Perfetto.

## Looking at the network itself

`rocprof-sys` can attribute traffic to individual network interfaces during MPI calls, which is how
the argument above becomes a measurement. The
[communication-performance section](https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/advanced/README.html#bonus-step-studying-communication-performance)
of the ROCm profiling-guide blog is the reference for the setup. The counters come through PAPI, so
`/proc/sys/kernel/perf_event_paranoid` has to be 2 or less, and CPU sampling has to be enabled.

Find the interface first, since the name differs from machine to machine:

```bash
rocprof-sys-avail -H -r net
```

Then configure through the environment and trace twenty steps on 16 ranks across four nodes:

```bash
salloc -N 4 -p LocalQ --exclusive --gres=gpu:4 -t 1:00:00

export ROCPROFSYS_NETWORK_INTERFACE=enp129s0
export ROCPROFSYS_PAPI_EVENTS="net:::enp129s0:rx:byte net:::enp129s0:rx:packet net:::enp129s0:tx:byte net:::enp129s0:tx:packet"
export ROCPROFSYS_TIMEMORY_COMPONENTS="wall_clock network_stats"
export ROCPROFSYS_USE_SAMPLING=true
export ROCPROFSYS_SAMPLING_FREQ=100

STEPS=$(printf "step_%s," {10..29})
mpirun -n 16 --map-by ppr:1:numa --bind-to numa ../gpu_bind.sh \
    rocprof-sys-run --preset=trace-hpc --flat-profile --selected-regions "${STEPS%,}" \
    -o nic -- ./shallow_mpi
```

Each rank writes `network_stats*.txt` and `papi_array*.txt` alongside the usual trace, and the traffic
appears as extra rows on the Perfetto timeline. Sampling plus counters is expensive, so treat this
run's wall clock as profiling overhead rather than a throughput measurement. The totals must not be
summed over ranks: every process on a node samples the same node-level interface, and its
process-level scope covers the whole run rather than the selected ranges.

We record the recipe rather than results. Running both stages under these counters and comparing the
bytes and packets each one puts on the wire belongs with the crossover measurement above, at the rank
counts where the two decompositions are expected to differ.

## What we learned

Pulling the whole tutorial together:

- Multi-GPU makes the diagnosis harder, not just the code. It was not obvious at the outset
  whether this application was communication-bound or compute-bound, and the answer changed as we
  optimized. Stage 0 was communication-bound because the problem was too small; stages 1 through 4
  were compute-bound; stage 4 ended communication-bound again, this time because the compute had got
  fast.
- Optimize the GPU-only part first. Not because it matters more, but because you cannot judge the
  communication's share of the runtime until the compute has stopped moving.
- Different questions need different tools. Counters found the block-shape problems, a kernel trace
  ranked the kernels, a thread trace found the redundant loads inside one kernel, and only a
  host-and-device timeline could see the MPI cost at all. No single tool would have found more than a
  fraction of this.
- Network optimization means rewriting code. Stages 5 and 6 are restructurings, not one-line
  changes, and stage 6 changes the meaning of a rank's data. This is the main reason communication
  problems are best identified early even if they are fixed late.
- Quantify every change, at more than one scale. Stage 5 looks worthless at two ranks and is worth
  1.09x at four. Stage 6 loses three percent at four and can only pay at rank counts this machine cannot
  reach. A single configuration would have given the wrong answer about both.
- Keep a correctness check that a race would break. The mass conservation check is the only reason
  the stale-halo bug in stage 5 was ever noticed, and it cost nothing to carry.

Cumulatively, from stage 1 where the problem first became a sensible size, four GPUs went from 79763
to 136645 MCUPS at stage 5, a 1.71x speedup: 1.57x of it from kernel work and 1.09x from
restructuring the communication.
