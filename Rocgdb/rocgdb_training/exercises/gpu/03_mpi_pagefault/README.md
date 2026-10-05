- [GPU Exercise 3 (MPI): find the rank that faults on the device](#orge5d7823)
  - [Build](#org50ac992)
  - [Reproduce and locate the fault (per-rank wrapper)](#orgdcf55ec)
  - [Diagnose](#org9203520)
  - [Fix and confirm](#org4ff42b2)


<a id="orge5d7823"></a>

# GPU Exercise 3 (MPI): find the rank that faults on the device

`mpi_saxmy.cpp` runs the `saxmy` kernel (`y[i] = a*x[i]*y[i]`, a scaled Hadamard product, deliberately NOT a BLAS routine) on *every* MPI rank, each on its own GPU. It crashes, but only on *one* rank. Our task: use `rocgdb` (driven per-rank, non-interactively) to find *which* rank and *why*.


<a id="org50ac992"></a>

## Build

```sh
amdclang++ -x hip -ggdb -O0 --offload-arch=gfx942 \
  $(mpicxx --showme:compile) $(mpicxx --showme:link) -o mpi_saxmy mpi_saxmy.cpp
```

-   Use the arch of your GPU (`gfx942` = MI300A, `gfx950` = MI355X, ...).
-   On Cray, the `CC` wrapper already carries the MPI flags: `CC -x hip -ggdb ...`.


<a id="orgdcf55ec"></a>

## Reproduce and locate the fault (per-rank wrapper)

There are no interactive GPU sessions on Hunter, so debug a real MPI run the way you would in a batch job: wrap every rank in `rocgdb` with the per-rank wrapper from the CPU MPI exercise (it writes one `gdb_out_<rank>.txt` per rank, no interleaving):

```sh
# the wrapper lives with the CPU MPI exercise; run it in place (no copy needed)
mpirun -np 2 ../../cpu/06_batch_mpi/gdb_mpi_wrapper_all_ranks.sh ./mpi_saxmy
tail -n +1 gdb_out_*.txt        # read each rank's log from the end
```

-   The wrapper sets `catch signal SIGSEGV` and `set amdgpu precise-memory`, runs, then dumps `thread apply all bt` and `info dispatches`.


<a id="org9203520"></a>

## Diagnose

-   Exactly one rank's log shows a GPU memory fault; the others just print `done`.
-   In the faulting rank's log, read the frame at `mpi_saxmy.cpp:24` (`y[i] = ...`): which pointer is null? (`print y`, or read the frame args.)
-   Look at `main`: `hipMalloc(&y, ...)` is guarded by `if (rank != size - 1)`, so the *last* rank never allocates `y`: it launches the kernel with a null `y` and faults on the store.


<a id="org4ff42b2"></a>

## Fix and confirm

-   Allocate `y` on every rank (drop the `if`, or allocate unconditionally). Then:

```sh
mpirun -np 2 ./mpi_saxmy      # every rank prints "rank N done", no fault
```
