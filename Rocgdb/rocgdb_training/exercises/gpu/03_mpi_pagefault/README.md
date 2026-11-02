# GPU Exercise 3 (MPI): find the rank that faults on the device

`mpi_saxmy.cpp` runs the `saxmy` kernel (`y[i] = a*x[i]*y[i]`, a scaled Hadamard product - deliberately NOT a BLAS routine) on **every** MPI rank, each on its own GPU. It crashes - but only on **one** rank. Your job: use `rocgdb` (driven per-rank, non-interactively) to find **which** rank and **why**.


## Build

`mpi_saxmy.cpp` is HIP C++, so it must be compiled by `amdclang++` (it needs `-x hip` and `--offload-arch`) *plus* the MPI include/link flags. The snag: the MPI wrapper `mpicxx` may be configured to call a *different* back-end compiler (often plain `g++` or `clang++`), which cannot build HIP. Pick whichever option fits your MPI:

a) **OpenMPI - drive `amdclang++` yourself.** Pull the MPI flags out of the wrapper with `--showme` and hand them to `amdclang++`. (`--showme` is OpenMPI-only; MPICH has no equivalent, so this option needs OpenMPI.)

```sh
amdclang++ -x hip -ggdb -O0 --offload-arch=gfx942 \
  $(mpicxx --showme:compile) $(mpicxx --showme:link) -o mpi_saxmy mpi_saxmy.cpp
```

b) **OpenMPI - point the wrapper at `amdclang++`.** `OMPI_CXX` overrides the compiler OpenMPI's `mpicxx` invokes (the MPICH equivalent is `MPICH_CXX`):

```sh
OMPI_CXX=amdclang++ mpicxx -x hip -ggdb -O0 --offload-arch=gfx942 -o mpi_saxmy mpi_saxmy.cpp
```

c) **`mpicxx` already wraps `amdclang++`** (e.g. Cray `PrgEnv-amd`, or a site module that sets it up) - just call it directly:

```sh
mpicxx -x hip -ggdb -O0 --offload-arch=gfx942 -o mpi_saxmy mpi_saxmy.cpp
```

-   Use the arch of your GPU (`gfx942` = MI300A, `gfx950` = MI355X, &#x2026;).
-   On Cray, the `CC` wrapper already carries the MPI flags: `CC -x hip -ggdb ...`.


## Reproduce and locate the fault (per-rank wrapper)

We debug an MPI job here as if it was a batch job: wrap every rank in `rocgdb` with the per-rank wrapper from the CPU MPI exercise (it writes one `gdb_out_<rank>.txt` per rank, no interleaving):

```sh
# the wrapper lives with the CPU MPI exercise; run it in place (no copy needed)
mpirun -np 2 ../../cpu/06_batch_mpi/gdb_mpi_wrapper_all_ranks.sh ./mpi_saxmy
tail -n +1 gdb_out_*.txt        # read each rank's log from the end
```

-   The wrapper sets `catch signal SIGSEGV` and `set amdgpu precise-memory`, runs, then dumps `thread apply all bt` and `info dispatches`.


## Diagnose

-   Exactly one rank's log shows a GPU memory fault; the others just print `done`.
-   In the faulting rank's log, read the frame at `mpi_saxmy.cpp:20` (`y[i] = ...`): which pointer is null? (`print y`, or read the frame args.)
-   Look at `main`: `hipMalloc(&y, ...)` is guarded by `if (rank !` size - 1)=, so the **last** rank never allocates `y` - it launches the kernel with a null `y` and faults on the store.


## Fix and confirm

-   Allocate `y` on every rank (drop the `if`, or allocate unconditionally). Then:

```sh
mpirun -np 2 ./mpi_saxmy      # every rank prints "rank N done", no fault
```
