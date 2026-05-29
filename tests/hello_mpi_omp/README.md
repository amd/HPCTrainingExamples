# hello_mpi_omp

Self-contained MPI + OpenMP "hello" program used by the `Affinity_MPI*` ctest
entries (`tests/affinity_mpi*.sh`). Vendored here because the upstream
`code.ornl.gov/olcf/hello_mpi_omp` mirror is not reachable from every CI host.

## Build

```bash
make                       # builds ./hello_mpi_omp with mpicc -fopenmp
make COMP=mpicc CFLAGS="-O2 -fopenmp"   # explicit
make clean
```

## Run

Each MPI rank spawns `OMP_NUM_THREADS` OpenMP threads; every thread prints
`MPI <rank> - OMP <tid>` (zero-padded to 3 digits), which is what the
`Affinity_MPI*` pass regexes match on.

```bash
OMP_NUM_THREADS=2 mpirun -np 4 ./hello_mpi_omp
```
