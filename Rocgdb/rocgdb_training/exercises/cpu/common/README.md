Build the shared programs from this directory:

```
$ make            # builds demo, floats, faulty, fusion (amdclang, -ggdb)
$ make mpi        # builds mpi_demo too (needs an MPI compiler)
```

-   `-ggdb` (C) / `-g` (Fortran; amdflang has no `-ggdb`) gives debug info; the debugging programs are built at `-O0` (easiest to step). `fusion` is deliberately built at `-O2` (Chapter 4's wrong-line exercise).
-   On a cluster, swap `amdclang=/=mpicc` for your site's compiler; nothing else is site-specific.
-   Check it worked: `./demo` prints `sum = 31, name length = ...`.
-   The exercises then run rocgdb from their own directory, on `../common/<program>`.
