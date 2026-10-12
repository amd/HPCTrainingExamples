- [Layout](#org6902795)

Hands-on labs for **Part II** (Basics of rocgdb). Each `NN_name/` directory is self-contained: it ships the buggy program to debug, a `*_fixed` reference, and a `README.org` with the build line and a walkthrough. There is no shared `common/`.

See `../README.org` for how to get onto a GPU node and load the toolchain.


<a id="org6902795"></a>

# Layout

-   `01_pagefault/` - a HIP `saxmy` that page-faults on the device; find out why.
-   `02_omp_offload/` - a Fortran and C `!$omp target` `saxmy` that page-faults because `x` is never allocated (a null pointer on the device); find and fix it.
-   `03_mpi_pagefault/` - an MPI + HIP `saxmy` where exactly one rank passes a null device pointer; find which rank and why (per-rank `rocgdb` wrapper).

Each lab: build the buggy program, reproduce the fault under `rocgdb`, locate it, then diff against the `*_fixed` version to confirm your diagnosis.
