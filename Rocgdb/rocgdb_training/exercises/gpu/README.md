Hands-on labs for **Part II** (Basics of rocgdb). Each `NN_name/` directory is self-contained: a `README` with the build line and a walkthrough, and (01-03) the buggy program to debug and a `*_fixed` reference; 04 debugs a program that is already in this repository. There is no shared `common/`.

See `../README` for how to get onto a GPU node and load the toolchain.


# Layout

-   `01_pagefault/` - a HIP `saxmy` that page-faults on the device; find out why.
-   `02_omp_offload/` - a Fortran and C `!$omp target` `saxmy` that page-faults because `x` is never allocated (a null pointer on the device); find and fix it.
-   `03_mpi_pagefault/` - an MPI + HIP `saxmy` where exactly one rank passes a null device pointer; find which rank and why (per-rank `rocgdb` wrapper).
-   `04_ghost_exchange/` - an MPI + HIP halo exchange (GhostExchange) that runs without an error but leaves some cells unset; find the bad kernel launch. Its code is not copied here: it is this repository's `MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver1WithBug`.

Each lab: build the buggy program, reproduce the problem under `rocgdb`, locate it, then confirm your diagnosis against the `*_fixed` version (04: against the fix at the end of its `README`).
