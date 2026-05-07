# OpenMP `target ... nowait` test (Fortran)

This test exercises the OpenMP `nowait` clause on a
`!$omp target teams distribute parallel do` construct, i.e. it checks
that when `nowait` is specified the host does NOT wait for the GPU
kernel to finish before continuing.

The pattern follows the OpenMP 6.0 specification example for the
`nowait` clause: the kernel is launched from inside a `!$omp parallel`
region by the masked thread, while the rest of the team (and
eventually the masked thread itself) executes a CPU `!$omp do` in
parallel with the still-running GPU kernel.

The program first times the same kernel run synchronously to obtain
`t_kernel`, then runs the spec-style pattern and measures
`t_target_return` -- the elapsed time between the masked thread
encountering the `target ... nowait` line and getting control back.
PASS iff (a) the kernel results match the synchronous run AND (b)
`t_target_return` is much smaller than `t_kernel`. If `nowait` is not
honored, `t_target_return` ~= `t_kernel` and the test prints `FAIL!`.

## Build & run

```bash
export FC=amdflang   # or flang / gfortran / ftn
make
OMP_NUM_THREADS=8 ./nowait
make clean
```

The Makefile follows the same compiler-detection style as the other
`Pragma_Examples/OpenMP/Fortran` tests and picks up `$FC`. The test
does not require `HSA_XNACK=1`: data is moved with explicit `map(to:)`
/ `map(from:)` clauses rather than unified shared memory.

## Pitfall: do not use `target ... nowait` outside a parallel region

`!$omp target ... nowait` generates a *target task*. The OpenMP
standard allows the runtime to execute that task either as a deferred
task picked up by another thread or as an *included* task that runs
synchronously in the encountering thread. With no surrounding
`!$omp parallel` region there is no other thread available, and
current AMD ROCm runtimes take the included path -- meaning the host
silently waits for the kernel to finish even though `nowait` was
written. The `parallel` + `masked` wrapping in this test is what
gives the runtime a team to defer the target task on, and is the
required idiom for actually obtaining host/GPU overlap.

## Reference

The pattern used in this test is taken from the OpenMP 6.0 examples
document, "nowait clause" example:
<https://www.openmp.org/wp-content/uploads/openmp-examples-6.0.pdf>.
