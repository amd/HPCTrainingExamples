# OpenMP nowait clause example (Fortran)

> This example shows that
> `target ... nowait` lets the **host** continue past a kernel launch
> instead of blocking until the GPU is done. 
> For other relevant examples see:
>
> ```
> Pragma_Examples/OpenMP/Fortran/3_vecadd/5_vecadd_async        (explicit map)
> Pragma_Examples/OpenMP/Fortran/3_vecadd/4_vecadd_usm_async    (USM variant)
> ```

## Build & run

```bash
export FC=amdflang   # or flang / gfortran / ftn
make
./nowait
make clean
```

Data is moved with explicit `map(to:)` / `map(from:)`, so `HSA_XNACK=1` should not be exported. 

## Note on kernel sizing

The arrays in this example are sized at `N = 2**20`, and the host
loop that runs alongside the kernels is a deliberately long serial
`sin()` accumulation. Both numbers are picked so that the GPU work
and the host work each take long enough to be measurable, which is
what makes the asynchronous behavior of `nowait` visible in
wall-clock time. 

## Reference

The pattern used in this example is taken from the OpenMP 6.0
examples document, "nowait clause" example:
<https://www.openmp.org/wp-content/uploads/openmp-examples-6.0.pdf>.
