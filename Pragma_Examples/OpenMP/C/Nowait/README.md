# OpenMP `target ... nowait` + `depend` example (C)

> **Scope of this example.** This example is here only to show that
> `target ... nowait` lets the **host** continue past a kernel launch
> instead of blocking until the GPU is done. It does **not** attempt
> to demonstrate kernel-to-kernel concurrency on the GPU. For kernel-to-kernel concurrency 
> see:
>
> ```
> Pragma_Examples/OpenMP/C/3_vecadd/5_vecadd_async        (explicit map)
> Pragma_Examples/OpenMP/C/3_vecadd/4_vecadd_async_usm    (USM variant)
> ```

## Build & run

```bash
export CC=amdclang   # or clang / gcc / cc
make
./nowait
make clean
```

Data is moved explicitly in this example so `HSA_XNACK=1` should not be exported. 

## Note on kernel sizing

The arrays in this example are sized at `N = 1 << 20`, and the host
loop that runs alongside the kernels is a deliberately long serial
`sin()` accumulation. Both numbers are picked so that the GPU work
and the host work each take long enough to be measurable, which is
what makes the asynchronous behavior of `nowait` visible in
wall-clock time. A long serial `sin()` loop on the CPU is **NOT** 
how you would write a real host code.

There is a companion regression test (`tests/openmp_c_nowait.sh`) sized
even more aggressively: each GPU thread does an inner serial chain
of many `sin()/cos()` ops. That makes the GPU kernel slow enough
that the host return time can be compared to it cleanly. It is not a
model of how to write performant GPU code -- the inner serial chain
deliberately under-uses the available parallelism so that the
*timing* property we want to test (host returns immediately) is
observable.

In short: this example and its regression test are about the
synchronization semantics of `nowait`, not about throughput. 

## Reference

The pattern used in this example is taken from the OpenMP 6.0
examples document, "nowait clause" example:
<https://www.openmp.org/wp-content/uploads/openmp-examples-6.0.pdf>.
