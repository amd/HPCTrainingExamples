- [GPU Exercise 2 (OpenMP offload): find the missing data mapping](#org0864a9c)
  - [Build](#org679f617)
  - [Reproduce the fault on the device](#orgb35673d)
  - [Diagnose](#orga80c800)
  - [Fix and confirm](#org6f00329)


<a id="org0864a9c"></a>

# GPU Exercise 2 (OpenMP offload): find the missing data mapping

`saxmy.f90` offloads `y(i) = a*x(i)*y(i)` to the GPU with a Fortran `!$omp target teams distribute`. It faults in the offload region. Find out why.


<a id="org679f617"></a>

## Build

```sh
amdflang -g -O0 -fopenmp --offload-arch=gfx942 -o saxmyf saxmy.f90
```

-   `amdflang` has no `-ggdb` (that is a Clang/GCC flag); `-g` is the Fortran way.
-   `-fopenmp --offload-arch=gfx942` turns the `!$omp target` region into a GPU kernel for MI300A.


<a id="orgb35673d"></a>

## Reproduce the fault on the device

Force the work onto the GPU so a fault is provably on the device (not a silent host fall-back):

```sh
OMP_TARGET_OFFLOAD=MANDATORY rocgdb ./saxmyf
```

```
(gdb) break saxmy.f90:26   # the compute line, y(i) = a*x(i)*y(i)
(gdb) run                  # (answer "y" to make the breakpoint pending)
(gdb) info threads         # the outlined kernel runs as GPU waves
(gdb) info dispatches      # the launch grid for the target region
```

-   A kernel breakpoint takes *two* locations (host fall-back + device). The hit with `lanes [0-...]` is the device one.


<a id="orga80c800"></a>

## Diagnose

-   The loop reads `x(i)` and writes `y(i)`. Look at the `!$omp target` clause: only `y` is mapped (`map(tofrom:y)`). `x` is *never copied to the device*, so the kernel reads an address that was never mapped: a memory fault.
-   `set amdgpu precise-memory on` before `run` pins the fault to the exact instruction if you want to see it land on the load of `x`.


<a id="org6f00329"></a>

## Fix and confirm

-   Add `map(to:x)` to the directive. Compare with `saxmy_fixed.f90`:

```sh
diff saxmy.f90 saxmy_fixed.f90
amdflang -g -O0 -fopenmp --offload-arch=gfx942 -o saxmyf_fixed saxmy_fixed.f90
OMP_TARGET_OFFLOAD=MANDATORY ./saxmyf_fixed    # runs clean: y(1) = 4.0
```
