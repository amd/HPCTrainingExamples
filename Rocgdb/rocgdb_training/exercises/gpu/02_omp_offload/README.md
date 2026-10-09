- [GPU Exercise 2 (OpenMP offload): a null-pointer page fault](#org8afb02e)
  - [Build and reproduce the fault](#org6efcdea)
  - [Find it under rocgdb](#org53b6dd9)
  - [Diagnose](#orgd3158fd)
  - [Fix and confirm](#org2e61a1c)
  - [Takeaway](#org5e45907)


<a id="org8afb02e"></a>

# GPU Exercise 2 (OpenMP offload): a null-pointer page fault

`saxmy` offloads `y(i) = a*x(i)*y(i)` to the GPU with an OpenMP `target teams distribute`. It faults in the offload region. Find out why, then fix it. The exercise ships in both languages - Fortran `saxmy.f90` and C `saxmy.c`; the reference solutions are `saxmy_fixed.f90` and `saxmy_fixed.c`.


<a id="org6efcdea"></a>

## Build and reproduce the fault

```sh
# Fortran
amdflang -g -O0 -fopenmp --offload-arch=gfx942 -o saxmyf saxmy.f90
OMP_TARGET_OFFLOAD=MANDATORY ./saxmyf
# C
amdclang -g -O0 -fopenmp --offload-arch=gfx942 -o saxmyc saxmy.c
OMP_TARGET_OFFLOAD=MANDATORY ./saxmyc
```

-   `amdflang` has no `-ggdb` (that is a Clang/GCC flag); `-g` is the Fortran way.
-   `OMP_TARGET_OFFLOAD=MANDATORY` makes a failed offload an error, so the fault is provably on the device (no silent host fall-back).

Both abort with a runtime memory fault at a *null* address (under any `HSA_XNACK`):

```text
OFFLOAD ERROR: memory access fault by GPU 1 ... at virtual address (nil).
```


<a id="org53b6dd9"></a>

## Find it under rocgdb

`set amdgpu precise-memory on` so the stop lands on the exact faulting load:

```sh
OMP_TARGET_OFFLOAD=MANDATORY rocgdb ./saxmyc        # or ./saxmyf
```

```text
(gdb) set amdgpu precise-memory on
(gdb) run
Thread 6 received signal SIGSEGV, Segmentation fault.
[Switching to thread 6, lane 0 (AMDGPU Lane 1:1:1:1/0 (0,0,0)[0,0,0])]
... __omp_offloading_..._main_l23 (n=..., y=0x7fffe4400000, a=..., x=0x0, ...)
    at saxmy.c:25
25          y[i] = a*x[i]*y[i];
(gdb) print x
$1 = (double *) 0x0
```

The frame (C) shows `x=0x0` outright, and `print x` confirms it: `x` is a null pointer. In Fortran the fault is identical - every wave stops at `saxmy.f90:24`, and the backtrace shows `x` as a descriptor whose data pointer is an invalid address. `info threads` shows all eight waves halted on the same line.


<a id="orgd3158fd"></a>

## Diagnose

-   The kernel reads `x(i)` / `x[i]`, but `x` is **never allocated** - it is a null pointer, so the device dereferences address 0.
-   Why doesn't simply *mapping* `x` save you? An *allocated* `x` on an MI300A is silently demand-paged onto the device even when it is unmapped, so a merely-unmapped (but allocated) `x` would NOT fault. The reliable bug - and the real one here - is the missing **allocation**.


<a id="org2e61a1c"></a>

## Fix and confirm

Allocate, initialise and map `x`. Compare with `saxmy_fixed`:

```sh
diff saxmy.f90 saxmy_fixed.f90
amdflang -g -O0 -fopenmp --offload-arch=gfx942 -o saxmyf_fixed saxmy_fixed.f90
OMP_TARGET_OFFLOAD=MANDATORY ./saxmyf_fixed         # y(1) = 4.0
```


<a id="org5e45907"></a>

## Takeaway

A GPU kernel that reads an unallocated (null) pointer faults at a null address, and rocgdb drops you right on the faulting load with the offending pointer visible in the frame. `set amdgpu precise-memory on` makes the stop precise; `info threads` shows every wave halted at the same line. Allocate *and* map the data your kernel touches - and remember that on a unified-memory APU an allocated-but-unmapped array can hide the mistake.
