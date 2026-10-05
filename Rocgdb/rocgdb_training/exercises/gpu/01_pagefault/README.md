- [GPU Exercise 1 (page fault): find the bad device pointer](#org12b71a5)
  - [Build](#org74adb78)
  - [Reproduce and locate the fault](#orgc03a359)
  - [Diagnose](#orgf815d8a)
  - [Fix and confirm](#org68ebab4)
  - [Keep the evidence (optional)](#org522c766)


<a id="org12b71a5"></a>

# GPU Exercise 1 (page fault): find the bad device pointer

`saxmy.cpp` launches a HIP kernel that computes `y[i] = a*x[i]*y[i]` on the device. It crashes. Our task: use `rocgdb` to find out *where* and *why*.


<a id="org74adb78"></a>

## Build

```sh
amdclang++ -x hip -ggdb -O0 --offload-arch=gfx942 -o saxmy saxmy.cpp
```

-   Use the arch of your GPU (`gfx942` = MI300A, `gfx950` = MI355X, ...).
-   `-ggdb -O0` keeps full debug info; `-x hip` selects the HIP language.


<a id="orgc03a359"></a>

## Reproduce and locate the fault

```sh
rocgdb ./saxmy
```

```
(gdb) set amdgpu precise-memory on   # fault at the exact instruction
(gdb) run                            # a GPU wave takes a SIGSEGV
(gdb) info threads                   # host threads AND the 4 GPU waves
```

-   The stop is in a *GPU wave* (rocgdb calls a wave a "thread"), at the `y[i] = a*x[i]*y[i]` line. Look at the frame's arguments.


<a id="orgf815d8a"></a>

## Diagnose

-   What are `x` and `y` in the faulting frame? (`print x`, `print y`, or read the frame line.) A null / unmapped pointer is the tell.
-   Cross to the host to see how it got here:

```
(gdb) thread 1
(gdb) where            # the CPU is parked in hipDeviceSynchronize, waiting
```

-   Now look at `main`: the device pointers `d_x`/`d_y` are *never allocated* (no `hipMalloc`), so the kernel dereferences garbage/null memory.


<a id="org68ebab4"></a>

## Fix and confirm

-   Add the missing allocation (and some initialisation). Compare with the reference `saxmy_fixed.cpp`:

```sh
diff saxmy.cpp saxmy_fixed.cpp
amdclang++ -x hip -ggdb -O0 --offload-arch=gfx942 -o saxmy_fixed saxmy_fixed.cpp
./saxmy_fixed       # runs clean, no fault
```


<a id="org522c766"></a>

## Keep the evidence (optional)

-   At the fault, `generate-core-file` (`gcore`) writes a core you can reload later with `rocgdb ./saxmy <core>` on any machine: no GPU needed for the post-mortem.
