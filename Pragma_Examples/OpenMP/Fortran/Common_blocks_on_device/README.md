## Common Blocks on Device

This example demonstrates why the `always` modifier is required on `map` clauses
when using Fortran common blocks with OpenMP offloading on discrete GPUs.

### Background

When a common block is declared as target (`!$omp declare target(/thing/)`),
the runtime maintains a device copy of the data. However, when using
`target enter data map(to:...)` and `target exit data map(from:...)` without the
`always` modifier, the runtime may skip the actual data transfer for COMMON and SAVE variables.

The solution is straightforward: add the `always` modifier to force the data transfer:
```
!$omp target enter data map(always,to:array)
! ... device computation ...
!$omp target exit data map(always,from:array)
```
Note concerning USM vs. map clauses: the problem only shows when run with `HSA_XNACK=0´, if you compile with -fopenmp-force-usm and/or run with `HSA_XNACK=1´ the problem is not there. The solution is important for discrete GPU portability.

### Problem

The `problem` folder contains a version **without** the `always` modifier.
The array is initialized to `1.0` on the host, then a GPU kernel multiplies
each element by `2.0`, and the result is copied back. A subsequent CPU
`parallel do` doubles the values again.

Without `always`, the host never receives the updated device data, so the
first print shows `1.0` (stale host value) and the second print shows `2.0`
(the CPU doubled the stale `1.0`).



### Solution

The `solution/` folder adds the `always` modifier to both `map(to:...)` and
`map(from:...)`. This forces the runtime to transfer the data, producing the
correct results: `2.0` after the GPU kernel and `4.0` after the CPU kernel.

### Build

```bash
module load rocm-afar  # or module load rocm depending on the system. Tested with rocm 7.2 and Fortran therock drop 23.1.0
export FC=amdflang

cd problem && make && cd ..
cd solution && make && cd ..
```

### Run

```bash
export HSA_XNACK=0

echo "=== Problem (without always) ==="
./problem/common_blocks

echo "=== Solution (with always) ==="
./solution/common_blocks
```

Note: if you `export HSA_XNACK=1` instead you will see correct results with any version. This shows how USM makes your life easier!

### Example Results (gfx942, rocm-afar/22.3.0)

**Problem (without `always` modifier) -- WRONG results:**
```
 First element:  1.
 Last element:   1.
 First element:  2.
 Last element:   2.
```

The first print should show `2.0` (1.0 * 2.0 on the GPU) but shows `1.0`
because the data was never transferred back from the device.
The second print should show `4.0` (2.0 * 2.0 on the CPU) but shows `2.0`
because the CPU operated on the stale host value of `1.0`.

**Solution (with `always` modifier) -- CORRECT results:**
```
 First element:  2.
 Last element:   2.
 First element:  4.
 Last element:   4.
```
