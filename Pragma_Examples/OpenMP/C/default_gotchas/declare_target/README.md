# `declare target` + `map(to/from)` without `always`

A static-lifetime variable that appears in `declare target` with the default `to` behavior is permanently placed in the device data environment with an infinite reference count. The OpenMP map rules only perform a transfer when the reference count transitions to 1 on enter or to 0 on exit, or when the `always` map-type modifier is present. For a declare-target global, those transitions never occur, so `map(to:...)` and `map(from:...)` can silently skip the copies and leave the host and device values out of sync.

Three independent fixes are demonstrated:

- `solution_force_usm/` compiles with `-fopenmp-force-usm`, which injects `requires unified_shared_memory`. The variable no longer has separate device storage. This requires `HSA_XNACK=1` at runtime.
- `solution_map_always/` adds the `always` modifier to the `map` clauses, forcing transfers even though the reference count remains infinite.
- `solution_declare_target_link/` uses the `link` clause so the variable is not pre-mapped. The reference count starts at 0, so normal map-enter and map-exit transitions trigger the copies.

This C example uses a C file-scope global array.
The same trap is also demonstrated for [Fortran](../../../Fortran/default_gotchas/declare_target/).

## Layout

```
declare_target/
├── problem/
├── solution_force_usm/
├── solution_map_always/
└── solution_declare_target_link/
```

Source files in `problem/` and `solution_force_usm/` are intentionally identical; the force-USM fix lives in the Makefile.

## Build and run

Run on a compute node, then load ROCm and build one subdirectory at a time:

```bash
salloc -N 1 --gpus=1 -p PPAC_MI300A_CPX
module load rocm
export CC=amdclang
export C_COMPILER=amdclang

cd problem              # or one of the solution_* directories
make ROCM_GPU=gfx942
./declare_target_global
```

Correct output from a solution:

```text
array[0] = 2.000000
array[0] = 4.000000
```

Wrong output from `problem/`:

```text
array[0] = 1.000000
array[0] = 2.000000
```

The host kernel that runs after the GPU kernel doubles the host value once more. If the GPU result reached the host, the first print shows 2 and the second 4. If it did not, the host doubles the stale value 1 to 2.

## Verifying with `LIBOMPTARGET_INFO=-1`

Set `LIBOMPTARGET_INFO=-1` to make the OpenMP runtime print per-clause diagnostics. In `problem/`, look for a mapping entry with `DynRefCount=INF` and no matching `Copying data from host to device` or `Copying data from device to host` lines for the array. In `solution_map_always/`, those copy lines appear even though the reference count remains infinite. In `solution_declare_target_link/`, the runtime creates a normal mapping with a finite reference count and removes it after the exit data region.
