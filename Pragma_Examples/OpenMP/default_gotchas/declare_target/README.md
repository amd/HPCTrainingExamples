# `declare target` + `map(to/from)` without `always`

A static-lifetime variable that appears in `declare target` (default `to`-style)
is permanently placed in the device data environment with an **infinite
reference count**. The OpenMP map rules only fire a transfer when the
reference count transitions to 1 (on enter) or to 0 (on exit), **or** when the
`always` map-type modifier is present. Neither transition can ever occur for a
declare-target global, so without `always` the runtime silently skips every
host-device transfer for that variable. The kernel runs against a stale
device buffer; the host never sees the result.

The trap is language-level: it applies equally to Fortran module variables,
Fortran SAVE locals, Fortran COMMON-block elements (see dedicated [example](https://github.com/amd/HPCTrainingExamples/tree/main/Pragma_Examples/OpenMP/Fortran/Common_blocks_on_device))
, and C/C++ file-scope or
static globals. The example uses a Fortran **module variable** (`Fortran/`)
and a C **file-scope global** (`C/`) — the simplest "normal" array form in
each language.

Three independent escapes are demonstrated, each in its own directory:

- `solution_force_usm/` — compile with `-fopenmp-force-usm`. The flag injects
  `requires unified_shared_memory` into every translation unit; the variable
  no longer has a separate device storage at all, only an 8-byte device-side
  reference pointer into host memory. Requires `HSA_XNACK=1` at runtime.
- `solution_map_always/` — add the `always` modifier on the `map` clauses.
  Refcount stays INF, but `always` overrides the refcount-based skip rule and
  forces the copies.
- `solution_declare_target_link/` — use the `link` clause on `declare target`.
  The variable is not pre-mapped, so the refcount starts at 0 and transitions
  0 to 1 on enter and 1 to 0 on exit; the OpenMP rules then fire the copies
  naturally, no `always` needed. Setting `HSA_XNACK=1` is optional: when it
  is on, the runtime detects the linked variable as eligible for auto
  zero-copy and skips the actual host-device data transfers entirely
  (correctness does not depend on it; see the trace comparison below).

## Layout

```
declare_target/
├── Fortran/   (module variable)
│   ├── problem/
│   ├── solution_force_usm/
│   ├── solution_map_always/
│   └── solution_declare_target_link/
└── C/         (file-scope global)
    ├── problem/
    ├── solution_force_usm/
    ├── solution_map_always/
    └── solution_declare_target_link/
```

Source files are `declare_target_module.f90` (Fortran) and
`declare_target_global.c` (C). Sources in `problem/` and `solution_force_usm/`
are identical; the only difference is the Makefile flag.

## Build and run

```bash
salloc -N 1 --gpus=1
module load rocm
module load amdclang

cd Fortran/<subdir>              # or  C/<subdir>
make
```

Correct output:

```
 First element:   2.             # array[0] = 2.000000
 First element:   4.             # array[0] = 4.000000
```

Wrong output (the bug):

```
 First element:   1.             # array[0] = 1.000000
 First element:   2.             # array[0] = 2.000000
```

The host kernel that runs after the GPU kernel doubles the host value once
more. If the GPU result reached the host, the first print shows `2` and the
second `4`. If it didn't, the host doubles the stale `1` to `2`, so the
prints show `1` and `2`.

## Verifying with `LIBOMPTARGET_INFO=-1`

Set `LIBOMPTARGET_INFO=-1` to make the OpenMP runtime print per-clause
diagnostics that show exactly which transfers happen and which mapping-table
entries exist. Address values below are abbreviated; rerun on your own
machine to obtain concrete addresses.

### `problem/` — the trap

```bash
cd Fortran/problem && make
LIBOMPTARGET_INFO=-1 ./declare_target_module 2>&1 | less
```

Output: `1.` then `2.` (wrong). Key lines:

```
omptarget device 0 info: Mapping exists with HstPtrBegin=0x..., TgtPtrBegin=0x..., Size=800, DynRefCount=INF (incremented), ... Name=_QMthing_modEarray_
omptarget device 0 info: Mapping exists with HstPtrBegin=0x..., TgtPtrBegin=0x..., Size=800, DynRefCount=INF (decremented), ... Name=_QMthing_modEarray_
```

What to look for:

- `DynRefCount=INF` is the trap.
- There is **no** `Copying data from host to device, ..., Name=_QMthing_modEarray_`
  line on enter and **no** `Copying data from device to host, ..., Name=_QMthing_modEarray_`
  line on exit. The 800-byte transfers that should happen are absent.
- The host pointer and the device pointer are different addresses.

### `solution_map_always/`

```bash
cd Fortran/solution_map_always && make
LIBOMPTARGET_INFO=-1 ./declare_target_module 2>&1 | less
```

Output: `2.` then `4.` (correct). Key lines:

```
omptarget device 0 info: Mapping exists with HstPtrBegin=0x..., TgtPtrBegin=0x..., Size=800, DynRefCount=INF (incremented), ... Name=_QMthing_modEarray_
omptarget device 0 info: Copying data from host to device, HstPtr=0x..., TgtPtr=0x..., Size=800, Name=_QMthing_modEarray_
...
omptarget device 0 info: Mapping exists with HstPtrBegin=0x..., TgtPtrBegin=0x..., Size=800, DynRefCount=INF (decremented), ... Name=_QMthing_modEarray_
omptarget device 0 info: Copying data from device to host, TgtPtr=0x..., HstPtr=0x..., Size=800, Name=_QMthing_modEarray_
```

What to look for:

- `DynRefCount` is still INF (the variable is still declare-target).
- But the `always` modifier produces the two `Copying data ... Size=800,
  Name=_QMthing_modEarray_` lines that were absent in `problem/`: one on
  enter (host → device), one on exit (device → host). That is the fix.

### `solution_declare_target_link/`

```bash
cd Fortran/solution_declare_target_link && make
LIBOMPTARGET_INFO=-1 ./declare_target_module 2>&1 | less
```

Output: `2.` then `4.` (correct). Key lines:

```
omptarget device 0 info: Creating new map entry with HstPtrBase=0x..., ..., Size=800, DynRefCount=1, ...
omptarget device 0 info: Copying data from host to device, HstPtr=0x..., TgtPtr=0x..., Size=800, ...
... kernel ...
omptarget device 0 info: Mapping exists with HstPtrBegin=0x..., ..., Size=800, DynRefCount=0 (decremented, delayed deletion), ...
omptarget device 0 info: Copying data from device to host, TgtPtr=0x..., HstPtr=0x..., Size=800, ...
omptarget device 0 info: Removing map entry with HstPtrBegin=0x..., ..., Size=800, ...
```

What to look for:

- `Creating new map entry ... DynRefCount=1` on enter (the variable was not
  pre-mapped, refcount transitions 0 → 1) and `DynRefCount=0 (decremented,
  delayed deletion)` followed by `Removing map entry` on exit (transitions
  1 → 0). The OpenMP rules fire the two `Copying data` transfers on those
  transitions; no `always` is needed.
- A separate 8-byte `_QMthing_modEarray_decl_tgt_ref_ptr` entry with
  `DynRefCount=INF` remains throughout — the device-side handle the compiler
  emits for a linked variable; it is populated when the data region is
  entered.

#### `HSA_XNACK=1` lets the runtime skip the 800-byte data motion

The link case is correct with `HSA_XNACK=0` or `HSA_XNACK=1`, but the
runtime behaviour differs. With XNACK on, the AMD plugin detects the
linked variable as eligible for *auto zero-copy* and avoids allocating a
device buffer or copying the 800 bytes in either direction. Run the same
binary back-to-back to see it:

```bash
cd Fortran/solution_declare_target_link && make
HSA_XNACK=0 LIBOMPTARGET_INFO=-1 ./declare_target_module 2>&1 | grep -E 'Copying data|map entry|data_alloc|data_submit|data_retrieve|data_delete|zero[-_]copy|Return HstPtrBegin'
HSA_XNACK=1 LIBOMPTARGET_INFO=-1 ./declare_target_module 2>&1 | grep -E 'Copying data|map entry|data_alloc|data_submit|data_retrieve|data_delete|zero[-_]copy|Return HstPtrBegin'
```

With `HSA_XNACK=0` (no zero-copy), the 800-byte traffic is real:

```
omptarget device 0 info: Creating new map entry with HstPtrBase=0x..., ..., Size=800, DynRefCount=1, ...
omptarget device 0 info: Copying data from host to device, HstPtr=0x..., TgtPtr=0x..., Size=800, Name=.../declare_target_module.f90
omptarget device 0 info: Copying data from host to device, HstPtr=0x..., TgtPtr=0x..., Size=8,   Name=_QMthing_modEarray_decl_tgt_ref_ptr
...
omptarget device 0 info: Copying data from device to host, TgtPtr=0x..., HstPtr=0x..., Size=800, Name=.../declare_target_module.f90
omptarget device 0 info: Removing map entry with HstPtrBegin=0x..., ..., Size=800, ...
Call                          data_alloc:      ...us 0x... (             0,            800, ..., 3)
Call                   data_submit_async:      ...us       0 (             0, 0x..., 0x...,            800, ...)
Call                 data_retrieve_async:      ...us       0 (             0, 0x..., 0x...,            800, ...)
Call                         data_delete:      ...us       0 (             0, 0x...,              3)
```

With `HSA_XNACK=1`, the same lines simply do not appear — there is no
800-byte allocation, no 800-byte H2D, no 800-byte D2H, no delete:

```
AMDGPU device 0 info: Application configured to run in zero-copy using auto zero-copy.
omptarget device 0 info: Return HstPtrBegin 0x... Size=800 for unified shared memory
omptarget device 0 info: Copying data from host to device, HstPtr=0x..., TgtPtr=0x..., Size=8, Name=_QMthing_modEarray_decl_tgt_ref_ptr
...
(no `Size=800` Copying-data lines, no `Creating new map entry ... Size=800`,
 no `Removing map entry ... Size=800`, no `data_alloc ... 800`,
 no `data_submit_async ... 800`, no `data_retrieve_async ... 800`,
 no `data_delete` for the 800-byte buffer)
```

The only persistent traffic is the 8-byte ref-ptr submit (pure runtime
bookkeeping; fixed cost regardless of array size). The `Return HstPtrBegin
... Size=800 for unified shared memory` line is how the runtime tells the
kernel "use the host address directly for this 800-byte region". The
program runs correctly either way; XNACK just makes the link case cheaper.

