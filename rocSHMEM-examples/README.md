# rocSHMEM One-sided and Collective

> [!NOTE]
> If you are on **AAC7**, replace the generic `module load openmpi rocm amdclang` steps with these modules to follow the rest of the exercise:
>
> ```
> module unload openmpi rocm
> module load rocm-therock/23.1.0 openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4-rocm-therock-23.1.0
> export CC=$(which amdclang); export CXX=$(which amdclang++); export FC=$(which amdflang)
> ```

## About these examples

[rocSHMEM](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocshmem) is AMD's GPU implementation of the OpenSHMEM PGAS model. Each PE (process) exposes part of its GPU memory as symmetric memory, and peers can directly access it from GPU kernels with one-sided operations.

This folder mirrors the structure of `../RCCL-examples`:


| Example          | MPI equivalent   | What it shows                                                                                          |
| ---------------- | ---------------- | ------------------------------------------------------------------------------------------------------ |
| `onesided.cpp`   |       | PE 0 pushes a GPU buffer into PE 1's symmetric buffer with `rocshmem_putmem` (one-sided communication) |
| `collective.cpp` | `collective.cpp` | PE 0 broadcasts a buffer to every PE with `rocshmem_ctx_int_broadcast_wg`                              |


## Build prerequisites

Use the system rocSHMEM module:

```
module load rocshmem/3.2.0-ro_ipc
```

This module sets `ROCSHMEM_PATH`, which the Makefile uses for includes and libraries.

## Build and run

Allocate at least 1 node with 2 GPUs (example):

```
salloc -p MI300A_SPX -N 1 --ntasks 2 --exclusive -t 1:00:00
```

Load modules and compile:

```
module load rocm openmpi rocshmem/3.2.0-ro_ipc
cd HPCTrainingExamples/rocSHMEM-examples
make
```

Run manually:

```
# Run with mpirun inside the interactive allocation
mpirun -np 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=2 ./onesided
mpirun -np 2 -mca pml ucx -mca osc ucx -x ROCSHMEM_MAX_NUM_CONTEXTS=2 ./collective
```

`ROCSHMEM_MAX_NUM_CONTEXTS=2` is required here because the collective example creates one extra work-group context in addition to the default context.

or simply:

```
make test
```

