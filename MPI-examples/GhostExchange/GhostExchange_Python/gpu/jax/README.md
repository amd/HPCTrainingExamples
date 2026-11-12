# Ghost exchange example in Python (JAX version)

This is the JAX counterpart of the CuPy / hip-python / RCCL versions. Instead of
a hand-written ghost cell exchange, the 2D stencil is expressed on a
single global array that is sharded over a device mesh (a 2D block / slab
decomposition), and the halo exchange is written explicitly with
[`jax.lax.ppermute`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html)
inside a [`jax.shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html):
each device shifts its edge rows and columns to its mesh neighbours. This is the
same single-program multiple-data (SPMD), multi-host programming model used by
`jax.distributed`; on ROCm, XLA lowers the `ppermute` to an RCCL
collective-permute.

This version exercises the **`jax.distributed` / XLA** path, which lowers the halo exchange to **RCCL** on ROCm.

See the top-level [README](../../README.md) for the common environment setup (virtual environment, netCDF4, the `-p` option and comparing solutions). In addition to `rocm`, load the jax module:

```
module load jax
```

`jax` is installed as a module with `extras/scripts/jax_setup.sh` in [HPCTrainingDock](https://github.com/amd/HPCTrainingDock), which builds JAX and its ROCm plugin from source for your ROCm installation (so `module load jax` provides a working GPU JAX). For JAX the `hdf5`, `netcdf-c` and `mpi4py` modules and the netCDF4 build are only needed for the optional `-p` output; the stencil itself only needs `rocm` and `jax`.

## Running

The device mesh is `-x` (devices along x) by `-y` (devices along y), and its size `-x * -y` must equal the total number of GPUs JAX sees, over all processes. Also make sure `-i` is divisible by `-x` and `-j` by `-y`, so the global grid shards evenly over the mesh.

JAX can drive the GPUs in two layouts, which compute exactly the same thing:

* **One process per node**: each process drives all the GPUs of its node. On a single node this needs no distributed setup at all:

  ```
  srun -N 1 -n 1 -c 128 python3 ghost_jax.py -x 4 -y 2 -i 8000 -j 8000 -I 500
  ```

  and across nodes (`-x * -y` is the total number of GPUs):

  ```
  srun -N 2 --ntasks-per-node=1 -c 128 python3 ghost_jax.py -x 4 -y 4 -i 8000 -j 8000 -I 500
  ```

  To use only some of the GPUs of a node, restrict the visible set, e.g. `HIP_VISIBLE_DEVICES=0,1,2,3 python3 ghost_jax.py -x 2 -y 2 ...`.

* **One process per GPU**, launched like the other variants:

  ```
  srun -N 2 -n 16 --ntasks-per-node=8 -c 16 --gpus-per-task=1 --gpu-bind=closest --gres-flags=allow-task-sharing \
       python3 ghost_jax.py -x 4 -y 4 -i 8000 -j 8000 -I 500
  ```

  Leaving all the GPUs of a node visible to every rank also works, and each rank then takes its own GPU by local rank, as in PyTorch's `LOCAL_RANK`: `srun -N 2 -n 16 --gpus 16 --gpus-per-node 8 --cpus-per-task=16 python3 ghost_jax.py ...`.

On run performed on Lux, both layouts gave the same solve time, but the start up of one process per GPU was much faster. For a 32768² grid and 10,000 iterations, the `Total` on 8 GPUs was 21 s with one process per GPU and 47 s with one process driving all 8 GPUs (the solve took 13 s in both).

When it is launched with several processes (Slurm, Open MPI, ...) the script calls `jax.distributed.initialize()` by itself. Under Slurm it takes the process count and id from `SLURM_NTASKS`/`SLURM_PROCID`, runs the coordinator on the first node of the `srun` step (set `JAX_COORDINATOR_ADDRESS=host:port` or `JAX_COORDINATOR_PORT` to override), and works out which GPUs each process drives: everything it can see if the launcher gave it its own GPU, otherwise its share of the GPUs of the node. It also clears the `http_proxy`-style environment variables first, because a proxy would otherwise intercept the coordinator's gRPC traffic and the start-up would time out.

### Give every process enough CPU cores

Slurm gives each task **one core by default**, shared by every thread the process starts, and XLA runs a host thread per GPU. With the default, a multi-GPU JAX process is slow without any error message (in a latency-bound test about 60 times slower per step on 8 GPUs and 120 times on 16 GPUs, and its start-up stalls), which looks exactly like "JAX does not scale". Always pass `-c` / `--cpus-per-task` (the examples above use a whole node for one process and 16 cores per GPU for one process per GPU). The same applies to the other variants, whose ranks also run helper threads.

### GPU isolation between processes

Some systems isolate each task to its own GPU when `--gpus-per-task` is used (for example OLCF's Lux). The processes then cannot reach each other's GPU memory and RCCL fails while initializing its communicators. Adding `--gres-flags=allow-task-sharing` to the `srun` line (not to `sbatch`/`salloc`) fixes this and keeps each rank on its NUMA-local GPU.

### The `-p` option with several processes

With `-p` the global array is gathered and written by process 0 as a serial netCDF file (the same `solution.nc` layout as the other variants: record 0 is the initial condition and record `maxIter` the final field). Two things to know:

* A netCDF4 built with parallel I/O imports `mpi4py`, whose default import calls `MPI_Init`; the script switches that off for its serial writer, so the other processes are not waited for.
* On Lux the gather works with up to 8 processes on one node, but with 16 processes over two nodes it fails during RCCL initialization (`ibv_reg_mr failed with error Bad address`). This is a known open issue. For larger runs compare the `Checksum:` line printed at the end instead: the sum and max of the field do not depend on the decomposition, so they must agree between runs of the same `-i`, `-j` and `-I`.

### Timing

`Total` includes a one-time start-up of a few seconds (JIT compilation and communication setup) that does not shrink with the number of GPUs, so for strong scaling of the computation itself look at `Solution Advancement`. The ghost exchange is fused into the stencil by XLA, so there is no separate halo time.

## Options

The `-h` (halo width, default 2) and `-c` (exchange the corner cells) options work as in the other versions: every step exchanges `nhalo` ghost layers in two phases (left/right, then top/bottom; with `-c` the top/bottom messages also carry the corners), and the domain boundary replicates the edge cells. As in the other versions the 5-point stencil only reads the innermost ghost layer, so neither option changes the result; they change how much data is exchanged.
