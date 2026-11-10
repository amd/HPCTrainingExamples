# Ghost exchange example in Python (JAX version)

This is the JAX counterpart of the CuPy / hip-python / RCCL versions. Instead of
a hand-written ghost cell exchange, the 2D Jacobi stencil is expressed on a
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

Unlike the other variants, JAX manages the GPUs itself and does not use `gpu_wrapper.sh`: on a single node one process uses all the visible GPUs, and for multi-node runs you launch one process per GPU with `srun --gpus-per-task=1` (the code calls `jax.distributed.initialize()` automatically when `SLURM_NTASKS > 1`).

The device mesh is `-x` (devices along x) by `-y` (devices along y), and its size `-x * -y` must equal the number of GPUs JAX sees. On a single node, either pick `-x`/`-y` to match the GPU count (for example `-x 4 -y 2` on an 8-GPU node), or restrict the visible set with `HIP_VISIBLE_DEVICES`:

```
# use 4 of the GPUs on one node
HIP_VISIBLE_DEVICES=0,1,2,3 python3 ghost_jax.py -x 2 -y 2 -i 200 -j 200 -I 100 -p
```

Also make sure `-i` is divisible by `-x` and `-j` by `-y`, so the global grid shards evenly over the mesh.

For multi-node runs, launch one process per GPU with `srun`:

```
srun -N 2 --ntasks-per-node=8 --gpus-per-task=1 python3 ghost_jax.py -x 4 -y 4 -i 8000 -j 8000 -I 500 -p
```

The `-c` (corners) and `-h` (halo width) options are accepted for command-line parity with the other versions but have no effect here: the 5-point stencil only needs a width-one, edge-neighbour halo.
