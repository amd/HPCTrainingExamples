# Ghost exchange example in Python (CuPy version)

These are the CuPy GPU versions of the ghost exchange example. The 2D stencil code runs on the GPU with CuPy; the two variants differ only in how the ghost cell exchange is done:

- `no_gpu_aware_mpi.py` stages the halo buffers through host memory before the MPI exchange, so it runs with any MPI.
- `gpu_aware_mpi.py` passes GPU buffers directly to MPI and therefore requires a GPU-aware MPI.

Both versions exercise the **mpi4py / MPI** transport for the halo exchange; `gpu_aware_mpi.py` additionally requires a **GPU-aware MPI**. For the non GPU-aware version, the timing report includes an extra `Host<->Device Transfer` line that accounts for the halo staging copies (the cost that `gpu_aware_mpi.py` avoids).

See the top-level [README](../../README.md) for the common environment setup (modules, virtual environment, netCDF4, GPU binding, the `-p` option and comparing solutions). In addition to the common modules, load the CuPy module:

```
module load cupy
```

`cupy` is installed as a module with `extras/scripts/cupy_setup.sh` in [HPCTrainingDock](https://github.com/amd/HPCTrainingDock).

Run from this directory with:

```
mpirun -n 4 ../gpu_wrapper.sh python3 ghost_cupy.py -x 2 -y 2 -i 200 -j 200 -c -I 100 -p
```
