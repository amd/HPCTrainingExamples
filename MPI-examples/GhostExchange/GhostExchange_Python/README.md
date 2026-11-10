# Ghost exchange example in Python

This is a 2D Jacobi stencil with a ghost (halo) cell exchange, written in Python.
A plain CPU/MPI reference lives in `cpu/orig.py`; the `gpu/` directory then
provides several GPU variants that compute the same stencil but differ in how
the halo exchange is done, each exercising a different layer of the open AMD
scaling stack:

- `gpu/cupy/` &mdash; CuPy GPU stencil with an MPI halo exchange (host-staged, plus a GPU-aware variant).
- `gpu/hip-python/` &mdash; the HIP runtime and HIPRTC kernels via hip-python, with an MPI halo exchange (host-staged and GPU-aware variants).
- `gpu/rccl/` &mdash; a direct GPU-to-GPU halo exchange with RCCL point-to-point (`cupy.cuda.nccl`).
- `gpu/jax/` &mdash; a sharded global array with the halo exchange expressed as `jax.lax.ppermute` (XLA lowers it to RCCL on ROCm).

Each variant has its own README describing what it does and how to run it; the
common environment setup below applies to all of them.

## Modules

Make sure to load the necessary modules:

```
module load rocm openmpi hdf5 netcdf-c mpi4py
```

Each GPU variant additionally loads one module for its framework (`cupy`,
`hip-python` or `jax`); see the variant README for which one.

Note that `hdf5` has to be built with parallel support based on the `openmpi` installation and `netcdf-c` has to be built using this parallel enabled `hdf5`. Furthermore, `mpi4py` also needs to be built using the same `openmpi`.

Scripts to build this software from source can be found in our [HPCTrainingDock](https://github.com/amd/HPCTrainingDock) repo.

## Virtual environment and netCDF4

Create and activate a virtual environment, with a couple of dependencies for the installation of the Python module of netCDF4:

```
python3 -m venv ghost_exchange_env
source ghost_exchange_env/bin/activate
pip3 install cython setuptools numpy cftime
```

Next, clone the `netcdf4-python` repo and install from source:

```
git clone https://github.com/Unidata/netcdf4-python.git
cd netcdf4-python
export HDF5_DIR=$HDF5_ROOT
export NETCDF4_DIR=$NETCDF_C_ROOT
python3 setup.py build
pip3 install .
```

## Running and the `-p` option

The CPU reference is run with:

```
mpirun -n 4 python3 cpu/orig.py -x 2 -y 2 -i 200 -j 200 -c -I 100 -p
```

Run each GPU variant from its own directory, using the command in that variant's README.

Note that unlike the C++ code, the `-p` option here is used to create the netCDF output file called `solution.nc`. Without the `-p` option no I/O is performed. For timing or scaling runs, omit `-p` so file I/O is not included in the reported `Total`.

## Checking that the variants agree

Every variant writes the same `solution.nc` when run with `-p`, so two runs (for example a GPU variant against the CPU reference) can be checked against each other. Difference the two output files into a `diff.nc` (for example with NCO's `ncdiff`) and run the `compute_max_abs.py` script one level up in the parent `GhostExchange/` directory, which reports the maximum absolute difference over the first and last time steps; a value at the level of floating-point roundoff means the solutions match:

```
ncdiff solution_cpu.nc solution_gpu.nc diff.nc
python3 ../compute_max_abs.py
```

The companion `print_solution.py` (same directory) renders a `solution.nc` to PNG images of the initial and final fields.

## Binding GPUs to MPI ranks

The `cupy`, `hip-python` and `rccl` variants launch one MPI rank per GPU but do not pin GPUs themselves. The `gpu/gpu_wrapper.sh` helper binds each rank to its node-local GPU by setting `HIP_VISIBLE_DEVICES` from `OMPI_COMM_WORLD_LOCAL_RANK` (OpenMPI) or `SLURM_LOCALID` (Slurm), so it works under both `mpirun` and `srun`; run it from a variant directory as `../gpu_wrapper.sh`. Under Slurm you can instead rely on the native `srun --gpu-bind=closest`. The `jax` variant manages the GPUs itself and does not use the wrapper (see its README).
