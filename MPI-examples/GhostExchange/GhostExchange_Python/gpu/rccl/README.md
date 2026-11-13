# Ghost exchange example in Python (RCCL halo exchange)

These versions keep the halo data on the GPU, like the GPU-aware MPI versions in
[`../mpi4py`](../mpi4py/README.md), but move it between GPUs with
[RCCL](https://github.com/ROCm/rccl) point-to-point calls (`ncclSend`/`ncclRecv`)
instead of MPI. The stencil code and the boundary conditions are the same as in
the mpi4py version of the same framework; only the ghost cell exchange differs.

| directory     | script               | RCCL binding used         |
|---------------|----------------------|---------------------------|
| `cupy/`       | `ghost_cupy_rccl.py` | CuPy's `cupy.cuda.nccl`   |
| `hip-python/` | `ghost_hip_rccl.py`  | hip-python's `hip.rccl`   |

The CuPy version is written in the NumPy array style, while the hip-python version
uses the kernels of the C++ HIP version; the mpi4py [README](../mpi4py/README.md)
describes that difference.

MPI is still used, but only to launch the ranks and to broadcast the RCCL unique
id that every rank needs to build the RCCL communicator, the same bootstrap idiom
used by `pt2pt.cpp` in
[`RCCL-examples`](https://github.com/amd/HPCTrainingExamples/tree/main/RCCL-examples).
MPI never touches the halo data.

For each direction (left/right, then top/bottom) the sends and receives are posted
between `ncclGroupStart` and `ncclGroupEnd`, so that RCCL runs them together and
neighbors that send to each other do not deadlock; the host then waits for the
stream before the received halos are used. Two differences from MPI show up in the
code:

- RCCL point-to-point calls have no message tags. The messages from one rank to a
  peer are matched to the peer's receives in the order they are posted, so both
  sides post them in the same order.
- RCCL has no `MPI_PROC_NULL`, so the calls for missing neighbors at the physical
  boundaries are skipped.

## Modules

See the top-level [README](../../README.md) for the common environment setup
(modules, virtual environment, netCDF4, GPU binding, the `-p` option and comparing
solutions). In addition to the common modules, load the module of the framework:

```
module load cupy        # for cupy/
module load hip-python  # for hip-python/
```

Both bindings call the RCCL library of the ROCm installation. `cupy` is installed
as a module with `extras/scripts/cupy_setup.sh` in
[HPCTrainingDock](https://github.com/amd/HPCTrainingDock) (on ROCm 7.x this builds
AMD's ROCm CuPy fork, whose `cupy.cuda.nccl` binding links against RCCL), and
`hip-python` with `extras/scripts/hip-python_setup.sh`.

## Running

Run from the directory of the framework, for example:

```
cd cupy
mpirun -n 4 ../../gpu_wrapper.sh python3 ghost_cupy_rccl.py -x 2 -y 2 -i 200 -j 200 -c -I 100 -p
```

```
cd hip-python
mpirun -n 4 ../../gpu_wrapper.sh python3 ghost_hip_rccl.py -x 2 -y 2 -i 200 -j 200 -c -I 100 -p
```

RCCL has to reach the GPUs of the other ranks on the node. If your Slurm setup
isolates each task to its own GPU, see the binding section of the top-level
README.
