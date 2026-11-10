# Ghost exchange example in Python (RCCL version)

This is the RCCL counterpart of the CuPy `gpu/cupy/gpu_aware_mpi.py` version:
both keep the packed halo buffers on the device and never stage them through host
memory. The difference is the transport &mdash; where the CuPy GPU-aware version
hands the device buffers to a GPU-aware MPI, this one moves the halos directly
between GPUs with [RCCL](https://github.com/ROCm/rccl) (`ncclSend`/`ncclRecv`).
The 2D Jacobi stencil and the boundary-condition update are identical; only the
ghost cell exchange differs. RCCL is driven from Python through CuPy's thin RCCL
binding, `cupy.cuda.nccl`. MPI is still used, but only to launch the ranks and to
broadcast the RCCL unique id needed to build the RCCL communicator &mdash; the
same bootstrap idiom used by the
[`RCCL-examples`](https://github.com/amd/HPCTrainingExamples/tree/main/RCCL-examples)
`pt2pt.cpp` &mdash; it never touches the GPU payload.

In terms of the open AMD scaling stack, this version exercises the **RCCL** point-to-point path directly (`ncclSend`/`ncclRecv`), with MPI used only for bootstrap.

See the top-level [README](../../README.md) for the common environment setup (modules, virtual environment, netCDF4, GPU binding, the `-p` option and comparing solutions). In addition to the common modules, load the CuPy module (its `cupy.cuda.nccl` binding links against RCCL):

```
module load cupy
```

`cupy` is installed as a module with `extras/scripts/cupy_setup.sh` in [HPCTrainingDock](https://github.com/amd/HPCTrainingDock) (on ROCm 7.x this builds AMD's ROCm CuPy fork, whose `cupy.cuda.nccl` binding links against RCCL).

Run from this directory with:

```
mpirun -n 4 ../gpu_wrapper.sh python3 ghost_rccl.py -x 2 -y 2 -i 200 -j 200 -c -I 100 -p
```
