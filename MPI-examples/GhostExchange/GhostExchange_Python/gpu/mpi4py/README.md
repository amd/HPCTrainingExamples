# Ghost exchange example in Python (MPI halo exchange with mpi4py)

These versions run the 2D stencil code on the GPU and exchange the ghost (halo)
cells with MPI through `mpi4py`. The GPU work is done with one of two frameworks,
CuPy or hip-python, and each comes in the same two variants of the exchange:

| directory     | host-staged MPI | GPU-aware MPI             |
|---------------|-----------------|---------------------------|
| `cupy/`       | `ghost_cupy.py` | `ghost_cupy_gpu_aware.py` |
| `hip-python/` | `ghost_hip.py`  | `ghost_hip_gpu_aware.py`  |

- The host-staged variants copy the halo data to host memory before the MPI
  exchange and back to the GPU afterwards, so they run with any MPI. `ghost_cupy.py`
  reports the time of those copies in an extra `Host<->Device Transfer` line.
- The GPU-aware variants hand device buffers straight to MPI and therefore need a
  GPU-aware MPI (for example OpenMPI built with UCX and ROCm support).

## CuPy and hip-python

The two frameworks start from different places:

- The CuPy versions are the NumPy CPU reference (`cpu/orig.py`) moved to the GPU.
  The arrays are CuPy arrays and the code keeps the NumPy array style: slice
  expressions for the stencil, the boundary conditions and the halo buffers. No GPU
  kernel is written by hand; CuPy launches the kernels for each array expression.
- The hip-python versions follow the C++ HIP version
  (`GhostExchange_ArrayAssign_HIP`) instead. [hip-python](https://rocm.docs.amd.com/projects/hip-python/en/latest/)
  provides AMD's Python bindings for HIP: device memory is managed with the HIP
  runtime and the kernels are compiled at run time with HIPRTC. The kernels have
  the same names and arguments as in the C++ code (`init_core`, `init_core2`,
  `blur`, `enforce_bcs_left`/`rght`/`bot`/`top`, `ghost_left_send`/`ghost_rght_send`,
  `ghost_rght_recv`/`ghost_left_recv`) and are launched with the same grid and
  block sizes. The only extra arguments are the halo depth `nhalo` and the row
  length `ld`: the C++ code indexes a `double**` with offset rows, while here the
  array is one flat device buffer. As in the C++ code, the left/right halo columns
  are packed into contiguous buffers by the `ghost_*_send` kernels and unpacked by
  the `ghost_*_recv` kernels, while the top/bottom halo rows, which are already
  contiguous, are exchanged directly from the array.

Expect the hip-python stencil to be faster: `blur` reads and writes each cell once
per iteration, while the CuPy slice expression runs several kernels, each making a
full pass through memory.

## GPU-aware MPI from Python

The GPU-aware exchange works as it would in C/C++, where you pass a device pointer
to `MPI_Isend`/`MPI_Irecv` and a GPU-aware MPI detects that the buffer lives on the
device and transfers it from there. The only extra step is on the Python side:
`mpi4py` does not take a raw `(pointer, count, datatype)` triple the way the C API
does; it takes a single Python object and reads those three facts from it. For
device memory it reads them from the object's `__cuda_array_interface__`. CuPy
arrays expose that interface, so the CuPy GPU-aware variant passes its arrays as
they are. A `hip-python` allocation does not, so `ghost_hip_gpu_aware.py` wraps
each device pointer in a small `__cuda_array_interface__` view (`_DeviceCAI`). The
view only advertises the pointer, length and data type to `mpi4py`; the transfer
itself is done by MPI, exactly as in C/C++.

GPU kernels run asynchronously with respect to the host, and MPI does not know
about them. Both GPU-aware variants therefore synchronize the GPU before handing
buffers to MPI, so that MPI neither reads data that a kernel has not finished
writing nor receives into memory that a pending kernel will still write.

## Modules

See the top-level [README](../../README.md) for the common environment setup
(modules, virtual environment, netCDF4, GPU binding, the `-p` option and comparing
solutions). In addition to the common modules, load the module of the framework:

```
module load cupy        # for cupy/
module load hip-python  # for hip-python/
```

Both are installed as modules by scripts in
[HPCTrainingDock](https://github.com/amd/HPCTrainingDock):
`extras/scripts/cupy_setup.sh` and `extras/scripts/hip-python_setup.sh` (the latter
installs the HIP bindings that match your ROCm installation).

## Running

Run from the directory of the framework, for example:

```
cd cupy
mpirun -n 4 ../../gpu_wrapper.sh python3 ghost_cupy.py -x 2 -y 2 -i 200 -j 200 -c -I 100 -p
```

```
cd hip-python
mpirun -n 4 ../../gpu_wrapper.sh python3 ghost_hip.py -x 2 -y 2 -i 200 -j 200 -c -I 100 -p
```

Swap in `ghost_cupy_gpu_aware.py` or `ghost_hip_gpu_aware.py` for the GPU-aware
exchange.
