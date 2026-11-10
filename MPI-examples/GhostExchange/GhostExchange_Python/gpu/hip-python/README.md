# Ghost exchange example in Python (hip-python version)

These are the hip-python versions of the ghost exchange example: the GPU work is
done with [`hip-python`](https://rocm.docs.amd.com/projects/hip-python/en/latest/)
(AMD's HIP bindings) instead of CuPy &mdash; device memory is managed with the HIP
runtime and the stencil and halo pack/unpack are HIP kernels compiled at runtime
with HIPRTC. As with the CuPy versions, the two variants differ only in how the
ghost cell exchange is done:

- `ghost_hip.py` stages the halo buffers through host memory before the MPI exchange, so it runs with any MPI.
- `ghost_hip_gpu_aware.py` keeps the packed halo buffers on the device and hands them straight to MPI, and therefore requires a GPU-aware MPI.

The GPU-aware exchange works exactly as it would in C/C++, where you pass a
`hipMalloc`-ed device pointer straight to `MPI_Isend`/`MPI_Irecv` and a GPU-aware
MPI detects that the buffer lives on the device and transfers it from there. The
only extra step is on the Python side: `mpi4py` does not take a raw
`(pointer, count, datatype)` triple the way the C API does &mdash; it takes a
single Python object and reads those three facts off it. For device memory it
reads them from the object's `__cuda_array_interface__`. CuPy arrays expose that
interface automatically (which is why the CuPy GPU-aware variant needs no extra
code), but a bare `hip-python` allocation does not, so `ghost_hip_gpu_aware.py`
wraps the device pointer in a tiny `__cuda_array_interface__` view. That view only
advertises the pointer, shape and dtype to `mpi4py`; the GPU-aware transfer itself
is still done by MPI, exactly as in the C/C++ case.

Both versions exercise the **HIP** runtime (device memory and HIPRTC kernels) together with the **mpi4py / MPI** transport; `ghost_hip_gpu_aware.py` additionally requires a **GPU-aware MPI**.

See the top-level [README](../../README.md) for the common environment setup (modules, virtual environment, netCDF4, GPU binding, the `-p` option and comparing solutions). In addition to the common modules, load the hip-python module:

```
module load hip-python
```

`hip-python` is installed as a module with `extras/scripts/hip-python_setup.sh` in [HPCTrainingDock](https://github.com/amd/HPCTrainingDock) (so `module load hip-python` pulls in the HIP bindings matching your ROCm installation).

Run from this directory with:

```
mpirun -n 4 ../gpu_wrapper.sh python3 ghost_hip.py -x 2 -y 2 -i 200 -j 200 -c -I 100 -p
```

Swap in `ghost_hip_gpu_aware.py` for the GPU-aware exchange (it needs a GPU-aware MPI).
