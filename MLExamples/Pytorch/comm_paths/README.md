# Communication paths: one all-reduce, four ways

`allreduce_comm_paths.py` runs the **same GPU all-reduce** over four different
communication paths and checks they all agree, so you can see which paths a
given PyTorch install actually supports:

| path | what it uses |
|------|--------------|
| `rccl` | `torch.distributed` default GPU backend (`nccl` == RCCL on ROCm) |
| `torch-mpi` | `torch.distributed` `backend="mpi"` |
| `mpi4py-torch` | `mpi4py` `Allreduce` on a torch CUDA tensor's device pointer |
| `mpi4py-cupy` | `mpi4py` `Allreduce` on a zero-copy CuPy array |

Each rank contributes `rank+1`; the reduced sum must equal `world*(world+1)/2`.

## Run

```bash
module load rocm pytorch mpi4py cupy   # or activate an env providing torch, mpi4py, cupy
./run_comm_paths.sh            # 2 ranks (one per GPU)
NRANKS=8 ./run_comm_paths.sh   # 8 ranks
```

## Notes

- `rccl` is the default and fastest GPU path; it is what `torch.distributed`
  uses for GPU tensors when you don't pass a backend.
- `torch-mpi` requires a PyTorch built with the MPI backend (a source build;
  plain wheels have no MPI backend). On ROCm it typically reports **N/A** even so:
  PyTorch's `ProcessGroupMPI` only enables GPU tensors when
  `MPIX_Query_cuda_support()` is true and never checks the ROCm equivalent, so it
  refuses a GPU tensor on AMD. This is a PyTorch-side limitation, **not** a sign
  the MPI lacks GPU support — the `mpi4py` paths below push GPU buffers through
  the very same MPI successfully.
- The `mpi4py` paths move GPU buffers directly when `mpi4py` is built against a
  GPU-aware MPI: `mpi4py-cupy` is zero-copy via CuPy's `__cuda_array_interface__`,
  while `mpi4py-torch` passes the tensor's `data_ptr()` explicitly (torch tensors
  don't expose that interface themselves).
- A path that isn't available on your install is reported (`N/A`) and does not
  fail the run; a genuine error or a wrong result does.
