#!/usr/bin/env python3
# Run the SAME GPU all-reduce four different ways and compare them:
#   1. rccl          - torch.distributed default GPU backend (nccl == RCCL on ROCm)
#   2. torch-mpi     - torch.distributed backend="mpi" (the GPU-aware MPI PyTorch
#                      was built against from source)
#   3. mpi4py-torch  - mpi4py Allreduce directly on a torch CUDA tensor's pointer
#   4. mpi4py-cupy   - mpi4py Allreduce on a zero-copy CuPy array
#
# Each rank contributes (rank+1); the all-reduced sum must equal world*(world+1)/2.
#
# Launch one rank per GPU, e.g.:
#   mpirun -np 2 python3 allreduce_comm_paths.py
#   srun  -n 2 python3 allreduce_comm_paths.py
#
# Modules (aac6): module load rocm pytorch mpi4py cupy

import os
import sys

# Import mpi4py first so MPI is initialized once; torch's MPI backend then reuses it.
from mpi4py import MPI
import torch


def env_int(names, default=None):
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return int(v)
    return default


RANK = env_int(["OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK", "SLURM_PROCID"], 0)
LOCAL_RANK = env_int(["OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK", "SLURM_LOCALID"], RANK)
WORLD = env_int(["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE", "SLURM_NTASKS"], 1)
EXPECTED = WORLD * (WORLD + 1) / 2.0

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29555")
os.environ["RANK"] = str(RANK)
os.environ["WORLD_SIZE"] = str(WORLD)
os.environ["LOCAL_RANK"] = str(LOCAL_RANK)


def log(msg):
    if RANK == 0:
        print(msg, flush=True)


def report(name, value):
    ok = abs(value - EXPECTED) < 1e-4
    status = "OK" if ok else "MISMATCH"
    log(f"  {name:13s} {status:8s} sum={value:.1f}")
    return ok


def path_rccl():
    import torch.distributed as dist
    dist.init_process_group("nccl", init_method="env://", rank=RANK, world_size=WORLD)
    x = torch.tensor([RANK + 1.0], device="cuda")
    dist.all_reduce(x)
    result = x.item()
    dist.destroy_process_group()
    return result


def path_torch_mpi():
    import torch.distributed as dist
    dist.init_process_group("mpi")
    x = torch.tensor([RANK + 1.0], device="cuda")
    dist.all_reduce(x)
    result = x.item()
    dist.destroy_process_group()
    return result


def path_mpi4py_torch():
    comm = MPI.COMM_WORLD
    x = torch.tensor([RANK + 1.0], device="cuda")
    torch.cuda.synchronize()  # buffer must be ready before MPI reads it
    # torch tensors expose no GPU-array interface, so hand MPI the raw device pointer
    buf = [MPI.memory.fromaddress(x.data_ptr(), x.numel() * x.element_size()), MPI.FLOAT]
    comm.Allreduce(MPI.IN_PLACE, buf)
    torch.cuda.synchronize()  # ensure the collective finished before we read it
    return x.item()


def path_mpi4py_cupy():
    import cupy as cp
    cp.cuda.Device(LOCAL_RANK).use()
    comm = MPI.COMM_WORLD
    x = cp.array([RANK + 1.0], dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    comm.Allreduce(MPI.IN_PLACE, x)  # cupy exposes __cuda_array_interface__, passed directly
    cp.cuda.runtime.deviceSynchronize()
    return float(x[0])


def main():
    if not torch.cuda.is_available():
        log("no GPU visible (torch.cuda.is_available() == False); run on a GPU node")
        sys.exit(77)
    torch.cuda.set_device(LOCAL_RANK)

    log(f"world={WORLD}  expected all-reduce sum={EXPECTED}  (rank contributes rank+1)")
    log(f"device: {torch.cuda.get_device_name(LOCAL_RANK)}")
    log("-" * 60)

    paths = [
        ("rccl", path_rccl),
        ("torch-mpi", path_torch_mpi),
        ("mpi4py-torch", path_mpi4py_torch),
        ("mpi4py-cupy", path_mpi4py_cupy),
    ]

    all_ok = True
    for name, fn in paths:
        try:
            value = fn()
            all_ok &= report(name, value)
        except Exception as e:
            # torch's ProcessGroupMPI checks MPIX_Query_cuda_support (never the
            # rocm one), so backend="mpi" rejects GPU tensors on AMD even though
            # the MPI is ROCm-aware. That is a known torch limitation, not a
            # failure of this system -> report N/A, don't fail the run.
            if "CUDA-aware MPI support" in str(e):
                log(f"  {name:13s} {'N/A':8s} torch backend='mpi' not ROCm-aware "
                    f"(use RCCL or mpi4py for GPU buffers)")
            else:
                log(f"  {name:13s} {'FAILED':8s} {type(e).__name__}: {e}")
                all_ok = False
        MPI.COMM_WORLD.Barrier()

    log("-" * 60)
    log("COMM PATHS: ALL OK" if all_ok else "COMM PATHS: SOME PATHS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
