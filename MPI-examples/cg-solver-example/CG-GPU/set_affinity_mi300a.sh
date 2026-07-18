#!/bin/bash
# set_affinity_mi300a.sh — Per-rank GPU and CPU affinity wrapper for MPI jobs
#                          on AMD Instinct MI300A nodes with 8 GPUs.
#
# Usage (as mpirun per-rank launcher):
#   mpirun -n 8 --bind-to none bash set_affinity_mi300a.sh <program> [args...]
#
# Each MPI rank is assigned exactly one GPU via ROCR_VISIBLE_DEVICES and is
# pinned to the CPU cores that share the same memory domain as that GPU.
# Limiting each rank to a single visible GPU also simplifies GPU assignment
# inside the application: hipGetDeviceCount() returns 1, so hipSetDevice(0)
# always selects the physically correct device regardless of rank count.
#
# Topology assumed (MI300A, 8 GPUs, 96 CPU cores):
#
#   96 CPU cores  — 8 groups × 12 cores  (one group per GPU die / GCD)
#   8 GPU devices — one GCD per group
#
#   GPU 0  →  cores   0-11
#   GPU 1  →  cores  12-23
#   GPU 2  →  cores  24-35
#   GPU 3  →  cores  36-47
#   GPU 4  →  cores  48-59
#   GPU 5  →  cores  60-71
#   GPU 6  →  cores  72-83
#   GPU 7  →  cores  84-95
#
# Override topology defaults with environment variables if your node differs:
#   NUM_CPUS   (default 96)
#   NUM_GPUS   (default 8)

export local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-0}

NUM_CPUS=${NUM_CPUS:-96}
NUM_GPUS=${NUM_GPUS:-8}
CORES_PER_GPU=$(( NUM_CPUS / NUM_GPUS ))   # 12 on a 96-core / 8-GPU node

my_gpu=${local_rank}
cpu_start=$(( local_rank * CORES_PER_GPU ))
cpu_end=$(( cpu_start + CORES_PER_GPU - 1 ))

# Give this rank exclusive visibility into its one GPU.
# The application's hipGetDeviceCount() will return 1 and hipSetDevice(0)
# will select the correct physical device.
export ROCR_VISIBLE_DEVICES=${my_gpu}

# Pin to the target CPU core range if it lies within the SLURM cpuset.
# On shared (non-exclusive) allocations SLURM may restrict the cpuset;
# taskset fails with EINVAL if any requested core is absent.  Falling back
# without CPU pinning is safe: ROCR_VISIBLE_DEVICES already routes each rank
# to the correct GPU, and CPU placement does not affect GPU-to-GPU bandwidth.
if taskset -c ${cpu_start}-${cpu_end} true 2>/dev/null; then
    exec taskset -c ${cpu_start}-${cpu_end} "$@"
else
    exec "$@"
fi
