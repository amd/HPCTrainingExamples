# Site environment for the shallow-water profiling tutorials on AAC6.
# Copy once, then edit for your account:
#   cp env_aac6.sh env.sh
#
# Set SLURM_PARTITION to the partition your account uses on AAC6.
# For the advanced track, pick the GPU_BIND / MPI_BIND pair that matches
# your node layout (see AAC6.md).

export SLURM_PARTITION=

source /etc/profile.d/lmod.sh
source /shared/apps/ubuntu/lmod/overridetcl2lmod.sh

module use /nfsapps/ubuntu-24.04-nightlies/modules/base
module load rocm/10.1.0a20260818
module load openmpi

# Two NUMA domains (typical SPX layout):
export GPU_BIND="../gpu_bind.sh"
export MPI_BIND="--map-by ppr:1:numa --bind-to numa"
# Single NUMA domain (typical CPX layout):
# export GPU_BIND="../gpu_bind_cpx.sh"
# export MPI_BIND="--map-by slot"

# NIC profiling (advanced stages 5–6). Set after: rocprof-sys-avail -H -r net
export ROCPROFSYS_NETWORK_INTERFACE=
if [ -n "${ROCPROFSYS_NETWORK_INTERFACE}" ]; then
    _nic=${ROCPROFSYS_NETWORK_INTERFACE}
    export ROCPROFSYS_PAPI_EVENTS="net:::${_nic}:rx:byte net:::${_nic}:rx:packet net:::${_nic}:tx:byte net:::${_nic}:tx:packet"
    export ROCPROFSYS_TIMEMORY_COMPONENTS="wall_clock network_stats"
    export ROCPROFSYS_USE_SAMPLING=true
    export ROCPROFSYS_SAMPLING_FREQ=100
fi

export ROOFLINE_EXTRACTOR=/nfsapps/ubuntu-24.04/opt/rooflineExtractor

# Novice profile.sh roofline backend: extractor (default) or rocprof-compute
export ROOFLINE_TOOL=extractor
# export ROOFLINE_TOOL=rocprof-compute

export ROOFLINE_VENV="${HOME}/roofline-venv"
if [ -f "${ROOFLINE_VENV}/bin/activate" ]; then
    source "${ROOFLINE_VENV}/bin/activate"
fi

# rocprof-compute analyze (see setup_rocprof_compute_venv.sh for one-time pip install).
export ROCprof_COMPUTE_VENV="${HOME}/rocprof-compute-venv"
if [ -f "${ROCprof_COMPUTE_VENV}/bin/activate" ]; then
    source "${ROCprof_COMPUTE_VENV}/bin/activate"
fi
