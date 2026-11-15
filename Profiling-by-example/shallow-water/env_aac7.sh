# Site environment for the shallow-water novice profiling tutorial on AAC7.
# Copy once:
#   cp env_aac7.sh env.sh
#
# Login shells load rocm/7.0.3 through the Cray programming environment.
# That module conflicts with rocm-new, so unload it before loading
# rocm-new/10.1.0. Do not module purge: that drops PrgEnv-cray.
#
# SLURM_PARTITION is the MI300A SPX partition (four GPUs per node).

export SLURM_PARTITION=192C4G1H_MI300A_RHEL9_A1

source /etc/profile.d/modules.sh
module unload rocm
module load rocm-new/10.1.0
# rocprof-compute needs Python >= 3.10; the system python3 is 3.9.
# rocm-new/10.1.0 requires this cray-python version and swaps out any other.
module load cray-python/3.12.12

# rocprof-compute analyze (see setup_rocprof_compute_venv.sh for one-time pip install).
export ROCprof_COMPUTE_VENV="${HOME}/rocprof-compute-venv"
if [ -f "${ROCprof_COMPUTE_VENV}/bin/activate" ]; then
    source "${ROCprof_COMPUTE_VENV}/bin/activate"
fi
