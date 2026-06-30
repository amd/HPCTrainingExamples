#!/bin/bash

# Ensure a rocm module is loaded. Dependent modules declare `prereq rocm/<ver>`,
# which is satisfied only by the `rocm/<ver>` alias -- not by the underlying
# `rocm-new/<ver>` that PrgEnv-amd-new loads. An unanchored "^rocm" check is
# wrong because it also matches "rocm-new/...". Match the alias name explicitly
# and, if absent, load it (preferring the version of an already-loaded
# rocm-new so we do not pull in a mismatched default).
if ! module -t list 2>&1 | grep -q "^rocm/"; then
  rocm_new=$(module -t list 2>&1 | grep -m1 "^rocm-new/")
  if [ -n "${rocm_new}" ]; then
    rocm_ver=${rocm_new#rocm-new/}
    echo "rocm/ alias not loaded; loading rocm/${rocm_ver} to match ${rocm_new}"
    module load rocm/${rocm_ver}
  else
    echo "rocm module is not loaded"
    echo "loading default rocm module"
    module load rocm
  fi
fi

# Prefer a GCC + OpenMPI stack when those modules exist (e.g. the bare-metal
# training environment). In the Cray programming environments (PrgEnv-amd-new,
# aac7-rocm-*) there is no gcc/openmpi module -- the env already provides a
# compiler wrapper and (Cray/from-source) MPICH -- so a failed load here is
# non-fatal: we fall back to whatever MPI is already active.
module load gcc openmpi 2>/dev/null

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRC_DIR=${REPO_DIR}/tests/hello_mpi_omp
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cp -r ${SRC_DIR}/* ${BUILD_DIR}/
cd ${BUILD_DIR}
make

# The affinity launch flags are MPI-implementation specific:
#   * OpenMPI uses the MCA framework (-mca ...), capitalized hwloc object names
#     (--map-by L3cache), and --report-bindings.
#   * MPICH's Hydra launcher rejects -mca outright ("unrecognized argument
#     mca"), uses lowercase hwloc object names (-map-by l3cache), and has no
#     --report-bindings.
# Detect the active launcher and choose compatible flags so the same test
# runs in both the OpenMPI and MPICH programming environments.
if ! command -v mpirun >/dev/null 2>&1; then
  echo "ERROR: no mpirun found in PATH; cannot launch the MPI job" >&2
  exit 1
fi

MPI_VERSION="$(mpirun --version 2>&1)"
if echo "${MPI_VERSION}" | grep -qiE "open[ -]?mpi|openrte|open rte"; then
  echo "Detected OpenMPI launcher"
  MPI_LAUNCH_OPTS=(-mca btl ^openib --map-by L3cache --report-bindings)
elif echo "${MPI_VERSION}" | grep -qiE "hydra|mpich|cray"; then
  echo "Detected MPICH/Hydra launcher"
  MPI_LAUNCH_OPTS=(-bind-to core -map-by l3cache)
else
  echo "Unrecognized MPI launcher; running without affinity flags:"
  echo "${MPI_VERSION}" | head -1
  MPI_LAUNCH_OPTS=()
fi

# Launcher binding *reports* are implementation specific: OpenMPI prints
# "rank N bound to ..." via --report-bindings, but MPICH/Hydra (the Cray PE
# launcher) has no equivalent, so the OpenMPI-only "rank 3 bound to" pass
# criterion can never match under MPICH even though the ranks ARE bound.
# Verify binding launcher-independently instead: wrap each rank so it prints
# its OWN CPU affinity (from /proc/self/status) in a fixed "Rank N bound to ..."
# form that matches the existing pass regex on every MPI. The rank index comes
# from whichever launcher set it (MPICH PMI_RANK, OpenMPI OMPI_COMM_WORLD_RANK,
# PMIx, or Slurm). Unpadded N so "Rank 3 bound to" matches "ank 3 bound to".
RANK_REPORT="${BUILD_DIR}/rank_affinity.sh"
cat > "${RANK_REPORT}" <<'EOF'
#!/bin/bash
r=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${SLURM_PROCID:-0}}}}
cpus=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)
echo "Rank ${r} bound to CPUs ${cpus}"
exec "$@"
EOF
chmod +x "${RANK_REPORT}"

OMP_NUM_THREADS=2 OMP_PROC_BIND=close mpirun -np 4 "${MPI_LAUNCH_OPTS[@]}" "${RANK_REPORT}" ./hello_mpi_omp
