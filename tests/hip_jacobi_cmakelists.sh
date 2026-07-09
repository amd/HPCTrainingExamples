#!/bin/bash

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
   export HIP_PLATFORM=amd
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/jacobi

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   module load libfabric
   # The Jacobi HIP target is compiled AND linked by ROCm's clang++ (HIP
   # language), NOT by the Cray CC wrapper. find_package(MPI) detects CXX=CC as
   # an MPI-capable compiler and returns EMPTY MPI_LIBRARIES (the wrapper only
   # injects cray-mpich when IT performs the link), so the clang++ link fails
   # with "undefined symbol: MPI_Irecv/MPI_Init/...". Load the from-source
   # mpich-wrappers MPI -- its mpicc/mpicxx expose real -I/-L/-lmpi via -show --
   # and point find_package(MPI) at them (below) so those flags land on the
   # clang++ link line. Loading the already-active version is idempotent.
   module load mpich-wrappers 2>/dev/null || true
else
   module load openmpi
fi

# Point cmake's find_package(MPI) at the resolved MPI wrappers so the legacy
# MPI_LIBRARIES / MPI_INCLUDE_PATH this CMakeLists uses are populated with
# explicit flags (mpicxx -show), which the HIP clang++ link then consumes.
# Without this, on Cray CXX=CC yields empty MPI_LIBRARIES (see above).
CMAKE_MPI_ARGS=()
if command -v mpicc  >/dev/null 2>&1; then CMAKE_MPI_ARGS+=("-DMPI_C_COMPILER=$(command -v mpicc)"); fi
if command -v mpicxx >/dev/null 2>&1; then CMAKE_MPI_ARGS+=("-DMPI_CXX_COMPILER=$(command -v mpicxx)"); fi

SRC_DIR=$(pwd)
# mktemp -d already creates the directory, so do NOT mkdir it again (that
# fails with "File exists", and the && short-circuit then skips the cd,
# leaving us in the source tree). Build out-of-tree and point cmake at the
# absolute source dir -- a relative ".." would resolve under /tmp, not here.
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cd "${BUILD_DIR}"
cmake "${CMAKE_MPI_ARGS[@]}" "${SRC_DIR}"
make

#salloc -p LocalQ --gpus=2 -n 2 -t 00:10:00
# Launch with the launcher that MATCHES the MPI the binary was linked against.
# mpich-wrappers and OpenMPI both ship their own mpirun/mpiexec next to mpicc;
# launching that MPI with srun gives each rank a singleton MPI_COMM_WORLD
# because srun's Cray PMI does not wire it up. Only a bare Cray MPICH (mpicc
# with no co-located launcher) falls back to srun.
MPI_BINDIR=$(dirname "$(command -v mpicc 2>/dev/null)" 2>/dev/null)
if [ -n "${MPI_BINDIR}" ] && [ -x "${MPI_BINDIR}/mpirun" ]; then
   LAUNCHER="${MPI_BINDIR}/mpirun"
elif [ -n "${MPI_BINDIR}" ] && [ -x "${MPI_BINDIR}/mpiexec" ]; then
   LAUNCHER="${MPI_BINDIR}/mpiexec"
else
   LAUNCHER="srun"
fi
"${LAUNCHER}" -n 2 ./Jacobi_hip -g 2
