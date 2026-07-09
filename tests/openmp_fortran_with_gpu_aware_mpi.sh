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
   export MPICH_GPU_SUPPORT_ENABLED=1
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdclang
   module load openmpi
fi

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/GPU_Aware_MPI

export AMDGPU_GFXMODEL=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *\(gfx[0-9,a-f]*\) *$/\1/'`

SRC_DIR=$(pwd)
echo "Source directory is ${SRC_DIR}"
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT
cp ${SRC_DIR}/* ${BUILD_DIR}
cd ${BUILD_DIR}

if [ -n "${CRAY_MPICH_VERSION:-}" ]; then
   MPI_PATH=$MPICH_WRAPPERS_DIR
fi
echo "MPI_PATH $MPI_PATH"

module list

# Collapse repeated slashes to make path comparisons robust against
# modulefiles whose MPI_PATH contains an embedded "//" (canonical
# `which mpifort` output uses single slashes; literal $INSTALL_PATH
# from the modulefile may not).
norm_path() { echo "$1" | tr -s '/'; }

echo ""
echo "=== Step 1: Verify mpifort points to our build ==="
MPIFORT_PATH=$(which mpifort)
echo "mpifort found at: $MPIFORT_PATH"
EXPECTED_MPIFORT=$(norm_path "$MPI_PATH/bin/mpifort")
if [[ "$(norm_path "$MPIFORT_PATH")" != "$EXPECTED_MPIFORT" ]]; then
    echo "FAIL: mpifort is not from our install ($EXPECTED_MPIFORT)"
    exit 1
fi
echo "PASS: mpifort is from our install"

echo ""
echo "=== Step 2: Verify wrapper calls amdflang with our lib/include ==="
SHOW_OUTPUT=$(mpifort -show)
echo "$SHOW_OUTPUT"
NORM_SHOW=$(norm_path "$SHOW_OUTPUT")

if ! echo "$NORM_SHOW" | grep -q "amdflang"; then
    echo "FAIL: mpifort does not wrap amdflang"
    exit 1
fi
echo "PASS: mpifort wraps amdflang"

if ! echo "$NORM_SHOW" | grep -qF "$(norm_path "$MPI_PATH/lib")"; then
    echo "FAIL: mpifort does not link against $(norm_path "$MPI_PATH/lib")"
    exit 1
fi
echo "PASS: mpifort links against our lib directory"

if ! echo "$NORM_SHOW" | grep -qF "$(norm_path "$MPI_PATH/include")"; then
    echo "FAIL: mpifort does not include $(norm_path "$MPI_PATH/include")"
    exit 1
fi
echo "PASS: mpifort includes our include directory"

echo ""
echo "=== Step 3: Verify .mod files exist in our install ==="
MOD_DIR=`find $MPI_PATH -name mpi.mod -print`
MOD_DIR=`dirname $MOD_DIR`

for mod in mpi.mod mpi_f08.mod; do
    if [ -f "$MOD_DIR/$mod" ]; then
        echo "PASS: $mod found"
    else
        echo "FAIL: $mod not found in $MOD_DIR"
        exit 1
    fi
done

if [[ ${AMDGPU_GFXMODEL} == "gfx942" ]]; then
        MPICH_GTL_DIRS=$PE_MPICH_GTL_DIR_amd_gfx942
        MPICH_GTL_LIBS=$PE_MPICH_GTL_LIBS_amd_gfx942
elif [[ ${AMDGPU_GFXMODEL} == "gfx90a" ]]; then
        MPICH_GTL_DIRS=$PE_MPICH_GTL_DIR_amd_gfx90a
        MPICH_GTL_LIBS=$PE_MPICH_GTL_LIBS_amd_gfx90a
else
    echo "gfx arch not included in this test"
    exit 1
fi

echo ""
echo "=== Step 4: Compile GPU-aware Fortran MPI test ==="
echo "mpifort -O3 -g \
    -fopenmp --offload-arch=${AMDGPU_GFXMODEL} \
    -L$ROCM_PATH/llvm/lib -lomp \
    -o test_gpu_aware_mpi test_gpu_aware_mpi.f90"

mpifort -O3 -g \
    -fopenmp --offload-arch=${AMDGPU_GFXMODEL} \
    -L$ROCM_PATH/llvm/lib -lomp \
    -o test_gpu_aware_mpi test_gpu_aware_mpi.f90

if [ $? -ne 0 ]; then
    echo "FAIL: compilation failed"
    exit 1
else
   echo "PASS: compilation succeeded"
fi

#if [ -n "${CRAY_MPICH_VERSION:-}" ]; then
#   echo ""
#   echo "=== Step 5: Verify binary links to Cray MPICH runtime and GPU transport ==="
#   LDD_OUTPUT=$(ldd ./test_gpu_aware_mpi)
#   echo "$LDD_OUTPUT"
#
#   if ! echo "$LDD_OUTPUT" | grep -q "libmpifort_gnu_112"; then
#       echo "FAIL: binary does not link to libmpifort_gnu_112"
#       exit 1
#   fi
#   echo "PASS: binary links to Cray MPICH Fortran runtime"
#
#   if ! echo "$LDD_OUTPUT" | grep -q "libmpi_gtl"; then
#       echo "FAIL: binary does not link to libmpi_gtl (GPU transport layer)"
#       exit 1
#   fi
#   echo "PASS: binary links to Cray GPU transport layer"
#fi

if [ -n "${CRAY_MPICH_VERSION:-}" ]; then
   MPIRUN=srun
else
   MPIRUN=mpirun
fi

echo ""
echo "=== Final Step 5: Run GPU-aware MPI test ==="
${MPIRUN} -n 2 ./test_gpu_aware_mpi

rm test_gpu_aware_mpi
