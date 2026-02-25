#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/Pragma_Examples/OpenMP/Fortran/GPU_Aware_MPI

MODULE_TO_LOAD="openmpi"
LIBFABRIC="0"

usage()
{
    echo ""
    echo "--help : prints this message"
    echo "--module : specifies the desired module to load, default is openmpi"
    echo "--libfabric : specifies whether we should load libfabrc, default is 0"
    echo ""
    exit
}

send-error()
{
    usage
    echo -e "\nError: ${@}"
    exit 1
}

reset-last()
{
   last() { send-error "Unsupported argument :: ${1}"; }
}

n=0
while [[ $# -gt 0 ]]
do
   case "${1}" in
      "--module")
          shift
	  MODULE_TO_LOAD=${1}
          reset-last
          ;;
      "--libfabric")
          shift
	  LIBFABRIC=${1}
          reset-last
          ;;
     "--help")
          usage
          ;;
      "--*")
          send-error "Unsupported argument at position $((${n} + 1)) :: ${1}"
          ;;
      *)
         last ${1}
         ;;
   esac
   n=$((${n} + 1))
   shift
done

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
   echo "rocm module is not loaded"
   echo "loading default rocm module"
   module load rocm
fi

export AMDGPU_GFXMODEL=`rocminfo |grep -m 1 -E gfx[^0]{1} | sed -e 's/ *Name: *\(gfx[0-9,a-f]*\) *$/\1/'`

if [[ ${MODULE_TO_LOAD} == *"mpich-wrappers"* ]]; then
   module unload rocm
fi

module load ${MODULE_TO_LOAD}
if [[ ${LIBFABRIC} == "1" ]]; then
   module load libfabric
fi   

if [[ ${MODULE_TO_LOAD} == *"mpich-wrappers"* ]]; then
   export MPICH_GPU_SUPPORT_ENABLED=1
   INSTALL_PATH=$MPICH_WRAPPERS_DIR
elif [[ ${MODULE_TO_LOAD} == *"openmpi"* ]]; then	
   INSTALL_PATH=$MPI_PATH  	
   module load amdclang
else
   echo "WARNING mpi module may not be currently supported by this test"
fi   

module list

echo ""
echo "=== Step 1: Verify mpifort points to our build ==="
MPIFORT_PATH=$(which mpifort)
echo "mpifort found at: $MPIFORT_PATH"
if [[ "$MPIFORT_PATH" != "$INSTALL_PATH/bin/mpifort" ]]; then
    echo "FAIL: mpifort is not from our install ($INSTALL_PATH/bin/mpifort)"
    exit 1
fi
echo "PASS: mpifort is from our install"

echo ""
echo "=== Step 2: Verify wrapper calls amdflang with our lib/include ==="
SHOW_OUTPUT=$(mpifort -show)
echo "$SHOW_OUTPUT"

if ! echo "$SHOW_OUTPUT" | grep -q "amdflang"; then
    echo "FAIL: mpifort does not wrap amdflang"
    exit 1
fi
echo "PASS: mpifort wraps amdflang"

if ! echo "$SHOW_OUTPUT" | grep -q "$INSTALL_PATH/lib"; then
    echo "FAIL: mpifort does not link against $INSTALL_PATH/lib"
    exit 1
fi
echo "PASS: mpifort links against our lib directory"

if ! echo "$SHOW_OUTPUT" | grep -q "$INSTALL_PATH/include"; then
    echo "FAIL: mpifort does not include $INSTALL_PATH/include"
    exit 1
fi
echo "PASS: mpifort includes our include directory"

echo ""
echo "=== Step 3: Verify .mod files exist in our install ==="
if [[ ${MODULE_TO_LOAD} == *"mpich-wrappers"* ]]; then
   MOD_DIR=$INSTALL_PATH"/include"	
else 
   MOD_DIR=$MPI_PATH"/lib"	
fi     	

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

if [[ ${MODULE_TO_LOAD} == *"mpich-wrappers"* ]]; then
   echo ""
   echo "=== Step 4: Compile GPU-aware Fortran MPI test ==="
   echo "mpifort -O3 -g \
       -fopenmp --offload-arch=${AMDGPU_GFXMODEL} \
       -L$ROCM_PATH/llvm/lib -lomp \
       -L$MPICH_DIR/lib -lmpifort_gnu_112 \
       $MPICH_GTL_DIRS $MPICH_GTL_LIBS \
       -o test_gpu_aware_mpi test_gpu_aware_mpi.f90"

   mpifort -O3 -g \
       -fopenmp --offload-arch=${AMDGPU_GFXMODEL} \
       -L$ROCM_PATH/llvm/lib -lomp \
       -L$MPICH_DIR/lib -lmpifort_gnu_112 \
       $MPICH_GTL_DIRS $MPICH_GTL_LIBS \
       -o test_gpu_aware_mpi test_gpu_aware_mpi.f90
   if [ $? -ne 0 ]; then
       echo "FAIL: compilation failed"
       exit 1
   else
      echo "PASS: compilation succeeded"
   fi
elif [[ ${MODULE_TO_LOAD} == *"openmpi"* ]]; then

     echo "     mpifort -O3 -g \
       -fopenmp --offload-arch=${AMDGPU_GFXMODEL} \
       -L$ROCM_PATH/llvm/lib -lomp \
       -o test_gpu_aware_mpi test_gpu_aware_mpi.f90"

     mpifort -O3 -g \
       -fopenmp --offload-arch=${AMDGPU_GFXMODEL} \
       -L$ROCM_PATH/llvm/lib -lomp \
       -o test_gpu_aware_mpi test_gpu_aware_mpi.f90
else
   echo "mpi module not considered in this test"
   exit 1
fi

if [[ ${MODULE_TO_LOAD} == *"mpich-wrappers"* ]]; then
   echo ""
   echo "=== Step 5: Verify binary links to Cray MPICH runtime and GPU transport ==="
   LDD_OUTPUT=$(ldd ./test_gpu_aware_mpi)
   echo "$LDD_OUTPUT"

   if ! echo "$LDD_OUTPUT" | grep -q "libmpifort_gnu_112"; then
       echo "FAIL: binary does not link to libmpifort_gnu_112"
       exit 1
   fi
   echo "PASS: binary links to Cray MPICH Fortran runtime"

   if ! echo "$LDD_OUTPUT" | grep -q "libmpi_gtl"; then
       echo "FAIL: binary does not link to libmpi_gtl (GPU transport layer)"
       exit 1
   fi
   echo "PASS: binary links to Cray GPU transport layer"
fi   

echo ""
echo "=== Final Step 6: Run GPU-aware MPI test ==="
if [[ ${MODULE_TO_LOAD} == *"mpich-wrappers"* ]]; then
    srun -n 2 ./test_gpu_aware_mpi
else
    mpirun -n 2 ./test_gpu_aware_mpi
fi

rm test_gpu_aware_mpi
