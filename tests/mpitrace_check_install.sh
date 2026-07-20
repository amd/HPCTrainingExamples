#!/bin/bash

# Quick install/sanity check for the IBM mpitrace MPI profiling library.
# Verifies: (1) the version-matched mpitrace module resolves, (2) its
# libmpitrace.so exists, (3) the module wires LD_PRELOAD, and (4) it actually
# profiles a trivial 2-rank MPI run (produces mpi_profile.* files).

# rocm is required (the openmpi module the mpitrace modulefile loads declares
# rocm as a prereq). Mirror the load-guard idiom used by the other tests.
module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
   echo "rocm module is not loaded"
   echo "loading default rocm module"
   module load rocm
fi

ROCM_VER=$(module -t list 2>&1 | grep '^rocm/' | head -1 | cut -d/ -f2)
if ! module load mpitrace/${ROCM_VER} 2>/tmp/mpitrace_check.$$.err; then
   cat /tmp/mpitrace_check.$$.err
   rm -f /tmp/mpitrace_check.$$.err
   # Canonical token so CTest's SKIP_REGULAR_EXPRESSION marks this SKIPPED
   # (not FAILED) when mpitrace is not installed for this rocm version.
   echo "Unable to locate a modulefile for 'mpitrace'"
   exit 0
fi
rm -f /tmp/mpitrace_check.$$.err

echo "=== mpitrace install check ==="
echo "ROCM_VER:      ${ROCM_VER}"
echo "MPITRACE_PATH: ${MPITRACE_PATH}"
echo "MPITRACE_LIB:  ${MPITRACE_LIB}"
echo "LD_PRELOAD:    ${LD_PRELOAD}"

# (1) library present
if [ ! -f "${MPITRACE_LIB}" ]; then
   echo "FAIL: MPITRACE_LIB does not point to an existing file (${MPITRACE_LIB})"
   exit 1
fi
# header is nice-to-have, not fatal
if [ ! -f "${MPITRACE_PATH}/include/mpitrace.h" ]; then
   echo "WARN: mpitrace.h not found under ${MPITRACE_PATH}/include"
fi

# (2) module wires LD_PRELOAD to libmpitrace.so
if [ "${LD_PRELOAD}" != "${MPITRACE_LIB}" ]; then
   echo "FAIL: LD_PRELOAD (${LD_PRELOAD}) is not set to libmpitrace.so (${MPITRACE_LIB})"
   exit 1
fi

# (3) basic functionality: profile a trivial MPI run and check output files
WORKDIR=$(mktemp -d -t mpitrace_check.XXXXXX)
cd "${WORKDIR}" || { echo "FAIL: could not enter workdir"; exit 1; }

cat > mpi_hello.c << 'EOF'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv)
{
   int rank, size;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Barrier(MPI_COMM_WORLD);
   if (rank == 0) printf("mpi_hello: %d ranks\n", size);
   MPI_Finalize();
   return 0;
}
EOF

if ! mpicc mpi_hello.c -o mpi_hello; then
   echo "FAIL: mpicc could not compile the MPI test program"
   cd / && rm -rf "${WORKDIR}"
   exit 1
fi

# Launch a trivial CPU-only MPI job (mpitrace profiles the host MPI layer; no
# GPU needed). Force pml ob1 + shared-mem/self btl so it runs on a GPU-less
# front-end too (the default UCX pml aborts where no HIP device is present).
# Require a real 2-rank run; the launch is time-boxed so a stuck launcher can't
# hang the test.
RC=1
LAUNCH="mpirun -np 2 --oversubscribe -mca pml ob1 -mca btl self,sm ./mpi_hello"
rm -f mpi_profile.*
echo "+ ${LAUNCH}"
timeout 90 ${LAUNCH}
if ls mpi_profile.* >/dev/null 2>&1; then
   echo "mpitrace produced profile output:"
   ls -1 mpi_profile.*
   echo "MPItrace Install Check: SUCCESS"
   RC=0
else
   echo "FAIL: no mpi_profile.* files produced (LD_PRELOAD profiling did not run)"
fi

cd / && rm -rf "${WORKDIR}"
exit ${RC}
