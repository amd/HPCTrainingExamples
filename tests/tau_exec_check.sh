#!/bin/bash

TAU_TRACE=0
TAU_PROFILE=0

usage()
{
    echo ""
    echo "--help : prints this message"
    echo "--tau-trace : sets TAU_TRACE=1 - default is TAU_TRACE=0"
    echo "--tau-profile : sets TAU_PROFILE=1 - default is TAU_PROFILE=0"
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
      "--tau-trace")
          shift
          TAU_TRACE=1
          reset-last
          ;;
      "--tau-profile")
          shift
          TAU_PROFILE=1
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

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
SRCDIR=${REPO_DIR}/HIP/jacobi

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
# On a Cray PE the MPI comes from the programming environment (cray-mpich +
# the mpich-wrappers), so there is no 'openmpi' module to load -- attempting it
# fails with "ERROR:105: Unable to locate a modulefile for 'openmpi'". Only
# load openmpi off-Cray.
if [[ -z "$CRAYPE_VERSION" && ! -f /etc/cray-release ]]; then
   module load openmpi
fi
module load tau

# The TAU modulefile sets TAU_PROFILE_FORMAT=merged (requested by the TAU
# author) so interactive users get a single tauprofile.xml instead of many
# profile.<node>.<ctx>.<thread> files. The pprof post-processing below (and the
# "profile.0" file the TAU_Profile_Check looks for) only understands the
# per-rank/thread profile.* text format, so this check requests that format for
# its own scratch run. The user-facing default (merged) is left untouched.
export TAU_PROFILE_FORMAT=profile

export TAU_PROFILE=${TAU_PROFILE}
export TAU_TRACE=${TAU_TRACE}

WORKDIR=$(mktemp -d -p ${SRCDIR} build_XXXXXX)
# Install the cleanup trap before the build gate below, which can exit early.
trap 'cd "${SRCDIR}" && rm -rf "${WORKDIR}"' EXIT
cp ${SRCDIR}/*.hip ${SRCDIR}/*.hpp ${SRCDIR}/*.h ${SRCDIR}/Makefile ${SRCDIR}/input.txt ${WORKDIR}/
cd ${WORKDIR}

# Stop the test if the build fails, otherwise tau_exec would go on to profile a
# binary that was never produced.
if ! make; then
   echo "TAU_CHECK: build FAILED, so nothing was profiled"
   exit 1
fi
if [ ! -x ./Jacobi_hip ]; then
   echo "TAU_CHECK: build reported success but ./Jacobi_hip is missing"
   exit 1
fi

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `

# Version-sorted comparison so two-digit majors (10.x, 11.x, ...) order
# correctly. The old `awk '$1>6.1.9'` compared version strings lexically, so
# "10.1.0" tested as LESS THAN "6.1.9" and the else-branch selected the retired
# `-T rocm,roctracer,rocprofiler` binding (removed in ROCm >= 10 in favour of
# rocprofiler-sdk), yielding "No matching binding".
ver_gt() { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" = "$1" ]; }
result=""; ver_gt "${ROCM_VERSION}" 6.1.9 && result="${ROCM_VERSION}"; echo $result

# MPI launcher: use the launcher that MATCHES the MPI the binary was linked
# against. This Jacobi is built by its Makefile with mpic++/mpicc from whatever
# MPI module is active (mpich-wrappers on Cray, openmpi off-Cray). Both ship
# their own mpirun/mpiexec next to mpicc. Launching such an MPI with srun gives
# each rank a SINGLETON MPI_COMM_WORLD (size 1) because srun's Cray PMI does not
# wire it up -- which then fails Jacobi's "-g 2 1" topology check
# ("MPI processes (2) doesn't match the topology size (1)"). Only a bare Cray
# MPICH (mpicc with no co-located launcher) falls back to srun. The MPICH hydra
# mpiexec also rejects OpenMPI's --oversubscribe, so add that flag for OpenMPI
# only (detected from the MPI wrapper below).
MPI_BINDIR=$(dirname "$(command -v mpicc 2>/dev/null)" 2>/dev/null)
if [ -n "${MPI_BINDIR}" ] && [ -x "${MPI_BINDIR}/mpirun" ]; then
   MPI_LAUNCH="${MPI_BINDIR}/mpirun -n 2"
elif [ -n "${MPI_BINDIR}" ] && [ -x "${MPI_BINDIR}/mpiexec" ]; then
   MPI_LAUNCH="${MPI_BINDIR}/mpiexec -n 2"
else
   MPI_LAUNCH="srun -n 2"
fi
# Ask the launch wrapper which MPI it is: only OpenMPI answers --showme:version
# and only MPICH answers -compile_info. This is more reliable than checking PATH,
# since a module can put one MPI's wrappers in front of another's.
MPI_FAMILY=unknown
if [ -n "${MPI_BINDIR}" ]; then
   if "${MPI_BINDIR}/mpicc" --showme:version >/dev/null 2>&1; then
      MPI_FAMILY=openmpi
   elif "${MPI_BINDIR}/mpicc" -compile_info >/dev/null 2>&1; then
      MPI_FAMILY=mpich
   fi
fi
echo "TAU_CHECK: launching with ${MPI_FAMILY} wrappers from ${MPI_BINDIR:-<none, using srun>}"
if [ "${MPI_FAMILY}" = "openmpi" ]; then
   MPI_LAUNCH="${MPI_LAUNCH} --oversubscribe"
fi

# Run both ranks on this node. The test needs two ranks on two GPUs, not two
# nodes, and Jacobi is built in a directory inside the checkout. If that checkout
# is on node-local storage, a multi-node launch could place a rank on a node that
# cannot see the build directory. Hydra takes -hosts, OpenMPI takes --host with a
# slot count. The srun fallback is left as is, because pinning it would also need
# the step's GRES respecified.
MPI_HOST_LOCAL="$(hostname -s)"
case "${MPI_LAUNCH}" in
   srun*) ;;
   *) case "${MPI_FAMILY}" in
         openmpi) MPI_LAUNCH="${MPI_LAUNCH} --host ${MPI_HOST_LOCAL}:2" ;;
         mpich)   MPI_LAUNCH="${MPI_LAUNCH} -hosts ${MPI_HOST_LOCAL}" ;;
         *) echo "TAU_CHECK: unknown MPI family, not pinning ranks to ${MPI_HOST_LOCAL}" ;;
      esac ;;
esac
echo "TAU_CHECK: launcher is ${MPI_LAUNCH}"

# Use a 1024x1024 local mesh so the TAU trace buffer fits in GPU memory
# (default 4096x4096 caused "HIP failure: 'out of memory'" during trace finalization).
if [[ "${result}" ]]; then
   ${MPI_LAUNCH} tau_exec -rocm -T rocm,rocprofsdk ./Jacobi_hip -g 2 1 -m 1024 1024
else
   ${MPI_LAUNCH} tau_exec -T rocm,roctracer,rocprofiler ./Jacobi_hip -g 2 1 -m 1024 1024
fi

ls
pprof 2>&1 | tee pprof.out

# Check for the produced files and print our own result strings, which CMake
# matches on. Matching the tool's own words was unreliable: "profile.0" also
# appears in messages like "Could not open profile.0.0.0".
tau_have() { ls $1 >/dev/null 2>&1; }
tau_status=0

if [ "${TAU_PROFILE}" = "1" ]; then
   if tau_have 'profile.*'; then
      echo "TAU_CHECK: profile artifact PRESENT"
   else
      echo "TAU_CHECK: profile artifact absent"
      tau_status=1
   fi
   # Reported but not fatal on their own: each is a different test's subject, so
   # the per-test expression decides rather than failing all of them together.
   if grep -q 'MPI_Allreduce' pprof.out; then
      echo "TAU_CHECK: MPI routines PRESENT in profile"
   else
      echo "TAU_CHECK: MPI routines absent from profile"
   fi
   if grep -q 'hipMemcpy' pprof.out; then
      echo "TAU_CHECK: HIP routines PRESENT in profile"
   else
      echo "TAU_CHECK: HIP routines absent from profile"
   fi
fi

if [ "${TAU_TRACE}" = "1" ]; then
   if tau_have 'tautrace.*'; then
      echo "TAU_CHECK: trace artifact PRESENT"
   else
      echo "TAU_CHECK: trace artifact absent"
      tau_status=1
   fi
fi

exit ${tau_status}
