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

export TAU_PROFILE=${TAU_PROFILE}
export TAU_TRACE=${TAU_TRACE}

WORKDIR=$(mktemp -d -p ${SRCDIR} build_XXXXXX)
cp ${SRCDIR}/*.hip ${SRCDIR}/*.hpp ${SRCDIR}/*.h ${SRCDIR}/Makefile ${SRCDIR}/input.txt ${WORKDIR}/
cd ${WORKDIR}

make

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
result=`echo ${ROCM_VERSION} | awk '$1>6.1.9'` && echo $result

# MPI launcher: use the launcher that MATCHES the MPI the binary was linked
# against. This Jacobi is built by its Makefile with mpic++/mpicc from whatever
# MPI module is active (mpich-wrappers on Cray, openmpi off-Cray). Both ship
# their own mpirun/mpiexec next to mpicc. Launching such an MPI with srun gives
# each rank a SINGLETON MPI_COMM_WORLD (size 1) because srun's Cray PMI does not
# wire it up -- which then fails Jacobi's "-g 2 1" topology check
# ("MPI processes (2) doesn't match the topology size (1)"). Only a bare Cray
# MPICH (mpicc with no co-located launcher) falls back to srun. The MPICH hydra
# mpiexec also rejects OpenMPI's --oversubscribe, so add that flag for OpenMPI
# only (detected via ompi_info).
MPI_BINDIR=$(dirname "$(command -v mpicc 2>/dev/null)" 2>/dev/null)
if [ -n "${MPI_BINDIR}" ] && [ -x "${MPI_BINDIR}/mpirun" ]; then
   MPI_LAUNCH="${MPI_BINDIR}/mpirun -n 2"
elif [ -n "${MPI_BINDIR}" ] && [ -x "${MPI_BINDIR}/mpiexec" ]; then
   MPI_LAUNCH="${MPI_BINDIR}/mpiexec -n 2"
else
   MPI_LAUNCH="srun -n 2"
fi
if command -v ompi_info >/dev/null 2>&1; then
   MPI_LAUNCH="${MPI_LAUNCH} --oversubscribe"
fi

# Use a 1024x1024 local mesh so the TAU trace buffer fits in GPU memory
# (default 4096x4096 caused "HIP failure: 'out of memory'" during trace finalization).
if [[ "${result}" ]]; then
   ${MPI_LAUNCH} tau_exec -rocm -T rocm,rocprofsdk ./Jacobi_hip -g 2 1 -m 1024 1024
else
   ${MPI_LAUNCH} tau_exec -T rocm,roctracer,rocprofiler ./Jacobi_hip -g 2 1 -m 1024 1024
fi

ls
pprof

cd ..
rm -rf ${WORKDIR}

