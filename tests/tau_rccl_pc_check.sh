#!/bin/bash

# TAU + RCCL PC-sampling check (from the TAU author's examples).
#
# Runs the rccl-tests `all_gather_perf` collective benchmark under TAU with GPU
# PC sampling (`-rocm_pc`) and, for the RCCL-routine view, event-based sampling
# (`-ebs`). The resulting `pprof` text profile is expected to contain three
# kinds of entries that the two example runs produce:
#
#   1. Host-side RCCL/NCCL routines  (mpirun -np 2 tau_exec -rocm_pc -ebs ...):
#        ncclCommInitRank, ncclGroupEnd, ncclAllGather, ncclCommDestroy, ...
#   2. GPU PC samples                (tau_exec -rocm_pc ...):
#        [rocm sample] Stall reason [WAITCNT] ...
#   3. GPU kernel execution          (tau_exec -rocm_pc ...):
#        [ROCm Kernel] ncclDevKernel_Generic_... [clone .kd]
#
# Usage:
#   ./tau_rccl_pc_check.sh                 # RCCL routine view (mpirun -np 2, -ebs)
#   ./tau_rccl_pc_check.sh --rccl-routines # same as default
#   ./tau_rccl_pc_check.sh --pc-sampling   # single-rank GPU PC-sampling view
#
# Both modes run `all_gather_perf` under `tau_exec -rocm_pc`; the full pprof
# output is printed so CTest can match the marker for the view under test. The
# test SKIPs (exit 0) when the tau or rccl-tests modules are unavailable or the
# node has fewer than 2 GPUs.

MODE="rccl-routines"

usage()
{
    echo ""
    echo "--help          : prints this message"
    echo "--rccl-routines : host-side RCCL/NCCL routine view (mpirun -np 2, -rocm_pc -ebs) [default]"
    echo "--pc-sampling   : single-rank GPU PC-sampling + kernel view (tau_exec -rocm_pc)"
    echo ""
    exit
}

send-error()
{
    usage
    echo -e "\nError: ${@}"
    exit 1
}

while [[ $# -gt 0 ]]
do
   case "${1}" in
      "--rccl-routines")
          MODE="rccl-routines"
          shift
          ;;
      "--pc-sampling")
          MODE="pc-sampling"
          shift
          ;;
      "--help")
          usage
          ;;
      *)
          send-error "Unsupported argument :: ${1}"
          ;;
   esac
done

# --- Modules -----------------------------------------------------------------
module -t list 2>&1 | grep -q "^rocm" || module load rocm

# On a Cray PE the MPI comes from the programming environment (cray-mpich +
# mpich-wrappers), so there is no 'openmpi' module to load. Only load openmpi
# off-Cray.
if [[ -z "$CRAYPE_VERSION" && ! -f /etc/cray-release ]]; then
   module load openmpi 2>/dev/null
fi

if ! module load rccl-tests 2>/tmp/tau_rccl_pc.$$.err; then
   cat /tmp/tau_rccl_pc.$$.err
   rm -f /tmp/tau_rccl_pc.$$.err
   echo "Unable to locate a modulefile for 'rccl-tests'"
   exit 0
fi
rm -f /tmp/tau_rccl_pc.$$.err

if ! module load tau 2>/tmp/tau_rccl_pc.$$.err; then
   cat /tmp/tau_rccl_pc.$$.err
   rm -f /tmp/tau_rccl_pc.$$.err
   echo "Unable to locate a modulefile for 'tau'"
   exit 0
fi
rm -f /tmp/tau_rccl_pc.$$.err

command -v tau_exec >/dev/null 2>&1 || { echo "Unable to locate a modulefile for 'tau'"; exit 0; }

# --- Locate the benchmark ----------------------------------------------------
ALL_GATHER="${RCCL_TESTS_PATH}/bin/all_gather_perf"
echo "all_gather_perf: ${ALL_GATHER}"
[ -x "${ALL_GATHER}" ] || { echo "Skip: all_gather_perf not found under RCCL_TESTS_PATH"; exit 0; }

# --- Require >= 2 GPUs (RCCL collectives need at least 2 ranks/GPUs) ----------
GPU_COUNT=$(rocminfo 2>/dev/null | grep -c "Device Type:             GPU")
echo "GPU count: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 2 ]; then
   echo "Skip: TAU RCCL PC-sampling check needs at least 2 GPUs (found ${GPU_COUNT})"
   exit 0
fi

# --- MPI launcher (match the MPI the benchmark was linked against) ------------
MPI_BINDIR=$(dirname "$(command -v mpicc 2>/dev/null)" 2>/dev/null)
if [ -n "${MPI_BINDIR}" ] && [ -x "${MPI_BINDIR}/mpirun" ]; then
   MPI_LAUNCH="${MPI_BINDIR}/mpirun -np 2"
elif [ -n "${MPI_BINDIR}" ] && [ -x "${MPI_BINDIR}/mpiexec" ]; then
   MPI_LAUNCH="${MPI_BINDIR}/mpiexec -np 2"
else
   MPI_LAUNCH="srun -n 2"
fi
# Detect the MPI flavour from the launch wrapper; see tau_exec_check.sh. Checking
# PATH instead could add OpenMPI's --oversubscribe to MPICH's mpiexec, which
# rejects it. This does not fix the case where TAU and the prebuilt rccl-tests
# binary come from different MPIs.
MPI_FAMILY=unknown
if [ -n "${MPI_BINDIR}" ]; then
   if "${MPI_BINDIR}/mpicc" --showme:version >/dev/null 2>&1; then
      MPI_FAMILY=openmpi
   elif "${MPI_BINDIR}/mpicc" -compile_info >/dev/null 2>&1; then
      MPI_FAMILY=mpich
   fi
fi
echo "MPI family of ${MPI_BINDIR:-<none, using srun>}: ${MPI_FAMILY}"
if [ "${MPI_FAMILY}" = "openmpi" ]; then
   MPI_LAUNCH="${MPI_LAUNCH} --oversubscribe"
fi

# --- Scratch profile dir -----------------------------------------------------
WORKDIR=$(mktemp -d)
export PROFILEDIR="${WORKDIR}"
export TAU_PROFILE=1
# The tau modulefile defaults TAU_PROFILE_FORMAT=merged (single tauprofile.xml,
# requested by the TAU author). pprof below only reads the per-rank/thread
# profile.* text format, so request it for this check's scratch run; the users'
# merged default is unaffected.
export TAU_PROFILE_FORMAT=profile
trap 'cd /; rm -rf "${WORKDIR}"' EXIT
cd "${WORKDIR}" || exit 1

# Keep the message range small so the run (and TAU finalization) stays quick.
BENCH_ARGS="-b 8 -e 8M -f 2 -g 1 -n 20"

echo "=== mode: ${MODE} ==="
if [ "${MODE}" = "pc-sampling" ]; then
   # Example B: single-rank GPU PC sampling + kernel view.
   echo "+ tau_exec -rocm_pc ${ALL_GATHER} ${BENCH_ARGS/-g 1/-g 2}"
   tau_exec -rocm_pc "${ALL_GATHER}" -b 8 -e 8M -f 2 -g 2 -n 20 \
      || echo "WARN: tau_exec returned nonzero (profile may still be usable)"
else
   # Example A: host-side RCCL/NCCL routine view across 2 ranks with EBS.
   echo "+ ${MPI_LAUNCH} tau_exec -rocm_pc -ebs ${ALL_GATHER} ${BENCH_ARGS}"
   ${MPI_LAUNCH} tau_exec -rocm_pc -ebs "${ALL_GATHER}" ${BENCH_ARGS} \
      || echo "WARN: tau_exec returned nonzero (profile may still be usable)"
fi

# --- Report ------------------------------------------------------------------
echo "=== profile files in ${PROFILEDIR} ==="
ls -la "${PROFILEDIR}"

echo "=== pprof ==="
pprof -a 2>/dev/null || pprof 2>/dev/null || echo "WARN: pprof produced no output"
