#!/bin/bash
#
# Run the full OpenMP Offloading Validation & Verification (OMPVV / SOLLVE)
# test suite on an AMD MI300A (gfx942) GPU.
#
# Project:  https://sollve.github.io/
# Suite:    https://github.com/OpenMP-Validation-and-Verification/OpenMP_VV
#           (formerly https://github.com/SOLLVE/sollve_vv)
#
# This script compiles and runs the C (amdclang), C++ (amdclang++) and Fortran
# (amdflang) tests across all OpenMP spec versions (4.5, 5.0, 5.1, 5.2).
#
# NOTE: this is the complete suite (thousands of tests x 4 versions) and can
# take a long time. For a quick Fortran-only check use
# sollve_vv_fortran_offload.sh instead.
#
# Usage:
#   ./sollve_vv_full.sh [ROCM_VERSION]
#
#   ROCM_VERSION  optional module version, e.g. 6.4.3 -> "module load rocm/6.4.3".
#                 If omitted, the currently loaded rocm module is used, or the
#                 default rocm module is loaded.
#
# Environment:
#   SOLLVE_TIMELIMIT  per-test runtime limit in seconds (default 180). Tests that
#                     run longer are reported as "FAIL: TEST HAS TIMEOUT".
#
# Examples:
#   ./sollve_vv_full.sh 6.4.3
#   ./sollve_vv_full.sh                       # use loaded / default rocm module
#   SOLLVE_TIMELIMIT=300 ./sollve_vv_full.sh  # raise per-test timeout to 300s
#

ROCM_VERSION="$1"

# ---------------------------------------------------------------------------
# Load compiler environment
# ---------------------------------------------------------------------------
if [ -n "${ROCM_VERSION}" ]; then
   echo "Loading rocm/${ROCM_VERSION}"
   module load rocm/${ROCM_VERSION}
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
      echo "rocm module is not loaded"
      echo "loading default rocm module"
      module load rocm
   fi
fi

module load amdflang-new >& /dev/null
if [ "$?" == "1" ]; then
   module load amdclang
fi

# ---------------------------------------------------------------------------
# Only run on MI300A (gfx942)
# ---------------------------------------------------------------------------
GFX_MODEL=`rocminfo | grep -m1 gfx | sed -e 's/Name://' | tr -d ' '`
if [ "${GFX_MODEL}" != "gfx942" ]; then
   echo "This test targets MI300A (gfx942), found '${GFX_MODEL}'"
   echo "Skip"
   exit 0
fi

# MI300A is an APU with unified (shared) memory; enable XNACK and mandate offload
export HSA_XNACK=1
export OMP_TARGET_OFFLOAD=MANDATORY

# Per-test runtime limit (seconds) enforced by the suite's run_test.sh via the
# `timeout` command. Tests exceeding this are reported as
# "FAIL: TEST HAS TIMEOUT" (exit code 124). Override by exporting SOLLVE_TIMELIMIT
# before invoking this script; otherwise default to 180s (upstream default is 60s).
export SOLLVE_TIMELIMIT=${SOLLVE_TIMELIMIT:-180}

# ---------------------------------------------------------------------------
# Working and results directories
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(dirname "$(readlink -fm "$0")")"
DATE_TAG=`date +%Y%m%d_%H%M%S`
RESULTS_DIR="${SCRIPT_DIR}/results_full_${DATE_TAG}"
mkdir -p "${RESULTS_DIR}"

WORKDIR=$(mktemp -d -p /tmp sollve_vv_XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

# ---------------------------------------------------------------------------
# Clone the suite
# ---------------------------------------------------------------------------
git clone --depth=1 https://github.com/OpenMP-Validation-and-Verification/OpenMP_VV ${WORKDIR}/OpenMP_VV
if [ ! -d "${WORKDIR}/OpenMP_VV" ]; then
   echo "ERROR: failed to clone OpenMP_VV"
   exit 1
fi
cd ${WORKDIR}/OpenMP_VV

# Target the local device explicitly. amdgpu-arch 'native' detection is
# unreliable at build time, so pin the offload arch to the detected device:
# amdflang hardcodes gfx90a and amdclang/amdclang++ default to native.
sed -i "s/--offload-arch=native/--offload-arch=${GFX_MODEL}/g; s/--offload-arch=gfx90a/--offload-arch=${GFX_MODEL}/g" sys/make/make.def

# Upstream run_test.sh references $2 under "set -u" even when no log argument is
# passed (every non --env test), which aborts the run step. Make it tolerant.
sed -i 's/if \[ -z \$2 \]; then/if [ -z "${2:-}" ]; then/' sys/scripts/run_test.sh

echo "==================================================================="
echo " OpenMP_VV full (C/C++/Fortran) run on ${GFX_MODEL}"
echo " ROCm:     `cat ${ROCM_PATH}/.info/version 2>/dev/null | head -1`"
echo " amdclang: `amdclang --version 2>/dev/null | head -1`"
echo " amdflang: `amdflang --version 2>/dev/null | head -1`"
echo "==================================================================="

# ---------------------------------------------------------------------------
# Build and run each OpenMP spec version in its own clean build / log dir
# ---------------------------------------------------------------------------
for v in 4.5 5.0 5.1 5.2; do
   echo ""
   echo "=================== OpenMP ${v} (C/C++/Fortran) ==================="
   # Pin the language level to the test set. The suite only sets OMPV for
   # 5.0/5.1/5.2, leaving 4.5 at the compiler default; override OMPV for every
   # set so 4.5 gets -fopenmp-version=45 too.
   OMPV_FLAG="-fopenmp-version=`echo ${v} | tr -d '.'`"
   make CC=amdclang CXX=amdclang++ FC=amdflang OMP_VERSION=${v} OMPV="${OMPV_FLAG}" \
        BINDIR=bin_${v} LOGDIRNAME=logs_${v} LOG_ALL=1 all

   echo ""
   echo "------------------- Summary OpenMP ${v} -------------------"
   make OMP_VERSION=${v} LOGDIRNAME=logs_${v} report_summary

   # Preserve logs for inspection
   if [ -d "logs_${v}" ]; then
      cp -r "logs_${v}" "${RESULTS_DIR}/"
   fi
done

echo ""
echo "==================================================================="
echo " Logs copied to: ${RESULTS_DIR}"
echo "==================================================================="
