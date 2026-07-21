#!/bin/bash
#
# Sweep the OpenMP Offloading Validation & Verification (OMPVV / SOLLVE) suite
# across several ROCm/compiler versions on an AMD MI300A (gfx942) GPU and report
# the results and the changes (regressions / fixes) between versions.
#
# Project:  https://sollve.github.io/
# Suite:    https://github.com/OpenMP-Validation-and-Verification/OpenMP_VV
#
# For each ROCm version the script loads the matching rocm module and then the
# amdclang module, which sets CC/CXX/FC to that ROCm's amdclang / amdclang++ /
# amdflang. The suite is built and run for all OpenMP spec versions
# (4.5, 5.0, 5.1, 5.2). Per-version results are emitted as CSV and then compared
# with sollve_vv_compare.py.
#
# Usage:
#   ./sollve_vv_rocm_sweep.sh [--fortran] [ROCM_VER ...]
#
#   --fortran     Fortran-only sweep (faster). Default: full C/C++/Fortran.
#   ROCM_VER ...  ROCm module versions to sweep. Default: 6.4.3 7.0.2 7.2.3
#
# Examples:
#   ./sollve_vv_rocm_sweep.sh                       # default 3 versions, full
#   ./sollve_vv_rocm_sweep.sh 6.3.4 6.4.3 7.0.2 7.2.3
#   ./sollve_vv_rocm_sweep.sh --fortran 6.4.3 7.0.2 7.2.3
#

FORTRAN_ONLY=0
if [ "$1" == "--fortran" ]; then
   FORTRAN_ONLY=1
   shift
fi

if [ "$#" -ge 1 ]; then
   ROCM_VERSIONS=("$@")
else
   ROCM_VERSIONS=(6.4.3 7.0.2 7.2.3)
fi

OMP_VERSIONS=(4.5 5.0 5.1 5.2)

# ---------------------------------------------------------------------------
# Confirm this is an MI300A (gfx942) before doing any heavy work
# ---------------------------------------------------------------------------
module load rocm >& /dev/null
GFX_MODEL=`rocminfo | grep -m1 gfx | sed -e 's/Name://' | tr -d ' '`
if [ "${GFX_MODEL}" != "gfx942" ]; then
   echo "This test targets MI300A (gfx942), found '${GFX_MODEL}'"
   echo "Skip"
   exit 0
fi

# MI300A is an APU with unified (shared) memory
export HSA_XNACK=1
export OMP_TARGET_OFFLOAD=MANDATORY

# ---------------------------------------------------------------------------
# Working and results directories
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(dirname "$(readlink -fm "$0")")"
DATE_TAG=`date +%Y%m%d_%H%M%S`
if [ "${FORTRAN_ONLY}" == "1" ]; then
   RESULTS_DIR="${SCRIPT_DIR}/results_sweep_fortran_${DATE_TAG}"
else
   RESULTS_DIR="${SCRIPT_DIR}/results_sweep_full_${DATE_TAG}"
fi
mkdir -p "${RESULTS_DIR}"

WORKDIR=$(mktemp -d -p /tmp sollve_vv_sweep_XXXXXX)
trap "rm -rf ${WORKDIR}" EXIT

# ---------------------------------------------------------------------------
# Clone the suite once and apply the MI300A adaptations
# ---------------------------------------------------------------------------
git clone --depth=1 https://github.com/OpenMP-Validation-and-Verification/OpenMP_VV ${WORKDIR}/OpenMP_VV
if [ ! -d "${WORKDIR}/OpenMP_VV" ]; then
   echo "ERROR: failed to clone OpenMP_VV"
   exit 1
fi
cd ${WORKDIR}/OpenMP_VV

# Pin the offload arch to the detected device explicitly (amdgpu-arch 'native'
# detection is unreliable at build time): amdflang hardcodes gfx90a and
# amdclang/amdclang++ default to native.
sed -i "s/--offload-arch=native/--offload-arch=${GFX_MODEL}/g; s/--offload-arch=gfx90a/--offload-arch=${GFX_MODEL}/g" sys/make/make.def
# run_test.sh references $2 under "set -u" even when no log arg is passed.
sed -i 's/if \[ -z \$2 \]; then/if [ -z "${2:-}" ]; then/' sys/scripts/run_test.sh

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
COMPARE_ARGS=()
for ROCM_VERSION in "${ROCM_VERSIONS[@]}"; do
   echo ""
   echo "###################################################################"
   echo "# ROCm ${ROCM_VERSION}"
   echo "###################################################################"

   # Switch the toolchain. Unload first so amdclang re-points CC/CXX/FC at the
   # newly loaded rocm version.
   module unload amdclang amdflang-new >& /dev/null
   module unload rocm >& /dev/null
   module load rocm/${ROCM_VERSION}
   if [ $? -ne 0 ]; then
      echo "WARNING: could not load rocm/${ROCM_VERSION}, skipping"
      continue
   fi
   module load amdclang
   echo "CC=${CC}"
   echo "CXX=${CXX}"
   echo "FC=${FC}"

   # Fortran-only sweep overrides CC/CXX to none so only amdflang tests build.
   LANG_FLAGS=""
   if [ "${FORTRAN_ONLY}" == "1" ]; then
      LANG_FLAGS="CC=none CXX=none"
   fi

   for v in "${OMP_VERSIONS[@]}"; do
      echo ""
      echo "----- ROCm ${ROCM_VERSION} / OpenMP ${v} -----"
      # Pin the language level to the test set (suite leaves 4.5 unpinned).
      OMPV_FLAG="-fopenmp-version=`echo ${v} | tr -d '.'`"
      make ${LANG_FLAGS} OMP_VERSION=${v} OMPV="${OMPV_FLAG}" \
           BINDIR=bin_${ROCM_VERSION}_${v} \
           LOGDIRNAME=logs_${ROCM_VERSION}_${v} LOG_ALL=1 all
   done

   # One CSV per ROCm version, aggregating all its OpenMP-version logs.
   CSV="${RESULTS_DIR}/results_${ROCM_VERSION}.csv"
   python3 sys/scripts/createSummary.py -r -f csv -o "${CSV}" \
           logs_${ROCM_VERSION}_*/* 2>/dev/null
   echo "Wrote ${CSV}"
   COMPARE_ARGS+=("${ROCM_VERSION}:${CSV}")

   # Per-version summary to the console / kept log
   echo "----- summary ROCm ${ROCM_VERSION} (all OpenMP versions) -----"
   python3 sys/scripts/createSummary.py -r -f summary logs_${ROCM_VERSION}_*/* 2>/dev/null
done

# ---------------------------------------------------------------------------
# Cross-version comparison
# ---------------------------------------------------------------------------
echo ""
echo "###################################################################"
echo "# Cross-version comparison"
echo "###################################################################"
MATRIX="${RESULTS_DIR}/comparison_matrix.csv"
python3 "${SCRIPT_DIR}/sollve_vv_compare.py" -o "${MATRIX}" "${COMPARE_ARGS[@]}" \
        | tee "${RESULTS_DIR}/comparison_report.txt"

echo ""
echo "==================================================================="
echo " Results directory: ${RESULTS_DIR}"
echo "   per-version CSVs:    results_<rocm>.csv"
echo "   comparison matrix:   comparison_matrix.csv"
echo "   comparison report:   comparison_report.txt"
echo "==================================================================="
