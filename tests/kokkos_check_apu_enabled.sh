#!/bin/bash

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
   echo "rocm module is not loaded"
   echo "loading default rocm module"
   module load rocm
fi
module load kokkos
module list

echo "=== Kokkos APU Capability Test ==="
echo "Kokkos install: ${Kokkos_DIR}"
echo ""

# --- Quick pre-flight check on the config header ---
CONFIG_H="${Kokkos_DIR}/include/KokkosCore_config.h"
if [ ! -f "${CONFIG_H}" ]; then
  echo "ERROR: ${CONFIG_H} not found"
  exit 1
fi

if grep -q "^#define KOKKOS_ARCH_AMD_GFX942_APU" "${CONFIG_H}"; then
  echo "Pre-flight: KOKKOS_ARCH_AMD_GFX942_APU found in KokkosCore_config.h"
else
  echo "FAIL: KOKKOS_ARCH_AMD_GFX942_APU not found in KokkosCore_config.h"
  exit 1
fi

SCRIPT_DIR=$REPO_DIR/Kokkos/kokkos_check_apu_enabled
BUILD_DIR=${SCRIPT_DIR}/kokkos_check_apu_enabled_build
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DKokkos_ROOT="${Kokkos_DIR}" \
  -DCMAKE_CXX_COMPILER=hipcc

cmake --build "${BUILD_DIR}" --verbose

if "${BUILD_DIR}/kokkos_check_apu_enabled"; then
  echo "Runtime test PASSED."
else
  echo ""
  echo "ERROR: no executable produced" 
fi

rm -rf ${BUILD_DIR}
