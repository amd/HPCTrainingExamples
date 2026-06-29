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

module load kokkos

echo "=== Kokkos gfx942 Capability Test ==="
echo "Kokkos install: ${Kokkos_ROOT}"
echo ""

CONFIG_H="${Kokkos_ROOT}/include/KokkosCore_config.h"
if [ ! -f "${CONFIG_H}" ]; then
  echo "ERROR: ${CONFIG_H} not found"
  exit 1
fi

# Determine which Kokkos AMD GPU arch this run must verify. MI300A is an APU
# (unified memory) built with AMD_GFX942_APU; MI300X is a discrete GPU built
# with AMD_GFX942. Both are gfx942 - only the _APU suffix differs. For any
# other/unknown AMD GPU, fall back to whatever AMD gfx arch the Kokkos install
# was actually configured with, so the test generalizes instead of skipping.
if rocminfo 2>/dev/null | grep -q MI300A; then
  EXPECTED_ARCH="AMD_GFX942_APU"; GPU_KIND="MI300A (APU)"
elif rocminfo 2>/dev/null | grep -q MI300X; then
  EXPECTED_ARCH="AMD_GFX942"; GPU_KIND="MI300X (discrete)"
else
  EXPECTED_ARCH="$(grep -oE '^#define KOKKOS_ARCH_AMD_GFX[0-9A-Za-z_]+' "${CONFIG_H}" | head -1 | sed 's/^#define KOKKOS_ARCH_//')"
  GPU_KIND="other/unknown (using install arch: ${EXPECTED_ARCH:-none})"
fi

if [ -z "${EXPECTED_ARCH}" ]; then
  echo "Skip: no AMD gfx942-class arch detected for this GPU or Kokkos install"
  exit 1
fi

echo "Detected GPU: ${GPU_KIND}; expecting Kokkos arch ${EXPECTED_ARCH}"

# --- Quick pre-flight check on the config header ---
# Exact-match anchor ($) so AMD_GFX942 does not spuriously match AMD_GFX942_APU.
if grep -q "^#define KOKKOS_ARCH_${EXPECTED_ARCH}$" "${CONFIG_H}"; then
  echo "Pre-flight: KOKKOS_ARCH_${EXPECTED_ARCH} found in KokkosCore_config.h"
else
  echo "FAIL: KOKKOS_ARCH_${EXPECTED_ARCH} not found in KokkosCore_config.h"
  echo "      (GPU is ${GPU_KIND} but the Kokkos install was not built for it)"
  exit 1
fi

SCRIPT_DIR=$REPO_DIR/Kokkos/kokkos_check_apu_enabled

# Build/run in a per-invocation /tmp scratch dir to avoid NFS metadata
# contention on the shared source tree when concurrent regression tests
# race against each other (CMake 3.31 try_compile scratch files would
# otherwise vanish mid-configure and break FindOpenMP).
BUILD_DIR=$(mktemp -d -p /tmp kokkos_check_apu_enabled_XXXXXX)
trap "rm -rf ${BUILD_DIR}" EXIT

# Pin OpenMP to the HOST runtime (libomp.so). Kokkos' OpenMP backend exports
# OpenMP::OpenMP_CXX in its link interface; under the Cray CC wrapper + ROCm
# clang, CMake's FindOpenMP auto-probe mis-resolves OpenMP_CXX_LIBRARIES to the
# amdgcn DEVICE archive (lib/llvm/lib/amdgcn-amd-amdhsa/libompdevice.a). That
# GPU bitcode archive then lands on the host link and ld.lld rejects it with
# "is incompatible with elf64-x86-64". Ask the compiler for the host libomp.so
# path and feed it to FindOpenMP as explicit hints so the broken probe (and the
# device archive) are bypassed.
# Pick the compiler used to resolve the host libomp.so path: the Cray CC
# wrapper on a Cray PE (matches the CXX the build uses there), and amdclang++
# (clang++ fallback) on a stock AMD ROCm stack such as AAC6.
if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
  OMP_CXX="${CXX:-$(command -v CC)}"
else
  OMP_CXX="${CXX:-$(command -v amdclang++ || command -v clang++)}"
fi
OMP_HOST_LIB="$(${OMP_CXX} -print-file-name=libomp.so 2>/dev/null)"
OMP_HINTS=()
if [ -n "${OMP_HOST_LIB}" ] && [ -f "${OMP_HOST_LIB}" ]; then
  OMP_HINTS=(
    -DOpenMP_CXX_FLAGS="-fopenmp=libomp"
    -DOpenMP_CXX_LIB_NAMES="omp"
    -DOpenMP_omp_LIBRARY="${OMP_HOST_LIB}"
  )
  echo "Pinning host OpenMP runtime: ${OMP_HOST_LIB}"
else
  echo "WARNING: could not resolve host libomp.so via ${OMP_CXX}; relying on FindOpenMP autodetect"
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DKokkos_ROOT="${Kokkos_ROOT}" \
  -DEXPECTED_KOKKOS_ARCH="${EXPECTED_ARCH}" \
  "${OMP_HINTS[@]}"

cmake --build "${BUILD_DIR}" --verbose

if "${BUILD_DIR}/kokkos_check_apu_enabled"; then
  echo "Runtime test PASSED."
else
  echo ""
  echo "ERROR: no executable produced" 
fi
