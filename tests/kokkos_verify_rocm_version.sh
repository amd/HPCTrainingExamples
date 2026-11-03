#!/bin/bash

# Verify the loaded Kokkos module was built against the loaded ROCm:
# `module load rocm/<v>` + `module load kokkos` must give a Kokkos compiled
# with rocm-<v>. An already-loaded rocm/kokkos is reused; a module is loaded
# only if none is loaded yet.

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded, loading default rocm module"
  module load rocm
fi

module -t list 2>&1 | grep -q "^kokkos"
if [ $? -eq 1 ]; then
  echo "kokkos module is not loaded, loading kokkos module"
  module load kokkos
fi

if [ -z "${ROCM_PATH}" ]; then
  echo "ROCM_PATH is not set; is a rocm module loaded?"
  echo "Failure"
  exit 1
fi

KOKKOS_CMAKE_DIR="${Kokkos_DIR:-${Kokkos_ROOT}/lib64/cmake/Kokkos}"
if [ ! -d "${KOKKOS_CMAKE_DIR}" ]; then
  echo "Could not find the Kokkos CMake dir; is a kokkos module loaded?"
  echo "Failure"
  exit 1
fi

# kokkos_setup.sh builds with CMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/amdclang++,
# so the compiler path baked into the installed CMake files reveals the build-time ROCm.
BUILT_ROCM=$(grep -rhoE "[^\";> ]*/llvm/bin/amdclang[^\";> ]*" "${KOKKOS_CMAKE_DIR}"/*.cmake 2>/dev/null \
             | head -1 | sed 's#/llvm/bin/amdclang.*##')

# Compare by ROCm version
LOADED_VER=$(basename "$(realpath -m "${ROCM_PATH}")"); LOADED_VER=${LOADED_VER#rocm-}
BUILT_VER=$(basename "${BUILT_ROCM}"); BUILT_VER=${BUILT_VER#rocm-}

echo "Loaded ROCm:               ${ROCM_PATH}  (version ${LOADED_VER})"
echo "Kokkos built against ROCm: ${BUILT_ROCM:-<unknown>}  (version ${BUILT_VER:-?})"

if [ -n "${BUILT_ROCM}" ] && [ "${BUILT_VER}" = "${LOADED_VER}" ]; then
  echo "Success"
else
  echo "Failure"
  exit 1
fi
