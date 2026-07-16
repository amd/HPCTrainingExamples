#!/bin/bash

# rocSHMEM install check: artifacts present, then run rocshmem_info on a GPU.
# rocSHMEM is a GPU library, so it SKIPs when no GPU is present (no CPU path).

module -t list 2>&1 | grep -q "^rocm" || module load rocm

if ! module load rocshmem 2>/dev/null; then
   echo "Unable to locate a modulefile for 'rocshmem'"
   exit 0
fi

[ -f "${ROCSHMEM_PATH}/lib/librocshmem.a" ] || { echo "FAIL: librocshmem.a missing"; exit 1; }
[ -d "${ROCSHMEM_PATH}/include/rocshmem" ] || { echo "FAIL: include/rocshmem missing"; exit 1; }
command -v rocshmem_info >/dev/null || { echo "FAIL: rocshmem_info not found"; exit 1; }

if [ "$(rocminfo 2>/dev/null | grep -c 'Device Type:             GPU')" -lt 1 ]; then
   echo "Skip: rocSHMEM needs a GPU to run rocshmem_info"
   exit 0
fi

timeout 120 rocshmem_info || { echo "FAIL: rocshmem_info failed"; exit 1; }

echo "rocSHMEM Install Check: SUCCESS"
