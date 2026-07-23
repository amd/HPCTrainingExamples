#!/bin/bash

# rccl-tests install check: run an in-node all-reduce across all GPUs and
# require data correctness plus a minimum bus bandwidth. GPU-only, so it SKIPs
# when fewer than 2 GPUs are present (no CPU fallback). Override the (generous)
# floor with RCCL_MIN_BUSBW_GBS (default 50).

module -t list 2>&1 | grep -q "^rocm" || module load rocm

if ! module load rccl-tests 2>/tmp/rccl_tests_check.$$.err; then
   cat /tmp/rccl_tests_check.$$.err
   rm -f /tmp/rccl_tests_check.$$.err
   echo "Unable to locate a modulefile for 'rccl-tests'"
   exit 0
fi
rm -f /tmp/rccl_tests_check.$$.err

echo "=== rccl-tests install check ==="
ALL_REDUCE="${RCCL_TESTS_PATH}/bin/all_reduce_perf"
echo "all_reduce_perf: ${ALL_REDUCE}"
[ -x "${ALL_REDUCE}" ] || { echo "FAIL: all_reduce_perf not found"; exit 1; }

GPU_COUNT=$(rocminfo 2>/dev/null | grep -c "Device Type:             GPU")
echo "GPU count: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 2 ]; then
   echo "Skip: rccl-tests needs at least 2 GPUs (found ${GPU_COUNT})"
   exit 0
fi

echo "+ ${ALL_REDUCE} -b 8M -e 512M -f 2 -g ${GPU_COUNT} -c 1 -n 50"
OUT=$(timeout 300 "${ALL_REDUCE}" -b 8M -e 512M -f 2 -g "${GPU_COUNT}" -c 1 -n 50) || { echo "${OUT}"; echo "FAIL: all_reduce_perf run failed"; exit 1; }
echo "${OUT}"
echo "${OUT}" | grep -q "Out of bounds values : 0 OK" || { echo "FAIL: data validation failed"; exit 1; }

BUSBW=$(echo "${OUT}" | grep "Avg bus bandwidth" | grep -oE "[0-9]+(\.[0-9]+)?" | tail -1)
THRESH=${RCCL_MIN_BUSBW_GBS:-50}
awk -v b="${BUSBW}" -v t="${THRESH}" 'BEGIN{ exit !(b+0 >= t+0) }' || { echo "FAIL: avg bus bandwidth ${BUSBW} GB/s < ${THRESH} GB/s"; exit 1; }

echo "Avg bus bandwidth: ${BUSBW} GB/s"
echo "RCCL-Tests Install Check: SUCCESS"
