#!/bin/bash
# torchrun --no-python wrapper: run ONLY local rank 0's Python under
# rocprof-sys-run so the host+GPU timeline is captured for one rank while the
# other ranks run bare (so the FSDP2 collectives still complete). Give rank 0 a
# dedicated ROCPROFSYS_OUTPUT_PATH.
#
# Usage (under torchrun):
#   torchrun --standalone --nproc_per_node=N --no-python \
#     ./rocsys_wrap.sh <output_path> fsdp2_bench.py [args...]
set -u
OUT="$1"; shift
if [ "${LOCAL_RANK:-0}" = "0" ]; then
  export ROCPROFSYS_OUTPUT_PATH="$OUT"
  exec rocprof-sys-run -- python -u "$@"
else
  exec python -u "$@"
fi
