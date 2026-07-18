#!/bin/bash
# Scaling sweep of the minGPT DDP benchmark, reporting RCCL all-reduce cost.
#
# For each GPU count it launches ddp_gpt_bench.py with one process per GPU via
# torchrun, then parses the RESULT line. Because the benchmark measures step time
# both with and without the gradient all-reduce (DDP no_sync), the reported
# comm_pct is the direct RCCL communication overhead; throughput scaling and
# efficiency show how that overhead affects end-to-end speed.
#
# Usage:
#   ./rccl_scaling_sweep.sh                       # GPU counts 1 2 4 8, gpt2-size model
#   GPUS="2 4 8" N_LAYER=24 N_EMBD=1024 ./rccl_scaling_sweep.sh
#
# Environment:
#   GPUS        space-separated GPU counts (default "1 2 4 8")
#   N_LAYER/N_HEAD/N_EMBD/BLOCK/BATCH   model + per-GPU batch (defaults 12/12/768/512/8)
#   BENCH       path to ddp_gpt_bench.py (default alongside this script)
#   NCCL_DEBUG  set INFO to log RCCL rings/trees (default WARN)
#   AFFINITY=1  bind each rank to its GPU's local NUMA node (via numactl); effect
#               is within noise here (see ../common/PERFORMANCE_NOTES.md)
set -euo pipefail

GPUS=${GPUS:-"1 2 4 8"}
N_LAYER=${N_LAYER:-12}
N_HEAD=${N_HEAD:-12}
N_EMBD=${N_EMBD:-768}
BLOCK=${BLOCK:-512}
BATCH=${BATCH:-8}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# The PPAC MI300A nodes are booted with "iommu=pt" (REQUIRED), so RCCL uses
# direct xGMI GPU-GPU P2P -- we set no P2P override here. Verify passthrough is
# present with: grep -o 'iommu=pt' /proc/cmdline. If you ever run on a node
# WITHOUT iommu=pt, RCCL P2P DMA can hang; the fallback is host-staged transport
# via NCCL_P2P_DISABLE=1 (slower, but clears the hang until the node is fixed).
[[ -n "${NCCL_P2P_DISABLE:-}" ]] && export NCCL_P2P_DISABLE

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH=${BENCH:-"$here/ddp_gpt_bench.py"}
[[ -f "$BENCH" ]] || { echo "ERROR: benchmark not found: $BENCH" >&2; exit 1; }

# Optional NUMA affinity: prepend the launcher so each rank binds to its GPU's
# local node. Empty by default (no behavior change).
AFFIN=""
if [[ "${AFFINITY:-0}" == "1" ]]; then
  LAUNCHER="$here/../common/affinity_launcher.py"
  if [[ -f "$LAUNCHER" ]]; then AFFIN="$LAUNCHER"; echo "AFFINITY=1: binding ranks to local NUMA nodes";
  else echo "WARN: $LAUNCHER not found; AFFINITY ignored" >&2; fi
fi

LOGDIR=${LOGDIR:-rccl_scaling_logs}
mkdir -p "$LOGDIR"

echo "minGPT DDP scaling: layers=$N_LAYER embd=$N_EMBD block=$BLOCK per_gpu_batch=$BATCH opts='${OPTS:-none}' gpus=[$GPUS]"
printf '%-6s %-12s %-12s %-10s %-14s %-10s %-10s\n' \
  "GPUs" "step_s" "nosync_s" "comm%" "tok_per_s" "speedup" "eff"

base=""
for n in $GPUS; do
  log="$LOGDIR/mingpt_${n}gpu.log"
  torchrun --standalone --nproc_per_node="$n" $AFFIN "$BENCH" \
    --n-layer "$N_LAYER" --n-head "$N_HEAD" --n-embd "$N_EMBD" \
    --block-size "$BLOCK" --batch-size "$BATCH" ${OPTS:-} > "$log" 2>&1 || {
      printf '%-6s %s\n' "$n" "FAILED (see $log)"; continue; }

  line=$(grep '^RESULT' "$log" | tail -n1)
  step=$(sed -n 's/.*step_sync_s=\([0-9.]*\).*/\1/p' <<<"$line")
  nosync=$(sed -n 's/.*step_nosync_s=\([0-9.]*\).*/\1/p' <<<"$line")
  commp=$(sed -n 's/.*comm_pct=\([0-9.]*\).*/\1/p' <<<"$line")
  tok=$(sed -n 's/.*tokens_per_s=\([0-9.]*\).*/\1/p' <<<"$line")
  if [[ -z "$tok" ]]; then printf '%-6s %s\n' "$n" "no RESULT (see $log)"; continue; fi

  [[ -z "$base" ]] && base=$tok
  speedup=$(awk -v a="$tok" -v b="$base" 'BEGIN{printf "%.2f", a/b}')
  g0=${GPUS%% *}
  eff=$(awk -v s="$speedup" -v n="$n" -v g="$g0" 'BEGIN{printf "%.0f%%", 100*s/(n/g)}')
  printf '%-6s %-12s %-12s %-10s %-14s %-10s %-10s\n' \
    "$n" "$step" "$nosync" "$commp" "$tok" "$speedup" "$eff"
done

echo
echo "comm% is the RCCL gradient all-reduce share of each step (via DDP no_sync)."
echo "Per-run logs (RCCL topology if NCCL_DEBUG=INFO): $LOGDIR/"
