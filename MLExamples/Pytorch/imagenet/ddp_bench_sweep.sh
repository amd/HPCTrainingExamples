#!/bin/bash
# torchrun-based DDP ResNet scaling sweep (robust replacement for driving the
# upstream main.py --dummy on nodes where its mp.spawn path hangs).
#
# It applies the environment fixes needed on MI300A (RCCL P2P + MIOpen cache),
# pre-warms the MIOpen kernel cache once (single process, no lock contention),
# then runs ddp_resnet_bench.py at each GPU count and reports throughput,
# speedup, weak-scaling efficiency, and the RCCL all-reduce share (comm%).
#
# Usage:
#   ./ddp_bench_sweep.sh                         # resnet50, GPUs 1 2 4, per-GPU batch 128
#   GPUS="2 4" OPTS="--channels-last --amp" ./ddp_bench_sweep.sh
#   ARCH=resnet101 BATCH=256 ./ddp_bench_sweep.sh
#
# Environment:
#   GPUS   GPU counts (default "1 2 4")     ARCH   torchvision model (default resnet50)
#   BATCH  per-GPU batch (default 128)       OPTS   extra bench flags (e.g. --channels-last --amp)
#   NCCL_P2P_DISABLE (unset => xGMI P2P; set 1 only as a fallback w/o iommu=pt),
#   MIOPEN_FIND_MODE (default FAST), MIOPEN_DB (cache dir)
set -euo pipefail

GPUS=${GPUS:-"1 2 4"}
ARCH=${ARCH:-resnet50}
BATCH=${BATCH:-128}
OPTS=${OPTS:-}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# PPAC MI300A nodes are booted with "iommu=pt" (REQUIRED), so RCCL uses direct
# xGMI P2P -- we set no P2P override. On a node without iommu=pt, RCCL P2P DMA
# can hang; the fallback is host-staged transport via NCCL_P2P_DISABLE=1.
[[ -n "${NCCL_P2P_DISABLE:-}" ]] && export NCCL_P2P_DISABLE
# Fast MIOpen kernel selection + a persistent, shared cache (override any per-job
# path the site module sets, so runs reuse one warm cache).
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export MIOPEN_USER_DB_PATH="${MIOPEN_DB:-/tmp/$USER/miopen-shared}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$here/ddp_resnet_bench.py"
WARM="$here/warm_miopen.py"
LOGDIR=${LOGDIR:-ddp_bench_logs}
mkdir -p "$LOGDIR"

# Pre-warm MIOpen once for this arch/batch (single process => no cache lock
# contention between ranks, which is what makes the first multi-rank run crawl).
if [[ -f "$WARM" ]]; then
  echo "Pre-warming MIOpen cache ($ARCH, batch $BATCH)..."
  python3 "$WARM" "$BATCH" "$ARCH" > "$LOGDIR/warm.log" 2>&1 || true
fi

echo "DDP $ARCH scaling: per_gpu_batch=$BATCH opts='${OPTS:-none}' gpus=[$GPUS]"
printf '%-6s %-12s %-12s %-10s %-14s %-10s %-10s\n' \
  "GPUs" "step_s" "nosync_s" "comm%" "img_per_s" "speedup" "eff"

base=""; g0=${GPUS%% *}
for n in $GPUS; do
  log="$LOGDIR/${ARCH}_${n}gpu.log"
  torchrun --standalone --nproc_per_node="$n" "$BENCH" -a "$ARCH" -b "$BATCH" $OPTS \
    > "$log" 2>&1 || { printf '%-6s %s\n' "$n" "FAILED (see $log)"; continue; }

  line=$(grep '^RESULT' "$log" | tail -n1)
  step=$(sed -n 's/.*step_sync_s=\([0-9.]*\).*/\1/p' <<<"$line")
  nosync=$(sed -n 's/.*step_nosync_s=\([0-9.]*\).*/\1/p' <<<"$line")
  commp=$(sed -n 's/.*comm_pct=\([0-9.]*\).*/\1/p' <<<"$line")
  img=$(sed -n 's/.*img_per_s=\([0-9.]*\).*/\1/p' <<<"$line")
  if [[ -z "$img" ]]; then printf '%-6s %s\n' "$n" "no RESULT (see $log)"; continue; fi

  [[ -z "$base" ]] && base=$img
  speedup=$(awk -v a="$img" -v b="$base" 'BEGIN{printf "%.2f", a/b}')
  eff=$(awk -v s="$speedup" -v n="$n" -v g="$g0" 'BEGIN{printf "%.0f%%", 100*s/(n/g)}')
  printf '%-6s %-12s %-12s %-10s %-14s %-10s %-10s\n' \
    "$n" "$step" "$nosync" "$commp" "$img" "$speedup" "$eff"
done

echo
echo "comm% is the RCCL gradient all-reduce share per step (via DDP no_sync)."
echo "Efficiency is weak-scaling relative to ${g0} GPU(s). Logs: $LOGDIR/"
