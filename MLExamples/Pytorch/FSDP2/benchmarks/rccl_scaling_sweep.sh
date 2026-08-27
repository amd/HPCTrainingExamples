#!/bin/bash
# Scaling sweep of the FSDP2 benchmark: throughput and per-GPU peak memory.
#
# For each GPU count it launches fsdp2_bench.py with one process per GPU via
# torchrun and parses the RESULT line. FSDP2 shards parameters across ranks, so
# watch two trends together:
#   * tokens_per_s / efficiency  -> how well the all-gather + reduce-scatter
#     collectives scale (the RCCL cost),
#   * peak_mem_mb                 -> should DROP as ranks increase (sharding).
#
# FSDP2 needs at least 2 GPUs, so the sweep starts at 2 by default.
#
# Usage:
#   ./rccl_scaling_sweep.sh                          # GPU counts 2 4 8
#   GPUS="2 4 8" N_LAYERS=24 DIM=2048 ./rccl_scaling_sweep.sh
#   MIXED_PRECISION=1 ./rccl_scaling_sweep.sh
#   OPTS="--compile" ./rccl_scaling_sweep.sh          # torch.compile the model
#
# Environment:
#   GPUS         space-separated GPU counts (default "2 4 8")
#   N_LAYERS/N_HEADS/DIM/SEQ/BATCH   model + per-GPU batch (defaults 16/16/1024/512/8)
#   MIXED_PRECISION   set to 1 to pass --mixed-precision
#   OPTS         extra bench flags passed through (e.g. --compile)
#   BENCH        path to fsdp2_bench.py (default alongside this script)
#   NCCL_DEBUG   set INFO to log RCCL rings/trees (default WARN)
#   AFFINITY=1   bind each rank to its GPU's local NUMA node (via numactl); effect
#                is within noise here (see ../../common/PERFORMANCE_NOTES.md)
set -euo pipefail

GPUS=${GPUS:-"2 4 8"}
N_LAYERS=${N_LAYERS:-16}
N_HEADS=${N_HEADS:-16}
DIM=${DIM:-1024}
SEQ=${SEQ:-512}
BATCH=${BATCH:-8}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# The PPAC MI300A nodes are booted with "iommu=pt" (REQUIRED), so RCCL uses
# direct xGMI GPU-GPU P2P -- we set no P2P override here. Verify passthrough is
# present with: grep -o 'iommu=pt' /proc/cmdline. If you ever run on a node
# WITHOUT iommu=pt, RCCL P2P DMA can hang; the fallback is host-staged transport
# via NCCL_P2P_DISABLE=1 (slower, but clears the hang until the node is fixed).
[[ -n "${NCCL_P2P_DISABLE:-}" ]] && export NCCL_P2P_DISABLE

mp_flag=""
[[ "${MIXED_PRECISION:-0}" == "1" ]] && mp_flag="--mixed-precision"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH=${BENCH:-"$here/fsdp2_bench.py"}
[[ -f "$BENCH" ]] || { echo "ERROR: benchmark not found: $BENCH" >&2; exit 1; }

# Optional NUMA affinity: prepend the launcher so each rank binds to its GPU's
# local node. Empty by default (no behavior change).
AFFIN=""
if [[ "${AFFINITY:-0}" == "1" ]]; then
  LAUNCHER="$here/../../common/affinity_launcher.py"
  if [[ -f "$LAUNCHER" ]]; then AFFIN="$LAUNCHER"; echo "AFFINITY=1: binding ranks to local NUMA nodes";
  else echo "WARN: $LAUNCHER not found; AFFINITY ignored" >&2; fi
fi

LOGDIR=${LOGDIR:-rccl_scaling_logs}
mkdir -p "$LOGDIR"

echo "FSDP2 scaling: layers=$N_LAYERS dim=$DIM seq=$SEQ per_gpu_batch=$BATCH gpus=[$GPUS] $mp_flag"
printf '%-6s %-12s %-14s %-14s %-10s %-10s\n' \
  "GPUs" "step_s" "tok_per_s" "peak_mem_MB" "speedup" "eff"

base=""; g0=${GPUS%% *}
for n in $GPUS; do
  log="$LOGDIR/fsdp2_${n}gpu.log"
  torchrun --standalone --nproc_per_node="$n" $AFFIN "$BENCH" \
    --n-layers "$N_LAYERS" --n-heads "$N_HEADS" --dim "$DIM" \
    --seq-len "$SEQ" --batch-size "$BATCH" $mp_flag ${OPTS:-} > "$log" 2>&1 || {
      printf '%-6s %s\n' "$n" "FAILED (see $log)"; continue; }

  line=$(grep '^RESULT' "$log" | tail -n1)
  step=$(sed -n 's/.*step_s=\([0-9.]*\).*/\1/p' <<<"$line")
  tok=$(sed -n 's/.*tokens_per_s=\([0-9.]*\).*/\1/p' <<<"$line")
  mem=$(sed -n 's/.*peak_mem_mb=\([0-9.]*\).*/\1/p' <<<"$line")
  if [[ -z "$tok" ]]; then printf '%-6s %s\n' "$n" "no RESULT (see $log)"; continue; fi

  [[ -z "$base" ]] && base=$tok
  speedup=$(awk -v a="$tok" -v b="$base" 'BEGIN{printf "%.2f", a/b}')
  eff=$(awk -v s="$speedup" -v n="$n" -v g="$g0" 'BEGIN{printf "%.0f%%", 100*s/(n/g)}')
  printf '%-6s %-12s %-14s %-14s %-10s %-10s\n' \
    "$n" "$step" "$tok" "$mem" "$speedup" "$eff"
done

echo
echo "Efficiency here is relative to the smallest GPU count in the sweep (${g0})."
echo "Peak memory should fall as GPUs increase (parameter sharding); the"
echo "throughput gap from ideal is the FSDP2 all-gather/reduce-scatter (RCCL) cost."
echo "Per-run logs (RCCL topology if NCCL_DEBUG=INFO): $LOGDIR/"
