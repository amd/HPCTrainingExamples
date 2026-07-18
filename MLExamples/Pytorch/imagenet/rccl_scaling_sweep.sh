#!/bin/bash
# Weak/strong scaling sweep of the PyTorch imagenet DDP example with --dummy data.
#
# With --dummy the input pipeline is essentially free (FakeData is generated on
# the fly), so per-step time is dominated by GPU compute plus the RCCL all-reduce
# of gradients. Comparing step time across GPU counts therefore exposes the RCCL
# communication overhead directly:
#
#   * weak scaling  (default): per-GPU batch is held constant (global batch grows
#     with GPU count). Ideal interconnect => flat step time; any growth is RCCL.
#   * strong scaling (MODE=strong): global batch is held constant. Speedup < N is
#     the combined effect of RCCL cost and shrinking per-GPU work.
#
# Usage:
#   ./rccl_scaling_sweep.sh                # weak scaling, resnet50, GPU counts 1 2 4 8
#   MODE=strong ./rccl_scaling_sweep.sh
#   ARCH=resnet101 GPUS="1 2 4" PERGPU_BATCH=128 ./rccl_scaling_sweep.sh
#
# Environment variables:
#   MAIN            path to pytorch imagenet main.py (auto-detected if unset)
#   ARCH            model architecture (default resnet50)
#   GPUS            space-separated GPU counts to test (default "1 2 4 8")
#   PERGPU_BATCH    per-GPU mini-batch for weak scaling (default 256)
#   GLOBAL_BATCH    global mini-batch for strong scaling (default 256)
#   MODE            weak|strong (default weak)
#   WARMUP_STEPS    steps to skip before reading steady-state time (default 20)
#   PORT            rendezvous port (default 23456)
#   NCCL_DEBUG      set to INFO to dump the RCCL topology/rings (default WARN)
set -euo pipefail

ARCH=${ARCH:-resnet50}
GPUS=${GPUS:-"1 2 4 8"}
PERGPU_BATCH=${PERGPU_BATCH:-256}
GLOBAL_BATCH=${GLOBAL_BATCH:-256}
MODE=${MODE:-weak}
WARMUP_STEPS=${WARMUP_STEPS:-20}
PORT=${PORT:-23456}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# The PPAC MI300A nodes are booted with "iommu=pt" (REQUIRED), so RCCL uses
# direct xGMI GPU-GPU P2P -- we set no P2P override here. Verify passthrough is
# present with: grep -o 'iommu=pt' /proc/cmdline. Without iommu=pt, RCCL P2P DMA
# can hang (and even destabilize the node); the fallback is host-staged transport
# via NCCL_P2P_DISABLE=1 (slower, but clears the hang until the node is fixed).
[[ -n "${NCCL_P2P_DISABLE:-}" ]] && export NCCL_P2P_DISABLE
# MIOpen's default solver search can take >10 min cold for ResNet convs on MI300;
# FAST (immediate) mode compiles in seconds. Keep a persistent cache so repeated
# runs in the sweep don't recompile (the pytorch module defaults to a per-job dir).
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
# Hard-override any per-job MIOpen cache the site module may set, so the sweep's
# runs share one warm cache instead of recompiling each time. Point MIOPEN_DB at
# a different dir to relocate it.
export MIOPEN_USER_DB_PATH="${MIOPEN_DB:-/tmp/$USER/miopen-shared}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Locate the upstream imagenet main.py (this example intentionally does not
# duplicate it). Override with MAIN=/path/to/main.py if needed.
if [[ -z "${MAIN:-}" ]]; then
  for cand in \
    "$HOME/pytorch_examples/imagenet/main.py" \
    "$HOME/examples/imagenet/main.py" \
    "./pytorch_examples/imagenet/main.py"; do
    [[ -f "$cand" ]] && MAIN="$cand" && break
  done
fi
if [[ -z "${MAIN:-}" || ! -f "$MAIN" ]]; then
  echo "ERROR: could not find imagenet main.py. Set MAIN=/path/to/main.py" >&2
  echo "  git clone --depth=1 https://github.com/pytorch/examples.git pytorch_examples" >&2
  exit 1
fi

LOGDIR=${LOGDIR:-rccl_scaling_logs}
mkdir -p "$LOGDIR"

echo "RCCL scaling sweep: arch=$ARCH mode=$MODE gpus=[$GPUS] main=$MAIN"
printf '%-6s %-14s %-12s %-12s %-10s\n' "GPUs" "step_time_s" "img_per_s" "speedup" "efficiency"

base_thru=""
for n in $GPUS; do
  if [[ "$MODE" == "strong" ]]; then
    batch=$GLOBAL_BATCH
  else
    batch=$(( PERGPU_BATCH * n ))
  fi

  # Restrict the number of visible GPUs; main.py uses device_count() for ranks.
  devs=$(seq -s, 0 $(( n - 1 )))
  log="$LOGDIR/${ARCH}_${MODE}_${n}gpu.log"

  # One epoch of FakeData is long; we only need steady-state step time, so run
  # and stop once we have enough progress lines past the warm-up window.
  # setsid so the whole spawn group can be signalled together.
  HIP_VISIBLE_DEVICES=$devs ROCR_VISIBLE_DEVICES=$devs \
    setsid stdbuf -oL python "$MAIN" -a "$ARCH" --dummy \
      --dist-url "tcp://127.0.0.1:${PORT}" --dist-backend nccl \
      --multiprocessing-distributed --world-size 1 --rank 0 \
      -b "$batch" -p 5 --epochs 1 > "$log" 2>&1 &
  pid=$!

  # Wait until we have collected steady-state samples, then stop the run.
  need=$(( WARMUP_STEPS + 40 ))
  for _ in $(seq 1 120); do
    lines=$(grep -c 'Epoch: \[0\]' "$log" 2>/dev/null || true)
    lines=${lines:-0}
    [[ "$lines" -ge "$need" ]] && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  kill -TERM -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true

  # Parse "Time  x.xxx (y.yyy)" -> steady-state avg is the last (avg) value.
  step_time=$(grep 'Epoch: \[0\]' "$log" \
    | sed -n 's/.*Time[[:space:]]\+[0-9.]\+[[:space:]]\+(\([0-9.]\+\)).*/\1/p' \
    | tail -n1)
  if [[ -z "$step_time" ]]; then
    printf '%-6s %-14s %-12s %-12s %-10s\n' "$n" "n/a" "n/a" "n/a" "(see $log)"
    continue
  fi

  img_per_s=$(awk -v b="$batch" -v t="$step_time" 'BEGIN{printf "%.1f", b/t}')
  if [[ -z "$base_thru" ]]; then base_thru=$img_per_s; fi
  speedup=$(awk -v a="$img_per_s" -v b="$base_thru" 'BEGIN{printf "%.2f", a/b}')
  eff=$(awk -v s="$speedup" -v n="$n" -v g="${GPUS%% *}" \
        'BEGIN{printf "%.0f%%", 100*s/(n/g)}')
  printf '%-6s %-14s %-12s %-12s %-10s\n' "$n" "$step_time" "$img_per_s" "$speedup" "$eff"
done

echo
echo "Per-run logs (with NCCL_DEBUG output if enabled) are in: $LOGDIR/"
echo "For weak scaling, ideal img/s grows linearly with GPUs; the gap is RCCL cost."
