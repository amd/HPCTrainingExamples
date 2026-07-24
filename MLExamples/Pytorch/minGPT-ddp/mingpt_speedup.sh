#!/bin/bash
# Summarize the minGPT-DDP scaling sweep from the RESULT lines the patched
# trainer.py prints (see the top-level README hands-on). Reads run_*.log (one per
# GPU count, written with `torchrun ... | tee run_<N>.log`), sorts by GPU count,
# and prints throughput, per-step time, peak memory, speedup, and weak-scaling
# efficiency.
#
#   * tokens_per_s / speedup below ideal is the DDP gradient all-reduce (RCCL) cost
#   * with PROFILE=1 the log also carries RCCL_TOTAL_MS (grepped out below)
#
# Usage:  ./mingpt_speedup.sh          # after the sweep writes run_2.log, run_4.log, ...
set -euo pipefail

tmp=$(mktemp)
shopt -s nullglob
for log in run_*.log; do
  line=$(grep '^RESULT' "$log" | tail -n1) || continue
  [ -z "$line" ] && continue
  ws=$(sed -n 's/.*world_size=\([0-9]*\).*/\1/p'      <<<"$line")
  step=$(sed -n 's/.*step_s=\([0-9.]*\).*/\1/p'       <<<"$line")
  tok=$(sed -n 's/.*tokens_per_s=\([0-9.]*\).*/\1/p'  <<<"$line")
  mem=$(sed -n 's/.*peak_mem_mb=\([0-9.]*\).*/\1/p'   <<<"$line")
  rccl=$(grep -h 'RCCL_TOTAL_MS' "$log" | tail -n1 | sed -n 's/.*RCCL_TOTAL_MS \([0-9.]*\).*/\1/p')
  [ -z "$ws" ] && continue
  echo "$ws $step $tok $mem ${rccl:--}"
done | sort -n > "$tmp"

if [ ! -s "$tmp" ]; then
  echo "No RESULT lines found in run_*.log (run the sweep first)." >&2
  rm -f "$tmp"; exit 1
fi

printf '%-6s %-12s %-14s %-14s %-10s %-8s %-12s\n' \
  "GPUs" "step_s" "tok_per_s" "peak_mem_MB" "speedup" "eff" "rccl_ms"
base=""; g0=""
while read -r ws step tok mem rccl; do
  [ -z "$g0" ]   && g0=$ws
  [ -z "$base" ] && base=$tok
  speedup=$(awk -v a="$tok" -v b="$base" 'BEGIN{printf "%.2f", a/b}')
  eff=$(awk -v s="$speedup" -v n="$ws" -v g="$g0" 'BEGIN{printf "%.0f%%", 100*s/(n/g)}')
  printf '%-6s %-12s %-14s %-14s %-10s %-8s %-12s\n' "$ws" "$step" "$tok" "$mem" "$speedup" "$eff" "$rccl"
done < "$tmp"
rm -f "$tmp"

echo
echo "Efficiency is relative to the smallest GPU count present (${g0})."
echo "The gap from ideal (and, with PROFILE=1, the rising rccl_ms) is the DDP"
echo "gradient all-reduce (RCCL) cost. Bigger models make it a larger share."
