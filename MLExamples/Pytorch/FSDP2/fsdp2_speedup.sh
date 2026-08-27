#!/bin/bash
# Summarize the FSDP2 scaling sweep from the RESULT lines the patched example.py
# prints (see the top-level README hands-on). Reads run_*.log (one per GPU count,
# written with `torchrun ... | tee run_<N>.log`), sorts by GPU count, and prints
# throughput, per-step time, peak memory, speedup, and weak-scaling efficiency.
#
# The two things to read together:
#   * peak_mem_MB should DROP as GPUs increase (parameter sharding, the FSDP2 win)
#   * speedup/eff below ideal is the all-gather + reduce-scatter (RCCL) cost
#
# Usage:  ./fsdp2_speedup.sh            # after the sweep writes run_2.log, run_4.log, ...
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
  [ -z "$ws" ] && continue
  echo "$ws $step $tok $mem"
done | sort -n > "$tmp"

if [ ! -s "$tmp" ]; then
  echo "No RESULT lines found in run_*.log (run the sweep first)." >&2
  rm -f "$tmp"; exit 1
fi

printf '%-6s %-12s %-14s %-14s %-10s %-8s\n' \
  "GPUs" "step_s" "tok_per_s" "peak_mem_MB" "speedup" "eff"
base=""; g0=""
while read -r ws step tok mem; do
  [ -z "$g0" ]   && g0=$ws
  [ -z "$base" ] && base=$tok
  speedup=$(awk -v a="$tok" -v b="$base" 'BEGIN{printf "%.2f", a/b}')
  eff=$(awk -v s="$speedup" -v n="$ws" -v g="$g0" 'BEGIN{printf "%.0f%%", 100*s/(n/g)}')
  printf '%-6s %-12s %-14s %-14s %-10s %-8s\n' "$ws" "$step" "$tok" "$mem" "$speedup" "$eff"
done < "$tmp"
rm -f "$tmp"

echo
echo "Efficiency is relative to the smallest GPU count present (${g0})."
echo "Peak memory should fall as GPUs increase (parameter sharding); the throughput"
echo "gap from ideal is the FSDP2 all-gather/reduce-scatter (RCCL) cost."
