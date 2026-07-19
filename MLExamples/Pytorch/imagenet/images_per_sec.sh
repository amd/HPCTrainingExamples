#!/bin/bash
# img/s = global batch / avg per-step Time, parsed from main.py --dummy logs.
# The GPU count N is taken from the log name run_<N>.log, and the global batch is
# N * PERGPU_BATCH (weak scaling: constant per-GPU batch). PERGPU_BATCH defaults
# to 128 (the SPX sweep); CPX sweeps export a smaller value.
# Speedup is relative to the smallest GPU count present (the baseline, run_1.log).
# Run after the sweep (which writes run_1.log, run_2.log, ...).
PERGPU_BATCH="${PERGPU_BATCH:-128}"
base=""
for log in $(ls run_*.log 2>/dev/null | sort -t_ -k2 -n); do
  n="${log#run_}"; n="${n%.log}"
  case "$n" in *[!0-9]*|"") continue ;; esac   # skip run_copy.log / run_migrate.log
  b=$(( n * PERGPU_BATCH ))
  read -r imgps step mem < <(awk -v b="$b" '
    /^Epoch:/     { gsub(/[()]/," "); for (i=1;i<=NF;i++) if ($i=="Time") a=$(i+2) }
    /^PEAK_MEM_MB/ { m=$2 }
    END { if (a>0) printf "%.4f %.4f %s\n", b/a, a, m }
  ' "$log")
  [ -z "$imgps" ] && continue
  [ -z "$base" ] && base="$imgps"
  awk -v f="$log" -v i="$imgps" -v s="$step" -v b="$b" -v m="$mem" -v base="$base" -v g="$n" \
    'BEGIN { printf "%-10s gpus=%-2d img/s=%.0f  step=%.4fs  batch=%d  peak_mem_mb=%s  speedup=%.2fx\n", f, g, i, s, b, m, i/base }'
done
