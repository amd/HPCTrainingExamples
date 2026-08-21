#!/bin/bash
# Measure the roofline CEILINGS with likwid-bench (timer-based, needs no PMU).
# Single-core, DP.  Pairs with the native likwid-perfctr app point (5.5.2+).
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load rocm/7.2.4 2>/dev/null
module load likwid 2>/dev/null
cd "$(dirname "$0")"
OUT=roofline_meas ; mkdir -p "$OUT"
RES="$OUT/likwid_bench.txt" ; : > "$RES"

flops() { # kernel -> MFlop/s (single thread, small L1-resident working set)
  likwid-bench -t "$1" -w S0:24kB:1 2>/dev/null | awk '/MFlops\/s:/{print $NF}';
}
bw() {    # kernel size -> MByte/s (single thread)
  likwid-bench -t "$1" -w "S0:$2:1" 2>/dev/null | awk '/MByte\/s:/{print $NF}';
}

echo "# compute peaks (MFlop/s, 1 core)" | tee -a "$RES"
for k in peakflops peakflops_sse_fma peakflops_avx_fma peakflops_avx512_fma; do
  echo "flops $k $(flops $k)" | tee -a "$RES"
done

echo "# load bandwidth by working set (MByte/s, 1 core)" | tee -a "$RES"
#   16kB < L1(32kB) ; 512kB < L2(1MB) ; 16MB < L3(32MB) ; 1GB >> caches (DRAM)
for w in 16kB 512kB 16MB 1GB; do
  echo "bw_load load_avx512 $w $(bw load_avx512 $w)" | tee -a "$RES"
done

echo "# triad (stream) DRAM (MByte/s + MFlop/s, 1 core)" | tee -a "$RES"
likwid-bench -t triad_avx512_fma -w S0:1GB:1 2>/dev/null | awk '/MByte\/s:|MFlops\/s:/{print "triad",$1,$NF}' | tee -a "$RES"
echo "DONE"
