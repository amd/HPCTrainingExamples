#!/bin/bash
# gpu_thermal_throttle_check.sh -- sustained-load thermal / throttle diagnostic.
#
# Loads every visible GPU with a sustained memory-bound kernel (HIP-STREAM triad)
# in repeated bursts, then flags a node that (a) rides its own thermal limit or
# (b) loses throughput burst-to-burst (thermal soak) -- i.e. a cooling or
# power-delivery anomaly, judged against the hardware's own spec.
#
# Nothing is hard-coded: the arch and the junction critical temperature are read
# at runtime (rocminfo; sysfs hwmon tempN_crit, exposed by every amdgpu), so it
# runs unchanged on any ROCm/HIP-capable AMD GPU that reports a junction/edge
# crit.  A node is flagged when peak junction rides within AMDGPU_TEMP_MARGIN of
# that crit -- a healthy GPU keeps ample headroom under this moderate load -- and
# the droop check is purely relative, so it needs no absolute reference.
#
# Run on an idle / exclusive node so the sampled temperature reflects this load.
#
# Tunables (env): AMDGPU_TEMP_MARGIN (C under crit, default 10), AMDGPU_DROOP_MAX
# (fraction, default 0.08), AMDGPU_NREPS (burst cap, default 20), AMDGPU_BURST_SEC
# (default 25), AMDGPU_PLATEAU_C (stop once junction rises < this many C over two
# bursts, default 1), AMDGPU_MINREPS (min bursts before a plateau exit, default 5).
# The load soaks to thermal steady state rather than a fixed
# time, so the verdict does not depend on test duration or the node's starting
# temperature -- a healthy node plateaus early (fast), a suspect one soaks longer.
# Prints a per-burst table + one greppable "RESULT: ... verdict=<...>" line, and
# "SUCCESS" only when healthy so it also plugs into the ctest harness.

set -u

# --- rocm environment (Cray PE or module) -----------------------------------
if [[ -n "${CRAYPE_VERSION:-}" || -f /etc/cray-release ]]; then
   [ -z "${HIPCC:-}" ] && export HIPCC=$(which hipcc)
   export HIP_PLATFORM=amd
else
   module -t list 2>&1 | grep -q "^rocm" || { echo "loading default rocm module"; module load rocm; }
fi
HIPCC=${HIPCC:-hipcc}

# --- one rocminfo snapshot: arch, GPU count, marketing name ------------------
RINFO=$(rocminfo 2>/dev/null)
GPU_ARCH=$(echo "$RINFO" | grep -m1 -E "gfx[^0]" | sed -e 's/ *Name: *//' -e 's/[[:blank:]]//g')
[ -z "$GPU_ARCH" ] && { echo "no AMD GPU detected (rocminfo)"; echo "FAILURE"; exit 1; }
NGPU=$(echo "$RINFO" | grep -cE "Name:[[:space:]]+gfx"); [ "$NGPU" -ge 1 ] || NGPU=1
GPU_NAME=$(echo "$RINFO" | grep -m1 "Marketing Name" | sed 's/.*Name: *//;s/[[:space:]]*$//')
# APUs share host memory: build xnack+ so page-fault access is valid (harmless elsewhere).
OFFLOAD="$GPU_ARCH"; echo "$RINFO" | grep -q "MI300A" && { OFFLOAD="${GPU_ARCH}:xnack+"; export HSA_XNACK=1; }

# --- build the repo HIP-STREAM load (explicit arch => no silent no-op) --------
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
BUILD_DIR=$(mktemp -d); trap "rm -rf ${BUILD_DIR}" EXIT
cp ${REPO_DIR}/HIP/hip-stream/stream.hip ${BUILD_DIR}/ 2>/dev/null \
   || { echo "cannot find HIP/hip-stream/stream.hip"; echo "FAILURE"; exit 1; }
${HIPCC} -O3 -DNDEBUG -x hip -munsafe-fp-atomics --offload-arch=${OFFLOAD} \
   ${BUILD_DIR}/stream.hip -o ${BUILD_DIR}/stream 2>${BUILD_DIR}/build.log \
   || { echo "build failed:"; cat ${BUILD_DIR}/build.log; echo "FAILURE"; exit 1; }

# --- device-reported junction sensor + critical temp (the grounding source) --
# hwmon tempN_{label,input,crit} are millidegrees C; prefer 'junction', else 'edge'.
TIN_FILES=(); TCRIT_MC=""
scan_label() {
   local want=$1 lf base
   for lf in /sys/class/drm/renderD*/device/hwmon/hwmon*/temp*_label; do
      [ -f "$lf" ] && [ "$(cat "$lf" 2>/dev/null)" = "$want" ] || continue
      base=${lf%_label}
      [ -f "${base}_input" ] && TIN_FILES+=("${base}_input")
      [ -z "$TCRIT_MC" ] && [ -f "${base}_crit" ] && TCRIT_MC=$(cat "${base}_crit" 2>/dev/null)
   done
}
scan_label junction; [ ${#TIN_FILES[@]} -eq 0 ] && scan_label edge
[ ${#TIN_FILES[@]} -eq 0 ] && { echo "no junction/edge hwmon sensor found -- cannot read HW thermal limit"; echo "FAILURE"; exit 1; }
[ -n "$TCRIT_MC" ] || { echo "device did not report a critical temp (tempN_crit) -- cannot ground the check"; echo "FAILURE"; exit 1; }
TCRIT=$(( TCRIT_MC / 1000 ))

# max live junction temp (C) across the device sensors, straight from hwmon
read_temp() { cat "${TIN_FILES[@]}" 2>/dev/null | sort -rn | head -1 | awk '{printf "%.0f",$1/1000}'; }

# --- preflight: confirm the load actually runs on the GPU --------------------
PF=$(ROCR_VISIBLE_DEVICES=0 ${BUILD_DIR}/stream 2>/dev/null | awk '/^Triad:/{print $2}')
awk "BEGIN{v=\"$PF\"; exit !(v!=\"inf\" && v+0>0)}" 2>/dev/null \
   || { echo "STREAM load did not execute (Triad='${PF:-none}') -- check GPU visibility/arch"; echo "FAILURE"; exit 1; }

# --- background sampler: epoch,maxJunctionC at ~1 Hz -------------------------
SMP=${BUILD_DIR}/samples.csv
( while :; do echo "$(date +%s),$(read_temp)"; sleep 1; done ) > $SMP 2>/dev/null &
SPID=$!; trap "kill $SPID 2>/dev/null; rm -rf ${BUILD_DIR}" EXIT

# --- sustained-load bursts: soak until junction plateaus (or NREPS cap) -------
# NREPS is a CAP; the loop stops early once junction stops climbing (rise < PLATEAU
# over two bursts), i.e. at thermal steady state -- so the verdict reflects where
# the node settles under load, not how long the test ran or how warm it started.
NREPS=${AMDGPU_NREPS:-20}; BURST=${AMDGPU_BURST_SEC:-25}; PLATEAU=${AMDGPU_PLATEAU_C:-1}; MINREPS=${AMDGPU_MINREPS:-5}; NODE=$(hostname)
echo "node: ${NODE} | GPU: ${GPU_NAME:-$GPU_ARCH} (${GPU_ARCH}) | visible=${NGPU} | junction_crit=${TCRIT}C (device-reported)"
echo "load: HIP-STREAM triad, up to ${NREPS} x ${BURST}s bursts on all ${NGPU} GPU(s), stop when junction plateaus (<${PLATEAU}C rise)"
declare -a BW TMAX; last=0
for ((r=1; r<=NREPS; r++)); do
   s=$(date +%s); sum=0; cnt=0
   while [ $(( $(date +%s) - s )) -lt $BURST ]; do
      pids=()   # wait only on the stream jobs, never on the background sampler
      for ((g=0; g<NGPU; g++)); do ROCR_VISIBLE_DEVICES=$g ${BUILD_DIR}/stream >${BUILD_DIR}/o.$g 2>/dev/null & pids+=($!); done
      wait "${pids[@]}"
      for ((g=0; g<NGPU; g++)); do
         t=$(awk '/^Triad:/{print $2}' ${BUILD_DIR}/o.$g)
         [ -n "$t" ] && sum=$(awk -v a=$sum -v b=$t 'BEGIN{print a+b}') && cnt=$((cnt+1))
      done
   done
   e=$(date +%s)
   BW[$r]=$(awk -v s=$sum -v c=$cnt 'BEGIN{printf "%.1f", c?s/c:0}')            # mean per-GPU Triad GiB/s
   TMAX[$r]=$(awk -F, -v a=$s -v b=$e '$1>=a&&$1<=b&&$2!=""{if($2>m)m=$2}END{print m+0}' $SMP)
   last=$r
   printf "  burst %d: Triad=%s GiB/s/GPU | peak_junction=%s C\n" $r "${BW[$r]}" "${TMAX[$r]}"
   # steady state: junction flat (< PLATEAU C rise) over the last TWO 2-burst
   # windows, after >= MINREPS soak, with both current readings valid (a dropped
   # 1 Hz sample makes TMAX 0, which must not be mistaken for a plateau).
   if [ $r -ge $MINREPS ] && [ "${TMAX[$r]}" -gt 0 ] && [ "${TMAX[$r-2]}" -gt 0 ]; then
      awk "BEGIN{exit !(${TMAX[$r]}-${TMAX[$r-2]} < $PLATEAU && ${TMAX[$r-1]}-${TMAX[$r-3]} < $PLATEAU)}" \
         && { echo "  (junction plateaued -- steady state reached)"; break; }
   fi
done

# --- verdict (thresholds device-derived or relative; nothing hard-coded) -----
MARGIN=${AMDGPU_TEMP_MARGIN:-10}; DROOP_MAX=${AMDGPU_DROOP_MAX:-0.08}
TLIMIT=$(( TCRIT - MARGIN ))
PEAKT=$(printf '%s\n' "${TMAX[@]}" | sort -rn | head -1)
DROOP=$(awk -v a="${BW[1]}" -v b="${BW[$last]}" 'BEGIN{if(a>0) printf "%.3f",(a-b)/a; else print "0.000"}')
DROOP_PCT=$(awk -v d=$DROOP 'BEGIN{printf "%.1f",d*100}')
printf "summary: peak_junction=%sC | crit=%sC margin=%sC -> flag>=%sC | droop=%s%% (limit %.1f%%)\n" \
   "$PEAKT" "$TCRIT" "$MARGIN" "$TLIMIT" "$DROOP_PCT" "$(awk -v d=$DROOP_MAX 'BEGIN{print d*100}')"
fail=0
awk "BEGIN{exit !($PEAKT>=$TLIMIT)}" && { echo "ANOMALY: peak junction ${PEAKT}C within ${MARGIN}C of device crit ${TCRIT}C -- cooling/power-delivery suspect"; fail=1; }
awk "BEGIN{exit !($DROOP>=$DROOP_MAX)}" && { echo "ANOMALY: throughput drooped ${DROOP_PCT}% burst-to-burst -- thermal soak"; fail=1; }
VERDICT=HEALTHY; [ $fail -ne 0 ] && VERDICT=ANOMALY
echo "RESULT: node=${NODE} part=${GPU_ARCH} visible=${NGPU} peak_junction=${PEAKT}C crit=${TCRIT}C droop=${DROOP_PCT}% verdict=${VERDICT}"
[ $fail -eq 0 ] && { echo "SUCCESS"; exit 0; }
echo "FAILURE"; exit 1
