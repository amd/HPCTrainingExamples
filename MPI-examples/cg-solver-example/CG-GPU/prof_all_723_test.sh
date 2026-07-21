#!/usr/bin/env bash
# Exercise EVERY ROCm 7.2.3 profiler on a real cg_gpu run, under whatever
# perf_event_paranoid the (prolog-tightened) node currently has. Reports, per
# tool: exit status, whether output was produced, and any perf_event_paranoid
# interaction.
#
# IMPORTANT: request GPUs with --gres (an --exclusive job alone hides /dev/dri
# in the cgroup, so rocminfo would see 0 GPUs). Run e.g.:
#   srun -p PPAC_MI300A_SPX --exclusive --gres=gpu:4 -t 00:25:00 \
#        --comment=paranoid2 ./CG-GPU/prof_all_723_test.sh
set -u
cd "$(dirname "$(readlink -fm "$0")")"

# ---------------------------------------------------------------------------
# Load the toolchain via modules. 'module load rocm/<ver>' already prepends
# rocm-patches-<ver>/rocprof-compute/bin, so rocprof-compute resolves to the
# self-contained Nuitka single-file build (bundling pandas/dash/matplotlib/...).
#
# CRITICAL: never pipe or command-substitute 'module' (e.g. `module load ... |
# tail`). A pipe runs the module shell function in a SUBSHELL, so its eval'd
# environment changes (PATH, ROCM_PATH, LD_LIBRARY_PATH) are discarded and the
# patches path never lands on PATH. Call module plainly.
# ---------------------------------------------------------------------------
ROCM_VER="${ROCM_VER:-7.2.3}"

module purge
module load "rocm/$ROCM_VER" openmpi

# If the site also ships a dedicated rocm_patches module, load it too.
PATCHES_MODULE=none
for m in "rocm_patches/$ROCM_VER" rocm_patches "rocm-patches/$ROCM_VER" rocm-patches; do
  if module is-avail "$m" 2>/dev/null; then
    module load "$m"; PATCHES_MODULE="$m"; break
  fi
done
echo "# rocm_patches module: $PATCHES_MODULE"

echo "############################################################"
echo "# node=$(hostname)  perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"
echo "# ROCM_PATH=$ROCM_PATH"
echo "# rocprofv3=$(command -v rocprofv3)"
echo "# rocprof-compute=$(command -v rocprof-compute)  ($(rocprof-compute --version 2>/dev/null | grep -m1 version))"
echo "# GPUs=$(rocminfo 2>/dev/null | grep -c 'Device Type:             GPU')  gfx=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')"
echo "############################################################"

MP="mpirun -n 1 --oversubscribe --bind-to none"
APP="./cg_gpu src/Dubcova2.pm staged"
OUT=prof723_out; rm -rf "$OUT" workloads; mkdir -p "$OUT"

pass=0; fail=0; skip=0
RESULTS=""
record() { RESULTS+="$(printf '%-26s %s\n' "$1" "$2")"$'\n'; }

have() { command -v "$1" >/dev/null 2>&1; }

# generic runner: name, output-dir-or-file to check, command...
# Success = the profiler PRODUCED OUTPUT. A nonzero exit *after* the data is
# written (e.g. rocprofiler-systems' known v1.3.0 glibc double-free at teardown)
# is reported as PASS-with-caveat, not a failure. Any perf_event_paranoid message
# is flagged explicitly (there were none in practice).
run_tool() {
  local name="$1" checkpath="$2"; shift 2
  local log="$OUT/${name}.log"
  echo; echo "=== $name ==="
  echo "\$ $*" | tee "$log"
  "$@" >>"$log" 2>&1; local rc=$?
  local para=""; grep -iqaE 'perf_event_paranoid|Access to performance monitoring' "$log" && para=" [PERF_PARANOID!]"
  local note=""
  if grep -qa 'double free or corruption' "$log"; then
    note=" (teardown double-free; data already written)"
  elif grep -qa 'roofline.*error while loading shared libraries\|roofline-ubuntu.*libamdhip64' "$log"; then
    note=" (counters OK; optional roofline microbench failed: Nuitka onefile scrubbed LD_LIBRARY_PATH for its helper)"
  fi
  local produced=no
  if [[ -z "$checkpath" ]]; then
    [[ $rc -eq 0 ]] && produced=yes
  elif [[ -e "$checkpath" ]] || [[ -n "$(ls -A "$checkpath" 2>/dev/null)" ]] || compgen -G "${checkpath}*" >/dev/null 2>&1; then
    produced=yes
  fi
  if [[ "$produced" == yes ]]; then
    if [[ $rc -eq 0 ]]; then echo "-> PASS (rc=0)$para"; record "$name" "PASS$para"
    else echo "-> PASS (output produced; nonzero teardown rc=$rc)$note$para"; record "$name" "PASS(output; rc=$rc)$note$para"; fi
    pass=$((pass+1))
  else
    echo "-> FAIL (rc=$rc)$para"; tail -6 "$log" | sed 's/^/     /'; record "$name" "FAIL(rc=$rc)$para"; fail=$((fail+1))
  fi
}

# ---------------------------------------------------------------------------
echo "=== build cg_gpu under rocm/7.2.3 ==="
make clean >/dev/null 2>&1; make >/dev/null 2>&1 && [[ -x ./cg_gpu ]] && echo "built OK" || { echo "BUILD FAILED"; exit 1; }

# 1) rocprofv3 — system trace (HIP + kernel + HSA)
if have rocprofv3; then
  run_tool rocprofv3_systrace "$OUT/rpv3_sys" \
    $MP rocprofv3 --sys-trace --output-format csv -d "$OUT/rpv3_sys" -- $APP
  # 2) rocprofv3 — hardware counters (PMC)
  run_tool rocprofv3_pmc "$OUT/rpv3_pmc" \
    $MP rocprofv3 --pmc SQ_WAVES GRBM_GUI_ACTIVE --output-format csv -d "$OUT/rpv3_pmc" -- $APP
else record rocprofv3 "SKIP(absent)"; skip=$((skip+2)); fi

# 3) rocprof-compute — GPU counter collection (omniperf successor)
if have rocprof-compute; then
  # success = GPU counter files under perfmon/ (roofline microbench is optional)
  run_tool rocprof-compute_profile "workloads/t723/*/perfmon/pmc_perf_0.txt" \
    rocprof-compute profile -n t723 -- $MP $APP
else record rocprof-compute "SKIP(absent)"; skip=$((skip+1)); fi

# 4) rocprofiler-systems — CPU/GPU sampling (THIS uses perf_event_open -> paranoid-sensitive)
if have rocprof-sys-sample; then
  ROCPROFSYS_OUTPUT_PATH="$OUT/rpsys_sample" \
  run_tool rocprof-sys-sample "$OUT/rpsys_sample" \
    $MP rocprof-sys-sample -- $APP
else record rocprof-sys-sample "SKIP(absent)"; skip=$((skip+1)); fi

# 5) rocprofiler-systems — binary-instrumentation runtime
if have rocprof-sys-run; then
  ROCPROFSYS_OUTPUT_PATH="$OUT/rpsys_run" \
  run_tool rocprof-sys-run "$OUT/rpsys_run" \
    $MP rocprof-sys-run -- $APP
else record rocprof-sys-run "SKIP(absent)"; skip=$((skip+1)); fi

echo
echo "############################################################"
echo "# SUMMARY  (paranoid=$(cat /proc/sys/kernel/perf_event_paranoid))"
echo "############################################################"
printf '%s' "$RESULTS"
echo "------------------------------------------------------------"
echo "PASS=$pass  FAIL=$fail  SKIP=$skip"
