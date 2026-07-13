#!/bin/bash

# Regression check for the ROCm HIP-runtime host-memory leak on the
# device-to-host (D2H) / tensor.item() path.
#
# The bug: on some ROCm HIP runtimes every synchronous D2H copy (the path
# behind .item(), .cpu(), copy_() into a host buffer) leaks a little HOST
# RAM -- ~0.8-1.1 KB per call. Invisible in a microbenchmark, but fatal for
# training, which does millions of such calls and eventually host-OOMs. The
# leak lives in the HIP runtime, not PyTorch: within one ROCm version every
# torch version behaves the same (one exception, see D_pinned below).
# Observed on MI250X and MI300A -- leaks on rocm 6.3.x / 7.0.x / 7.1.x /
# 7.2.0 / 7.12.x; clean on 6.4.x / 7.2.1-7.2.4 / 7.13.x (fix landed at the
# 7.2.1 point release, hip build 7.2.53211).
#
# Discriminator: native host-RSS growth per call (bytes/call), measured as
# delta VmRSS from /proc/self/status across a tight loop, after a warmup
# that settles one-time allocations. tracemalloc stays ~0 (native, not a
# Python-heap leak). Clean runtimes report ~0 B/call; leaky ones ~800-1100.
# The two populations are ~100x apart, so the pass threshold is not delicate
# (default 100 B/call == 0.1 KB/call).
#
# The operations we time (see the printed legend for the same list):
#   item      N x tensor.item()          -- headline .item() leak (reproducer 1)
#   A_sync    cuda.synchronize()         -- CONTROL: pure sync, no D2H (want ~0)
#   B_gpuop   x.add_(0.0)                -- CONTROL: GPU kernel, no D2H (want ~0)
#   C_item    tensor.item()              -- scalar D2H copy (the primary bug path)
#   D_pinned  copy_() into pinned host   -- D2H into a REUSED pinned buffer.
#                                           NB: on rocm 7.2.0 + torch 2.12.0 the
#                                           leak MOVES here while .item() looks
#                                           clean, so we must gate on D, not C.
#   E_page    copy_() into pageable host -- D2H into a REUSED pageable buffer
#   F_cpu     tensor.cpu()               -- D2H that ALLOCATES a new host tensor
#   G_event   cuda.Event().record()      -- event create/record. A DISTINCT leak
#                                           path (e.g. rocm 6.4.x leaks here while
#                                           D2H is clean); reported for info, does
#                                           not gate the verdict.
#
# Verdict: PASS if max bytes/call over the D2H family {item, C, D, E, F} is
# <= THRESHOLD_B; otherwise FAIL. SKIP if no pytorch module / torch import
# fails / no GPU. A,B are controls and G is reported for info only (none gate
# the verdict). Edit the three constants below if you ever need to.
#
# NOTE: this test assumes PyTorch has been installed according to the model
# installation repo: https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

# Iteration counts: how many times each op runs so the accumulated leak rises
# well above VmRSS sampling noise (~a few kB). At these counts a leaky runtime
# grows tens/hundreds of MB while a clean one stays flat; smaller counts still
# work but shrink the margin, larger ones just make the test slower.
ITEM_N=300000     # tensor.item() calls in reproducer 1 (the .item() headline)
ISO_N=100000      # calls per op in reproducer 2 (the A-G per-primitive breakdown)
# Pass/fail line in bytes-leaked-per-call. Clean runtimes measure ~0 B/call and
# leaky ones ~800-1100 B/call, so 100 (= 0.1 KB/call) sits ~10x above clean
# noise and ~8x below a real leak -- anywhere in ~20..400 gives the same verdict.
THRESHOLD_B=100

if ! type module >/dev/null 2>&1; then
   [ -r /etc/profile.d/lmod.sh ]         && . /etc/profile.d/lmod.sh
   [ -r /usr/share/lmod/lmod/init/bash ] && . /usr/share/lmod/lmod/init/bash
fi

# Load rocm and pytorch, but do NOT reload either if it is already loaded
# (hierarchical modules: pytorch only resolves once rocm is in).
module -t list 2>&1 | grep -q "^rocm/"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module -t list 2>&1 | grep -q "^pytorch/"
if [ $? -eq 1 ]; then
  echo "pytorch module is not loaded"
  echo "loading default pytorch module"
  module load pytorch 2>/dev/null
fi

if ! python3 -c "import torch" 2>/dev/null; then
   echo "MEM LEAK CHECK: SKIPPED (pytorch module not available or torch import failed)"
   exit 77
fi

echo
echo "--- pytorch host-leak regression-check environment ---"
if type module >/dev/null 2>&1; then
   echo "  loaded modules =" $(module -t list 2>&1 | tr '\n' ' ')
fi
python3 - <<'PY'
import torch
print(f"  torch.__version__ = {torch.__version__}")
print(f"  torch.version.hip = {torch.version.hip}")
print(f"  cuda_available    = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device            = {torch.cuda.get_device_name(0)}")
PY
echo "  item_N=${ITEM_N}  iso_N=${ISO_N}  threshold=${THRESHOLD_B} B/call"
echo "------------------------------------------------------"
echo

# ======================================================================
# Reproducer 1 (inlined item_leak.py): N x tensor.item(), report native
# host-RSS growth in bytes/call. VmRSS is read in kB (finer than MB) so
# bytes/call = 1024 * dkB / N.
# ======================================================================
echo "=== item_leak: ${ITEM_N} x .item() ==="
ITEM_OUT=$(python3 - "${ITEM_N}" <<'PY'
import sys, torch
def rss_kb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS:"):
                return int(l.split()[1])   # VmRSS is reported in kB
    return -1
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
if not torch.cuda.is_available():
    print("LEAKCHECK nogpu"); sys.exit(77)
torch.cuda.set_device(0)
x = torch.randn(1, device="cuda")
torch.cuda.synchronize()
base = rss_kb()
for _ in range(N):
    _ = x.item()                 # synced D2H scalar copy (the gdb path)
torch.cuda.synchronize()
end = rss_kb()
per_call_B = 1024.0 * (end - base) / N
print(f"[item] N={N} start_kB={base} end_kB={end} "
      f"leak_MB={(end-base)//1024} per_call_B={per_call_B:.2f}")
print(f"LEAKCHECK item per_call_B={per_call_B:.3f} leak_MB={(end-base)//1024}")
PY
)
RC_ITEM=$?
echo "${ITEM_OUT}"
if [ ${RC_ITEM} -eq 77 ] || echo "${ITEM_OUT}" | grep -q "LEAKCHECK nogpu"; then
   echo "MEM LEAK CHECK: SKIPPED (no GPU available)"
   exit 77
fi
if [ ${RC_ITEM} -ne 0 ]; then
   echo "MEM LEAK CHECK: FAILED (item_leak reproducer rc=${RC_ITEM})"
   exit 1
fi

# ======================================================================
# Reproducer 2 (inlined leak_isolate.py): isolate which primitive leaks.
# Each sub-bench warms up (settles one-time allocs), resets baseline, then
# loops N times; the leak is monotonic so a clean op reports ~0 and a
# leaking op a clear per-call cost. tracemalloc confirms native growth.
# ======================================================================
echo
echo "=== leak_isolate: ${ISO_N} iters/op ==="
ISO_OUT=$(python3 - "${ISO_N}" <<'PY'
import gc, sys, tracemalloc, torch
def rss_kb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS:"):
                return int(l.split()[1])   # kB
    return -1
if not torch.cuda.is_available():
    print("LEAKCHECK nogpu"); sys.exit(77)
torch.cuda.set_device(0)
dev = "cuda:0"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000

x   = torch.randn(1, device=dev)          # gpu scalar
v   = torch.randn(1024, device=dev)       # small gpu vector
hb  = torch.empty(1, pin_memory=True)     # reused pinned host buffer
hb0 = torch.empty(1)                      # reused pageable host buffer

def bench(name, fn, need_sync=True):
    for _ in range(2000):                 # warmup (fill caches / one-time allocs)
        fn()
    torch.cuda.synchronize(); gc.collect()
    tracemalloc.start()
    b = rss_kb(); tb, _ = tracemalloc.get_traced_memory()
    for _ in range(N):
        fn()
    if need_sync:
        torch.cuda.synchronize()
    e = rss_kb(); te, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    per_call_B = 1024.0 * (e - b) / N
    print(f"{name:20s} N={N} leak_MB={(e-b)//1024:5d} per_call_B={per_call_B:8.2f} "
          f"py_tracemalloc_dB={te-tb}")
    print(f"LEAKCHECK prim {name} per_call_B={per_call_B:.3f}")

bench("A_pure_synchronize",  lambda: torch.cuda.synchronize())
bench("B_gpu_op_only",       lambda: x.add_(0.0), need_sync=True)   # GPU kernel, no D2H
bench("C_item_scalar",       lambda: x.item())                     # synced D2H scalar (.item)
bench("D_d2h_into_pinned",   lambda: hb.copy_(x))                   # D2H into REUSED pinned buf
bench("E_d2h_into_pageable", lambda: hb0.copy_(x))                  # D2H into REUSED pageable buf
bench("F_cpu_alloc_copy",    lambda: v.cpu())                       # D2H that ALLOCATES host tensor
bench("G_event_record",      lambda: torch.cuda.Event().record())  # event create/record only
print("DONE_ISOLATE")
PY
)
RC_ISO=$?
echo "${ISO_OUT}"
if [ ${RC_ISO} -eq 77 ] || echo "${ISO_OUT}" | grep -q "LEAKCHECK nogpu"; then
   echo "MEM LEAK CHECK: SKIPPED (no GPU available)"
   exit 77
fi
if [ ${RC_ISO} -ne 0 ] || ! echo "${ISO_OUT}" | grep -q "DONE_ISOLATE"; then
   echo "MEM LEAK CHECK: FAILED (leak_isolate reproducer rc=${RC_ISO}, incomplete)"
   exit 1
fi

# ----------------------------------------------------------------------
# Build the summary table and decide the verdict. Only the D2H family
# {item, C, D, E, F} gates PASS/FAIL; A,B are controls and G is info.
# ----------------------------------------------------------------------
get_val() {   # $1 = anchored grep key, $2 = text blob -> prints per_call_B float
   echo "${2}" | awk -v k="$1" '
      $0 ~ k { for (i=1;i<=NF;i++) if ($i ~ /^per_call_B=/) { split($i,a,"="); print a[2] } }' | head -1
}
row() {       # category, operation, role, value -> table row + "LEAK"/"ok"
   local res; res=$(awk -v v="${4:-0}" -v t="${THRESHOLD_B}" 'BEGIN{print (v+0>t)?"LEAK":"ok"}')
   printf "  | %-8s | %-30s | %-7s | %10s | %-4s |\n" "$1" "$2" "$3" "${4:-?}" "${res}"
}

ITEM_PC=$(get_val "^LEAKCHECK item "                    "${ITEM_OUT}")
A_PC=$(get_val    "^LEAKCHECK prim A_pure_synchronize "  "${ISO_OUT}")
B_PC=$(get_val    "^LEAKCHECK prim B_gpu_op_only "       "${ISO_OUT}")
C_PC=$(get_val    "^LEAKCHECK prim C_item_scalar "       "${ISO_OUT}")
D_PC=$(get_val    "^LEAKCHECK prim D_d2h_into_pinned "   "${ISO_OUT}")
E_PC=$(get_val    "^LEAKCHECK prim E_d2h_into_pageable " "${ISO_OUT}")
F_PC=$(get_val    "^LEAKCHECK prim F_cpu_alloc_copy "    "${ISO_OUT}")
G_PC=$(get_val    "^LEAKCHECK prim G_event_record "      "${ISO_OUT}")

echo
echo "  host-RSS growth per call (bytes/call). gated ops must be <= ${THRESHOLD_B}; clean ~0, leaky ~800-1100."
echo "  role: gated = counts toward PASS/FAIL; control = pure-GPU, must stay ~0; info = separate leak path."
printf "  | %-8s | %-30s | %-7s | %10s | %-4s |\n" "category" "operation" "role" "B/call" "res"
printf "  |----------|--------------------------------|---------|------------|------|\n"
row "item"     "N x tensor.item()"             "gated"   "${ITEM_PC}"
row "A_sync"   "cuda.synchronize()"            "control" "${A_PC}"
row "B_gpuop"  "x.add_(0.0)"                   "control" "${B_PC}"
row "C_item"   "tensor.item() scalar D2H"      "gated"   "${C_PC}"
row "D_pinned" "copy_() -> pinned host buf"    "gated"   "${D_PC}"
row "E_page"   "copy_() -> pageable host buf"  "gated"   "${E_PC}"
row "F_cpu"    "tensor.cpu() (alloc + D2H)"    "gated"   "${F_PC}"
row "G_event"  "cuda.Event().record()"         "info"    "${G_PC}"
echo

read WORST OFFENDER < <(awk -v vals="${ITEM_PC} ${C_PC} ${D_PC} ${E_PC} ${F_PC}" \
   -v names="item C_item D_pinned E_page F_cpu" '
   BEGIN { nv=split(vals,v," "); split(names,n," "); m=-1; who="none";
           for (i=1;i<=nv;i++){ x=v[i]+0; if (x>m){ m=x; who=n[i] } }
           printf "%.3f %s", m, who }')

if awk -v w="${WORST}" -v t="${THRESHOLD_B}" 'BEGIN{exit !(w>t)}'; then
   echo "MEM LEAK CHECK: FAILED  (${OFFENDER} leaks ${WORST} B/call > ${THRESHOLD_B} B/call)"
   echo "  -> this ROCm HIP runtime has the D2H/.item() host-memory leak."
   echo "  -> known-clean runtimes: rocm 7.2.1-7.2.4, 7.13.x (hip build >= 7.2.53211)."
   exit 1
else
   echo "MEM LEAK CHECK: PASSED  (max ${OFFENDER} ${WORST} B/call <= ${THRESHOLD_B} B/call; D2H host-leak absent)"
   exit 0
fi
