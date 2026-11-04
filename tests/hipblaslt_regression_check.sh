#!/bin/bash

# Note, this test is checking the default track
# which for MI300A is the case when TENSILE_SOLUTION_SELECTION_METHOD=0
# NOTE that on MI350, the TENSILE_SOLUTION_SELECTION_METHOD flag has no effect
# and StreamK is always the track that is picked
# For details: https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/env-variables.html#origami-with-stream-k-configuration

# Fast regression check for hipBLASLt heuristic/dispatch bugs on MI300-class GPUs.
# Two phases:
#   Phase 1 (per-dtype log sweep): runs the forward + 2 backward GEMMs of
#     nn.Linear(hidden, num_classes) over a (batch, num_classes, hidden) sweep,
#     captures the hipBLASLt API log at LOG_LEVEL=5, and fails on any
#     `returnAlgoCount=0` heuristic miss.
#   Phase 2 (layout timing sweep): times small GEMMs across the full transpose
#     matrix (mm/bmm/linear, NN/NT/TN/TT) and fails on any cell slower than
#     ${SLOW_MS} ms -- catches the generic-fallback dispatch defect the log
#     sweep is blind to (see the Phase 2 banner below).
#
# Knobs (env vars):
#   HIPBLASLT_REGRESS_SWEEP    quick (12 configs) | full (120, default)
#   HIPBLASLT_REGRESS_DTYPES   comma-separated, default fp16,fp32,bf16
#   HIPBLASLT_REGRESS_ARCH     force the shape family (gfx942|gfx950); the
#                              .dat library is still chosen by real hardware
#   HIPBLASLT_REGRESS_SLOW_MS  Phase 2 slow-cell threshold in ms, default 50
#
# Each dtype consults an INDEPENDENT Tensile .dat (HH=fp16, SS=fp32, BB=bf16),
# so each needs its own coverage. fp64 is excluded: it rarely dispatches through
# hipBLASLt and its DD_DD libraries are catch-all rows that cannot miss, so it
# would pass vacuously.
#
# Shapes are arch-aware (read from rocminfo, override with HIPBLASLT_REGRESS_ARCH):
#   gfx942 (CDNA3, MI300A/X): ResNet classifier-head FC shapes, centered on
#                             (256,100,2048) -- the point that surfaced the
#                             original 2026-05 regression.
#   gfx950 (CDNA4, MI355X)  : Llama-/Mistral-style FFN + output-projection shapes.
#   other                   : small blended fallback (warns).
# Detection is arch-agnostic: a zero-solution tuple (Phase 1) or a slow cell
# (Phase 2) is a regression on whatever partition (SPX/CPX) the run lands on.
#
# Runtime ~10 s healthy; each regressed shape adds ~6 s of Tensile compile stall.
#
# Assumes PyTorch installed per:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

if ! type module >/dev/null 2>&1; then
   [ -r /etc/profile.d/lmod.sh ]         && . /etc/profile.d/lmod.sh
   [ -r /usr/share/lmod/lmod/init/bash ] && . /usr/share/lmod/lmod/init/bash
fi

module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load pytorch 2>/dev/null

if ! python3 -c "import torch" 2>/dev/null; then
   echo "REGRESSION CHECK: SKIPPED (pytorch module not available or torch import failed)"
   exit 77
fi

# Self-diagnostic header
echo
echo "--- hipBLASLt regression-check environment ---"
echo "  HIPBLASLT_TENSILE_LIBPATH = ${HIPBLASLT_TENSILE_LIBPATH:-<unset, will use default <rocm>/lib/hipblaslt/library>}"
echo "  HIPBLASLT_OVERLAY         = ${HIPBLASLT_OVERLAY:-<unset>}"
echo "  ROCM_PATH                 = ${ROCM_PATH:-<unset>}"
if type module >/dev/null 2>&1; then
   echo "  loaded modules            =" $(module -t list 2>&1 | tr '\n' ' ')
fi
echo "----------------------------------------------"
echo

# HIPBLASLT_REGRESS_ARCH override: changes which SHAPE FAMILY is
# probed, NOT which .dat library hipBLASLt opens. The .dat library is
# selected by the actual hardware at runtime -- setting
# HIPBLASLT_REGRESS_ARCH=gfx950 on a gfx942 node will probe Llama-style
# shapes against the gfx942 heuristic library (a useful but distinct
# question), not against a gfx950 library that doesn't exist on that
# node. To actually test a gfx950 .dat library, you need gfx950
# hardware.
if [ -n "${HIPBLASLT_REGRESS_ARCH:-}" ]; then
   GFX_ARCH="${HIPBLASLT_REGRESS_ARCH}"
elif command -v rocminfo >/dev/null 2>&1; then
   GFX_ARCH=$(rocminfo 2>/dev/null \
      | grep -E '^\s*Name:\s*gfx' \
      | sed -e 's/^\s*Name:\s*//' -e 's/\s*$//' -e 's/:.*//' \
      | sort -u | head -1)
fi
GFX_ARCH=${GFX_ARCH:-unknown}
echo "detected GPU arch: ${GFX_ARCH}"

# hipBLASLt LOG_LEVEL semantics (NOT a simple verbosity counter --
# higher levels are bitmask-progressive over distinct log channels):
#     1 -> ERROR
#     2 -> ERROR + TRACE
#     3 -> ERROR + TRACE + HINTS
#     4 -> + INFO            <- [Info][initialize] line lives here
#     5 -> + API             <- [Api][rocblaslt_matmul_algo_get_heuristic]
#                                returnAlgoCount=N lines live here
# We need BOTH the API channel (to see heuristic misses) and the INFO
# channel (to confirm which .dat library was opened), so LEVEL=5 is
# the minimum that gives the parser anything to work with.
export HIPBLASLT_LOG_LEVEL=5

# Sweep width. The sweep is intentionally wider than the overlay's coverage, so
# a FAIL on a patched system is useful signal that the upstream gap is wider than
# what we patch -- don't narrow it just to satisfy CTest.
SWEEP=${HIPBLASLT_REGRESS_SWEEP:-full}

DTYPES_RAW=${HIPBLASLT_REGRESS_DTYPES:-fp16,fp32,bf16}

# Slow-cell threshold (ms) for the layout/dispatch timing sweep (Phase 2).
# A healthy small GEMM dispatches in well under 1 ms; the general-path
# dispatch defect produces a flat ~400-600 ms, so 50 ms cleanly separates
# the two with a wide margin.
SLOW_MS=${HIPBLASLT_REGRESS_SLOW_MS:-50}

OVERALL_FAIL=0
PER_DTYPE_RESULT=""
LOGS_TO_CLEAN=""
cleanup_logs() {
   if [ -n "${LOGS_TO_CLEAN}" ]; then
      # shellcheck disable=SC2086
      rm -f ${LOGS_TO_CLEAN}
   fi
}
trap cleanup_logs EXIT

# Phase 1, one dtype: run the harness + parser against a per-dtype log file
# (per-dtype so the parser's miss list isn't cross-contaminated). The dtype is
# passed as argv[3] so the harness can select .half()/.float()/.bfloat16().
run_dtype_sweep() {
   local DTYPE="$1"

   local LOG
   LOG=$(mktemp -t "hipblaslt_regress_${DTYPE}.XXXXXX.log")
   LOGS_TO_CLEAN="${LOGS_TO_CLEAN} ${LOG}"
   export HIPBLASLT_LOG_FILE=${LOG}

   echo
   echo "=========================================================="
   echo "  REGRESSION SWEEP  dtype=${DTYPE}  sweep=${SWEEP}  arch=${GFX_ARCH}"
   echo "=========================================================="

   python3 - "${SWEEP}" "${GFX_ARCH}" "${DTYPE}" <<'PY'
import os, sys, time, itertools
import torch
import torch.nn as nn

sweep = sys.argv[1] if len(sys.argv) > 1 else "quick"
arch  = sys.argv[2] if len(sys.argv) > 2 else "unknown"
dtype_label = sys.argv[3] if len(sys.argv) > 3 else "fp16"
_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp64": torch.float64,
}
if dtype_label not in _DTYPE_MAP:
    print(
        f"HIPBLASLT_REGRESSION_CHECK: unknown dtype '{dtype_label}' "
        f"(expected one of: {', '.join(sorted(_DTYPE_MAP))})",
        file=sys.stderr,
    )
    sys.exit(2)
torch_dtype = _DTYPE_MAP[dtype_label]
if not torch.cuda.is_available():
    print("HIPBLASLT_REGRESSION_CHECK: no GPU available, skipping", file=sys.stderr)
    sys.exit(77)  # CTest "skip" convention

# Per-arch sweep ranges. The (M, N, K) tuples that fall out of
# nn.Linear(hidden, num_classes) forward+backward on batch=bs are:
#     fwd:        (num_classes, bs, hidden)
#     bwd grad:   (hidden, bs, num_classes)
#     bwd weight: (hidden, num_classes, bs)
# Pick (bs, num_classes, hidden) ranges that cover the production
# workload pattern for the target arch.
if arch == "gfx942":
    # CDNA3 (MI300A/X): ResNet-final-FC pattern (original regression).
    # The sweep is intentionally wider than the existing overlay's
    # coverage so the test surfaces upstream-heuristic family gaps,
    # not just the 3 cells the overlay patches.
    if sweep == "full":
        batches = [32, 64, 128, 256, 512, 1024]
        classes = [10, 100, 256, 1000, 2048]
        hiddens = [512, 1024, 2048, 4096]
    else:                                       # quick
        batches = [64, 128, 256, 512]
        classes = [10, 100, 1000]
        hiddens = [2048]
elif arch == "gfx950":
    # CDNA4 (MI355X): LLM-style FFN / attention output projection
    # shapes. hidden = model dim, nc = ffn-dim or vocab-tile, bs =
    # tokens/microbatch. Sized to span Llama-3-8B (hidden=4096),
    # Llama-3-70B (hidden=8192), Mistral-7B (hidden=4096).
    if sweep == "full":
        batches = [1, 8, 32, 128, 1024, 4096]
        classes = [4096, 8192, 11008, 14336, 32768]
        hiddens = [2048, 4096, 8192]
    else:                                       # quick
        batches = [1, 32, 512, 4096]
        classes = [4096, 11008, 32768]
        hiddens = [4096]
else:
    # Unknown / future arch: small blended fallback.
    print(f"  WARN: unrecognized arch '{arch}', using generic fallback sweep",
          file=sys.stderr)
    batches = [64, 256, 1024]
    classes = [100, 1000, 4096]
    hiddens = [2048]

dev = torch.device("cuda")
configs = list(itertools.product(batches, classes, hiddens))
print(f"arch={arch} sweep={sweep} dtype={dtype_label}: {len(configs)} configurations, "
      f"{len(configs)*3} GEMM (M,N,K) tuples")
print(f"  batches={batches}")
print(f"  classes={classes}")
print(f"  hiddens={hiddens}")
print(f"  device     = {torch.cuda.get_device_name(0)}")
print(f"  torch.hip  = {torch.version.hip}")

torch.manual_seed(0)
# Warmup at the canonical (bs=256, nc=100, hd=2048) point: a known fp16
# heuristic hit, so it confirms the overlay is being consulted and triggers the
# `[Info][initialize]` line our library-path diagnostic keys off, without
# polluting the miss list with a synthetic warmup shape.
_warm = nn.Linear(2048, 100).to(dtype=torch_dtype, device=dev)
_warm(torch.randn(256, 2048, dtype=torch_dtype, device=dev)).sum().backward()
torch.cuda.synchronize()

t0 = time.time()
for bs, nc, hd in configs:
    net = nn.Linear(hd, nc).to(dtype=torch_dtype, device=dev)
    x   = torch.randn(bs, hd, dtype=torch_dtype, device=dev)
    t   = time.time()
    out = net(x)
    out.sum().backward()
    torch.cuda.synchronize()
    dt  = time.time() - t
    print(f"  [{dtype_label}] bs={bs:>4} nc={nc:>4} hd={hd:>4}  fwd+bwd {dt*1000:>7.1f} ms")
print(f"total compute time: {time.time()-t0:.2f} s")
PY
   local RC_PY=$?

   if [ ${RC_PY} -eq 77 ]; then
      echo "REGRESSION CHECK [dtype=${DTYPE}]: SKIPPED (no GPU)"
      return 77
   fi
   if [ ${RC_PY} -ne 0 ]; then
      echo "REGRESSION CHECK [dtype=${DTYPE}]: FAILED (python harness rc=${RC_PY})"
      return 1
   fi

   # Sanity-check the log file is non-empty BEFORE running the parser.
   echo
   echo "--- hipBLASLt log sanity check [dtype=${DTYPE}] ---"
   echo "  log file       = ${LOG}"
   if [ -s "${LOG}" ]; then
      local LOG_BYTES LOG_LINES API_CALLS
      LOG_BYTES=$(stat -c %s "${LOG}" 2>/dev/null || echo "?")
      LOG_LINES=$(wc -l < "${LOG}")
      echo "  size           = ${LOG_BYTES} bytes, ${LOG_LINES} lines"
      echo "  unique tags    =" $(grep -oE '\[(Error|Trace|Hints|Info|Api|Profile)\]' "${LOG}" | sort -u | tr '\n' ' ')
      API_CALLS=$(grep -c 'rocblaslt_matmul_algo_get_heuristic' "${LOG}" 2>/dev/null || echo 0)
      echo "  algo_get_heuristic calls captured = ${API_CALLS}"
      if [ "${API_CALLS}" -eq 0 ]; then
         echo "  WARN: no algo_get_heuristic events in log -- parser will report"
         echo "        zero misses regardless of true state. Check HIPBLASLT_LOG_LEVEL."
      fi
   else
      echo "  log file is empty or missing -- hipBLASLt logging is not configured."
      echo "  Expected env: HIPBLASLT_LOG_LEVEL=${HIPBLASLT_LOG_LEVEL:-<unset>},"
      echo "                HIPBLASLT_LOG_FILE=${HIPBLASLT_LOG_FILE:-<unset>}."
   fi

   # Extract the authoritative library path from the hipBLASLt init line.
   echo
   echo "--- hipBLASLt library path actually used (from log) [dtype=${DTYPE}] ---"
   local INIT_LINE
   INIT_LINE=$(grep -m1 'initialize' "${LOG}" 2>/dev/null)
   if [ -n "${INIT_LINE}" ]; then
      echo "  ${INIT_LINE}"
      if echo "${INIT_LINE}" | grep -q 'rocm-patches'; then
         echo "  -> overlay ACTIVE (patched .dat files in use)"
      else
         echo "  -> overlay NOT in use (default upstream .dat files)"
      fi
   else
      echo "  (no [initialize] line in log)"
      # If we're at LOG_LEVEL>=4 (so Info channel is on) and the file is
      # non-empty but still has no init line, that's worth flagging: it
      # may mean hipBLASLt never actually ran (e.g. all GEMMs went via
      # rocBLAS) which is a third class of false PASS.
      if [ "${HIPBLASLT_LOG_LEVEL:-0}" -ge 4 ] && [ -s "${LOG}" ]; then
         echo "  WARN: log is non-empty + LOG_LEVEL=${HIPBLASLT_LOG_LEVEL} >= 4 (INFO on),"
         echo "        yet no [initialize] line. hipBLASLt may not have been"
         echo "        invoked by the probe -- check first 5 log lines:"
         head -5 "${LOG}" | sed 's/^/        /'
      fi
   fi
   echo "-------------------------------------------------------"
   echo

   # Parse the log: every `returnAlgoCount=0` is a heuristic miss.  For
   # each miss, reconstruct (M, N, K) from the 3 layout_create lines that
   # immediately precede it.
   local MISSES RC_PARSE
   MISSES=$(python3 - "${LOG}" <<'PY'
import re, sys, os
log = sys.argv[1]
re_lay  = re.compile(r'rocblaslt_matrix_layout_create.*rows=(\d+) cols=(\d+) ld=\d+')
re_tA   = re.compile(r'MATMUL_DESC_TRANSA.*bufData=0x([0-9a-fA-F]+)')
re_tB   = re.compile(r'MATMUL_DESC_TRANSB.*bufData=0x([0-9a-fA-F]+)')
re_miss = re.compile(r'rocblaslt_matmul_algo_get_heuristic.*returnAlgoCount=(\d+)')
tA = tB = None
layouts = []
seen = {}
if not os.path.exists(log):
    sys.exit("log file not found")
with open(log) as f:
    for ln in f:
        m = re_tA.search(ln);  ta = int(m.group(1),16) if m else None
        m = re_tB.search(ln);  tb = int(m.group(1),16) if m else None
        if ta is not None: tA = ta
        if tb is not None: tB = tb
        m = re_lay.search(ln)
        if m:
            layouts.append((int(m.group(1)), int(m.group(2))))
            if len(layouts) > 3: layouts = layouts[-3:]
        m = re_miss.search(ln)
        if m and int(m.group(1)) == 0 and len(layouts) >= 3:
            a, b, c = layouts[-3:]
            M, N = c
            K = a[0] if a[1] == M else a[1] if a[0] == M else max(a)
            op = lambda x: 'N' if x == 0x6f else 'T' if x == 0x70 else f'0x{x:02x}' if x else '?'
            key = (M, N, K, op(tA), op(tB))
            seen[key] = seen.get(key, 0) + 1
            layouts = []
for (M,N,K,a,b), n in sorted(seen.items()):
    print(f"  MISS x{n}  M={M:<5} N={N:<5} K={K:<5} tA={a} tB={b}")
sys.exit(0 if not seen else 1)
PY
)
   RC_PARSE=$?

   echo "${MISSES}"

   if [ ${RC_PARSE} -eq 0 ]; then
      echo "REGRESSION CHECK [dtype=${DTYPE}]: PASSED  (zero heuristic misses across the sweep)"
      return 0
   else
      echo "REGRESSION CHECK [dtype=${DTYPE}]: FAILED  (heuristic misses listed above)"
      echo "  -> rebuild / extend the hipblaslt/patched overlay for these shapes."
      echo "  -> patcher: hpctd repo, rocm/scripts/hipblaslt_patch_setup.sh"
      return 1
   fi
}

# ─────────────────────────────────────────────────────────────────────
# Phase 2: transpose-layout / dispatch TIMING sweep.
#
# The per-dtype sweep above is a LOG-based detector: it flags only
# `returnAlgoCount=0` heuristic misses from nn.Linear (which emits a
# single transpose layout, NT). It is blind to a whole class of
# dispatch regression that the 2026-11 MI300A investigation surfaced:
#
#   * The generic (no-CU) gfx942 fallback libraries ship solutions whose
#     workGroupMappingXCC (4/8/16/32) does not divide the partition CU
#     count (228 SPX / 38 CPX). The runtime predicate rejects them, the
#     heuristic search empties, the empty result is NOT cached, and the
#     full search is paid on EVERY call -> a flat ~400-600 ms that is
#     independent of GEMM size, spans fp16/bf16/fp32, and -- critically --
#     appears on transpose layouts nn.Linear never produces (mm/bmm
#     NN/TN/TT). A log-miss counter can also UNDER-report it when the
#     slow search eventually returns a (valid) solution.
#
# So this phase measures wall-clock dispatch time directly, across the
# full transpose matrix and both dtypes/sizes (this is the content of
# check_fp16_gemm.py, Daniele Bagni / HLRS, 2026-11), and FAILS on any
# cell slower than ${SLOW_MS} ms. It is architecture-agnostic: a slow
# cell is a regression on whatever partition the run lands on.
#
# Logging is deliberately unset for this phase so the API-log overhead
# does not perturb the timings.
run_layout_timing_sweep() {
   echo
   echo "=========================================================="
   echo "  PHASE 2: layout/dispatch TIMING sweep  (slow > ${SLOW_MS} ms)"
   echo "=========================================================="
   (
      unset HIPBLASLT_LOG_LEVEL HIPBLASLT_LOG_FILE
      python3 - "${SLOW_MS}" <<'PY'
import os, sys, time, statistics
import torch

slow_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
if not torch.cuda.is_available():
    print("  no GPU available, skipping", file=sys.stderr)
    sys.exit(77)

DTYPES = [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]
# Part 1: the OPT-125M fused-QKV projection (where this first surfaced
# under vLLM) plus one large shape to show big GEMMs are unaffected.
CASES = [
    ("small skinny  x[1,768] @ w[2304,768]", 1, 768, 2304),
    ("small batched x[8,768] @ w[2304,768]", 8, 768, 2304),
    ("large         x[4096,4096] @ w[4096,4096]", 4096, 4096, 4096),
]
# Part 2: transpose layouts. The defect is NOT dtype-bound and NOT
# shape-bound -- it is selected by transpose layout, so every op/layout
# combination must be probed.
LAYOUT_SHAPES = [(256, 256, 128), (1024, 1024, 1024)]
BATCH = 8
WARMUP, REPEATS = 5, 10
dev = torch.device("cuda")

print(f"  device   = {torch.cuda.get_device_name(0)}")
print(f"  torch    = {torch.__version__}  hip = {torch.version.hip}")
print(f"  libpath  = {os.environ.get('HIPBLASLT_TENSILE_LIBPATH', '(unset)')}")
print(f"  overlay  = {os.environ.get('HIPBLASLT_OVERLAY', '(unset)')}")

def measure(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(REPEATS):
        torch.cuda.synchronize(); t = time.perf_counter()
        fn(); torch.cuda.synchronize()
        s.append((time.perf_counter() - t) * 1000.0)
    return min(s), statistics.median(s)

def measure_adaptive(fn):
    """measure(), but cheap when a single call already costs ~0.5 s."""
    fn(); torch.cuda.synchronize()
    t = time.perf_counter(); fn(); torch.cuda.synchronize()
    warm, rep = (1, 3) if (time.perf_counter() - t) > 0.05 else (WARMUP, REPEATS)
    for _ in range(warm):
        fn()
    s = []
    for _ in range(rep):
        torch.cuda.synchronize(); t = time.perf_counter()
        fn(); torch.cuda.synchronize()
        s.append((time.perf_counter() - t) * 1000.0)
    return min(s), statistics.median(s)

def layout_ops(m, n, k, dtype, batch):
    def t(*shape):
        return torch.randn(*shape, dtype=dtype, device=dev)
    a_n, a_t = t(m, k), t(k, m)
    b_n, b_t = t(k, n), t(n, k)
    ba_n, ba_t = t(batch, m, k), t(batch, k, m)
    bb_n, bb_t = t(batch, k, n), t(batch, n, k)
    return [
        ("mm", "NN", lambda: torch.mm(a_n, b_n)),
        ("mm", "NT", lambda: torch.mm(a_n, b_t.t())),
        ("mm", "TN", lambda: torch.mm(a_t.t(), b_n)),
        ("mm", "TT", lambda: torch.mm(a_t.t(), b_t.t())),
        ("bmm", "NN", lambda: torch.bmm(ba_n, bb_n)),
        ("bmm", "NT", lambda: torch.bmm(ba_n, bb_t.transpose(1, 2))),
        ("bmm", "TN", lambda: torch.bmm(ba_t.transpose(1, 2), bb_n)),
        ("linear", "NT", lambda: torch.nn.functional.linear(a_n, b_t, None)),
    ]

torch.manual_seed(0)
slow = []

print()
print(f"  {'case':<44} {'dtype':<6} {'min':>11} {'median':>11}")
print("  " + "-" * 72)
for label, m, k, n in CASES:
    for dname, dtype in DTYPES:
        x = torch.randn(m, k, dtype=dtype, device=dev)
        w = torch.randn(n, k, dtype=dtype, device=dev)
        lo, med = measure(lambda: torch.nn.functional.linear(x, w, None))
        mark = "  <== SLOW" if med > slow_ms else ""
        print(f"  {label:<44} {dname:<6} {lo:9.3f}ms {med:9.3f}ms{mark}")
        if mark:
            slow.append(f"{label} {dname}")
        del x, w
        torch.cuda.empty_cache()

print()
print(f"  Part 2: transpose layouts. MxNxK, bmm batch {BATCH}.")
print(f"  {'shape':<16} {'op':<7} {'layout':<7} {'dtype':<6} {'min':>11} {'median':>11}")
print("  " + "-" * 72)
for m, n, k in LAYOUT_SHAPES:
    shape = f"{m}x{n}x{k}"
    for dname, dtype in DTYPES:
        for op, layout, fn in layout_ops(m, n, k, dtype, BATCH):
            lo, med = measure_adaptive(fn)
            mark = "  <== SLOW" if med > slow_ms else ""
            print(f"  {shape:<16} {op:<7} {layout:<7} {dname:<6} {lo:9.3f}ms {med:9.3f}ms{mark}")
            if mark:
                slow.append(f"{shape} {op} {layout} {dname}")
        torch.cuda.empty_cache()
    print()

ncells = len(CASES) * len(DTYPES) + len(LAYOUT_SHAPES) * len(DTYPES) * 8
print(f"  Slow cells (median > {slow_ms:g} ms): {len(slow)} of {ncells}")
for s in slow:
    print("     slow:", s)
sys.exit(0 if not slow else 1)
PY
   )
}

# Main dtype loop: comma-separated DTYPES_RAW -> "fp16 fp32" etc.
IFS=',' read -r -a DTYPES_ARR <<< "${DTYPES_RAW}"
ALL_SKIP=1
for DT in "${DTYPES_ARR[@]}"; do
   DT_TRIM=$(echo "${DT}" | tr -d '[:space:]')
   case "${DT_TRIM}" in
      fp16|fp32|bf16|fp64) ;;
      *) echo "REGRESSION CHECK: unknown dtype '${DT_TRIM}' (expected one of: fp16, fp32, bf16, fp64)"
         OVERALL_FAIL=1
         PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  ${DT_TRIM}: BAD_DTYPE\n"
         continue;;
   esac
   run_dtype_sweep "${DT_TRIM}"
   RC=$?
   case ${RC} in
      0)  PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  ${DT_TRIM}: PASS\n"
          ALL_SKIP=0;;
      77) PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  ${DT_TRIM}: SKIP (no GPU)\n";;
      *)  PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  ${DT_TRIM}: FAIL\n"
          OVERALL_FAIL=1
          ALL_SKIP=0;;
   esac
done

# Phase 2: layout/dispatch timing sweep (complements the per-dtype log sweep).
run_layout_timing_sweep
LAYOUT_RC=$?
case ${LAYOUT_RC} in
   0)  PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  layout-timing: PASS\n"
       ALL_SKIP=0;;
   77) PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  layout-timing: SKIP (no GPU)\n";;
   *)  PER_DTYPE_RESULT="${PER_DTYPE_RESULT}  layout-timing: FAIL\n"
       echo "REGRESSION CHECK [layout-timing]: FAILED  (slow dispatch cells listed above)"
       echo "  -> likely the generic-fallback workGroupMappingXCC divisibility defect."
       echo "  -> fix: hpctd repo, rocm/scripts/hipblaslt_patch_setup.sh (general-xcc path)."
       OVERALL_FAIL=1
       ALL_SKIP=0;;
esac

echo
echo "=========================================================="
echo "  REGRESSION CHECK summary (dtypes = ${DTYPES_RAW})"
echo "=========================================================="
echo -en "${PER_DTYPE_RESULT}"
echo

if [ "${ALL_SKIP}" -eq 1 ]; then
   echo "REGRESSION CHECK: SKIPPED  (every dtype reported no-GPU)"
   exit 77
fi
if [ ${OVERALL_FAIL} -eq 0 ]; then
   echo "REGRESSION CHECK: PASSED  (zero heuristic misses across every dtype)"
   exit 0
else
   echo "REGRESSION CHECK: FAILED  (one or more dtypes had heuristic misses; see per-dtype list above)"
   exit 1
fi
