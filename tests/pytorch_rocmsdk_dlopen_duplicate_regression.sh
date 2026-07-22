#!/bin/bash
#
# Regression test for AMD's split "rocm-sdk" pip PyTorch wheels shipping only
# VERSIONED shared libraries (e.g. libamdhip64.so.7, libhsa-runtime64.so.1)
# with no UNVERSIONED "dev" symlink (libamdhip64.so, libhsa-runtime64.so).
#
# BACKGROUND
# ----------
# Several system components dlopen() ROCm shared libraries by their
# unversioned name at runtime rather than linking against them directly --
# e.g. HPE Cray's libfabric CXI provider dlopen()s "libhsa-runtime64.so" (no
# version suffix) to pull in hsa_memory_copy/hsa_amd_memory_async_copy for
# GPU-direct memory registration. Classic, monolithic ROCm PyTorch wheels
# (rocm <= 6.4) ship both the versioned file AND the unversioned dev symlink,
# so that dlopen() call resolves back to the SAME shared object PyTorch
# itself is already using.
#
# AMD's newer split "rocm-sdk" pip wheels (_rocm_sdk_core,
# _rocm_sdk_libraries_gfx90a, ...) only ship the versioned file. If an
# unversioned symlink is not added next to it, any dlopen() of the
# unversioned name falls through the loader's search path and can resolve to
# a DIFFERENT installed copy of the library elsewhere on the system (e.g. a
# system-wide ROCm install earlier/later in LD_LIBRARY_PATH). That loads a
# SECOND, independent runtime instance (HIP/HSA context) into the same
# process. Since HSA runtime tracks GPU pointer/agent state internally, a
# device pointer allocated by PyTorch under the first instance is invalid
# under the second, silently corrupting anything that instance touches --
# e.g. GPU memory registration for RDMA fails with "Bad address", or worse,
# silent data corruption.
#
# This test does NOT require any network hardware/plugin to reproduce the
# defect: it forces HIP/HSA runtime init via a real GPU allocation, then
# simulates the kind of unversioned dlopen() call an external library would
# make, and asserts that doing so does not introduce a second, distinct copy
# of any ROCm shared library PyTorch has already loaded.
#
# It runs two sub-tests and reports PASS only if BOTH pass:
#   1. symlink-exists - every ROCm shared library PyTorch has already loaded
#                        from a "rocm_sdk" pip package has a same-directory
#                        unversioned symlink pointing back to the exact file
#                        already loaded.
#   2. dlopen-no-dup  - dlopen()-ing each such library by its unversioned
#                        name does not add a new, distinct realpath to the
#                        process's memory map (i.e. it resolves back to the
#                        library already loaded, not a duplicate elsewhere).
#
# This complements pytorch_rccl_distributed_regression.sh, which checks that
# collectives themselves succeed, by specifically catching the underlying
# "duplicate ROCm runtime instance from an unversioned dlopen()" defect class
# -- a bug that can silently reappear with future pip rocm-sdk wheel updates.
#
# Requirements: a node with at least 1 visible GPU. Run directly on a GPU
# node:
#     ./pytorch_rocmsdk_dlopen_duplicate_regression.sh
#
# NOTE: assumes PyTorch was installed per the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

set -u

export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Load a rocm module if none is loaded, then pytorch -- but only load the
# DEFAULT of either if the caller has not already loaded a specific one.
# This lets the test be invoked as-is (loads the site defaults) or after
# the caller has already done e.g. `module load rocm/6.3.0 pytorch/2.7.1`
# to target a specific combination: an unconditional `module load pytorch`
# would otherwise silently swap a pinned, non-default pytorch back to the
# default for whatever rocm tree is loaded.
# (Match the "rocm/" alias explicitly so "rocm-new/..." does not count.)
# ---------------------------------------------------------------------------
if ! module -t list 2>&1 | grep -q "^rocm/"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
if ! module -t list 2>&1 | grep -q "^pytorch/"; then
  echo "pytorch module is not loaded"
  echo "loading default pytorch module"
  module load pytorch
fi

# ---------------------------------------------------------------------------
# Skip (CTest exit code 77) unless the allocated node exposes >= 1 GPU.
# ---------------------------------------------------------------------------
NGPU=$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)
NGPU=${NGPU:-0}
echo "visible GPUs: ${NGPU}"
if [ "${NGPU}" -lt 1 ]; then
  echo "REGRESSION RESULT: SKIPPED (need >= 1 visible GPU, found ${NGPU})"
  exit 77
fi

# ---------------------------------------------------------------------------
# Inline the python programs into a scratch dir.
# ---------------------------------------------------------------------------
WORKDIR=$(mktemp -d)
trap 'rm -rf "${WORKDIR}"' EXIT

# Shared helper: force real HIP/HSA runtime init, then scan /proc/self/maps
# for shared objects loaded from an AMD pip "rocm_sdk" package directory.
#
# We intentionally restrict this to a curated allowlist of libraries known to
# be dlopen()d by their UNVERSIONED name by components outside PyTorch itself
# (e.g. network fabric providers, MPI GPU-transport layers). Normal linking
# (DT_NEEDED + SONAME) never needs the unversioned symlink, so checking every
# rocm-sdk library indiscriminately would flag many libraries that are not
# actually at risk -- and, worse, actually dlopen()-ing some of them a second
# time (e.g. the LLVM/MIOpen JIT compiler stack) aborts the process outright
# via unrelated global-singleton state (LLVM CommandLine option registry),
# which is a different failure mode than the one this test targets. Extend
# this set if other externally-dlopen()d libraries are identified.
COMMON='
import ctypes
import os
import sys

import torch

EXTERNAL_DLOPEN_TARGETS = {"libamdhip64.so", "libhsa-runtime64.so", "librccl.so"}

# Force real HIP/HSA runtime init (context creation, agent enumeration, and
# memory pool setup) -- a bare "import torch" is not enough to reproduce this.
_ = torch.zeros(1, device="cuda")
torch.cuda.synchronize()


def unversioned_name(base):
    # e.g. libamdhip64.so.7 -> libamdhip64.so ; libhsa-runtime64.so.1.21.0 -> libhsa-runtime64.so
    idx = base.find(".so")
    return base if idx == -1 else base[: idx + 3]


def rocm_sdk_libs():
    """{(dir, unversioned_name): realpath} for every EXTERNAL_DLOPEN_TARGETS
    library, from a pip rocm-sdk package, currently mapped into this
    process. Used to check symlink hygiene of OUR pip install."""
    libs = {}
    with open(f"/proc/{os.getpid()}/maps") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            path = parts[-1]
            if "rocm_sdk" not in path or ".so" not in path:
                continue
            real = os.path.realpath(path)
            d, base = os.path.split(real)
            uname = unversioned_name(base)
            if uname not in EXTERNAL_DLOPEN_TARGETS:
                continue
            libs[(d, uname)] = real
    return libs


def realpaths_for(uname):
    """All distinct realpaths currently mapped ANYWHERE in this process
    (not just rocm_sdk paths -- a duplicate can come from a system ROCm
    install too) whose basename is uname itself or a versioned variant of
    it (e.g. uname="libamdhip64.so" matches "libamdhip64.so.7")."""
    paths = set()
    with open(f"/proc/{os.getpid()}/maps") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            p = parts[-1]
            base = os.path.basename(p)
            if base == uname or base.startswith(uname + "."):
                paths.add(os.path.realpath(p))
    return paths
'

cat > "${WORKDIR}/symlink_exists.py" <<PY
${COMMON}

libs = rocm_sdk_libs()
if not libs:
    print("WARNING: none of EXTERNAL_DLOPEN_TARGETS found mapped in this "
          "process from a rocm_sdk pip package -- is PyTorch built from "
          "AMD's pip rocm-sdk wheels?", flush=True)
    print("SUBTEST_RESULT: OK", flush=True)
    sys.exit(0)

ok = True
for (d, unversioned), real in sorted(libs.items()):
    link = os.path.join(d, unversioned)
    if not os.path.exists(link):
        print(f"FAIL: missing unversioned symlink {link} (loaded lib: {real})", flush=True)
        ok = False
        continue
    resolved = os.path.realpath(link)
    if resolved != real:
        print(f"FAIL: {link} resolves to {resolved}, expected {real}", flush=True)
        ok = False
        continue
    print(f"OK: {link} -> {real}", flush=True)

print(f"SUBTEST_RESULT: {'OK' if ok else 'FAIL'}", flush=True)
if not ok:
    sys.exit(1)
PY

cat > "${WORKDIR}/dlopen_no_duplicate.py" <<PY
${COMMON}

# Note: unlike symlink_exists.py, the before/after comparison below
# deliberately does NOT restrict itself to rocm_sdk paths -- a duplicate
# instance loaded because of this bug typically comes from a DIFFERENT
# location entirely (e.g. a system-wide ROCm install elsewhere in
# LD_LIBRARY_PATH / ld.so.cache), which is exactly what we need to catch.
before = {name: realpaths_for(name) for name in EXTERNAL_DLOPEN_TARGETS}
before = {name: paths for name, paths in before.items() if paths}
if not before:
    print("WARNING: none of EXTERNAL_DLOPEN_TARGETS found mapped in this "
          "process from a rocm_sdk pip package -- is PyTorch built from "
          "AMD's pip rocm-sdk wheels?", flush=True)
    print("SUBTEST_RESULT: OK", flush=True)
    sys.exit(0)

# Simulate what an external component (e.g. a network fabric provider) does
# when it dlopen()s a ROCm library by its UNVERSIONED name at runtime.
for name in sorted(before):
    try:
        ctypes.CDLL(name)
        print(f"dlopen({name}) succeeded", flush=True)
    except OSError as e:
        print(f"WARNING: dlopen({name}) failed: {e}", flush=True)

ok = True
for name, paths_before in before.items():
    paths_after = realpaths_for(name)
    new_paths = paths_after - paths_before
    if new_paths:
        print(f"FAIL: dlopen({name}) loaded {len(new_paths)} NEW duplicate "
              f"copy(ies) not already in the process: {sorted(new_paths)} "
              f"(already loaded: {sorted(paths_before)})", flush=True)
        ok = False
    else:
        print(f"OK: dlopen({name}) reused the already-loaded copy "
              f"({sorted(paths_before)})", flush=True)

print(f"SUBTEST_RESULT: {'OK' if ok else 'FAIL'}", flush=True)
if not ok:
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# Run the two sub-tests. PASS only if both pass.
# ---------------------------------------------------------------------------
overall=0

echo "### [1/2] unversioned dev symlinks exist next to pip rocm-sdk libraries"
if python3 "${WORKDIR}/symlink_exists.py"; then symlink_exists=PASS; else symlink_exists=FAIL; overall=1; fi

echo "### [2/2] dlopen() of the unversioned name does not create a duplicate runtime instance"
if python3 "${WORKDIR}/dlopen_no_duplicate.py"; then dlopen_no_dup=PASS; else dlopen_no_dup=FAIL; overall=1; fi

echo "----------------------------------------"
echo "symlink-exists : ${symlink_exists}"
echo "dlopen-no-dup  : ${dlopen_no_dup}"
if [ "${overall}" -eq 0 ]; then
  echo "REGRESSION RESULT: PASS"
  exit 0
else
  echo "REGRESSION RESULT: FAIL"
  exit 1
fi
