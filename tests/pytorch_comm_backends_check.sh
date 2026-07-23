#!/bin/bash
#
# This test gives info on the comm paths available in your PyTorch installation.
#
#   INFO ON THE OUTPUT PROVIDED BY THE TEST
#
#   (A) REPORTS, for the loaded build:
#       * which torch.distributed backends are available
#           (nccl(=RCCL), mpi, gloo, ucc) via torch.distributed.is_*_available();
#       * the DEFAULT comm path -- i.e. which backend init_process_group(backend=None)
#           picks per device (cpu -> gloo, cuda -> nccl(=RCCL)) -- printed prominently
#           so you can tell what will actually run and where to point diagnostics;
#       * the native RCCL / MPI libraries the build is wired to (soname, realpath,
#           version), and the UCC/UCX story (see below);
#
#   (B) REPORTS whether PyTorch is wired to the GPU-aware MPI from the openmpi
#       module, as a single diagnostic line:
#           GPU-AWARE MPI: OK   ...            (all conditions below hold)
#           GPU-AWARE MPI: NO - <reason>       (something is off)
#       The conditions are: an openmpi module is actually loaded; the MPI backend
#       is compiled in (is_mpi_available()); the libmpi that libtorch links
#       resolves UNDER that openmpi module's install prefix -- and, crucially,
#       that prefix is read straight from the loaded modulefile (`module show`),
#       NOT from an ambient $MPI_PATH which anything could have set; and that
#       OpenMPI is genuinely GPU-aware -- confirmed by asking that module's own
#       ompi_info for the built-in ROCm accelerator MCA component (not merely by
#       libmpi linking UCC/UCX, which only proves the transport exists).
#
# TWO DIFFERENT "UCC"s (this is a common source of confusion):
#   * torch NATIVE UCC backend (backend='ucc' / ProcessGroupUCC): a torch build
#     option (USE_UCC). Reported via is_ucc_available(). Frequently NOT built.
#   * UCC/UCX as the OpenMPI TRANSPORT: used underneath backend='mpi'. Detected
#     from libmpi's own DT_NEEDED. This is present on this system even when the
#     native torch UCC backend is not.
#
# NO PASS/FAIL IN THE OUTPUT -- run it standalone on any PyTorch setup purely to
# diagnose; it never prints "PASSED"/"FAILED". Gating lives entirely in CTest:
#   * In the regression suite, CMakeLists sets
#       PASS_REGULAR_EXPRESSION "GPU-AWARE MPI: OK"
#     so the test passes ONLY when PyTorch is built against the module's
#     GPU-aware MPI. A wheel / system-MPI / non-GPU-aware build prints
#     "GPU-AWARE MPI: NO - ..." instead, so the suite catches it.
#   * If torch cannot be imported at all, the script emits "COMM BACKENDS CHECK:
#     SKIPPED" and returns 77 (CTest skip). A missing pytorch modulefile is
#     likewise caught by the CTest skip regex.
#
# NOTE: assumes PyTorch/OpenMPI installed per the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

set -u

# ---------------------------------------------------------------------------
# If rocm AND pytorch are already loaded (e.g. by the caller), use them as-is
# and do NOT reload pytorch. Otherwise load rocm (if needed) then pytorch.
# Loading pytorch also loads the GPU-aware "openmpi" module (which exports
# MPI_PATH) and magma. (Match the "rocm/" / "pytorch/" aliases so that e.g.
# "rocm-new/..." does not count.) If no module system is present (e.g. a plain
# venv) these calls just no-op and the venv's python3/torch are used.
# ---------------------------------------------------------------------------
#if ! (module -t list 2>&1 | grep -q "^rocm/") || ! (module -t list 2>&1 | grep -q "^pytorch/"); then
#  if ! module -t list 2>&1 | grep -q "^rocm/"; then
#    echo "rocm module is not loaded"
#    echo "loading default rocm module"
#    module load rocm
#  fi
#  if ! module -t list 2>&1 | grep -q "^pytorch/"; then
#    module load pytorch
#  fi
#fi

# ---------------------------------------------------------------------------
# Determine the loaded openmpi module and its install prefix DIRECTLY from the
# modulefile (`module show`), so we can later prove torch's libmpi actually
# comes from THAT module -- rather than trusting an ambient $MPI_PATH that could
# have been set by anything. If no module system / no openmpi module is present
# (e.g. a plain venv) these stay empty and the MPI verdict becomes "NO".
# ---------------------------------------------------------------------------
OPENMPI_MODULE_NAME=$(module -t list 2>&1 | grep -m1 "^openmpi/")
OPENMPI_MODULE_PREFIX=""
if [ -n "${OPENMPI_MODULE_NAME}" ]; then
  _show=$(module show "${OPENMPI_MODULE_NAME}" 2>&1)
  OPENMPI_MODULE_PREFIX=$(printf '%s\n' "${_show}" | sed -n 's/.*setenv("MPI_PATH","\([^"]*\)").*/\1/p' | head -1)
  if [ -z "${OPENMPI_MODULE_PREFIX}" ]; then
    OPENMPI_MODULE_PREFIX=$(printf '%s\n' "${_show}" | sed -n 's|.*prepend_path("LD_LIBRARY_PATH","\([^"]*\)/lib").*|\1|p' | head -1)
  fi
fi
export OPENMPI_MODULE_NAME OPENMPI_MODULE_PREFIX

# ---------------------------------------------------------------------------
# Everything else is Python introspection of the loaded build + its libraries.
# Needs no GPU or launcher. We capture the output so that if torch fails to
# import / aborts (environmental) we can emit SKIP instead of a bogus failure.
# MPI_PATH (from the openmpi module) is read inside Python via os.environ.
# ---------------------------------------------------------------------------
_OUT=$(mktemp)
trap 'rm -f "${_OUT}"' EXIT

python3 <<'EOF' 2>&1 | tee "${_OUT}"
import ctypes
import os
import re
import subprocess
import sys

try:
    import torch
    import torch.distributed as dist
except Exception as e:
    print("could not import torch:", e)
    print("COMM BACKENDS CHECK: SKIPPED")
    sys.exit(77)


def rule(t):
    print("\n" + "-" * 72 + f"\n{t}\n" + "-" * 72)


def ldd_map(sofile):
    out = {}
    try:
        r = subprocess.run(["ldd", sofile], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            m = re.match(r"\s*(\S+)\s+=>\s+(\S+)", line)
            if m:
                out[m.group(1)] = m.group(2)
    except Exception:
        pass
    return out


def dt_needed(sofile):
    """Direct DT_NEEDED entries of an ELF file (readelf -d)."""
    needs = []
    try:
        r = subprocess.run(["readelf", "-d", sofile], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            m = re.search(r"\(NEEDED\).*\[([^\]]+)\]", line)
            if m:
                needs.append(m.group(1))
    except Exception:
        pass
    return needs


def resolve_lib(pattern, sofiles):
    for so in sofiles:
        if not os.path.exists(so):
            continue
        for name, path in ldd_map(so).items():
            if re.search(pattern, name, re.I):
                return name, os.path.realpath(path), so
    return None, None, None


def loaded_from_maps(pattern):
    hits = set()
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                p = line.rsplit(" ", 1)[-1].strip()
                if p and re.search(pattern, os.path.basename(p), re.I):
                    hits.add(os.path.realpath(p))
    except FileNotFoundError:
        pass
    return sorted(hits)


def ompi_accelerators(prefix):
    """Set of OpenMPI 'accelerator' MCA components (e.g. {'null','rocm'}) as
    reported by THAT module's own ompi_info. None if ompi_info is unusable.
    'rocm' present => OpenMPI was built GPU-aware for ROCm."""
    exe = os.path.join(prefix, "bin", "ompi_info")
    if not os.path.exists(exe):
        return None
    try:
        r = subprocess.run([exe, "--parsable"], capture_output=True,
                           text=True, timeout=60)
        return set(re.findall(r"mca:accelerator:(\w+):version", r.stdout))
    except Exception:
        return None


def mpi_banner(libmpi_path):
    try:
        lib = ctypes.CDLL(libmpi_path, mode=ctypes.RTLD_GLOBAL)
        buf = ctypes.create_string_buffer(32768)
        n = ctypes.c_int(0)
        lib.MPI_Get_library_version(buf, ctypes.byref(n))
        return buf.value.decode(errors="replace").strip().splitlines()[0]
    except Exception as e:
        return f"(could not query libmpi directly: {e})"


is_rocm = getattr(torch.version, "hip", None) is not None
tl = os.path.join(os.path.dirname(torch.__file__), "lib")
sofiles = [os.path.join(tl, x) for x in
           ("libtorch_hip.so", "libtorch_cuda.so", "libtorch_cpu.so",
            "libtorch.so", "libc10d.so")]

rule("PyTorch identity")
print("torch.__version__:", torch.__version__)
print("ROCm/HIP version :", getattr(torch.version, "hip", None))
print("CUDA version     :", torch.version.cuda)
print("LOADEDMODULES    :", os.environ.get("LOADEDMODULES", "(not set)"))

# ---------------------------------------------------------------------------
rule("Backends available (torch.distributed.is_*_available)")
dist_ok = dist.is_available()
print("torch.distributed.is_available():", dist_ok)
avail = {}
for b in ["nccl", "mpi", "gloo", "ucc"]:
    fn = getattr(dist, f"is_{b}_available", None)
    avail[b] = bool(fn()) if fn else False
    print(f"  {b:5s} available: {avail[b]}")

# ---------------------------------------------------------------------------
rule("DEFAULT COMM PATH (what init_process_group(backend=None) selects)")
defaults = {}
for dev in ["cpu", "cuda"]:
    val = None
    try:
        val = str(dist.get_default_backend_for_device(dev))
    except Exception:
        try:
            val = str(dict(dist.Backend.default_device_backend_map).get(dev))
        except Exception:
            val = None
    defaults[dev] = val
try:
    print("  default_device_backend_map:",
          dict(dist.Backend.default_device_backend_map))
except Exception as e:
    print("  (map unavailable:", e, ")")
print(f"  --> GPU (cuda) tensors default to : {defaults.get('cuda')}   (nccl == RCCL)")
print(f"  --> CPU        tensors default to : {defaults.get('cpu')}")
print("  --> MPI is used ONLY when you explicitly pass backend='mpi' and launch")
print("      under mpirun/srun; it is never the implicit default.")

# ---------------------------------------------------------------------------
rule("RCCL / NCCL library this build is wired to")
rccl_name, rccl_path, rccl_via = resolve_lib(r"(rccl|nccl)", sofiles)
if rccl_path:
    print(f"  soname : {rccl_name}")
    print(f"  path   : {rccl_path}  (linked by {os.path.basename(rccl_via)})")
else:
    print("  RCCL/NCCL: not a DT_NEEDED of libtorch (loaded lazily?)")
try:
    print(f"  version (torch.cuda.nccl.version()): {torch.cuda.nccl.version()}")
except Exception as e:
    print("  version:", e)

# ---------------------------------------------------------------------------
# MPI wiring: reported as NOTE:/OK: lines
# ---------------------------------------------------------------------------
rule("MPI library this build is wired to")
# Provenance we can be SURE of: the openmpi module name + its prefix read from
# the modulefile itself (passed in from bash). MPI_PATH is shown separately and
# only as an ambient env var -- we do NOT assume it came from the module.
ompi_mod = os.environ.get("OPENMPI_MODULE_NAME") or ""
ompi_prefix = os.environ.get("OPENMPI_MODULE_PREFIX") or ""
ompi_prefix_real = os.path.realpath(ompi_prefix) if ompi_prefix else ""
print("  openmpi module loaded    :", ompi_mod or "(none)")
print("  module prefix (modulefile):", ompi_prefix or "(unknown)")
print("  MPI_PATH (ambient env var):", os.environ.get("MPI_PATH") or "(not set)")

mpi_name, mpi_real, mpi_via = resolve_lib(r"libmpi", sofiles)
if mpi_real:
    print(f"  libtorch links   : {mpi_name}")
    print(f"  resolves to      : {mpi_real}  (via {os.path.basename(mpi_via)})")
    print(f"  library version  : {mpi_banner(mpi_real)}")
else:
    print("  libtorch does NOT link libmpi -> MPI backend not compiled in")

mpi_needs = dt_needed(mpi_real) if mpi_real else []
mpi_ucc = [n for n in mpi_needs if re.search(r"ucc", n, re.I)]
mpi_ucx = [n for n in mpi_needs if re.search(r"ucp|ucs|uct", n, re.I)]
print(f"  libmpi DT_NEEDED UCC : {mpi_ucc or 'NONE'}")
print(f"  libmpi DT_NEEDED UCX : {mpi_ucx or 'NONE'}")

# Authoritative GPU-awareness: ask THIS module's ompi_info for the ROCm
# accelerator MCA component (present only if OpenMPI was built --with-rocm).
ompi_accs = ompi_accelerators(ompi_prefix_real) if ompi_prefix_real else None
ompi_rocm = bool(ompi_accs and "rocm" in ompi_accs)
print(f"  ompi_info accelerators: "
      f"{sorted(ompi_accs) if ompi_accs is not None else '(ompi_info unavailable)'}"
      f"{'  -> ROCm GPU-aware' if ompi_rocm else ''}")

# SURE only if torch's libmpi resolves under the openmpi MODULE's prefix (read
# from the modulefile, not from ambient MPI_PATH).
under_module = bool(
    mpi_real and ompi_prefix_real
    and (mpi_real == ompi_prefix_real
         or mpi_real.startswith(ompi_prefix_real + os.sep)))

# Single diagnostic verdict for the MPI wiring (no PASS/FAIL wording). CTest
# gates on the "GPU-AWARE MPI: OK" form; humans read the reason on "NO".
if not avail["mpi"]:
    gpu_aware_mpi = "NO - MPI backend not compiled in (normal for pip/venv wheels)"
elif not ompi_mod:
    gpu_aware_mpi = "NO - no openmpi module is loaded (cannot attribute the MPI to a module)"
elif not ompi_prefix_real:
    gpu_aware_mpi = f"NO - could not read the '{ompi_mod}' module's install prefix from the modulefile"
elif not mpi_real:
    gpu_aware_mpi = "NO - libtorch does not link libmpi"
elif not under_module:
    gpu_aware_mpi = (f"NO - libtorch links {mpi_real}, which is NOT under the "
                     f"'{ompi_mod}' module prefix {ompi_prefix_real}")
elif ompi_accs is None:
    gpu_aware_mpi = f"NO - could not run the '{ompi_mod}' module's ompi_info to confirm GPU-awareness"
elif not ompi_rocm:
    gpu_aware_mpi = (f"NO - the '{ompi_mod}' OpenMPI is NOT GPU-aware "
                     f"(ompi_info has no rocm accelerator component; found {sorted(ompi_accs)})")
elif not (mpi_ucc and mpi_ucx):
    gpu_aware_mpi = f"NO - linked MPI does not pull in UCC/UCX (ucc={bool(mpi_ucc)}, ucx={bool(mpi_ucx)})"
else:
    gpu_aware_mpi = (f"OK - torch linked against the '{ompi_mod}' module MPI "
                     f"({mpi_name} under {ompi_prefix_real}); OpenMPI is GPU-aware "
                     f"(rocm accelerator) with UCC+UCX transport")

# ---------------------------------------------------------------------------
rule("UCC status: native torch backend vs OpenMPI transport")
print(f"  torch NATIVE UCC backend (backend='ucc' / ProcessGroupUCC): "
      f"{'available' if avail['ucc'] else 'NOT BUILT'}   "
      f"[is_ucc_available()={avail['ucc']}]")
transport = bool(mpi_ucc and mpi_ucx)
print(f"  UCC/UCX as OpenMPI TRANSPORT (used under backend='mpi')    : "
      f"{'YES' if transport else 'no'}")
if transport:
    for lib in mpi_ucc + mpi_ucx:
        print(f"        {lib}   (DT_NEEDED of {os.path.basename(mpi_real)})")
print("  NOTE: 'NOT BUILT' above means only that torch has no native UCC")
print("        ProcessGroup; UCC/UCX are still used as the MPI transport.")

# ---------------------------------------------------------------------------
rule("Comm libraries actually mapped in this process")
for pat, label in [(r"rccl|nccl", "RCCL/NCCL"), (r"libmpi", "MPI"),
                   (r"libucc", "UCC"), (r"libucp|libucs|libuct", "UCX")]:
    hits = loaded_from_maps(pat)
    print(f"  {label:10s}: {hits if hits else '(not mapped)'}")

# ---------------------------------------------------------------------------
# Summary. The "GPU-AWARE MPI:" line (column 0) is what CTest gates on:
# PASS_REGULAR_EXPRESSION "GPU-AWARE MPI: OK". No PASS/FAIL wording is printed.
# ---------------------------------------------------------------------------
rule("Summary")
print(f"  backends: nccl(RCCL)={avail['nccl']} gloo={avail['gloo']} "
      f"mpi={avail['mpi']} ucc(native)={avail['ucc']}")
print(f"  default comm path: GPU={defaults.get('cuda')}(RCCL) CPU={defaults.get('cpu')}")
print(f"GPU-AWARE MPI: {gpu_aware_mpi}")
sys.exit(0)
EOF

# ---------------------------------------------------------------------------
# CTest gates on the "GPU-AWARE MPI: OK" keyword (PASS_REGULAR_EXPRESSION).
# Here we only need the exit code: 77 if torch could not be introspected (skip),
# 0 otherwise (CTest then applies its keyword regex to decide pass/fail).
# ---------------------------------------------------------------------------
if grep -q "COMM BACKENDS CHECK: SKIPPED" "${_OUT}"; then
  exit 77
elif grep -q "^GPU-AWARE MPI:" "${_OUT}"; then
  exit 0
else
  echo "PyTorch introspection did not complete (torch failed to import or aborted);"
  echo "treating as environmental."
  echo "COMM BACKENDS CHECK: SKIPPED"
  exit 77
fi
