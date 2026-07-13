#!/bin/bash
#
# Regression test for the ROCm PyTorch distributed / RCCL collective path.
#
# It runs three sub-tests (the python programs are inlined below) and prints a
# single verdict. It reports PASS only if ALL THREE sub-tests pass:
#
#   1. gloo         - N-rank CPU all_reduce (control: rendezvous + CPU collective).
#   2. nccl-default - N-rank RCCL/NCCL GPU all_reduce, default intra-node transport.
#   3. nccl-nop2p   - N-rank RCCL/NCCL GPU all_reduce with NCCL_SOCKET_IFNAME=lo
#                     and NCCL_P2P_DISABLE=1 (the constrained repro configuration).
#
# HOW IT IS LAUNCHED / PROCESS PLACEMENT
# --------------------------------------
# The distributed sub-tests use N ranks (default 2, override with NRANKS).
# Placement is ONE RANK PER GPU (per GCD/XCD): each rank calls
# torch.cuda.set_device(local_rank), where local_rank comes from the launcher
# (OMPI_COMM_WORLD_LOCAL_RANK under mpirun, SLURM_LOCALID under srun). So rank i
# is pinned to logical device i -- there is no "everyone on device 0". This
# mirrors the original sbatch, which had --gres=gpu:2 (both devices visible) and
# did the actual per-rank binding in python via set_device(local_rank).
#
# Requirements: a node with at least NRANKS visible GPUs (GCDs/XCDs) and either
# mpirun (preferred) or srun available. Run it directly on a GPU node:
#     ./pytorch_rccl_distributed_regression.sh
# or inside a SLURM allocation with >= NRANKS tasks/GPUs, or with more ranks:
#     NRANKS=8 ./pytorch_rccl_distributed_regression.sh
#
# NOTE: assumes PyTorch was installed per the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

set -u

# ---------------------------------------------------------------------------
# Configuration (all overridable from the environment)
# ---------------------------------------------------------------------------
NRANKS=${NRANKS:-2}                       # ranks for the distributed sub-tests
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29542}
export PYTHONUNBUFFERED=1
# NCCL_DEBUG is intentionally NOT set: leaving it unset keeps RCCL quiet
# (setting it to WARN/INFO produces benign "alt_rsmi ... Could not read node"
# topology spam). Export NCCL_DEBUG=INFO yourself if you need to debug.

# ---------------------------------------------------------------------------
# Load a rocm module if none is loaded, then pytorch.
# (Match the "rocm/" alias explicitly so "rocm-new/..." does not count.)
# ---------------------------------------------------------------------------
if ! module -t list 2>&1 | grep -q "^rocm/"; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
module load pytorch

# ---------------------------------------------------------------------------
# Skip (CTest exit code 77) unless the allocated node exposes >= NRANKS GPUs.
# The distributed tests place one rank per GPU, so fewer visible devices than
# ranks cannot run (e.g. login node with no GPU, or gres too small).
# ---------------------------------------------------------------------------
NGPU=$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)
NGPU=${NGPU:-0}
echo "visible GPUs: ${NGPU} (ranks required: ${NRANKS})"
if [ "${NGPU}" -lt "${NRANKS}" ]; then
  echo "REGRESSION RESULT: SKIPPED (need ${NRANKS} visible GPUs, found ${NGPU})"
  exit 77
fi

# ---------------------------------------------------------------------------
# Inline the python programs into a scratch dir.
# ---------------------------------------------------------------------------
WORKDIR=$(mktemp -d)
trap 'rm -rf "${WORKDIR}"' EXIT

# Launcher-agnostic rank detection; sets MASTER_ADDR/PORT and expected sum.
DIST_PREAMBLE='
import os, sys, datetime

def env_int(names, default=None):
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return int(v)
    if default is None:
        raise KeyError(f"none of {names} set")
    return default

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29542")
rank = env_int(["OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK", "SLURM_PROCID"])
local_rank = env_int(["OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "LOCAL_RANK", "SLURM_LOCALID"], default=rank)
world = env_int(["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE", "SLURM_NTASKS"])
os.environ["RANK"] = str(rank)
os.environ["LOCAL_RANK"] = str(local_rank)
os.environ["WORLD_SIZE"] = str(world)
expected = world * (world + 1) / 2.0
'

# Common NCCL body: one rank per GPU, init, all_reduce, verify.
NCCL_BODY='
import torch
import torch.distributed as dist

ngpu = torch.cuda.device_count()
if local_rank >= ngpu:
    print(f"[rank {rank}] FAIL: local_rank={local_rank} >= visible GPUs={ngpu}", flush=True)
    sys.exit(1)
torch.cuda.set_device(local_rank)  # one rank per GCD/XCD
print(f"[rank {rank}] set_device device={local_rank} ngpu={ngpu}", flush=True)

dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=120))
x = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
ok = abs(x.item() - expected) < 1e-6
print(f"[rank {rank}] nccl all_reduce x={x.item()} device={local_rank} expected={expected} ok={ok}", flush=True)
dist.destroy_process_group()
if not ok:
    sys.exit(1)
if rank == 0:
    print("DISTRIBUTED_RESULT: OK", flush=True)
'

cat > "${WORKDIR}/gloo.py" <<PY
${DIST_PREAMBLE}
import torch
import torch.distributed as dist

dist.init_process_group("gloo", init_method="env://", rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=120))
x = torch.tensor([rank + 1.0], device="cpu")
dist.all_reduce(x)
ok = abs(x.item() - expected) < 1e-6
print(f"[rank {rank}] gloo all_reduce x={x.item()} expected={expected} ok={ok}", flush=True)
dist.destroy_process_group()
if not ok:
    sys.exit(1)
if rank == 0:
    print("DISTRIBUTED_RESULT: OK", flush=True)
PY

# Default-transport NCCL: no forced NCCL_SOCKET_IFNAME / NCCL_P2P_DISABLE.
cat > "${WORKDIR}/nccl_default.py" <<PY
${DIST_PREAMBLE}
${NCCL_BODY}
PY

# Constrained NCCL: force loopback socket + disable P2P (baked in so it does not
# depend on the launcher propagating env vars to the ranks).
cat > "${WORKDIR}/nccl_nop2p.py" <<PY
${DIST_PREAMBLE}
os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
${NCCL_BODY}
PY

# ---------------------------------------------------------------------------
# Pick a launcher for the distributed sub-tests: prefer mpirun, else srun.
# ---------------------------------------------------------------------------
if command -v mpirun >/dev/null 2>&1; then
  LAUNCH=(mpirun -np "${NRANKS}")
  echo "launcher: mpirun -np ${NRANKS}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  LAUNCH=(srun --nodes=1 --ntasks="${NRANKS}")
  echo "launcher: srun --ntasks=${NRANKS}"
else
  echo "ERROR: neither mpirun nor srun available to launch ${NRANKS} ranks" >&2
  echo "REGRESSION RESULT: FAIL"
  exit 1
fi

# ---------------------------------------------------------------------------
# Run the three sub-tests. PASS only if all three pass.
# ---------------------------------------------------------------------------
overall=0

echo "### [1/3] gloo CPU all_reduce (${NRANKS} ranks)"
if "${LAUNCH[@]}" python3 "${WORKDIR}/gloo.py"; then gloo=PASS; else gloo=FAIL; overall=1; fi

echo "### [2/3] nccl/rccl GPU all_reduce, default transport (${NRANKS} ranks, one rank per GPU)"
if "${LAUNCH[@]}" python3 "${WORKDIR}/nccl_default.py"; then nccl_default=PASS; else nccl_default=FAIL; overall=1; fi

echo "### [3/3] nccl/rccl GPU all_reduce, NCCL_SOCKET_IFNAME=lo + NCCL_P2P_DISABLE=1 (${NRANKS} ranks)"
if "${LAUNCH[@]}" python3 "${WORKDIR}/nccl_nop2p.py"; then nccl_nop2p=PASS; else nccl_nop2p=FAIL; overall=1; fi

echo "----------------------------------------"
echo "gloo         : ${gloo}"
echo "nccl-default : ${nccl_default}"
echo "nccl-nop2p   : ${nccl_nop2p}"
if [ "${overall}" -eq 0 ]; then
  echo "REGRESSION RESULT: PASS"
  exit 0
else
  echo "REGRESSION RESULT: FAIL"
  exit 1
fi
