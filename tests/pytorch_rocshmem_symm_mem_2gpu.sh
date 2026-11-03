#!/bin/bash
#
# 2-GPU validation of the ROCm rocSHMEM symmetric-memory backend in PyTorch
# (torch.distributed._symmetric_memory, "NVSHMEM" backend). This exercises the
# device kernels in torch/csrc/distributed/c10d/symm_mem/rocshmem_extension.cu
# that HPCTrainingDock/extras/scripts/pytorch_setup.sh patches for the ROCm
# rocSHMEM 3.3.0 device-API skew (rocshmem_barrier_wg -> rocshmem_barrier_all_wg;
# device ROCSHMEM_TEAM_WORLD threaded in as the world_team kernel parameter).
#
# It drives torch.ops.symm_mem.all_to_all_vdev across 2 ranks/GPUs and checks
# the result against a dist.all_to_all_single reference (same shape as PyTorch's
# own test/distributed/test_nvshmem.py::test_all_to_all_vdev). A correct result
# proves the patched kernels (exchangeSplitAndOffset + allToAllV, incl. the
# world_team PE translation and the work-group barrier) run correctly, not just
# that they compiled.
#
# Outcome keywords (CTest gates on these):
#   "ROCSHMEM SYMMMEM: OK"    -> pass (exit 0)
#   "ROCSHMEM SYMMMEM: FAIL"  -> the backend is available but the op/result is
#                                wrong (exit 1)
#   "ROCSHMEM SYMMMEM: SKIPPED" -> not applicable here (no torch, < 2 GPUs, no
#                                P2P, or the rocSHMEM/NVSHMEM backend was not
#                                built into this torch) -> CTest skip (exit 77)
#
# NOTE: this test assumes PyTorch was installed per the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/pytorch_setup.sh

set -u

# ── bring Lmod's `module` into scope if needed, load rocm + pytorch, but never
#    fail the test over modules (a plain venv with torch must still work) ──────
if ! type module >/dev/null 2>&1; then
  [ -r /etc/profile.d/lmod.sh ]         && . /etc/profile.d/lmod.sh
  [ -r /usr/share/lmod/lmod/init/bash ] && . /usr/share/lmod/lmod/init/bash
fi
if type module >/dev/null 2>&1; then
  if ! module -t list 2>&1 | grep -q "^rocm/"; then
    echo "loading default rocm module"
    module load rocm 2>/dev/null || echo "  (could not load a rocm module; continuing with current environment)"
  fi
  if ! module -t list 2>&1 | grep -q "^pytorch/"; then
    echo "loading default pytorch module"
    module load pytorch 2>/dev/null || echo "  (could not load a pytorch module; continuing with current environment)"
  fi
else
  echo "no module system detected; using the current environment / venv"
fi

_OUT=$(mktemp)
_WORKER=$(mktemp --suffix=.py)
trap 'rm -f "${_OUT}" "${_WORKER}"' EXIT

# ── pre-flight (single process): decide SKIP vs RUN before launching 2 ranks ──
python3 <<'EOF' 2>&1 | tee "${_OUT}"
import sys
try:
    import torch
except Exception as e:
    print("could not import torch:", e)
    print("ROCSHMEM SYMMMEM: SKIPPED")
    sys.exit(77)

if not torch.cuda.is_available():
    print("no GPU available (torch.cuda.is_available() == False)")
    print("ROCSHMEM SYMMMEM: SKIPPED")
    sys.exit(77)

n = torch.cuda.device_count()
print("torch.__version__ :", torch.__version__)
print("torch.version.hip :", getattr(torch.version, "hip", None))
print("GPU device count  :", n)
if n < 2:
    print("need >= 2 GPUs for the symmetric-memory all_to_all_vdev test")
    print("ROCSHMEM SYMMMEM: SKIPPED")
    sys.exit(77)

# P2P access between the two GPUs is required by the symm-mem backend.
try:
    if not torch.cuda.can_device_access_peer(0, 1):
        print("GPU 0 cannot peer-access GPU 1 (no P2P)")
        print("ROCSHMEM SYMMMEM: SKIPPED")
        sys.exit(77)
except Exception as e:
    print("P2P check failed:", e)
    print("ROCSHMEM SYMMMEM: SKIPPED")
    sys.exit(77)

# Is the rocSHMEM / NVSHMEM symmetric-memory backend compiled into this torch?
try:
    import torch.distributed._symmetric_memory as symm_mem
    avail = bool(getattr(symm_mem, "is_nvshmem_available", lambda: False)())
except Exception as e:
    print("symmetric_memory import/availability check failed:", e)
    avail = False
print("rocSHMEM/NVSHMEM symm-mem backend available:", avail)
if not avail:
    print("  (torch was built without the rocSHMEM backend, or the SDK shipped no rocshmem)")
    print("ROCSHMEM SYMMMEM: SKIPPED")
    sys.exit(77)

print("PREFLIGHT: RUN")
sys.exit(0)
EOF
_pf=${PIPESTATUS[0]}
if [ "${_pf}" -eq 77 ]; then
  exit 77
elif [ "${_pf}" -ne 0 ] || ! grep -q "PREFLIGHT: RUN" "${_OUT}"; then
  echo "ROCSHMEM SYMMMEM: FAIL - preflight error"
  exit 1
fi

# ── the 2-rank worker (launched by torchrun). Mirrors PyTorch's own
#    test_nvshmem.py::test_all_to_all_vdev shape. ───────────────────────────────
cat > "${_WORKER}" <<'EOF'
import os, sys
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

def main():
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        # Select the rocSHMEM/NVSHMEM symmetric-memory backend.
        if hasattr(symm_mem, "set_backend"):
            symm_mem.set_backend("NVSHMEM")

        group_name = dist.group.WORLD.group_name
        torch.manual_seed(1234 + rank)
        dtype = torch.float

        k = 10  # per-peer element count is random in [0, k)
        inp_splits = torch.randint(k, (world_size,), device=device)
        inp_numel = inp_splits.sum().item()

        out_splits = torch.zeros_like(inp_splits)
        dist.all_to_all_single(out_splits, inp_splits)
        out_numel = out_splits.sum().item()

        max_inp_numel = k * world_size
        overflow_factor = world_size          # worst case: one rank gets everything
        max_out_numel = max_inp_numel * overflow_factor

        inp = symm_mem.empty(max_inp_numel, dtype=dtype, device=device).copy_(
            torch.randn(max_inp_numel, dtype=dtype, device=device)
        )
        out = symm_mem.empty(max_out_numel, dtype=dtype, device=device).fill_(-1)
        in_splits = symm_mem.empty(world_size, dtype=torch.int64, device=device)
        out_splits_offsets = symm_mem.empty((2, world_size), dtype=torch.int64, device=device)
        in_splits.copy_(inp_splits)

        dist.barrier()

        # ---- the patched rocSHMEM kernels run here ----
        torch.ops.symm_mem.all_to_all_vdev(inp, out, in_splits, out_splits_offsets, group_name)

        # input splits (row 0 of the 2xW tensor the op fills) must be preserved
        torch.testing.assert_close(in_splits, inp_splits)
        # output splits (row 0 of out_splits_offsets) must match the a2a-single exchange
        torch.testing.assert_close(out_splits_offsets[0], out_splits)
        # output offsets (row 1) are an exclusive prefix scan of out_splits
        out_offsets = torch.cumsum(out_splits, dim=0)
        assert out_splits_offsets[1][0].item() == 0
        torch.testing.assert_close(out_splits_offsets[1][1:], out_offsets[:-1])
        # data must equal a reference all_to_all_single
        expected = torch.empty(out_numel, dtype=dtype, device=device)
        dist.all_to_all_single(expected, inp[:inp_numel], out_splits.tolist(), inp_splits.tolist())
        torch.testing.assert_close(out[:out_numel], expected)

        dist.barrier()
        if rank == 0:
            print("ROCSHMEM SYMMMEM: OK - all_to_all_vdev verified on {} GPUs".format(world_size))
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("ROCSHMEM SYMMMEM: FAIL - result mismatch: {}".format(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Backend was reported available but the op path errored (e.g. a bad
        # device-side team translation would surface here) -> real failure.
        print("ROCSHMEM SYMMMEM: FAIL - {}: {}".format(type(e).__name__, e), file=sys.stderr)
        sys.exit(1)
EOF

echo "launching 2-rank torchrun for all_to_all_vdev ..."
# Prefer torchrun; fall back to `python -m torch.distributed.run`.
if command -v torchrun >/dev/null 2>&1; then
  _LAUNCH=(torchrun)
else
  _LAUNCH=(python3 -m torch.distributed.run)
fi
"${_LAUNCH[@]}" --standalone --nnodes=1 --nproc_per_node=2 "${_WORKER}" 2>&1 | tee -a "${_OUT}"
_rc=${PIPESTATUS[0]}

# ── gate ──────────────────────────────────────────────────────────────────────
if grep -q "ROCSHMEM SYMMMEM: SKIPPED" "${_OUT}"; then
  exit 77
elif grep -q "ROCSHMEM SYMMMEM: OK" "${_OUT}" && [ "${_rc}" -eq 0 ]; then
  exit 0
else
  grep -q "ROCSHMEM SYMMMEM: FAIL" "${_OUT}" || echo "ROCSHMEM SYMMMEM: FAIL - torchrun exit ${_rc}"
  exit 1
fi
