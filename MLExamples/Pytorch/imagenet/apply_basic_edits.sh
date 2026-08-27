#!/bin/bash

# Modifications to the example source code

# Fix warning destroy_process_group

# Upstream main.py inits the NCCL process group but never destroys it, so PyTorch
# warns at exit ("`destroy_process_group()` was not called ... can leak resources").
# Register an atexit handler right after `init_process_group` so every worker
# (incl. mp.spawn children) cleans up on a normal exit.

sed -i '/world_size=args.world_size, rank=args.rank)/a\        import atexit as _ax, torch.distributed as _d; _ax.register(lambda: _d.destroy_process_group() if _d.is_initialized() else None)' main.py

# Keep the demo (and the profiler trace) short.

sed -i '/^        data_time.update(time.time() - end)/a\
        if i >= 100: break' main.py

# Add GPU peak memory instrumentation

# Print per-GPU peak memory once at the end of train() (matches README `peak_mem_mb`)

sed -i '/^def validate(/i\    torch.cuda.is_available() and getattr(args,"rank",0)<=0 and print(f"PEAK_MEM_MB {torch.cuda.max_memory_allocated()/1e6:.0f}")' main.py

# Enable the profiler and print total RCCL time

# Start a `torch.profiler` at the top of `train()` and stop it just before
# `validate()`. The on-GPU time of the `nccl*` collective kernels is summed and
# printed as `RCCL_TOTAL_MS` (~0 at 1 GPU, growing with GPU count). The `break`
# from the short-run edit keeps the captured trace short.

# Start the profiler at the top of `train()`:

sed -i '/^    model.train()/a\
    import torch.profiler as _tp\
    _prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA], acc_events=True); _prof.start()' main.py

# Stop the profiler and print the total RCCL kernel time (inserted at the end of `train()`, right before `def validate(`):

sed -i '/^def validate(/i\
    _prof.stop()\
    _rccl_ms = sum(e.self_device_time_total for e in _prof.key_averages() if "nccl" in e.key.lower())/1e3\
    _ws = getattr(args,"world_size","?")\
    getattr(args,"rank",0)<=0 and print(f"RCCL_TOTAL_MS {_rccl_ms:.3f} gpus={_ws}")' main.py

# Compare `.to` (copy) vs `.migrate` (zero-copy) staging

# The migrate path (`STAGE=migrate`) aliases the batch instead of copying it; it
# needs `COMMON_DIR` (for `zerocopy.Stager`), `HSA_XNACK=1`, and pageable
# (non-pinned) host memory. When `STAGE` is unset these edits are inert, so the
# plain scaling runs are unaffected.

# Point `COMMON_DIR` at the shared helpers and enable XNACK. This script runs
# from a temporary work dir, so use $BASE_DIR (the repo imagenet dir) rather than
# the README's relative "../common" (which only resolves when run in-place).

export COMMON_DIR="$BASE_DIR/../common"
export HSA_XNACK=1

# Set up the staging counters and (for `STAGE=migrate`) the zero-copy `Stager` at the top of `train()`:

sed -i '/^    model.train()/a\
    _stage_ms = 0.0; _stage_n = 0; _stager = None\
    if os.environ.get("STAGE") == "migrate":\
        import sys; sys.path.insert(0, os.environ["COMMON_DIR"]); from zerocopy import Stager\
        _stager = Stager(device, method="register")' main.py

# Time the host->device staging, but only when `STAGE` is set (copy vs migrate):

sed -i '/^        images = images.to(device, non_blocking=True)/c\
        if os.environ.get("STAGE"):\
            _e0 = torch.cuda.Event(enable_timing=True); _e1 = torch.cuda.Event(enable_timing=True)\
            _e0.record()\
            images = _stager.to_device(images) if _stager is not None else images.to(device, non_blocking=True)\
            _e1.record(); torch.cuda.synchronize()\
            _stage_ms += _e0.elapsed_time(_e1); _stage_n += 1\
        else:\
            images = images.to(device, non_blocking=True)' main.py

# Print the per-step staging time (inserted at the end of `train()`, right before `def validate(`):

sed -i '/^def validate(/i\
    _stg = os.environ.get("STAGE",""); _ws2 = getattr(args,"world_size","?")\
    getattr(args,"rank",0)<=0 and _stage_n and print(f"STAGE_MS_PER_STEP {_stage_ms/_stage_n:.4f} gpus={_ws2} stage={_stg}")' main.py

# Make pin_memory graceful (pageable only when zero-copy is actually live)

# The register path needs *pageable* host memory (hipHostRegister fails on
# already-pinned buffers), but the plain `.to()` fallback (discrete GPU,
# HSA_XNACK!=1, or the extension failing to build) copies faster from *pinned*
# memory. So instead of forcing pin_memory=False unconditionally, gate it on
# whether the zero-copy register path is actually available -- mirroring the
# Stager's own graceful fallback. When STAGE is unset this is inert: pin_memory
# stays True (upstream behavior), so the plain scaling runs get fast pinned copies.

# Inject the gate helper right after the imports:

sed -i '/^from torch.utils.data import Subset/a\
def _zero_copy_active():\
    """pin_memory gate: pageable only when the STAGE=migrate register path is live."""\
    if os.environ.get("STAGE") != "migrate":\
        return False\
    try:\
        import sys; sys.path.insert(0, os.environ["COMMON_DIR"]); from zerocopy import unified_memory_available\
        return unified_memory_available()\
    except Exception:\
        return False' main.py

# Use pageable buffers only when zero-copy is active, pinned otherwise:

sed -i 's/pin_memory=True/pin_memory=not _zero_copy_active()/g' main.py
