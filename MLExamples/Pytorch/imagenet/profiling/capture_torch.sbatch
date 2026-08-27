#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=imagenet_torch_capture
#SBATCH --output=capture_torch-%j.out
#SBATCH --error=capture_torch-%j.out
#
# Capture a REAL compute-vs-communication timeline for the README's profiling section using
# torch.profiler (Path A). The TAU path (Path B) lives in its own dedicated
# script, profiling/capture_tau.sbatch.
#
#  Run : 4-GPU main.py under torch.profiler   -> per-rank Chrome/Perfetto JSON
#  Then: merge per-rank JSON                  -> imagenet_4gpu.perfetto.json.gz
#        headless matplotlib render           -> figs/timeline_4gpu.png
#        drop raw JSON traces to profiling/traces/ for manual Perfetto viewing
#
# Submit from the imagenet dir:  sbatch profiling/capture_torch.sbatch

# Where to write results back to (the repo imagenet dir).
OUT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PROFILING_DIR="${OUT_DIR}/profiling"
FIGS_DIR="${OUT_DIR}/figs"
TRACES_DIR="${PROFILING_DIR}/traces"
mkdir -p "${FIGS_DIR}" "${TRACES_DIR}"

echo "=== capture_torch: OUT_DIR=${OUT_DIR} on $(hostname) ==="

# Disposable work dir in /tmp.
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}" || exit 1

# uv venv with system site packages (matches run_imagenet_uv.sbatch).
uv init imagenet_test >/dev/null 2>&1
cd imagenet_test || exit 1
uv venv --system-site-packages

# Match run_imagenet_uv.sbatch: use the default rocm (and pytorch) so the profiled
# run reflects the tutorial's actual software stack.
module load rocm openmpi pytorch
# Activate AFTER module load so .venv/bin is first on PATH (module load prepends its own
# python); otherwise the venv's matplotlib below is shadowed and the render fails.
source .venv/bin/activate

# The system matplotlib is built against NumPy 1.x but the pytorch module ships
# NumPy 2.x, which crashes on import. Install a NumPy-2-compatible matplotlib into
# the venv (takes precedence over the system one) for the headless render.
uv pip install --quiet matplotlib || echo "WARN: matplotlib install failed (render may fail)"

# MIOpen: fast solver + a writable cache dir; clean up at exit.
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export MIOPEN_LOG_LEVEL=3
export KINETO_LOG_LEVEL=3
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "${WORK_DIR}" "${MIOPEN_USER_DB_PATH}"' EXIT

# Upstream example.
git clone --depth=1 https://github.com/pytorch/examples.git ./pytorch_examples
cp pytorch_examples/imagenet/* .

# ---------------------------------------------------------------------------
# Instrument main.py: clean process-group teardown, short run (20 steps), and
# torch.profiler capture with per-rank Chrome/Perfetto export.
# ---------------------------------------------------------------------------
sed -i '/world_size=args.world_size, rank=args.rank)/a\        import atexit as _ax, torch.distributed as _d; _ax.register(lambda: _d.destroy_process_group() if _d.is_initialized() else None)' main.py
sed -i '/^        data_time.update(time.time() - end)/a\
        if i >= 20: break' main.py

# torch.profiler: start at top of train().
sed -i '/^    model.train()/a\
    import torch.profiler as _tp\
    _prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA], acc_events=True); _prof.start()' main.py

# torch.profiler: stop + per-rank Chrome/Perfetto export + RCCL_TOTAL_MS.
sed -i '/^def validate(/i\
    _prof.stop()\
    _rk = getattr(args,"rank",0)\
    _prof.export_chrome_trace(f"torch_trace_rank{_rk}.json")\
    _rccl_ms = sum(e.self_device_time_total for e in _prof.key_averages() if "nccl" in e.key.lower())/1e3\
    _ws = getattr(args,"world_size","?")\
    _rk<=0 and print(f"RCCL_TOTAL_MS {_rccl_ms:.3f} gpus={_ws}")' main.py

# Sanity: GPUs visible?
python3 -c 'import torch; print("cuda?", torch.cuda.is_available(), "devs", torch.cuda.device_count())'

# Warm the MIOpen cache single-process so autotuning is out of the traced window.
HIP_VISIBLE_DEVICES=0 python -c "import torch,torchvision.models as M; \
   d=torch.device('cuda'); n=M.resnet50().to(d); c=torch.nn.CrossEntropyLoss().to(d); \
   x=torch.randn(256,3,224,224,device=d); y=torch.randint(0,1000,(256,),device=d); \
   [c(n(x),y).backward() for _ in range(3)]; torch.cuda.synchronize(); print('warm done')"

# ---------------------------------------------------------------------------
# Capture: torch.profiler on 4 GPUs.
# ---------------------------------------------------------------------------
echo "=== torch.profiler (4 GPUs) ==="
HIP_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50 --dummy \
  --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
  --multiprocessing-distributed --world-size 1 --rank 0 -b 512 -p 20 --epochs 1 \
  |& tee "${OUT_DIR}/capture_run_torch.log"

ls -la torch_trace_rank*.json 2>/dev/null || echo "WARN: no torch traces produced"
cp -f torch_trace_rank*.json "${TRACES_DIR}/" 2>/dev/null

# Merge the per-rank torch traces into ONE 4-lane Perfetto trace (Chrome JSON,
# gzipped). This is the reliable Perfetto artifact and needs only the stdlib.
echo "=== Merge torch traces -> imagenet_4gpu.perfetto.json.gz ==="
python3 "${PROFILING_DIR}/merge_perfetto.py" \
  --glob 'torch_trace_rank*.json' \
  --out "${TRACES_DIR}/imagenet_4gpu.perfetto.json.gz" \
  || echo "WARN: perfetto merge failed"

# ---------------------------------------------------------------------------
# Render the committed README figure headlessly (guaranteed, no display).
# matplotlib was already installed into the venv right after the module load above.
# ---------------------------------------------------------------------------
echo "=== Render matplotlib timeline -> figs/timeline_4gpu.png ==="
python3 "${PROFILING_DIR}/render_timeline.py" \
  --out "${FIGS_DIR}/timeline_4gpu.png" \
  --glob 'torch_trace_rank*.json' \
  || echo "WARN: matplotlib render failed (traces are in ${TRACES_DIR}; render later)"

# The Perfetto view is produced manually: download the merged trace below and open
# it in https://ui.perfetto.dev

echo "=== DONE ==="
echo "Figures:"; ls -la "${FIGS_DIR}" 2>/dev/null
echo "Traces (download imagenet_4gpu.perfetto.json.gz -> open in ui.perfetto.dev):"; ls -la "${TRACES_DIR}" 2>/dev/null
deactivate 2>/dev/null || true
