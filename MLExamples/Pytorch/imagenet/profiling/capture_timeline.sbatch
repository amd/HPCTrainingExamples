#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=imagenet_tau_capture
#SBATCH --output=capture_timeline-%j.out
#SBATCH --error=capture_timeline-%j.out
#
# Capture a REAL compute-vs-communication timeline for README section 13.
#  Run A: 4-GPU main.py under torch.profiler  -> per-rank Chrome/Perfetto JSON
#  Run B: 4-GPU main.py under tau_exec -rocm  -> TAU trace -> tau_trace2json (Perfetto JSON)
#  Then: headless matplotlib render -> figs/timeline_4gpu.png
#        best-effort headless Perfetto screenshot -> figs/perfetto_4gpu.png
#        drop raw JSON traces to profiling/traces/ for manual download either way
#
# Submit from the imagenet dir:  sbatch profiling/capture_timeline.sbatch

# Where to write results back to (the repo imagenet dir).
OUT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PROFILING_DIR="${OUT_DIR}/profiling"
FIGS_DIR="${OUT_DIR}/figs"
TRACES_DIR="${PROFILING_DIR}/traces"
mkdir -p "${FIGS_DIR}" "${TRACES_DIR}"

echo "=== capture_timeline: OUT_DIR=${OUT_DIR} on $(hostname) ==="

# Disposable work dir in /tmp.
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}" || exit 1

# uv venv with system site packages (matches run_imagenet_uv.sbatch).
uv init imagenet_test >/dev/null 2>&1
cd imagenet_test || exit 1
uv venv --system-site-packages
source .venv/bin/activate

# Pin rocm/6.4.3 so the TAU module's prereq is satisfied, plus openmpi/pytorch/tau.
module load rocm/6.4.3 openmpi pytorch/2.12.0
module load tau || echo "WARN: tau module failed to load (TAU run will be skipped)"

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
# Build main.py (torch.profiler instrumented) and main_tau.py (plain, for TAU).
# ---------------------------------------------------------------------------
cp main.py main_tau.py

# Common edits (both files): clean process-group teardown + short run (20 steps).
for f in main.py main_tau.py; do
  sed -i '/world_size=args.world_size, rank=args.rank)/a\        import atexit as _ax, torch.distributed as _d; _ax.register(lambda: _d.destroy_process_group() if _d.is_initialized() else None)' "$f"
  sed -i '/^        data_time.update(time.time() - end)/a\
        if i >= 20: break' "$f"
done

# torch.profiler only in main.py: start at top of train().
sed -i '/^    model.train()/a\
    import torch.profiler as _tp\
    _prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA], acc_events=True); _prof.start()' main.py

# torch.profiler only in main.py: stop + per-rank Chrome/Perfetto export + RCCL_TOTAL_MS.
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
# Run A: torch.profiler on 4 GPUs.
# ---------------------------------------------------------------------------
echo "=== Run A: torch.profiler (4 GPUs) ==="
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
# Run B: TAU (best-effort) on 4 GPUs. TAU output is for Jumpshot (slog2) or
# Vampir (OTF2) -- NOT Perfetto (tau_trace2json emits a TAU-specific schema the
# Perfetto UI cannot read), so the Perfetto artifact stays the torch merge above.
# ---------------------------------------------------------------------------
if command -v tau_exec >/dev/null 2>&1; then
  echo "=== Run B: TAU tau_exec -rocm (4 GPUs) ==="
  export TAU_TRACE=1
  export TAU_PROFILE=1
  export TRACEDIR="${PWD}/tau_trace"
  export PROFILEDIR="${PWD}/tau_trace"
  mkdir -p "${TRACEDIR}"
  HIP_VISIBLE_DEVICES=0,1,2,3 tau_exec -T rocm,rocprofsdk -rocm python main_tau.py -a resnet50 --dummy \
    --dist-url 'tcp://127.0.0.1:23457' --dist-backend nccl \
    --multiprocessing-distributed --world-size 1 --rank 0 -b 512 -p 20 --epochs 1 \
    |& tee "${OUT_DIR}/capture_run_tau.log" || echo "WARN: TAU run returned nonzero (trace may still be usable)"
  ( cd "${TRACEDIR}" \
    && tau_treemerge.pl \
    && tau2otf2 tau.trc tau.edf traces.otf2 ) \
    && cp -f "${TRACEDIR}"/traces.otf2 "${TRACEDIR}"/*.def "${TRACES_DIR}/" 2>/dev/null \
    && echo "TAU/OTF2 (Vampir) -> ${TRACES_DIR}/traces.otf2" \
    || echo "WARN: TAU->OTF2 conversion failed; skipping TAU artifact"
else
  echo "WARN: tau_exec not on PATH; skipping TAU run"
fi

# ---------------------------------------------------------------------------
# Render the committed README figure headlessly (guaranteed, no display).
# The system matplotlib is built against NumPy 1.x but the pytorch module ships
# NumPy 2.x; install a compatible matplotlib into the venv if import fails.
# ---------------------------------------------------------------------------
echo "=== Render matplotlib timeline -> figs/timeline_4gpu.png ==="
if ! python3 -c "import matplotlib" >/dev/null 2>&1; then
  uv pip install matplotlib || python3 -m pip install --user matplotlib || echo "WARN: matplotlib install failed"
fi
python3 "${PROFILING_DIR}/render_timeline.py" \
  --out "${FIGS_DIR}/timeline_4gpu.png" \
  --glob 'torch_trace_rank*.json' \
  || echo "WARN: matplotlib render failed (traces are in ${TRACES_DIR}; render later)"

# ---------------------------------------------------------------------------
# Best-effort headless Perfetto screenshot (optional bonus figure).
# ---------------------------------------------------------------------------
echo "=== Best-effort Perfetto screenshot ==="
SHOT_TRACE="${TRACES_DIR}/imagenet_4gpu.perfetto.json.gz"
[ -f "${SHOT_TRACE}" ] || SHOT_TRACE="torch_trace_rank0.json"
if [ -f "${SHOT_TRACE}" ]; then
  if uv pip install --quiet playwright >/dev/null 2>&1 \
     && timeout 300 python3 -m playwright install chromium >/dev/null 2>&1; then
    timeout 240 python3 "${PROFILING_DIR}/perfetto_shot.py" "${SHOT_TRACE}" \
      --out "${FIGS_DIR}/perfetto_4gpu.png" \
      && echo "Perfetto screenshot -> ${FIGS_DIR}/perfetto_4gpu.png" \
      || echo "WARN: Perfetto screenshot failed; use the dropped trace in ${TRACES_DIR}"
  else
    echo "WARN: could not install playwright/chromium; skipping Perfetto screenshot"
  fi
fi

echo "=== DONE ==="
echo "Figures:"; ls -la "${FIGS_DIR}" 2>/dev/null
echo "Traces (download these to open in ui.perfetto.org):"; ls -la "${TRACES_DIR}" 2>/dev/null
deactivate 2>/dev/null || true
