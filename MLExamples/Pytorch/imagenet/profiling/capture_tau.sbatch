#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --job-name=imagenet_tau_only
#SBATCH --output=capture_tau-%j.out
#SBATCH --error=capture_tau-%j.out
#
# Produce REAL TAU artifacts for the 4-GPU run. Unlike the torch path, TAU
# wants distinct MPI ranks, so we launch one process per GPU with `mpirun -n 4`
# (env-less tcp:// rendezvous) rather than the README's `mp.spawn`; that gives TAU a
# proper 4-rank profile instead of collapsing every worker onto node 0.
#
# Artifacts are written to a PERSISTENT dir (profiling/tau/), not the /tmp work dir:
#   profile.* + tau_pprof.txt      ParaProf profile (+ text summary)
#   tautrace.*.trc / events.*.edf  raw TAU event trace (TAU_TRACE=1)
#   imagenet_4gpu.slog2            Jumpshot timeline (tau_treemerge.pl + tau2slog2)
#
# Submit from the imagenet dir:  sbatch profiling/capture_tau.sbatch

OUT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
TAU_OUT="${OUT_DIR}/profiling/tau"
mkdir -p "${TAU_OUT}"
echo "=== capture_tau: OUT_DIR=${OUT_DIR} TAU_OUT=${TAU_OUT} on $(hostname) ==="

WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}" || exit 1

uv init imagenet_test >/dev/null 2>&1
cd imagenet_test || exit 1
uv venv --system-site-packages
source .venv/bin/activate

module load rocm openmpi pytorch
module load tau || { echo "ERROR: tau module failed to load"; exit 1; }
command -v tau_exec >/dev/null 2>&1 || { echo "ERROR: tau_exec not on PATH"; exit 1; }

export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export MIOPEN_LOG_LEVEL=3
export KINETO_LOG_LEVEL=3
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "${WORK_DIR}" "${MIOPEN_USER_DB_PATH}"' EXIT

git clone --depth=1 https://github.com/pytorch/examples.git ./pytorch_examples
cp pytorch_examples/imagenet/* .

# Plain main.py, short run (20 steps), clean process-group teardown. No torch
# profiler edits -- TAU does the instrumentation here.
cp main.py main_tau.py
sed -i '/world_size=args.world_size, rank=args.rank)/a\        import atexit as _ax, torch.distributed as _d; _ax.register(lambda: _d.destroy_process_group() if _d.is_initialized() else None)' main_tau.py
sed -i '/^        data_time.update(time.time() - end)/a\
        if i >= 20: break' main_tau.py

# Per-rank wrapper: pin one GPU per MPI rank and run the launched (non-spawn) path.
cat > tau_wrapper.sh <<'EOF'
#!/bin/bash
export HIP_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
exec python main_tau.py -a resnet50 --dummy \
  --dist-url 'tcp://127.0.0.1:23459' --dist-backend nccl \
  --world-size ${OMPI_COMM_WORLD_SIZE} --rank ${OMPI_COMM_WORLD_RANK} \
  --gpu 0 -b 128 -p 20 --epochs 1
EOF
chmod +x tau_wrapper.sh

python3 -c 'import torch; print("cuda?", torch.cuda.is_available(), "devs", torch.cuda.device_count())'

# Warm the MIOpen cache single-process.
HIP_VISIBLE_DEVICES=0 python -c "import torch,torchvision.models as M; \
   d=torch.device('cuda'); n=M.resnet50().to(d); c=torch.nn.CrossEntropyLoss().to(d); \
   x=torch.randn(256,3,224,224,device=d); y=torch.randint(0,1000,(256,),device=d); \
   [c(n(x),y).backward() for _ in range(3)]; torch.cuda.synchronize(); print('warm done')"

# TAU profiling AND tracing into the persistent dir. TAU_PROFILE default is 1;
# TAU_TRACE=1 additionally emits the per-rank event trace needed for the Jumpshot
# timeline. PROFILEDIR gets the profile.* files; TRACEDIR gets tautrace.*/events.*.
export TAU_PROFILE=1
export TAU_TRACE=1
export PROFILEDIR="${TAU_OUT}"
export TRACEDIR="${TAU_OUT}"

echo "=== TAU 4-rank run (mpirun -n 4 tau_exec -rocm) ==="
# One rank per MI300A APU (each APU is its own NUMA domain: GPU + 24 cores), bound
# to that NUMA node so it matches HIP_VISIBLE_DEVICES=local_rank. Under mpirun the
# binding comes from these --map-by/--bind-to flags, NOT Slurm --cpus-per-task.
# --report-bindings prints the per-rank core mask so you can confirm rank i -> APU i.
mpirun -n 4 --map-by numa --bind-to numa --report-bindings \
  tau_exec -T rocm,rocprofsdk -rocm ./tau_wrapper.sh \
  |& tee "${OUT_DIR}/capture_tau_run.log" \
  || echo "WARN: TAU run returned nonzero (trace may still be usable)"

echo "=== TAU ParaProf profile in ${TAU_OUT} ==="
cd "${TAU_OUT}" || exit 1
ls -la
# ParaProf profile: profile.* files are already here from TAU_PROFILE=1.
pprof -a > tau_pprof.txt 2>/dev/null && echo "pprof text summary -> ${TAU_OUT}/tau_pprof.txt" || echo "WARN: pprof failed"

# ---------------------------------------------------------------------------
# Jumpshot timeline: merge the per-rank event traces and convert to SLOG-2.
# (These converters work on the tau/dev ROCm 7.2.4 module; older ROCm builds
# produced empty .slog2 files.) The raw tautrace.*/events.* are kept so the
# conversion can be re-run; the large merged tau.trc intermediate is removed.
# ---------------------------------------------------------------------------
echo "=== Merge TAU traces + convert to SLOG-2 (Jumpshot) ==="
if ls tautrace.*.trc >/dev/null 2>&1; then
  tau_treemerge.pl && \
  tau2slog2 tau.trc tau.edf -o imagenet_4gpu.slog2 && \
  echo "Jumpshot trace -> ${TAU_OUT}/imagenet_4gpu.slog2" || \
  echo "WARN: tau_treemerge.pl / tau2slog2 failed"
  rm -f tau.trc   # drop the large merged intermediate; raw traces remain
else
  echo "WARN: no tautrace.*.trc found (was TAU_TRACE=1 honored?)"
fi

echo "=== DONE. TAU artifacts: ==="
ls -la "${TAU_OUT}"
echo "View: 'paraprof' on profile.* or 'jumpshot imagenet_4gpu.slog2' (see README 13.5/13.7 for VNC),"
echo "      or render figs/tau_profile_4gpu.png from tau_pprof.txt with profiling/render_tau_profile.py."
deactivate 2>/dev/null || true
