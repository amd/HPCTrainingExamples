#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --job-name=imagenet_tau_only
#SBATCH --output=capture_tau-%j.out
#SBATCH --error=capture_tau-%j.out
#
# Produce REAL TAU artifacts for the 4-GPU run, viewable in Jumpshot / Vampir /
# ParaProf. Unlike the torch path, TAU wants distinct MPI ranks, so we launch one
# process per GPU with `mpirun -n 4` (env-less tcp:// rendezvous) rather than the
# README's `mp.spawn`; that gives TAU a proper 4-rank trace instead of collapsing
# every worker onto node 0.
#
# Artifacts are written to a PERSISTENT dir (profiling/tau/), not the /tmp work dir:
#   tau.trc / tau.edf         merged TAU trace
#   traces.otf2 (+ *.def)     OTF2 -> open in Vampir
#   imagenet_4gpu.slog2       SLOG2 -> open in Jumpshot
#   profile.* + tau_pprof.txt ParaProf profile (+ text summary)
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

module load rocm/6.4.3 openmpi pytorch/2.12.0
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

# TAU tracing + profiling into the persistent dir.
export TAU_TRACE=1
export TAU_PROFILE=1
export TRACEDIR="${TAU_OUT}"
export PROFILEDIR="${TAU_OUT}"

echo "=== TAU 4-rank run (mpirun -n 4 tau_exec -rocm) ==="
mpirun -n 4 --oversubscribe tau_exec -T rocm,rocprofsdk -rocm ./tau_wrapper.sh \
  |& tee "${OUT_DIR}/capture_tau_run.log" \
  || echo "WARN: TAU run returned nonzero (trace may still be usable)"

echo "=== Merge + convert TAU artifacts in ${TAU_OUT} ==="
cd "${TAU_OUT}" || exit 1
ls -la
tau_treemerge.pl || echo "WARN: tau_treemerge.pl failed"
if [ -f tau.trc ] && [ -f tau.edf ]; then
  tau2otf2  tau.trc tau.edf traces.otf2        && echo "OTF2 (Vampir)  -> ${TAU_OUT}/traces.otf2"   || echo "WARN: tau2otf2 failed"
  tau2slog2 tau.trc tau.edf imagenet_4gpu.slog2 && echo "SLOG2 (Jumpshot)-> ${TAU_OUT}/imagenet_4gpu.slog2" || echo "WARN: tau2slog2 failed"
fi
# ParaProf profile: profile.* files are already here from TAU_PROFILE=1.
pprof -a > tau_pprof.txt 2>/dev/null && echo "pprof text summary -> ${TAU_OUT}/tau_pprof.txt" || echo "WARN: pprof failed"

echo "=== DONE. TAU artifacts: ==="
ls -la "${TAU_OUT}"
echo "View: Jumpshot on imagenet_4gpu.slog2, Vampir on traces.otf2, or 'paraprof' on profile.* (see README 13.5 for VNC)."
deactivate 2>/dev/null || true
