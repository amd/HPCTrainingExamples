#!/bin/bash
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=submit_timeline_traces.out
#SBATCH --error=submit_timeline_traces.out
#
# Generate the profiler traces and the committed PNG screenshots for the FSDP2
# per-profiler guides (README.md, torch-profiler.md, rocprofv3.md,
# rocprofiler-systems.md, tensorboard.md, tracelens.md).
#
# Produces, under this directory:
#   torch_prof/  tl_bf16/   torch.profiler Kineto traces (fp32 + bf16)
#   prof_stats/              rocprofv3 --kernel-trace --stats CSV (RCCL totals)
#   prof_tl/                 rocprofv3 --sys-trace pftrace (small, for the timeline PNG)
#   rsys/                    rocprofiler-systems host+GPU .proto (rank 0)
#   torch_prof/nccl_csvs/    TraceLens multi-rank collective report (bandwidth + skew)
#   figs/*.png               headless Perfetto / matplotlib / TensorBoard screenshots
#
# Large trace binaries (*.pftrace, *.proto, *.pt.trace.json) are NOT committed;
# rerun this job to regenerate them and the PNGs.
#
# Usage:
#   sbatch --partition=<gpu_partition> submit_timeline_traces.sbatch
#   ROCM_VER=6.4.3 PT_VER=2.12.0 sbatch ... submit_timeline_traces.sbatch   # pin a combo
#
# NOTE: FSDP2 needs PyTorch >= 2.5 and >= 2 GPUs (4 gives a clearer RCCL signal).

set -u

GPN=${GPN:-4}                      # GPUs per run
BENCH=../benchmarks/fsdp2_bench.py

module load rocm/${ROCM_VER:-7.2.3} openmpi pytorch/${PT_VER:-2.12.0}
# rocprof-sys-run and rocprofv3 ship inside the rocm module (no separate module).
command -v rocprof-sys-run >/dev/null 2>&1 \
  && echo "[ok] rocprof-sys-run: $(command -v rocprof-sys-run)" \
  || echo "[info] rocprof-sys-run not on PATH; rocprof-sys traces will be skipped"
# TraceLens is available as a module here; prefer it over a git pip install.
module load tracelens/dev 2>/dev/null \
  && echo "[ok] tracelens module loaded" \
  || echo "[info] no tracelens module; will pip-install into the venv"
module load google-chrome/stable 2>/dev/null || echo "[info] no google-chrome module; set CHROME_BIN"

# bf16 on ROCm 7.2.x can stall in hipBLASLt; force rocBLAS for the bf16 runs.
export TORCH_BLAS_PREFER_HIPBLASLT=0

if [[ ! -f "$HOME/pytorch_examples/distributed/FSDP2/model.py" ]]; then
  git clone --depth=1 https://github.com/pytorch/examples.git "$HOME/pytorch_examples"
fi
export UPSTREAM="$HOME/pytorch_examples/distributed/FSDP2"

cd "$SLURM_SUBMIT_DIR"
mkdir -p figs

# A venv layered on the module for the screenshot driver (playwright), TensorBoard,
# matplotlib, and (fallback) TraceLens.  Reused if it already exists.
VENV=${VENV:-$HOME/venvs/fsdp2-profiling}
TL_PIP=""
command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1 \
  || TL_PIP="git+https://github.com/AMD-AGI/TraceLens.git"
if [[ ! -d "$VENV" ]]; then
  python -m venv --system-site-packages "$VENV"
  source "$VENV/bin/activate"
  pip install --quiet playwright tensorboard torch-tb-profiler matplotlib tqdm $TL_PIP \
    || echo "[warn] pip install partial"
  python -m playwright install chromium 2>/dev/null || true
else
  source "$VENV/bin/activate"
fi

export CHROME_BIN=${CHROME_BIN:-$(command -v google-chrome || true)}

# shot <trace-glob> <out.png> <title> <label> [zoomout]
# Picks the LARGEST matching trace (the busiest rank) and screenshots it.
shot() {
  local t; t=$(ls -S $1 2>/dev/null | head -1)
  [[ -n "$t" && -e "$t" ]] || { echo "[skip] no trace for $1"; return; }
  python screenshot_perfetto.py "$t" "$2" "$3" 18000 "$4" "${5:-5}" \
    || echo "[warn] screenshot failed for $t"
}

echo "===== 1. torch.profiler (Kineto) — fp32 baseline + bf16 ====="
torchrun --standalone --nproc_per_node=$GPN "$BENCH" --profile --profile-dir ./torch_prof || true
torchrun --standalone --nproc_per_node=$GPN "$BENCH" --profile --profile-dir ./tl_bf16 --mixed-precision || true
# RCCL 2.27 fuses the collectives into ncclDevKernel_Generic_* (not _AllGather_*).
shot "torch_prof/rank0/*.pt.trace.json" figs/fsdp2_torch_allgather.png \
     "FSDP2 torch.profiler ${GPN} GPUs" ncclDevKernel_Generic 6

echo "===== 2. rocprofv3 — kernel stats (RCCL totals) + small sys-trace timeline ====="
rocprofv3 --kernel-trace --stats --truncate-kernels --output-format csv -d prof_stats -- \
  torchrun --standalone --nproc_per_node=$GPN "$BENCH" --warmup 3 --iters 10 || true
echo "--- RCCL kernel rows ---"
grep -hiE 'nccl|AllGather|ReduceScatter' prof_stats/*_kernel_stats.csv 2>/dev/null | head
# small trace so headless Chrome can load it (sys-trace of a full run is ~100 MB
# and crashes the tab); clean any prior large traces first so `shot` picks a small one
rm -rf prof_tl
rocprofv3 --sys-trace --output-format pftrace -d prof_tl -- \
  torchrun --standalone --nproc_per_node=2 "$BENCH" --warmup 1 --iters 2 || true
shot "prof_tl/*/*_results.pftrace" figs/fsdp2_rocprofv3_timeline.png \
     "FSDP2 rocprofv3 timeline" ncclDevKernel_Generic 5

echo "===== 3. TraceLens — perf report (roofline) + multi-rank collective (RCCL) ====="
ARCH=../../../TraceLens/mi300.json
if command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1; then
  R0=$(ls torch_prof/rank0/*.pt.trace.json | head -1)
  TraceLens_generate_perf_report_pytorch --profile_json_path "$R0" \
    ${ARCH:+--gpu_arch_json_path $ARCH} || echo "[warn] perf report failed"
  # The multi-rank collective report expects rank{N}_trace.json in the dir.
  for n in $(seq 0 $((GPN-1))); do
    f=$(ls torch_prof/rank$n/*.pt.trace.json 2>/dev/null | head -1)
    [[ -n "$f" ]] && ln -sf "rank$n/$(basename "$f")" "torch_prof/rank${n}_trace.json"
  done
  TraceLens_generate_multi_rank_collective_report_pytorch --trace_dir ./torch_prof \
    --world_size $GPN --gpus_per_node $GPN \
    --output_xlsx_path ./torch_prof/nccl_analysis_report.xlsx \
    --output_csvs_dir ./torch_prof/nccl_csvs || echo "[warn] collective report failed"
  # Render the collective bandwidth/skew + gpu_timeline figure from the CSVs.
  python make_tracelens_fig.py figs/fsdp2_tracelens_collective.png || echo "[warn] figure failed"
else
  echo "[skip] TraceLens not installed"
fi

echo "===== 4. TensorBoard Overview (headless) ====="
# put all ranks in one run dir so the Distributed view is available too
mkdir -p tb_logs/fsdp2
for n in $(seq 0 $((GPN-1))); do
  f=$(ls torch_prof/rank$n/*.pt.trace.json 2>/dev/null | head -1)
  [[ -n "$f" ]] && ln -sf "$(readlink -f "$f")" "tb_logs/fsdp2/rank$n.pt.trace.json"
done
python shot_tensorboard.py ./tb_logs figs/fsdp2_tb_overview.png figs/fsdp2_tb_second.png 6007 \
  || echo "[warn] tensorboard screenshot failed"

echo "===== 5. rocprofiler-systems — host+GPU timeline (rank 0; may hang headless) ====="
# NOTE: on ROCm 7.2.3 wrapping the distributed launcher headlessly can hang in
# rocprof-sys init, so this is LAST and timeout-guarded; collect interactively for
# a clean rocprof-sys .proto (see rocprofiler-systems.md). The committed GPU-side
# figure is the rocprofv3 --sys-trace timeline from step 2.
if command -v rocprof-sys-run >/dev/null 2>&1; then
  export ROCPROFSYS_USE_SAMPLING=ON ROCPROFSYS_SAMPLING_FREQ=1000 \
         ROCPROFSYS_USE_PROCESS_SAMPLING=OFF ROCPROFSYS_TIME_OUTPUT=OFF
  chmod +x ./rocsys_wrap.sh
  timeout 600 torchrun --standalone --nproc_per_node=$GPN --no-python \
    ./rocsys_wrap.sh "$SLURM_SUBMIT_DIR/rsys" "$BENCH" --warmup 2 --iters 3 \
    || echo "[warn] rocprof-sys run failed/timed out (expected headless on 7.2.3)"
  shot "rsys/**/*.proto" figs/fsdp2_rocsys_timeline.png \
       "FSDP2 rocprof-sys rank0" ncclDevKernel_Generic 5
else
  echo "[skip] rocprof-sys-run not found"
fi

echo "Finished. PNGs in figs/ ; trace binaries left in place (not committed)."
ls -la figs/*.png 2>/dev/null
