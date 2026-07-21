# Testing a ROCm Nightly Build with PyTorch (CIFAR-100 train + profile)

This guide walks you through testing a **nightly ROCm build** together with
PyTorch on an AMD GPU. You will:

1. Build a self-contained Python virtual environment with nightly ROCm, PyTorch,
   and the ROCm profilers.
2. Create a `setup_rocm.sh` script that activates that environment.
3. Run the CIFAR-100 training workload through each profiling tool using the
   provided SLURM scripts, and check that everything works end to end.

The workload itself is the same `train_cifar_100.py` used throughout this
directory — a small vision model trained on CIFAR-100. It is intentionally
short so that a nightly build can be validated quickly.

> This guide targets an AMD MI300A GPU (`gfx942`) on a SLURM cluster. For a
> different GPU, change the architecture (`device-gfx942` / `--arch`) and the
> SLURM `--partition` in the scripts accordingly.

---

## Step 1 — Build the virtual environment

Follow [`ROCM_PYTORCH_PIP_VENV_SETUP.md`](./ROCM_PYTORCH_PIP_VENV_SETUP.md) to
create the `rocm-pytorch-pip` venv. In short, it:

- creates a venv at `~/venvs/rocm-pytorch-pip`,
- installs nightly ROCm + PyTorch + profilers from the multi-arch nightly index,
- installs `transformers` (required by the training script),
- runs `rocm-sdk init` to extract the development headers and device code.

To test a **specific nightly**, set the ROCm version pin in that guide, e.g.:

```bash
ROCM_VERSION=7.15.0a20260716
```

Change this value to the nightly date you want to validate.

## Step 2 — Verify `setup_rocm.sh`

The SLURM scripts in each sub-directory activate the environment by sourcing
`../setup_rocm.sh` — i.e. the `setup_rocm.sh` shipped in **this**
(`PyTorch_Profiling/`) directory. It is already provided; just verify (and edit
if needed) that its `VENV` points at the venv you built in Step 1. Its full
contents, and a GPU-node sanity check, are covered in steps 5-7 of
[`ROCM_PYTORCH_PIP_VENV_SETUP.md`](./ROCM_PYTORCH_PIP_VENV_SETUP.md).

Once verified, a quick check that the nightly build runs GPU kernels through
PyTorch:

```bash
source setup_rocm.sh
srun -n1 --gpus=1 python3 -c "import torch; print('torch', torch.__version__); \
x = torch.ones(4, device='cuda:0'); print('device ok:', (x+1).sum().item())"
```

If you see `device ok:`, you are ready to profile.

## Step 3 — Run the SLURM scripts

Each sub-directory contains a single-process SLURM script that sources
`../setup_rocm.sh`, pre-downloads the dataset if needed, and runs the workload
under one tool. All of them use a single GPU and a short run
(`--batch-size 32 --max-steps 5`) so a nightly can be checked quickly.

**Submit each script from its own directory** (the scripts use
`SLURM_SUBMIT_DIR` to locate themselves):

| Tool | Directory | Script | What it produces |
|------|-----------|--------|------------------|
| None (baseline) | `no-profiling/` | `slurm_single_process_noprofile.sh` | Plain training run — confirms the workload runs without a profiler. |
| ROCm Compute Profiler | `rocm-compute-profiler/` | `slurm_single_process_profile.sh` | Hardware-counter profile under `workloads/` (analyze with `rocprof-compute analyze`). |
| RocProfiler (kernels) | `rocprofv3/` | `slurm_single_process_kernels.sh` | Kernel stats + trace CSVs under `single_process/`. |
| RocProfiler (traces) | `rocprofv3/` | `slurm_single_process_traces.sh` | System timeline trace (`.pftrace`) under `single_process/`. |
| ROCm Systems Profiler | `rocm-systems-profiler/` | `slurm_single_process.sh` | Sampling profile + trace under `rocprofsys-python3-output/`. |
| Roofline Extractor | `roofline-extractor/` | `slurm_single_process.sh` | Per-kernel roofline analysis + interactive HTML plot under `output/`. |

Example (baseline sanity check first, then a profiler):

```bash
cd no-profiling
sbatch slurm_single_process_noprofile.sh

cd ../rocprofv3
sbatch slurm_single_process_kernels.sh
```

Check job status and output:

```bash
squeue --me
# stdout/stderr land in the submit directory as <name>_<jobid>.out / .err
```

### Analyzing the results

> **Note — running `rocprof-compute analyze`:** run the analysis step only
> **after** its profile job has finished (the counter database must exist).
> `rocprof-compute analyze` requires `numpy==1.26.4`, which conflicts with the
> `numpy>=2.0` needed by the training/profiling jobs. To avoid disturbing the
> shared venv, the analysis script uses a separate, isolated venv
> (`~/venvs/rocprof-compute-analyze`) that holds only rocprof-compute's pinned
> requirements, while reusing the ROCm install from the shared venv. The shared
> venv is never modified, so the analysis can run even while training or
> profiling jobs are using it.

- **ROCm Compute Profiler:** submit the companion analysis job from
  `rocm-compute-profiler/` (it locates the workload and runs the analysis for
  you, using the isolated analysis venv described above):

```bash
cd rocm-compute-profiler
sbatch slurm_single_process_analyze.sh
```

  Or run it by hand: `rocprof-compute analyze -p rocm-compute-profiler/workloads/cifar_100_single_proc/<subdir>`
- **RocProfiler:** open the CSVs (kernels) or load the `.pftrace` in
  [ui.perfetto.dev](https://ui.perfetto.dev) (traces).
- **ROCm Systems Profiler:** load the `perfetto-trace-*.proto` from
  `rocprofsys-python3-output/<timestamp>/` in [ui.perfetto.dev](https://ui.perfetto.dev).
- **Roofline Extractor:** open the generated `.html` in `roofline-extractor/output/`.

---

## Notes

- If a job fails to start, check the SLURM `--partition` and time limits in the
  script headers match your cluster.
- The scripts derive a per-job rendezvous port from the SLURM job ID, so
  multiple jobs can share a node without port collisions.
- To validate a different nightly, rebuild the venv (Step 1) with a new
  `ROCM_VERSION` and re-run the scripts.
