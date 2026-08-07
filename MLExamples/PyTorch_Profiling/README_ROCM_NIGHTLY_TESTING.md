# Testing a ROCm Nightly Build with PyTorch (CIFAR-100 train + profile)

This guide walks you through testing a **nightly ROCm build** together with
PyTorch on an AMD GPU. You will:

1. Describe your cluster in a single `local.env` file.
2. Build a self-contained Python virtual environment with nightly ROCm, PyTorch,
   and the ROCm profilers, exposed as an Lmod module.
3. Run the CIFAR-100 training workload through each profiling tool using the
   provided SLURM scripts, and check that everything works end to end.

The workload itself is the same `train_cifar_100.py` used throughout this
directory — a small vision model trained on CIFAR-100. It is intentionally
short so that a nightly build can be validated quickly.

The scripts run unmodified on any SLURM cluster and on any supported GPU:
everything machine-specific lives in `local.env`, and no script needs editing.

---

## Step 1 — Describe your site in `local.env`

```bash
cp local.env.example local.env
$EDITOR local.env
```

`env.sh` sources it, or `$SITE_ENV` if set, and derives every path from it.

| Key | Meaning |
|-----|---------|
| `PROJECT` / `PARTITION` | SLURM account and partition, exported as `SBATCH_ACCOUNT` / `SBATCH_PARTITION` |
| `VENV_BASE` | Shared directory holding the venvs, modulefiles and examples |
| `GPU_ARCH` / `ROOFLINE_ARCH` | Wheel extra `device-<arch>`, and the roofline extractor's counter set |
| `ROCM_VERSION` / `ROCM_INDEX_URL` | Nightly build to validate, and the wheel index |
| `PYTHON_MODULE` / `BASE_PYTHON` | Base interpreter used to create the venv |
| `LMOD_INIT` | Lmod init script, for sites where `module` is undefined in non-login shells |
| `PROXY` | Outbound proxy, for sites whose nodes have no direct internet |
| `MIOPEN_LOCAL_DB` / `MIOPEN_TMP_BASE` | Keep MIOpen's SQLite databases on node-local storage |

Two worked examples ship in `local.env.example`: OLCF Frontier (MI250X, Lustre,
proxied compute nodes) and AAC6 (MI300A, NFS, direct internet).

`VENV_BASE` must be visible from **both** the login and the compute nodes, since
the login node builds the venv and the jobs read it. Prefer the fastest shared
filesystem available; node-local paths such as `/tmp` or `/dev/shm` do not work
however fast they are.

To validate a different nightly later, change `ROCM_VERSION` and rebuild.

## Step 2 — Build the environment

Run this **on a login node** (one with internet access):

```bash
bash install_rocm_pytorch.sh
```

It is idempotent, and it:

- creates the venv under `${VENV_BASE}/venvs/rocm-pytorch-pip` and installs
  nightly ROCm + PyTorch + profilers from the multi-arch nightly index, plus
  `transformers` (required by the training script),
- runs `rocm-sdk init` to extract the development headers and device code,
- generates an Lmod modulefile at
  `${VENV_BASE}/modulefiles/rocm-pytorch-pip/${ROCM_VERSION}.lua`,
- pre-stages everything the jobs need, so compute nodes need no internet:
  `rooflineExtractor` and its requirements, the isolated
  `rocprof-compute analyze` venv (`numpy==1.26.4`), and the CIFAR-100 dataset.

[`ROCM_PYTORCH_PIP_VENV_SETUP.md`](./ROCM_PYTORCH_PIP_VENV_SETUP.md) walks
through the same venv build by hand, for adapting it or debugging a failure.

Confirm the nightly runs GPU kernels through PyTorch before profiling:

```bash
source setup_rocm.sh
srun -n1 --gpus=1 python3 -c "import torch; print('torch', torch.__version__); \
x = torch.ones(4, device='cuda:0'); print('device ok:', (x+1).sum().item())"
```

`setup_rocm.sh` loads the generated module when it exists and otherwise
activates the venv directly; the SLURM scripts source it as `../setup_rocm.sh`.
If you see `device ok:`, you are ready to profile.

## Step 3 — Run the SLURM scripts

Each sub-directory contains a single-process SLURM script that sources
`../setup_rocm.sh`, pre-downloads the dataset if needed, and runs the workload
under one tool. All of them use a single GPU and a short run
(`--batch-size 32 --max-steps 5`) so a nightly can be checked quickly.

`run_all.sh` submits the whole suite from a login node, each script from its own
directory, chaining the analyze job after its profile job with
`--dependency=afterok`. It prints one `<tool> <jobid>` line per submission:

```bash
bash run_all.sh
```

| Tool | Directory | Script | What it produces |
|------|-----------|--------|------------------|
| None (baseline) | `no-profiling/` | `slurm_single_process_noprofile.sh` | Plain training run — confirms the workload runs without a profiler. |
| ROCm Compute Profiler (profile) | `rocm-compute-profiler/` | `slurm_single_process_profile.sh` | Hardware-counter profile under `workloads/`. |
| ROCm Compute Profiler (analyze) | `rocm-compute-profiler/` | `slurm_single_process_analyze.sh` | Analysis report from the profile above. |
| RocProfiler (kernels) | `rocprofv3/` | `slurm_single_process_kernels.sh` | Kernel stats + trace CSVs under `single_process/`. |
| RocProfiler (traces) | `rocprofv3/` | `slurm_single_process_traces.sh` | System timeline trace (`.pftrace`) under `single_process/`. |
| ROCm Systems Profiler | `rocm-systems-profiler/` | `slurm_single_process.sh` | Sampling profile + trace under `rocprofsys-python3-output/`. |
| Roofline Extractor | `roofline-extractor/` | `slurm_single_process.sh` | Per-kernel roofline analysis + interactive HTML plot under `output/`. |

To submit one tool on its own, do it **from that tool's directory** (the scripts
use `SLURM_SUBMIT_DIR` to locate themselves), having sourced `env.sh` so the
account and partition reach `sbatch`:

```bash
source env.sh
cd no-profiling
sbatch slurm_single_process_noprofile.sh
```

Check job status and output:

```bash
squeue --me
# stdout/stderr land in the submit directory as <name>_<jobid>.out / .err
```

### Analyzing the results

> **Note — running `rocprof-compute analyze`:** run it only **after** its
> profile job has finished (the counter database must exist); `run_all.sh`
> chains it for you. Because it needs `numpy==1.26.4` (vs the shared venv's
> `numpy>=2.0`), the analysis script uses its own isolated venv
> (`${VENV_BASE}/venvs/rocprof-compute-analyze`) and never touches the shared venv.

- **ROCm Compute Profiler:** submit the companion analysis job from
  `rocm-compute-profiler/` (it locates the workload and runs the analysis for you):

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

- **Account and partition are not in the `#SBATCH` headers.** SLURM parses those
  directives before the script runs, so they cannot expand variables. `env.sh`
  exports `SBATCH_ACCOUNT` and `SBATCH_PARTITION` instead, which `sbatch` honours
  at submit time. Source `env.sh` (or use `run_all.sh`) before submitting by hand.
- **MIOpen on NFS/Lustre.** MIOpen's SQLite perf and kernel databases need real
  POSIX file locking, and `conv2d` fails with
  `RuntimeError: miopenStatusInternalError` without it. `setup_rocm.sh` points
  `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` at node-local storage; set
  `MIOPEN_LOCAL_DB=0` if your shared filesystem locks correctly.
- **Pre-fetch the dataset with `download_only_nogpus.py`.**
  `train_cifar_100.py --download-only` calls `dist.init_process_group("nccl", ...)`
  even in download-only mode, which hangs on a GPU-less login node.
- If a job fails to start, check the time limits in the script headers and the
  `PARTITION` in `local.env` against your cluster's limits.
- The scripts derive a per-job rendezvous port from the SLURM job ID, so
  multiple jobs can share a node without port collisions.
