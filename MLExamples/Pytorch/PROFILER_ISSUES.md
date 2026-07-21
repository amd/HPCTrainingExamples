# Profiler issues — PyTorch examples (status & action items)

Issues encountered while adding and running the profilers documented under
[`common/profilers/`](common/profilers/README.md) for the `imagenet`, `minGPT-ddp`,
and `FSDP2` examples, with the current workaround and what still needs to be
addressed. Companion doc for the CG solver:
[`MPI-examples/cg-solver-example/docs/PROFILER_ISSUES.md`](../../MPI-examples/cg-solver-example/docs/PROFILER_ISSUES.md).

**Verified platform:** AMD MI300A (AAC6 `PPAC_MI300A_SPX`), ROCm 7.2.3,
PyTorch 2.12.0, Score-P 11.0-dev, 2 GPUs.

Status legend: **OPEN** (needs a fix we do not control) · **WORKAROUND** (documented,
usable today) · **RESOLVED** (fixed in this repo).

> **Re-evaluated 2026-07-18** after a site image update. Newly resolved by the site:
> the shared graphics/GUI deps (`Xvfb`, `libturbojpeg0`, `libxcb-cursor0`, a JRE) are
> installed — a headless CubeGUI screenshot now works (§4, and CG doc §2.3). Re-tested
> and **still open**: the bf16 hipBLASLt transformer hang on ROCm 7.2.3 (§5.2 — still
> hangs >200 s; `TORCH_BLAS_PREFER_HIPBLASLT=0` still fixes it, 132,320 tok/s), and
> Score-P GPU/RCCL capture on 7.2.x (§2.3, same adapter abort as the CG solver).

---

## 1. Environment / module hygiene (bit every profiler)

### 1.1 An active virtualenv shadows the module Python — **RESOLVED (documented)**
- **Symptom:** `ModuleNotFoundError: No module named 'torch'` under `srun`/`sbatch`
  even after `module load pytorch/2.12.0`.
- **Cause:** an inherited venv (e.g. the matplotlib `figs` venv, or a Score-P venv)
  is first on `PATH`/`VIRTUAL_ENV`, so `python` resolves to the venv, not the module
  interpreter (`/nfsapps/.../pytorch-v2.12.0/pytorch/bin/python`).
- **Fix:** scrub the venv before loading modules —
  `unset VIRTUAL_ENV PYTHONHOME` and drop venv dirs from `PATH`. Use `python`
  (the module provides `python`, not necessarily `python3`).
- **To address:** none (documented); keep the scrub in any batch driver.

### 1.2 Batch (non-login) shell has no `MODULEPATH` — **RESOLVED (documented)**
- **Symptom:** `module load rocm/... openmpi pytorch/...` silently fails in an
  `sbatch` script; nothing downstream works.
- **Cause:** the module hierarchy (rocm → openmpi → pytorch) is set up by the login
  profile; a plain `#!/bin/bash` batch shell that only sources `modules.sh` does not
  get the base `MODULEPATH`.
- **Fix:** use a **login shell** (`#!/bin/bash -l`) in job scripts, then scrub the
  venv (§1.1).

### 1.3 `libpapi.so.7.1` on compute nodes only — **WORKAROUND**
- Score-P Python bindings need `libpapi.so.7.1`, present on **compute** nodes but not
  the login node. Build the Score-P venv on **login** (needs network), **run** on a
  compute node. Documented in [`common/profilers/scorep.md`](common/profilers/scorep.md) §1.

---

## 2. Score-P (Python)

### 2.1 Instrumentation bindings not in the module — **RESOLVED (documented)**
- **Symptom:** `import scorep` fails; the `scorep` module ships only the **OTF2**
  Python bindings, not the Score-P Python instrumentation bindings.
- **Fix:** `pip install scorep` into a venv layered on the PyTorch module (build on
  login, run on compute). Documented.

### 2.2 Automatic Python instrumentation is too heavy — **RESOLVED (documented)**
- **Symptom:** the run hangs / produces an enormous trace under `python -m scorep`.
- **Cause:** automatic Python instrumentation intercepts every Python call — far too
  much for PyTorch.
- **Fix:** run with `--nopython` + a few **user regions**
  ([`common/scorep_ml.py`](common/scorep_ml.py), opt-in via `SCOREP_ML=1`); drive it
  with `torchrun --no-python` → `python -m scorep` per rank
  ([`common/scorep_launch.sh`](common/scorep_launch.sh)).

### 2.3 No GPU/RCCL capture under Score-P here — **OPEN (upstream) / WORKAROUND** · re-confirmed 2026-07-18
- **Symptom:** Score-P profiles show Python regions only — no GPU kernels, no RCCL.
- **Cause:** the PyTorch module is on **ROCm 7.2.x**, where the Score-P ROCm adapter
  aborts (same root cause as the CG solver, §1.1 there), and `torchrun` uses RCCL,
  not MPI.
- **Re-test (2026-07-18):** the CG solver's `SCOREP_GPU=1` run on **ROCm 7.2.4 still
  aborts** (same adapter), so nothing changed for the GPU dimension here.
- **Workaround:** use [torch.profiler](common/profilers/torch-profiler.md),
  [rocprofv3](common/profilers/rocprofv3.md), or
  [rocprofiler-systems](common/profilers/rocprofiler-systems.md) for GPU/RCCL detail;
  Score-P is for per-rank Python phase timing.
- **To address:** revisit when a Score-P release supports rocprofiler-sdk on 7.2.x.

### 2.4 Per-rank experiment dirs must pre-exist — **RESOLVED**
- `Can't create experiment directory .../rank_0` — the launcher `mkdir -p`s the base
  dir; hand-launches must create `$SCOREP_EXP_BASE` first.

---

## 3. Other ML profilers

### 3.1 TensorBoard / `torch-tb-profiler` not in the module — **WORKAROUND**
- Install into a venv layered on the module (`pip install tensorboard
  torch-tb-profiler`). Documented in [`common/profilers/tensorboard.md`](common/profilers/tensorboard.md).

### 3.2 DeepSpeed pip wheel aborts without a CUDA toolkit — **WORKAROUND**
- **Symptom:** a pip-installed DeepSpeed aborts at import with
  `MissingCUDAException: CUDA_HOME does not exist`.
- **Fix:** prefer the **site ROCm-built DeepSpeed** (imports cleanly); if using a pip
  wheel, `export CUDA_HOME=${ROCM_PATH:-/opt/rocm}` (the FlopsProfiler is pure
  Python). Documented in [`common/profilers/deepspeed-flops.md`](common/profilers/deepspeed-flops.md).

### 3.3 First-step MIOpen autotune inflates measured regions — **RESOLVED (documented)**
- The first training steps include one-off MIOpen autotune; with `--warmup` steps
  also running synchronized, `train_step_sync` looks inflated. Raise
  `--warmup`/`--iters` for steady state; set `MIOPEN_FIND_MODE=FAST`. Documented.

---

## 4. GUI / graphics viewers — shared with the CG solver
Native CubeGUI/Vampir modules are still not installed (use the **CubeGui AppImage**;
the `ghcr.io/scalasca/cubegui` container never existed). **Update (2026-07-18):** the
site image now ships `Xvfb`, `libturbojpeg0`, `libxcb-cursor0`, and a JRE (OpenJDK 21),
so a **headless CubeGUI screenshot now works** and Java GUIs (paraprof) can start —
see the CG doc [§2.3](../../MPI-examples/cg-solver-example/docs/PROFILER_ISSUES.md#23-headless-cubegui-screenshot--resolved-2026-07-18-site-deps-installed)
(resolved) and the verified capture
[`cg_cubegui.png`](../../MPI-examples/cg-solver-example/docs/profilers/figs/cg_cubegui.png).
For interactive click-through use TurboVNC / noVNC / `ssh -X` (`man aac6_vnc` /
`aac6_novnc` / `aac6_x11`), or the committed matplotlib figures in
[`common/profilers/figs/`](common/profilers/figs/).

---

## 5. hipBLASLt on the PyTorch examples

### 5.1 `hipblaslt/patched` has no measurable effect — **RESOLVED (measured)**
- A/B (stock vs patched) shows no change: ResNet-50 `--amp` 244 vs 245 img/s; GEMM
  microbenches identical. The module is a narrow `HIPBLASLT_TENSILE_LIBPATH` overlay
  for **skinny fp16 GEMMs**, which these workloads don't hit. Full data:
  [`common/hipblaslt-notes.md`](common/hipblaslt-notes.md) §1.

### 5.2 bf16 transformer hang in hipBLASLt on ROCm 7.2.x — **OPEN (upstream) / WORKAROUND** · re-confirmed 2026-07-18
- **Symptom:** `minGPT-ddp --amp` and `FSDP2 --mixed-precision` **stall for minutes**
  in hipBLASLt on the first bf16 GEMMs (fp32 is fine; ResNet `--amp` is fine;
  `hipblaslt/patched` does **not** fix it).
- **Cause:** the 7.2.x hipBLASLt path for these transformer bf16 GEMM shapes. bf16
  works normally on ROCm 6.4.3.
- **Re-test (2026-07-18, ROCm 7.2.3 / PyTorch 2.12.0, 2 GPU, n_layer=12 n_embd=768):**
  default (hipBLASLt) bf16 **still hangs** — killed at a 200 s timeout; with
  `TORCH_BLAS_PREFER_HIPBLASLT=0` the same run finishes in **23 s at 132,320 tok/s**.
  No change — still open, workaround still valid.
- **Workaround:** `export TORCH_BLAS_PREFER_HIPBLASLT=0` (routes to rocBLAS) — bf16
  then runs and is ~2.3× faster than fp32 (~132–138k vs 60,723 tok/s). Details:
  [`common/hipblaslt-notes.md`](common/hipblaslt-notes.md) §2.
- **To address:** file an upstream hipBLASLt/PyTorch issue with the reproducing
  transformer shapes on ROCm 7.2.x; retest on later ROCm.

---

## Action-item checklist
- [ ] **Score-P GPU/RCCL capture on ROCm 7.2.x** (§2.3) — still open (re-confirmed 2026-07-18); track the same upstream fix as the CG solver.
- [ ] **bf16 transformer hipBLASLt hang** (§5.2) — still open (re-confirmed 2026-07-18); file upstream issue; retest on later ROCm; keep the `TORCH_BLAS_PREFER_HIPBLASLT=0` note.
- [x] **Compute-image graphics deps + JRE** (§4) — `xvfb`, `libturbojpeg0`, `libxcb-cursor0`, OpenJDK 21 **installed** (site, 2026-07-18); headless screenshot works.
- [ ] **Native CubeGUI/Vampir modules** (§4) — nice-to-have; AppImage now covers it.
- [ ] Keep venv-scrub + login-shell boilerplate (§1.1–1.2) in any new job scripts.
