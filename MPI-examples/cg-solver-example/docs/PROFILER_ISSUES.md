# Profiler issues — CG solver (status & action items)

Issues encountered while adding and running the profilers documented under
[`docs/profilers/`](profilers/README.md) for the CG solver, with the current
workaround and what still needs to be addressed. Companion doc for the PyTorch
examples: [`MLExamples/Pytorch/PROFILER_ISSUES.md`](../../../MLExamples/Pytorch/PROFILER_ISSUES.md)
(shared items — Score-P GPU on ROCm 7.2.x, CubeGUI viewers, headless graphics — are
covered once there and cross-referenced).

**Verified platform:** AMD MI300A (AAC6 `PPAC_MI300A_SPX`), ROCm 6.4.3 / 7.2.3 /
7.2.4 / 7.13.0, Score-P 9.4 (and 11.0-dev), OpenMPI 5.0.10.

Status legend: **OPEN** (needs a fix we do not control) · **WORKAROUND** (documented,
usable today) · **RESOLVED** (fixed in this repo).

---

## 1. Score-P

### 1.1 ROCm adapter aborts on ROCm 7.2.x — **OPEN (upstream)**
- **Symptom:** with GPU capture enabled, the run **SIGABRTs** before finalize; the
  experiment dir is empty (no `profile.cubex`).
- **Cause:** the Score-P ROCm (roctracer/rocprofiler) adapter is incompatible with
  the rocprofiler-sdk in ROCm 7.2.x.
- **Impact:** no GPU/HIP kernel attribution on the toolchain the rest of the stack
  uses (7.2.x).
- **Workaround:** full GPU+HIP+MPI capture is reliable on **ROCm 6.4.3**; on 7.2.x
  run **MPI-only** (`SCOREP_GPU=0`). Documented in
  [`profilers/scorep.md`](profilers/scorep.md) §7.
- **To address:** re-test each Score-P release against ROCm ≥7.2.x; file/track an
  upstream Score-P + rocprofiler-sdk compatibility issue; revisit when a fixed
  Score-P lands so GPU capture works on the production ROCm.

### 1.2 `cube_dump -c region` hangs — **WORKAROUND**
- **Symptom:** `cube_dump` hangs (no output) on some `profile.cubex` files.
- **Workaround:** use `cube_stat -p -m time` (with a `timeout` wrapper) for the flat
  region/time table; this is what `run_scorep.sh` captures.
- **To address:** confirm the CubeLib version pairing (`cube_dump` needs a
  version-matched CubeLib for the writer that produced the file); document the
  matching binary or drop `cube_dump` from the workflow permanently.

### 1.3 `SCOREP_TOTAL_MEMORY` / compiler instrumentation — **RESOLVED (documented)**
- **Symptom:** `No free memory page … increase SCOREP_TOTAL_MEMORY`, or huge
  overhead with compiler instrumentation.
- **Fix:** build with `scorep --nocompiler --mpp=mpi` (the `Makefile SCOREP=1`
  path), raise `SCOREP_TOTAL_MEMORY` (e.g. `256M`). Documented.

### 1.4 `libpapi.so.7.1` missing on login node — **WORKAROUND**
- Score-P runs need `libpapi.so.7.1`, present only on **compute nodes**. Build/link
  anywhere; **run** in an allocation. Documented.

### 1.5 Cross-ROCm binary incompatibility — **RESOLVED (documented)**
- A binary linked under one ROCm major fails at runtime on another
  (`undefined symbol: hsa_amd_memory_get_preferred_copy_engine`). Rebuild per
  toolchain — the `SCOREP=1` build and the sweeps already do.

---

## 2. GUI / graphics viewers (shared with the ML examples)

### 2.1 No native Cube/Vampir GUI; TAU paraprof unusable — **WORKAROUND**
- **Symptom:** CubeGUI and Vampir are not installed; TAU's `paraprof` will not start
  (**no Java runtime** on the nodes).
- **Workaround:** use text tools (`scorep-score`, `cube_stat`, `pprof`) and the
  committed matplotlib figures ([`profilers/figs/`](profilers/figs/)); view a real
  GUI via the CubeGui AppImage in a remote desktop (below).
- **To address:** request a site install of CubeGUI/Vampir and a JRE (for paraprof),
  or the Cube 4.9.1 JupyterLab (WASM) service.

### 2.2 CubeGUI container reference was wrong — **RESOLVED**
- **Was:** docs told users to `apptainer pull docker://ghcr.io/scalasca/cubegui:latest`,
  which **404/403s (image does not exist)**.
- **Fixed:** docs now use the official **`CubeGui-4.9.1.AppImage`** (verified:
  downloaded, `--appimage-extract`, `usr/bin/cube` present) plus the `cube_server` +
  `ssh -L` remote option. See [`profilers/scorep.md`](profilers/scorep.md) §6.

### 2.3 Headless CubeGUI screenshot not possible on the frontend — **OPEN (env)**
- **Symptom:** cannot auto-capture a CubeGUI screenshot from a login shell.
- **Cause:** no `Xvfb`; the TurboVNC `Xvnc` is missing `libturbojpeg.so.0`
  (not installed anywhere on the node); `cube` is Qt/xcb and needs a real X display
  + `libxcb-cursor0`.
- **Workaround:** open CubeGUI in an interactive **TurboVNC / noVNC / `ssh -X`**
  session (`man aac6_vnc` / `aac6_novnc` / `aac6_x11`) and screenshot there.
- **To address:** install `libturbojpeg0` (fixes TurboVNC `Xvnc`) and/or `xvfb` +
  `libxcb-cursor0` on the compute image so screenshots can be scripted headlessly.

---

## 3. hipBLASLt `hipblaslt/patched` — no effect on this solver — **RESOLVED (measured)**
- **Finding:** loading `hipblaslt/patched` (auto on 7.2.x, manual on 7.13.0) does
  **not** recover the 6.4.3→7.x compute regression: solve time moves ≤0.5 % (within
  noise). The solver uses **rocSPARSE SpMV + rocBLAS level-1**; the patch is a
  narrow **skinny-fp16-GEMM** Tensile overlay with no GEMM to accelerate here.
- **Where:** measured A/B in `CG-GPU/STUDY_REPORT.md` §5.3
  (`sweep_rocm_versions.sh HBL=both`). The sweeps and `run_scorep.sh` still load it
  automatically (correct default; harmless).
- **To address (real regression):** the ~2.6–2.8× 6.4.3→7.x slowdown lives in the
  **rocSPARSE CSR SpMV kernel** (§5/§5.1). Next step: one-iteration `rocprofv3`
  kernel trace 6.4.3 vs 7.2.4 to identify the kernel and file a rocSPARSE issue.

---

## 4. Other profilers (documented, spot-checked)
The remaining per-profiler pages ([`rocprofv3`](profilers/rocprofv3.md),
[`rocprof-compute`](profilers/rocprof-compute.md),
[`rocprofiler-systems`](profilers/rocprofiler-systems.md),
[`tau`](profilers/tau.md), [`hpctoolkit`](profilers/hpctoolkit.md),
[`likwid`](profilers/likwid.md), [`uprof`](profilers/uprof.md),
[`perf`](profilers/perf.md), etc.) carry verified commands. Known cross-cutting
items: TAU needs a JRE for the GUI (§2.1); GUI/timeline viewers (Perfetto,
rocprof-compute `--gui`) require the remote-desktop methods in §2. No blocking
issues beyond those recorded above.

---

## Action-item checklist
- [ ] **Score-P GPU capture on ROCm ≥7.2.x** (§1.1) — track upstream fix; retest per release.
- [ ] **`cube_dump` version-matched CubeLib** (§1.2) — pin or remove from workflow.
- [ ] **Site GUI install** (§2.1) — CubeGUI/Vampir + JRE, or Cube JupyterLab WASM.
- [ ] **Compute-image graphics deps** (§2.3) — `libturbojpeg0`, `xvfb`, `libxcb-cursor0`.
- [ ] **rocSPARSE SpMV 7.x regression** (§3) — `rocprofv3` kernel trace + upstream report.
