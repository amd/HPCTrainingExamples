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

> **Re-evaluated 2026-07-18** after a site image update. Newly resolved by the site:
> the graphics/GUI dependencies (`Xvfb`, `libturbojpeg0`, `libxcb-cursor0`, a JRE) are
> now installed — a **headless CubeGUI screenshot works** (§2.3) and **paraprof** can
> start (§2.1). Re-tested and **still open**: the Score-P ROCm adapter still aborts on
> ROCm 7.2.4 (§1.1); the bf16 hipBLASLt transformer hang persists (ML doc §5.2).

---

## 1. Score-P

### 1.1 ROCm adapter aborts on ROCm 7.2.x — **OPEN (upstream)** · re-confirmed 2026-07-18
- **Symptom:** with GPU capture enabled, the run **SIGABRTs** before finalize; the
  experiment dir is empty (no `profile.cubex`).
- **Cause:** the Score-P ROCm (roctracer/rocprofiler) adapter is incompatible with
  the rocprofiler-sdk in ROCm 7.2.x.
- **Impact:** no GPU/HIP kernel attribution on the toolchain the rest of the stack
  uses (7.2.x).
- **Re-test (2026-07-18):** `run_scorep.sh` with `ROCM_VERSION=7.2.4 SCOREP_GPU=1`
  **still aborts** (no `profile.cubex`); the same run on **6.4.3 still succeeds**
  (fresh `profile.cubex`, HIP 38.9 % / MPI 24.1 %). No change — still open.
- **Workaround:** full GPU+HIP+MPI capture is reliable on **ROCm 6.4.3**; on 7.2.x
  run **MPI-only** (`SCOREP_GPU=0`). Documented in
  [`profilers/scorep.md`](profilers/scorep.md) §7.
- **To address:** re-test each Score-P release against ROCm ≥7.2.x; file/track an
  upstream Score-P + rocprofiler-sdk compatibility issue; revisit when a fixed
  Score-P lands so GPU capture works on the production ROCm.

### 1.2 `cube_dump` not provided by the `scorep` module — **WORKAROUND**
- **Symptom:** earlier `cube_dump -c region` hung on some files; on re-test
  (2026-07-18) `cube_dump` is simply **not on `PATH`** from the `scorep/9.4` module
  (`cube_dump: command not found`) — only `cube_stat` / `scorep-score` ship with it.
- **Workaround:** use `cube_stat -p -m time` (with a `timeout` wrapper) for the flat
  region/time table; this is what `run_scorep.sh` captures. Verified working.
- **To address:** if `cube_dump` is needed, load a matching **CubeLib** module/AppImage
  (`squashfs-root/usr/bin/` from the CubeGUI AppImage carries the tools); otherwise
  `cube_stat` covers the workflow.

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

### 2.1 No native Cube/Vampir GUI; paraprof JRE — **WORKAROUND** (JRE now present, 2026-07-18)
- **Symptom:** CubeGUI and Vampir are still not installed as native modules.
- **Update (2026-07-18):** a **JRE is now installed** (`java` → OpenJDK 21), so TAU's
  `paraprof` and other Java GUIs can start (they still need an X display, §2.3).
- **Workaround:** use text tools (`scorep-score`, `cube_stat`, `pprof`) and the
  committed matplotlib figures ([`profilers/figs/`](profilers/figs/)); view a real
  GUI via the CubeGui AppImage (native Cube/Vampir remain a site-install ask).
- **To address:** a site install of native CubeGUI/Vampir modules would be convenient
  but is no longer blocking — the AppImage renders headlessly now (§2.3).

### 2.2 CubeGUI container reference was wrong — **RESOLVED**
- **Was:** docs told users to `apptainer pull docker://ghcr.io/scalasca/cubegui:latest`,
  which **404/403s (image does not exist)**.
- **Fixed:** docs now use the official **`CubeGui-4.9.1.AppImage`** (verified:
  downloaded, `--appimage-extract`, `usr/bin/cube` present) plus the `cube_server` +
  `ssh -L` remote option. See [`profilers/scorep.md`](profilers/scorep.md) §6.

### 2.3 Headless CubeGUI screenshot — **RESOLVED (2026-07-18, site deps installed)**
- **Was:** could not auto-capture a CubeGUI screenshot from a login shell — no `Xvfb`,
  the TurboVNC `Xvnc` was missing `libturbojpeg.so.0`, and Qt/xcb needed `libxcb-cursor0`.
- **Fixed by the site image:** `Xvfb` + `xvfb-run`, `libturbojpeg.so.0`, and
  `libxcb-cursor.so.0` are now all installed (`Xvnc` also resolves all libs).
- **Now:** the helper [`CG-GPU/shot_cubegui.sh`](../CG-GPU/shot_cubegui.sh) starts
  `Xvfb :99`, launches the CubeGui AppImage on `profile.cubex`, and captures the
  screen with Pillow (xcb) — verified end-to-end, producing
  [`profilers/figs/cg_cubegui.png`](profilers/figs/cg_cubegui.png) (real 3-pane
  metric/call/system view). Embedded in [`profilers/scorep.md`](profilers/scorep.md) §6.
- **Note:** interactive **TurboVNC / noVNC / `ssh -X`** (`man aac6_vnc` /
  `aac6_novnc` / `aac6_x11`) is still the way to *click through* trees; the headless
  path is for scripted/CI screenshots (no window manager, so trees render collapsed).

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
- [ ] **Score-P GPU capture on ROCm ≥7.2.x** (§1.1) — still open (re-confirmed 2026-07-18 on 7.2.4); track upstream fix; retest per release.
- [ ] **`cube_dump` from a CubeLib module/AppImage** (§1.2) — optional; `cube_stat` covers the workflow.
- [x] **Compute-image graphics deps** (§2.3) — `libturbojpeg0`, `xvfb`, `libxcb-cursor0` **installed** (site, 2026-07-18); headless screenshot works.
- [x] **JRE for paraprof/Java GUIs** (§2.1) — **installed** (OpenJDK 21, 2026-07-18).
- [ ] **Native CubeGUI/Vampir modules** (§2.1) — nice-to-have; AppImage now covers it.
- [ ] **rocSPARSE SpMV 7.x regression** (§3) — `rocprofv3` kernel trace + upstream report.
