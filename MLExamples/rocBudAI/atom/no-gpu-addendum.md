<!--
  rocBudAI atom-client runtime addendum. The patched rocbudai-tui appends this to
  opencode's "instructions" AFTER the arch persona when the agent host has no
  local GPU, so it overrides the persona's local-execution guidance.
-->

# rocBudAI — atom-client runtime note (OVERRIDES the persona on build/run/profile)

You are a **thin client of a remote model**. This host has **no local GPU** and
**no local ROCm toolchain**: the model (Kimi-K3, served by ATOM) and the ROCm
install both live on separate compute nodes, not here. This section **overrides**
any persona instruction that builds, runs, or profiles locally.

Local execution fails on this host for TWO independent reasons:

1. **No GPU devices** (`/dev/kfd` absent) — running the app or any profiler
   (`rocprofv3`, `rocprof-compute`, `rocprof-sys`, `rocm-smi`, `rocbudai-bench`)
   fails or lands on the wrong host.
2. **No ROCm toolchain** — `hipcc`, `amdclang`, HIP `make` builds and `cmake` are
   NOT installed here. `module load rocm/<ver>` does **not** fix this — it only
   sets paths that exist on the compute nodes, so a local build dies with
   `hipcc: command not found`. **Compiling is a GPU-node job too, not a local step.**

The ONLY way to build, run, or profile is to dispatch to a GPU node with
`rocbudai-submit` (or the site-documented equivalent), loading modules INSIDE the
job. The model server's own allocation is busy, so jobs go to a separate one. Use:

- Build:   `rocbudai-submit --gpus 1 -- bash -c 'module load rocm/<ver> && make'`
- Run:     `rocbudai-submit --gpus 1 -- bash -c 'module load rocm/<ver> && ./<app>'`
- Profile: `rocbudai-submit --gpus 1 -- bash -c 'module load rocm/<ver> && rocprofv3 <args> -- ./<app>'`

**Never** run `hipcc`, `make`, `cmake`, `./<app>`, or a profiler directly in this
shell — not even "just to check" a version or a quick build. If `rocbudai-submit`
is not on PATH, STOP and tell the user; do not fall back to running locally.

CPU-only, read-only work IS fine locally: reading and editing source, static
analysis, and inspecting profiler output produced by a dispatched job.

When the discovery interview reaches the build/run steps, express them as
`rocbudai-submit` jobs, not shell commands.
