# 8. Profiling: measuring communication and compute

The solver already reports a coarse **communication vs. compute** split from
`MPI_Wtime` timers (`g_halo_time`, `g_allreduce_time` in `cg.cpp`; see the
`comm total` / `halo exchange` / `dot allreduce` / `compute (rest)` lines). This
chapter shows how to go deeper with the profilers installed on the system —
attributing time to individual **kernels**, **transports** (SDMA/blit copies,
RCCL, MPI point-to-point/collective), and **hardware limits** (HBM/DRAM
bandwidth, FLOP/s) — for **both** the CPU reference and the GPU solver.

## Which profiler for which job

| Profiler | Load | Scope | Measures here |
|----------|------|-------|---------------|
| **rocprofv3** | `module load rocm` | GPU (`CG-GPU`) | Per-kernel time (compute); memory-copy, HIP, and **RCCL** API traces (communication); **ATT** instruction-level trace (`--att`, §A.4) |
| **rocprof-compute** | `module load rocm` | GPU | Roofline: achieved HBM bandwidth vs peak → proves SpMV is memory-bound |
| **rocprofiler-systems** | `module load rocprofiler-systems` | GPU + host | Perfetto **timeline**: halo-exchange overlap, SDMA vs blit copies, RCCL/MPI on the host thread |
| **TAU** | `module load tau` | MPI + GPU | Per-call MPI time (`MPI_Allreduce`, `MPI_Isend`, `MPI_Alltoallv`) + communication matrix + GPU kernels |
| **HPCToolkit** | `module load hpctoolkit` | MPI + GPU | Call-path sampling: MPI wait time vs compute, GPU kernel attribution |
| **likwid** | `module load likwid` | CPU (`CG-CPU`) | DRAM bandwidth (`MEM_DP`) and `FLOPS_DP` → CPU roofline of the reference solver |
| **AMD uProf** | add `.../AMDuProf_5.3-518/bin` to `PATH` | CPU (`CG-CPU`) | CPU hotspots (SpMV loop, dot) and memory/bandwidth analysis |
| **Linux perf** | none (in-box; run in an allocation) | CPU (`CG-CPU`) | Baseline HW counters: IPC + cache-miss rate (`perf stat`); cycle hotspots (`perf record/report`) (§K) |
| **cachegrind** (Valgrind) | `/usr/bin/valgrind` (compute node) | CPU (`CG-CPU`) | *Deterministic* simulated cache model: D1/LL miss counts per function/line (§L) |
| **IntelliKit** | `module load rocm intellikit` | GPU (`CG-GPU`) | *Decoded* metrics (`metrix`), kernel isolation (`kerncap`), correctness validation (`accordo`), source-level SQTT (`linex`) — agent/MCP-driven (§H) |
| **roofline-extractor** | `module load rocm roofline-extractor` | GPU (`CG-GPU`) | Automated per-kernel **percent-of-peak roofline** (AI @ HBM/L2/L1/LDS, limiter, % of roof, HTML plot) (§I) |
| **rocBudAI** | `salloc --comment=ollama; module load rocbudai` | GPU + host | **AI assistant** that drives the whole stack (build→profile→analyse→optimise loop, airgapped, writes `report.md`) (§J) |

The GPU-native tools (rocprofv3, rocprof-compute, rocprofiler-systems,
roofline-extractor, IntelliKit) target `CG-GPU`; the CPU tools (Linux `perf`,
Valgrind cachegrind, uProf; likwid where supported) target `CG-CPU`. **TAU** and
**HPCToolkit** are whole-application and work on either (they are the best way to
see the *MPI* communication that the ROCm GPU tools do not trace). **rocBudAI**
(§J) can drive any of them for you.

> **Verified on AAC6 / MI300A (ROCm 6.4.3, OpenMPI 5.0.10, 4 ranks, `Dubcova2.pm`).**
> All commands below were run against these solvers. Confirmed working:
> rocprofv3 (kernel + sys trace), rocprof-compute (roofline + Top-Kernels — SpMV
> `rocsparse::csrmvn` and the rocBLAS `axpy`/`dot`/`scal` kernels), rocprofiler-
> systems (Perfetto trace), TAU (MPI breakdown via `pprof`), HPCToolkit (`hpcrun`
> → `hpcstruct` → `hpcprof` database), and AMD uProf (hotspots — top is the CSR
> SpMV). **Exception:** `likwid-perfctr` counters are **not supported** on the
> MI300A APU CPU in this build (see §F).

## Ground rules (or your numbers are noise)

Everything from [chapter 3](03-correct-measurement.md) still applies under a
profiler — more so, because profiling adds overhead:

- Profile **inside an affinity-bound, `--exclusive` allocation** using
  `gpu_bind.sh` / `set_affinity_mi300a.sh`. Bad affinity dominates everything.
- **Fix the RHS** with `CG_SEED` so every profiled run solves the same system.
- Profile **one transport at a time** (`staged`, `isend`, `rccl`, `alltoallv`,
  `alltoallv_staged`, and the APU zero-copy `staged_unified` / `alltoallv_unified`
  with `HSA_XNACK=1`) and compare across runs.
- Give **each rank its own output directory** (use `$OMPI_COMM_WORLD_RANK`).

> **Launch note (verified on AAC6 / MI300A, ROCm 6.4.3, OpenMPI 5.0.10).** Use an
> `sbatch` job (or `salloc`) that requests GPUs, e.g.
> `sbatch -N1 --exclusive --gpus=4`. An `sbatch` script body runs *on the compute
> node*, which matters for the **CPU counter tools** (uProf) — they must execute
> where the ranks run, so launch them **per rank under `mpirun`**, not by wrapping
> `mpirun` itself. If `mpirun` complains about slots under a one-task step, add
> `--oversubscribe` (or `export OMPI_MCA_rmaps_base_oversubscribe=true`). The
> GPU-native tools (rocprofv3/rocprof-compute/rocprofiler-systems) run fine under
> `mpirun` and write per-rank output.

> **Tip — label the timeline.** For sharper GPU traces, wrap the halo and
> all-reduce regions in ROCTx ranges (`#include <roctracer/roctx.h>`;
> `roctxRangePush("halo")` / `roctxRangePop()`), then link `-lroctx64`. rocprofv3
> and rocprofiler-systems will show these named ranges next to the kernels.

> **`module load rocm/<ver>` pulls in the patched profilers.** The `rocm/<ver>`
> modulefile prepends `rocm-patches-<ver>/rocprof-compute/bin`, so `rocprof-compute`
> resolves to the self-contained **Nuitka single-file executable** (it bundles
> pandas/dash/matplotlib/…) rather than the plain `rocm-<ver>/bin` Python script
> that would fail on missing deps. If the site also ships a dedicated `rocm_patches`
> module, load it too. **Gotcha:** never pipe or command-substitute `module`
> (e.g. `module load rocm/7.2.3 | tail`) — a pipe runs the `module` function in a
> **subshell**, discarding its `eval`'d `PATH`/`ROCM_PATH`/`LD_LIBRARY_PATH` changes,
> after which `rocprof-compute` silently falls back to the unpatched script. Call
> `module` plainly. See the verified end-to-end driver
> [`CG-GPU/prof_all_723_test.sh`](../CG-GPU/prof_all_723_test.sh) and
> [chapter 9](09-perf-security-demo.md#do-the-rocm-723-profilers-still-work-at-paranoid2--yes-verified).

---

## A. CG-GPU — rocprofv3 (kernels + transports)

### A.1 Compute: per-kernel trace

```bash
module load rocm openmpi
cd CG-GPU && make
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh \
  rocprofv3 --kernel-trace --output-format csv \
  -d prof_kern_rank_${OMPI_COMM_WORLD_RANK:-0} \
  -- ./cg_gpu src/Dubcova2.pm rccl
```

The kernel CSV shows where GPU compute goes: `rocsparse_spmv` (dominant), the
`rocsparse_dgthr` gather that packs the send buffer, and the rocBLAS
`ddot`/`daxpy`/`dscal` kernels. Roll up total ns per kernel to rank the compute.

### A.2 Communication: memory-copy + RCCL/HIP traces

```bash
# staged / alltoallv_staged: see the D->H and H->D staging copies
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh \
  rocprofv3 --memory-copy-trace --hip-trace --output-format pftrace \
  -d prof_comm_rank_${OMPI_COMM_WORLD_RANK:-0} \
  -- ./cg_gpu src/Dubcova2.pm staged

# rccl: --sys-trace additionally captures the RCCL API (ncclSend/ncclRecv)
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh \
  rocprofv3 --sys-trace --output-format pftrace \
  -d prof_rccl_rank_${OMPI_COMM_WORLD_RANK:-0} \
  -- ./cg_gpu src/Dubcova2.pm rccl
```

Open the per-rank `*.pftrace` at <https://ui.perfetto.dev>. For `staged` you see
the D↔H staging copies around each SpMV; for `rccl` you see `ncclSend`/`ncclRecv`
on the `rccl_stream` overlapping the on-proc SpMV on the default stream. This is
the halo-exchange cost the solver rolls into `g_halo_time`, now resolved by
transport. (rocprofv3 traces RCCL and HIP but **not** MPI itself — use TAU or
HPCToolkit in sections D/E for the `MPI_Isend`/`MPI_Alltoallv`/`MPI_Allreduce`
timing.)

### A.3 Bandwidth counters (optional)

```bash
printf 'pmc: FETCH_SIZE WRITE_SIZE L2CacheHit VALUBusy\n' > counters.txt
mpirun -n 4 ./gpu_bind.sh rocprofv3 -i counters.txt --output-format csv \
  -d prof_pmc_rank_${OMPI_COMM_WORLD_RANK:-0} -- ./cg_gpu src/Dubcova2.pm rccl
```

`FETCH_SIZE`+`WRITE_SIZE` ÷ kernel time = achieved HBM bandwidth per kernel — or
let `rocprof-compute` do the math (next section).

### A.4 Instruction-level: Advanced Thread Trace (ATT)

The traces above stop at the *kernel* boundary. **Advanced Thread Trace** (ATT,
`rocprofv3 --att`) goes one level deeper: it records wavefront execution at the
**ISA-instruction** granularity on selected compute units, so you get a
per-instruction hotspot/hit map, stall reasons, VALU/VMEM issue behaviour, and
occupancy for a single kernel — the tool for asking *why* the SpMV is
bandwidth-bound (memory-stall dominated) rather than just *that* it is.

ATT produces an enormous amount of data, so you **must** target it:

- **One kernel** — filter with `--kernel-include-regex` (here the `rocsparse`
  SpMV, e.g. `csrmv`), optionally `--att-consecutive-kernels N` to bound how many
  invocations are captured.
- **One rank** — run `-n 1`; the compute per rank is representative.
- **A few CUs / SIMDs** — `--att-target-cu` (default 1) and `--att-simd-select`
  (default `0xF`), plus `--att-shader-engine-mask` (default `0x1`).

MI300A is **gfx942 (gfx9)**, so the gfx9-only options apply:
`--att-perfcounters` (a counter list sampled alongside the trace) and
`--att-activity 8` (HW activity counters; AMD's recommended period).

```bash
module load rocm/7.13.0 openmpi   # 7.13.0 (or 7.12.0) ships the ATT decoder
cd CG-GPU && make
# Instruction trace of the SpMV kernel on CU 1 of one rank:
CG_SEED=12345 mpirun -n 1 --oversubscribe ./gpu_bind.sh \
  rocprofv3 --att \
    --att-library-path $ROCM_PATH/lib \
    --att-target-cu 1 \
    --att-shader-engine-mask 0x1 \
    --att-simd-select 0xF \
    --att-activity 8 \
    --att-consecutive-kernels 1 \
    --kernel-include-regex 'csrmv' \
    -d att_spmv \
    -- ./cg_gpu src/Dubcova2.pm rccl
```

The decoded output lands under the `-d` directory: `*_gfx942_code_object_id_*.out`
files hold the **ISA disassembly annotated with per-instruction hit counts**, and
a `ui_output_agent_*_dispatch_*/` folder holds per-wavefront JSON for the ROCm ATT
viewer (a `*_results.db` rocpd SQLite database is written too). Read it to confirm
the SpMV spends its cycles on `global_load`/`flat_load` (VMEM) waits rather than in
VALU — the instruction-level restatement of "CG is memory-bound".

> **Decoder library — use ROCm ≥ 7.12 (measured).** ATT *collection* is built into
> `rocprofv3`, but *decoding* needs the separate closed-source
> `rocprof-trace-decoder` library. On this cluster it ships with **`rocm/7.12.0`
> and `rocm/7.13.0`** (`$ROCM_PATH/lib/librocprof-trace-decoder.so`) but **not**
> with `rocm/7.2.4` or earlier — there the run aborts with
> `Fatal error: rocprof-trace-decoder library path not found`. So load a 7.12+
> `rocm` module and point `--att-library-path` at `$ROCM_PATH/lib` (or install the
> package / grab a release from
> <https://github.com/ROCm/rocprof-trace-decoder> and point at that). **Verified
> decode-to-ISA on MI300A with `rocm/7.13.0`** (decoder `0.1.7`). The other
> rocprofv3 modes (A.1–A.3) need no decoder.

---

## B. CG-GPU — rocprof-compute (roofline)

The compute per rank is representative, so characterize the kernels on a single
rank:

```bash
module load rocm
rocprof-compute profile -n cg_spmv -- ./cg_gpu src/Dubcova2.pm rccl 12345
rocprof-compute analyze -p workloads/cg_spmv/MI300* | less
```

Read the **roofline** and **memory chart**: the SpMV and rocBLAS kernels sit on
the **HBM-bandwidth roof** (low arithmetic intensity) — CG is memory-bound, so
the achieved GB/s vs. peak is the practical compute ceiling. `--roof-only` is a
faster, fewer-pass variant.

---

## C. CG-GPU — rocprofiler-systems (timeline of overlap)

```bash
module load rocprofiler-systems
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh \
  rocprof-sys-run -- ./cg_gpu src/Dubcova2.pm rccl
```

This produces a per-rank Perfetto trace. Line up the GPU-kernel row, the
memory-copy row, and the host row to see:

- the **on-proc SpMV overlapping** the in-flight halo messages (the "exposed
  wait" the README mentions for `isend`/`rccl`/`alltoallv`),
- **SDMA vs blit** copies for GPU-Aware transports — rerun with
  `HSA_ENABLE_SDMA=0` (blit) vs `=1` (SDMA) and compare the copy rows, tying the
  timeline back to the [SDMA vs blit chapter](05-sdma-vs-blit.md).

---

## D. CG-GPU / CG-CPU — TAU (the MPI communication view)

TAU intercepts MPI (and, with `-rocm`, ROCm) so it directly measures the
communication the ROCm tools miss:

```bash
module load tau rocm openmpi
# GPU solver: MPI + ROCm
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh \
  tau_exec -T MPI,ROCM -rocm ./cg_gpu src/Dubcova2.pm rccl
# CPU reference: MPI only
mpirun -n 4 tau_exec -T MPI ./cg_cpu src/Dubcova2.pm

pprof            # text summary, or:
paraprof         # GUI: per-call MPI time + communication matrix
```

TAU reports time in `MPI_Allreduce` (the dot-product reduction →
`g_allreduce_time`), `MPI_Isend`/`MPI_Irecv`/`MPI_Waitall` or `MPI_Alltoallv`
(the halo exchange → `g_halo_time`), plus GPU kernel time — the same split the
solver prints, now attributed per MPI call and shown as a **communication
matrix** across ranks. (For this small test matrix `MPI_Init` dominates the
profile; use a larger matrix for a meaningful loop breakdown. TAU may return a
nonzero exit at teardown after ROCm tracking — the `profile.*` files are still
written and readable by `pprof`/`paraprof`.)

---

## E. CG-GPU / CG-CPU — HPCToolkit (call-path sampling)

```bash
module load hpctoolkit rocm openmpi
# GPU solver: CPU call paths + AMD GPU operations
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh \
  hpcrun -e CPUTIME -e gpu=amd -o cg_gpu.m ./cg_gpu src/Dubcova2.pm rccl
# CPU reference:
mpirun -n 4 hpcrun -e CPUTIME -o cg_cpu.m ./cg_cpu src/Dubcova2.pm

hpcstruct ./cg_gpu                       # recover program structure
hpcprof -S cg_gpu.hpcstruct cg_gpu.m -o cg_gpu.d
hpcviewer cg_gpu.d                       # call-path profile; hpctraceviewer for timeline
```

HPCToolkit shows how much time each rank spends in MPI wait vs. compute vs. GPU
kernels along the full call path — ideal for spotting a load-imbalanced halo
exchange or a rank stalling in `MPI_Allreduce`. (Run `hpcstruct`/`hpcprof` on the
compute node so their runtime libs are present; if `hpcprof` warns about a partial
struct-file match, pass the suggested `-R '<build>'='<build>/.'` remap.)

---

## F. CG-CPU — likwid (CPU roofline)

On a supported CPU, likwid gives hardware-counter bandwidth/FLOP/s per rank:

```bash
module load likwid openmpi
cd CG-CPU && make
likwid-mpirun -mpi openmpi -np 4 -g MEM_DP  ./cg_cpu src/Dubcova2.pm   # DRAM bandwidth
likwid-mpirun -mpi openmpi -np 4 -g FLOPS_DP ./cg_cpu src/Dubcova2.pm  # DP MFLOP/s
```

Compare the summed DRAM bandwidth to the node's STREAM peak: a memory-bound CG
should land near it. To isolate the SpMV from the dot products, add the likwid
**Marker API** (`likwid_markerStartRegion("spmv")` / `...StopRegion`) and build
with `-DLIKWID_PERFMON -llikwid`.

> **⚠ MI300A caveat (measured).** On the MI300A APU node, `likwid-perfctr`
> (v5.5.1) reports *"Unsupported AMD Zen Processor / K19"* and cannot initialize
> its event groups — the integrated Zen4 CPU is not in this build's perfmon map,
> so `MEM_DP`/`FLOPS_DP` are unavailable there. `likwid-topology` still works on
> supported hosts. **For CPU counters/hotspots on MI300A, use Linux `perf` (§K),
> AMD uProf (§G), or `rocprof-compute` for the GPU roofline instead.** likwid
> remains useful on conventional EPYC login/compute nodes.

---

## G. CG-CPU — AMD uProf

uProf **does** work on the MI300A node. Profile it **per rank under `mpirun`** (so
uProf runs on the compute node where the ranks execute), each writing to its own
directory, then generate the report from that **directory**:

```bash
export PATH=$PATH:/nfsapps/ubuntu-24.04-nightlies/opt/AMDuProf_5.3-518/bin
cd CG-CPU
# Time-based (hotspot) profiling, one uProf per rank:
mpirun --oversubscribe -n 4 bash -c \
  'AMDuProfCLI collect --config tbp -o uprof_r${OMPI_COMM_WORLD_RANK} ./cg_cpu src/Dubcova2.pm'
# Report from the collection DIRECTORY (not the session.uprof file):
AMDuProfCLI report -i uprof_r0        # writes uprof_r0/report.csv
```

Measured on MI300A, `report.csv` lists the hottest functions — for this solver
the top entry is the CSR SpMV (`spmv(double, ParMat&, ...)`), then `main` and the
`std::map` column-index lookups. The memory/bandwidth analysis config (see
`AMDuProfCLI collect --help`) adds DRAM traffic and cache behavior — the CPU-side
analog of the rocprof-compute roofline. (Requires `perf_event_paranoid` low
enough to sample; the MI300A compute nodes are configured permissively.)

---

## H. CG-GPU — IntelliKit (agent-first: metrix / kerncap / accordo / linex)

[IntelliKit](https://github.com/AMDResearch/intellikit) (module `intellikit/main`,
loaded **after** a `rocm` module) is an "agent-first" ROCm profiling and
validation toolkit built on rocprofiler-sdk. Where the tools above emit raw
counters and traces, IntelliKit gives **decoded, human-readable** metrics and
**kernel isolation/validation** — and every tool also ships as an MCP server so an
AI agent can drive it. Installed tools: `metrix`, `kerncap`, `accordo` (CLIs);
`linex`, `nexus` (SQTT / HSA, MCP-driven); `rocm_mcp`, `uprof_mcp` (MCP servers).

> **Invocation note (this build).** Only `accordo` is installed as a `bin` script.
> Invoke the others through the module's Python entrypoints — e.g. define shell
> wrappers once per session:
>
> ```bash
> module load rocm/7.2.4 intellikit/main
> kerncap(){ python -c "from kerncap.cli import main; main()" "$@"; }
> metrix(){  python -c "from metrix.cli.main import main; main()" "$@"; }
> # accordo is on PATH directly
> ```

### H.1 metrix — GPU profiling, *decoded*

`metrix` collects rocprofiler-sdk counters and reports them as named metrics
(bandwidths, cache hit rates, arithmetic intensity) instead of raw PMC — the
low-friction way to answer "is the SpMV bandwidth-bound?".

```bash
# Decoded HBM/L2 bandwidth for just the SpMV kernel, top 5 dispatches
metrix profile -p memory_bandwidth -k 'csrmv' --top 5 \
  "./cg_gpu src/Dubcova2.pm rccl 12345"
# profiles: quick | memory | memory_bandwidth | memory_cache | compute
# or a custom set:  metrix profile -m memory.hbm_bandwidth,memory.l2_hit_rate ...
```

Verified on MI300A (ROCm 7.2.4) against the `rocsparse::csrmvn` SpMV:

```
Dispatch #1420: csrmvn_general_kernel   Duration avg=8.84 μs
  HBM Bandwidth Utilization      4.75 Percent
  HBM Read Bandwidth           245.55 GB/s
  L2 Cache Bandwidth Util.     668.79 GB/s
```

Reading it: for this small test matrix the SpMV is largely **L2-resident** — L2
bandwidth (~669 GB/s) far exceeds HBM (~245 GB/s) and HBM utilization is only
~4.7%. Scale the matrix up and the working set spills to HBM, pushing the kernel
onto the HBM roof — the decoded restatement of the [rocprof-compute
roofline](#b-cg-gpu--rocprof-compute-roofline) in §B. `metrix` and rocprof-compute
agree; `metrix` is faster to read, rocprof-compute plots the roofline.

### H.2 kerncap — isolate the SpMV kernel

`kerncap` ranks kernels and, more usefully, **extracts a single kernel into a
standalone reproducer** you can iterate on without the full solver:

```bash
kerncap profile -- ./cg_gpu src/Dubcova2.pm rccl 12345      # rank kernels by time
kerncap extract csrmvn_general --cmd './cg_gpu src/Dubcova2.pm rccl 12345' \
  --language hip -o spmv_repro/                              # standalone reproducer
kerncap replay  spmv_repro/                                 # VA-faithful re-dispatch
kerncap validate spmv_repro/                                # outputs match capture?
```

Verified `kerncap profile` on MI300A ranks the CG kernels by time — the
`rocsparse::csrmvn` SpMV plus the rocBLAS `axpy`/`dot`/`scal` and the rocclr
fill/copy buffers — the same compute breakdown as §A.1, produced in one command.

### H.3 accordo — correctness validation of an optimization

When you optimize the SpMV, `accordo` intercepts a kernel and compares outputs
between a **reference** and an **optimized** binary within tolerance:

```bash
accordo validate --kernel-name csrmvn_general_kernel \
  --ref-binary ./cg_gpu_ref --opt-binary ./cg_gpu_opt \
  --atol 1e-8 --rtol 1e-5
```

This is the guard-rail for the optimization path in
[chapter 7](07-optimization-path.md): prove a faster kernel still computes the
same result before trusting its speedup.

### H.4 linex / nexus and the MCP servers

- **`linex`** — *source-level SQTT*: maps GPU cycles back to your **source lines**.
  It is the **decoded companion to the raw ATT/SQTT trace in [§A.4](#a4-instruction-level-advanced-thread-trace-att)** — same
  underlying thread trace, presented against source. (SQTT decoding needs the same
  `rocprof-trace-decoder` library noted in §A.4.)
- **`nexus`** — HSA packet / kernel source extractor (research tool for
  intercepting and analysing dispatched kernels).
- **MCP servers** — `linex-mcp`, `nexus-mcp`, `kerncap-mcp`, `metrix-mcp`,
  `accordo-mcp`, plus `rocm_mcp` (`rocminfo`/`amd-smi`/HIP compiler & docs) and
  `uprof_mcp` (AMD uProf). These expose the tools to an MCP client so an agent can
  profile, extract, and validate CG kernels conversationally.

> **Verified on PPAC / MI300A (gfx942, ROCm 7.2.4, `intellikit/main`).** `metrix
> profile` (memory_bandwidth on the SpMV) and `kerncap profile`/`extract` run
> against `cg_gpu` and use rocprofiler-sdk counters directly (no extra decoder).
> `linex`/`nexus` are SQTT/HSA and MCP-server-first; `linex` source mapping needs
> the `rocprof-trace-decoder` library (see §A.4).

---

## I. CG-GPU — roofline-extractor (percent-of-peak roofline)

`roofline-extractor` (module `roofline-extractor`, load after `rocm`) is an
automated **percent-of-peak** roofline analyser. It drives `rocprofv3` for you
(four counter passes + one kernel-trace pass), then reports, per kernel,
arithmetic intensity at each memory level (HBM/L2/L1/LDS), the active performance
limiter, achieved-vs-peak throughput, and % of roofline reached — with an optional
interactive HTML plot. It is the
turnkey complement to [rocprof-compute (§B)](#b-cg-gpu--rocprof-compute-roofline):
less to drive, and it prints the headline number directly ("you are at X% of the
HBM roof").

```bash
module load rocm
module load roofline-extractor
# End-to-end: profiles cg_gpu and prints the per-kernel roofline breakdown.
roofline-extractor-profile -o rex_out -- ./cg_gpu src/Dubcova2.pm rccl 12345
```

Verified on MI300A (gfx942, ROCm 7.13.0) against `cg_gpu`:

```
<rocBLAS kernel>  contribution 10.7% GPU time
  Arithmetic intensity (HBM): 0.5752    limiter: HBM_BW (MI300A)
  Achieved 129.4 GFLOPS/s vs 2121 peak  →  6.1% of linear roofline
  ...
Average percent of linear roofline achieved: 17.2 %
Total application GPU time on MI300A: 7.48e6 ns
```

The low arithmetic intensity (~0.58 FLOP/byte) and the `HBM_BW` limiter are the
percent-of-peak restatement of the memory-bound result from §B / §H.1: CG lives on
the bandwidth roof. Outputs (`counters.csv`, `*_EXTRACTED*.csv`, `counters.html`)
land in `rex_out/`; open the HTML to see each kernel plotted on the roofline. For a
multi-rank/phase view, collect per-run bundles and use
`roofline-extractor-extract -D <bundle_dir>`. See `man roofline-extractor` and the
`METRICS_SUMMARY.md` / `METRICS_DETAILED.md` under the install root for metric
formulas and per-architecture peaks.

> **Arch note.** The tool needs an MI-series arch it recognizes (MI250/MI300A/
> MI300X/MI325X/MI350X/…). On a **CPX**-mode node the virtual GPUs may not match a
> known arch string — run on **SPX**, or pin `--arch MI300A`.

---

## J. CG-GPU — rocBudAI (AI profiling assistant)

`rocBudAI` (module `rocbudai`) is an AI performance-engineering assistant that
**drives the whole stack above for you**. It wires the OpenCode TUI to a
locally-hosted model (Ollama) pre-seeded with the AMD profiling toolchain
(rocprofv3, rocprof-compute, rocprof-sys, rocpd, rocm-smi), then runs a
build → profile → analyse → optimise → re-profile loop, writing a running
`report.md` whose claims are tagged `[FACT]` / `[INFERENCE]` / `[OPINION]`.

It runs **only on a Slurm-allocated compute node**, and the `--comment=ollama`
allocation is air-gapped (source, prompts, and model output stay on the node):

```bash
# 1. Allocate an SPX MI300A node with the local model daemon started for you:
salloc -p PPAC_MI300A_SPX --exclusive --comment=ollama --time=02:00:00
# 2. Point it at this example:
cd ~/HPCTrainingExamples/MPI-examples/cg-solver-example/CG-GPU
# 3. Launch the assistant (auto-starts the TUI, pre-configured for AMD profiling):
module load rocbudai
```

A seven-question interview (app type, ROCm version, modules, build command, run
command, figure of merit) configures the run. For this example answer roughly:
*HIP/C++ + MPI*; `rocm/7.13.0` + `openmpi`; build `make`; run
`mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm rccl`; FoM = *CG solve time*
(or *% of roofline*). The agent then profiles with the tools in this chapter and
proposes optimizations. Use it when you want the workflow orchestrated and
explained rather than run by hand — but still check its conclusions against the raw
tool output above. See `man rocbudai` for airgap enforcement, session management,
and ASK vs AUTO-RUN permission modes; verify the sandbox with
`rocbudai-airgap-check`.

---

## K. CG-CPU — Linux perf (always-available CPU baseline)

`perf` needs no modules and no special build — the zero-friction CPU baseline, and
the natural fallback where likwid's counters are unsupported on MI300A (§F). The
MI300A **compute** nodes run with `perf_event_paranoid = -1`, so hardware counters
are fully available (the **login** node is restricted — run perf inside an
allocation).

```bash
cd CG-CPU && make CXXFLAGS="-O3 -g -std=c++17"   # -g for symbol resolution
# Per-rank counter summary (IPC + cache behaviour), one file per rank:
mpirun -n 4 bash -c 'perf stat -o perf_r${OMPI_COMM_WORLD_RANK}.txt \
  -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
  ./cg_cpu src/Dubcova2.pm 12345'
# Hotspot sampling → where the cycles go:
mpirun -n 1 perf record -g -o perf.data ./cg_cpu src/Dubcova2.pm 12345
perf report -i perf.data        # interactive; or `perf annotate` for source+asm
```

Verified single-rank on MI300A (`perf stat`):

```
    726,966,836  cycles
    602,835,327  instructions            #  0.83  insn per cycle
     18,270,624  cache-references
      2,410,388  cache-misses            #  13.19% of all cache refs
    205,921,430  L1-dcache-loads
      4,193,649  L1-dcache-load-misses   #   2.04% of all L1-dcache accesses
```

The low IPC (0.83) and the non-trivial last-level cache-miss rate (~13%) are the
CPU signature of the memory-bound SpMV — the CPU analog of the GPU HBM-bound
roofline (§B/§I). `perf record`/`perf report` then attribute cycles to the SpMV
row-loop and the CSR index gather. (`perf list | grep -i cache` shows the memory
events your CPU exposes; wrap `perf stat` under `mpirun` for per-rank files as
above rather than measuring the launcher.)

---

## L. CG-CPU — Valgrind cachegrind (deterministic cache model)

Where perf reads noisy hardware counters, **cachegrind** *simulates* the cache
hierarchy, so its miss counts are **deterministic and repeatable** — ideal for
comparing two SpMV implementations without run-to-run counter jitter. It is a CPU
binary-instrumentation tool from the Valgrind suite; `valgrind` is installed on the
MI300A **compute** nodes (`/usr/bin/valgrind`; not on the login node). Expect a
~20–50× slowdown, so use a fixed, small matrix.

```bash
cd CG-CPU && make CXXFLAGS="-O2 -g -std=c++17"   # -g (and -O2) for clean annotation
valgrind --tool=cachegrind --cache-sim=yes \
  --cachegrind-out-file=cg.cachegrind.out \
  ./cg_cpu src/Dubcova2.pm 12345
cg_annotate cg.cachegrind.out            # file:function miss breakdown
cg_annotate cg.cachegrind.out src/cg.cpp # annotate a source file line-by-line
```

Verified on MI300A:

```
D refs:      46,177,102  (32,130,783 rd + 14,046,319 wr)
D1  misses:     653,773        D1  miss rate: 1.4%
LLd misses:     136,346        LLd miss rate: 0.3%
```

`cg_annotate` breaks the D1/LL misses down **per function and per source line**, so
you can see the misses land in the CSR SpMV row-loop and the column-index-driven
gather of `x[col[j]]` — the classic irregular-access hotspot of sparse mat-vec.
(Because CG reads a gzipped matrix, `gzgets`/decompression can dominate a tiny run;
fix the seed and use a larger matrix, or focus on the solve-loop functions, for a
representative picture. `--branch-sim=yes` adds branch-misprediction stats.)

---

## M. perf_events security & access control

`perf` (§K) reads hardware performance counters, execution-context registers, and
sampled addresses — data that can leak sensitive information from *other*
processes. The kernel therefore gates `perf_events` behind an access-control
model you must understand to (a) know why a command is denied and (b) request the
right, least-privilege fix from your admins. This section summarises the
[kernel perf-security guide](https://docs.kernel.org/admin-guide/perf-security.html)
as it applies to the CG example; the runnable demo is
[`CG-CPU/perf_security_demo.sh`](../CG-CPU/perf_security_demo.sh) and the full
implementation/demo plan is [chapter 9](09-perf-security-demo.md).

### The `perf_event_paranoid` scope ladder

An unprivileged user's scope is set by `/proc/sys/kernel/perf_event_paranoid`:

| Value | What an unprivileged user may do |
|:-----:|----------------------------------|
| **-1** | Everything; per-cpu `perf_event_mlock_kb` limit ignored (**least secure**) |
| **>= 0** | Per-process **and** system-wide (`-a`); excludes raw/ftrace tracepoints |
| **>= 1** | Per-process **only** (no system-wide); user + kernel events |
| **>= 2** | Per-process, **user-space events only** (no kernel-space sampling) |

On this cluster the **exclusive compute nodes run `paranoid = -1`** (full access —
verified), while the **login node is restricted** (`paranoid = 4`, i.e. `> 2`).
That is why the rule throughout this chapter is *profile inside an allocation*.
`perf stat`/`perf record` on your own `cg_cpu` (per-process) work at any level `<=
2`; `perf stat -a` (system-wide, e.g. uncore/IMC memory-bandwidth counters) needs
`<= 0`.

### CAP_PERFMON — the least-privilege alternative

Lowering `paranoid` system-wide is blunt. The secure alternative is the
**`CAP_PERFMON`** capability (Linux ≥ 5.9), which grants perf scope *without* full
root and is preferred over the legacy `CAP_SYS_ADMIN`. Admins typically gate it to
a group and stamp it on the `perf` binary:

```bash
groupadd perf_users
chgrp perf_users "$(command -v perf)" && chmod o-rwx "$(command -v perf)"
setcap "cap_perfmon,cap_sys_ptrace,cap_syslog=ep" "$(command -v perf)"
getcap "$(command -v perf)"   # cap_sys_ptrace,cap_syslog,cap_perfmon+ep
```

Members of `perf_users` then profile at any paranoid level. (`cap_syslog` allows
resolving kernel symbols via `/proc/kallsyms`; `cap_sys_ptrace` is not needed on
≥ 5.9 but is harmless; add `cap_ipc_lock` for `perf top`.) Where `setcap` is not
possible (e.g. `nosuid` filesystem) the guide's `capsh`/`sudo` "privileged perf
shell" achieves the same via the ambient capability set.

The [demo](09-perf-security-demo.md) automates this as a **least-privilege overlay
on one ROCm release**: [`perf_users_setup.sh`](../CG-CPU/perf_users_setup.sh)
creates the group, adds users, and gates (`chgrp`+`chmod o-rwx`) + `setcap`s the
profilers of a single release, while **leaving `perf_event_paranoid` open** so
profilers from *other* releases keep working unchanged. A real-world wrinkle it
handles: the local `perf` ELF is on **xfs** (`setcap` works), but ROCm is on
**NFS**, which cannot hold file-capability xattrs — there group members get
`CAP_PERFMON` via the `capsh` privileged shell instead. (The script uses the lean
`cap_perfmon,cap_syslog=ep`; `cap_sys_ptrace` from the guide's example is dropped
as unneeded on ≥ 5.9.) To **prove** the overlay,
[`perf_paranoid_test.sh`](../CG-CPU/perf_paranoid_test.sh) (run once under `sudo`
in an exclusive allocation) *raises* `perf_event_paranoid` and shows an ordinary
`perf_users` member still lands a system-wide `perf stat -a` via `CAP_PERFMON`,
then restores the original value.

### Resource limits that bite multi-rank runs

Two per-process resource limits (not security *scope*, but they cause the same
"cannot proceed" symptom) matter when profiling all MPI ranks at once:

- **`perf_event_mlock_kb`** (here `516` KiB, *per cpu*) — the ring-buffer budget.
  The first `perf` process can grab it all and starve the other ranks. Cap each
  rank's buffer with `perf record --mmap-pages=N` (or `perf top -m N`).
- **`RLIMIT_NOFILE`** (`ulimit -n`) — perf opens ≥ `events × cpus` file
  descriptors; large event lists on many cores can exhaust it.

`CAP_IPC_LOCK` lifts the `mlock` limit for privileged perf users.

### Why this does not block on the ROCm version

`perf_events` security is a **kernel** feature — orthogonal to ROCm. The
[demo](09-perf-security-demo.md) therefore pins a **single** `rocm` module (only to
build `cg_cpu`) and does **not** sweep ROCm versions: the paranoid ladder,
`CAP_PERFMON`, and the resource limits behave identically under every ROCm module
on a given kernel. Better still, because the setup **leaves `perf_event_paranoid`
open**, you can apply the `perf_users`/`CAP_PERFMON` overlay to **one release's**
profilers and every *other* release keeps working untouched — a safe, one-at-a-time
rollout rather than a global change. (Contrast the *GPU* ATT decoder in §A.4, which
genuinely needs `rocm >= 7.12`.)

---

## Mapping tools to the tutorial's questions

| Question | GPU (`CG-GPU`) | CPU (`CG-CPU`) |
|----------|----------------|----------------|
| Which transport is fastest for the halo exchange? | rocprofv3 `--sys-trace` + rocprofiler-systems timeline; TAU MPI matrix | TAU / HPCToolkit MPI time |
| How costly is the dot-product all-reduce? | TAU / HPCToolkit `MPI_Allreduce` time (→ `g_allreduce_time`) | same |
| Is SpMV hitting the bandwidth ceiling? | rocprof-compute roofline; roofline-extractor % of peak (§I); IntelliKit `metrix` (decoded, §H.1) | perf cache-miss rate + IPC (§K); uProf memory (likwid perfmon unsupported on MI300A) |
| Which functions/lines cause cache misses? | ATT / linex (§A.4/§H.4) | Valgrind cachegrind `cg_annotate` (§L) |
| How far from peak overall? | roofline-extractor % of peak (§I) | n/a |
| Want the workflow driven & explained end-to-end? | rocBudAI AI assistant (§J) | rocBudAI (§J) |
| *Why* is SpMV memory-bound (which ISA lines stall)? | rocprofv3 ATT (`--att`, §A.4); IntelliKit `linex` (source-level SQTT, §H.4) | n/a |
| Did my SpMV optimization stay correct / faster? | IntelliKit `accordo` validate + `kerncap` extract/replay (§H) | n/a |
| SDMA vs blit copies? | rocprofiler-systems timeline + `HSA_ENABLE_SDMA` sweep | n/a |
| Where does each rank stall (load imbalance)? | HPCToolkit call-path | HPCToolkit / TAU |

> Flag names and event lists vary across releases. If a flag is rejected, check
> `rocprofv3 --help`, `rocprof-compute --help`, `rocprof-sys-run --help`,
> `tau_exec -help`, `hpcrun -L`, `likwid-perfctr -a`, and
> `AMDuProfCLI collect --help` on the loaded module.
