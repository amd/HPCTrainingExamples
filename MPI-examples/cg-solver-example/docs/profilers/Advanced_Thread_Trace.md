# Advanced Thread Trace (ATT) — CG-GPU (instruction-level thread trace)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first.

**Advanced Thread Trace (ATT)** is `rocprofv3 --att` plus the closed-source
`rocprof-trace-decoder` library. It records wavefront execution at the
**ISA-instruction** granularity on selected compute units and decodes it to a
per-instruction hit map with **latency, stall, and idle cycles**, occupancy, and
per-wavefront timelines for a **single kernel**. Where [rocprof-compute](rocprof-compute.md)
and [roofline-extractor](roofline-extractor.md) tell you *that* the SpMV is
bandwidth-bound, ATT shows you *why* — which exact ISA lines spend the cycles.

**Verified on MI300A** (`gfx942`, SPX mode, ROCm 7.13.0, `rocprof-trace-decoder`
0.1.7) against `cg_gpu` on `src/Dubcova2.pm`, single rank, `rccl`, tracing the
rocSPARSE `csrmvn_general_kernel` (SpMV) on one CU. Absolute cycle counts drift
run-to-run; the *shape* (memory waits dominate) is the point.

---

## 1. Build and run

ATT dumps an enormous amount of data, so you **must** target it: one kernel, one
rank, a few CUs/SIMDs.

```bash
module load rocm/7.13.0 openmpi   # 7.13.0 (or 7.12.0) ships the ATT decoder
cd CG-GPU && make
export HSA_XNACK=1
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

The targeting knobs:

- **One kernel** — `--kernel-include-regex 'csrmv'` (the rocSPARSE SpMV);
  `--att-consecutive-kernels 1` traces a single dispatch of it.
- **One rank** — `mpirun -n 1`; per-rank compute is representative.
- **A few CUs / SIMDs** — `--att-target-cu 1` (default 1), `--att-simd-select 0xF`
  (all 4 SIMDs, gfx9), `--att-shader-engine-mask 0x1` (one SE).
- **gfx9 sampling** — MI300A is **gfx942 (gfx9)**, so `--att-activity 8` (AMD's
  recommended period) applies. `gpu_bind.sh` pins the single rank to one GPU/NUMA
  node so the target is unambiguous.

> **Launch through `mpirun -n 1`.** `cg_gpu` is an MPI program; a bare `./cg_gpu`
> or direct `srun` aborts in `MPI_Init` (no PMI). `rocprofv3` runs *inside* the
> launcher.

> **Decoder library — use ROCm >= 7.12 (measured).** ATT *collection* is built into
> `rocprofv3`, but *decoding* needs the separate closed-source
> `rocprof-trace-decoder` library. On this cluster it ships with **`rocm/7.12.0`
> and `rocm/7.13.0`** (`$ROCM_PATH/lib/librocprof-trace-decoder.so`) but **not**
> with `rocm/7.2.4` or earlier — there the run aborts with
> `Fatal error: rocprof-trace-decoder library path not found`.

A reproducible batch version is in
[`CG-GPU/submit_att_spmv.sbatch`](../../CG-GPU/submit_att_spmv.sbatch)
(`sbatch submit_att_spmv.sbatch`). It preflights the decoder library, runs the
trace, and summarizes the decoded ISA into `att_spmv/ISA_top.txt`.

## 2. Reading the decoded ISA

The `-d att_spmv` directory holds three kinds of output:

- `att_spmv/<node>/<pid>_gfx942_code_object_id_*.out` — the raw **ELF code
  objects** the viewer disassembles (binary, not meant to be read directly).
- `att_spmv/<node>/<pid>_results.db` — a **rocpd** SQLite database.
- `att_spmv/ui_output_agent_<pid>_dispatch_<n>/` — the **ATT viewer** data as JSON.
  The key file is `code.json`: the decoded ISA with per-instruction counters. Its
  schema (from the file header) is:

  ```text
  ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle
  ```

`Latency`/`Stall` are cycle counts attributed to each instruction. Summing them by
instruction class (the batch script writes this to `att_spmv/ISA_top.txt`) gives
the instruction-level verdict for the SpMV:

```text
## cycles by instruction class (% of total latency)
  wait   Latency=176868   Stall=176868   (65.6%)
  vmem   Latency=60916    Stall=59128    (22.6%)
  salu   Latency=23172    Stall=15648    (8.6%)
  valu   Latency=8864     Stall=1104     (3.3%)

## top 12 instructions by latency (cycles)   Hit  Latency  Stall  instruction
  88     65592     65592      s_waitcnt vmcnt(1)
  88     54984     54984      s_waitcnt vmcnt(0)
  88     25936     25496      global_load_dwordx2 v[12:13], v[12:13], off
  88     20184     20184      s_waitcnt vmcnt(1)
  32     16640     16640      s_waitcnt lgkmcnt(0)
  88     12196     11756      global_load_dword v12, v[12:13], off
  32     12088     11960      global_load_dword v11, v[0:1], off
  ...
```

How to read it:

- **~66 % of cycles are `s_waitcnt vmcnt(...)`** — the wavefront is *stalled waiting
  for VMEM loads to return*. Another **~23 % is the `global_load_*` instructions**
  themselves. Together **~88 % of the SpMV's cycles are memory**, and the two most
  expensive instructions in the whole kernel are the `vmcnt` waits after the CSR
  column/value gathers.
- **VALU (actual floating-point compute) is only ~3 %.** The multiply-adds are
  nearly free; the kernel is waiting on `global_load` almost the entire time.
- This is the **instruction-level restatement of "CG is memory-bound"**: the
  roofline says the SpMV runs at ~94 % of the (curved) HBM roof; ATT shows the
  cycles are spent on `s_waitcnt vmcnt` behind the matrix/vector gathers, not in
  the FMA units.

## 3. The ATT viewer (occupancy + per-wavefront timeline)

`ui_output_agent_*/` also carries `occupancy.json`, `se0_perfcounter.json`, and one
`se0_sm<simd>_sl<slot>_wv<wave>.json` per traced wavefront — the wave-slot timeline
the GUI renders. To explore it interactively, load the `ui_output_agent_*` folder in
the **ROCm ATT viewer** inside a remote graphical session:

- `man aac6_vnc` — TurboVNC desktop, then start the ATT viewer and open the folder
- `man aac6_novnc` — the same desktop in your local browser
- `man aac6_x11` — `ssh -X` for a single window

The viewer is a GUI (no headless/CLI screenshot path), so this guide leads with the
text `code.json`/`ISA_top.txt` evidence above, which needs no display.

## 4. Participant exercises

Work these on an **SPX** MI300A node (`salloc -p SH5_MI300A_SPX --gres=gpu:1`) after
the build in §1, `module load rocm/7.13.0 openmpi`. Each is a small change to the
command or a query into `code.json`.

1. **Confirm the limiter at the instruction level.** From `att_spmv/ISA_top.txt`,
   add up the `wait` + `vmem` latency percentages. *What fraction of the SpMV's
   cycles is memory-related, and how much is VALU?* Explain why the two costliest
   instructions are `s_waitcnt vmcnt(...)` rather than an FMA.

2. **Prove the decoder gate.** Re-run the §1 command with `module load rocm/7.2.4`
   instead of 7.13.0. *What error do you get, and at which stage (collection vs
   decode)?* (Expect `rocprof-trace-decoder library path not found`.) Then switch
   back to 7.13.0 and confirm `$ROCM_PATH/lib/librocprof-trace-decoder.so` exists.

3. **Widen the trace aperture.** Re-run with `--att-target-cu 2` and then with a
   narrower `--att-simd-select 0x1` (one SIMD). *How do the per-wavefront JSON count
   and the occupancy change?* Relate the number of `se0_sm*_sl*_wv*.json` files to
   how many wavefronts landed on the traced CU.

4. **Sweep the sampling period.** Compare `--att-activity 8` (default here) to a
   larger value. *How do the total trace size and the per-instruction `Hit` counts
   change?* Discuss the sampling-rate vs overhead / data-volume trade-off, and why
   ATT must be targeted to one kernel.

5. **Trace a different kernel and contrast.** Change `--kernel-include-regex` to a
   BLAS-1 vector kernel (`axpy` or `scal`). *Is its `vmem`+`wait` fraction as high as
   the SpMV's?* Tie the answer back to arithmetic intensity from the
   [roofline-extractor](roofline-extractor.md) table (SpMV AI ~1.8 vs `axpy` ~0.7).

6. **Cross-check against the roofline.** Put this ISA breakdown next to the
   [rocprof-compute](rocprof-compute.md) / [roofline-extractor](roofline-extractor.md)
   verdict (`HBM_BW`-limited, ~94 % of curved roof). *Do the two tools agree, and
   what does ATT add that the roofline cannot?* (Hint: the roofline gives the number;
   ATT names the stalling instruction.)

## Viewing the results remotely

- **Text** (`att_spmv/ISA_top.txt`, `code.json`) needs no display — inspect on the node.
- **ATT viewer** (`ui_output_agent_*` folder): load it in the ROCm ATT viewer inside
  a VNC / noVNC / X11 desktop (see §3 and the man pages above).
- **rocpd database** (`*_results.db`): query with `sqlite3`, or convert with
  `rocpd2pftrace` and open the timeline at <https://ui.perfetto.dev>.

## See also

- [rocprofv3](rocprofv3.md) — the parent tool (kernels, transports); ATT is its
  instruction-level mode
- [rocprof-compute](rocprof-compute.md) / [roofline-extractor](roofline-extractor.md)
  — the roofline that ATT explains at the instruction level
