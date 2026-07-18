# rocprofv3 — CG-GPU (kernels + transports)

> Part of the [CG profiler guides](README.md). Read the shared
> [ground rules](../08-profiling.md#ground-rules-or-your-numbers-are-noise) first;
> load the patched profilers with `module load rocm/<ver>` (see the
> [index note](README.md#the-patched-hipblaslt-performance-runs)).

`rocprofv3` is the in-box ROCm profiler. It traces **GPU kernels** (compute),
**memory copies / HIP / RCCL** (communication), and — with `--att` — decodes a
single kernel to the **ISA-instruction** level. It does **not** trace MPI itself;
use [TAU](tau.md) or [HPCToolkit](hpctoolkit.md) for `MPI_Isend`/`MPI_Allreduce`.

## 1. Compute: per-kernel trace

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

## 2. Communication: memory-copy + RCCL/HIP traces

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

For `staged` you see the D↔H staging copies around each SpMV; for `rccl` you see
`ncclSend`/`ncclRecv` on the `rccl_stream` overlapping the on-proc SpMV on the
default stream. This is the halo-exchange cost the solver rolls into
`g_halo_time`, now resolved by transport.

## 3. Bandwidth counters (optional)

```bash
printf 'pmc: FETCH_SIZE WRITE_SIZE L2CacheHit VALUBusy\n' > counters.txt
mpirun -n 4 ./gpu_bind.sh rocprofv3 -i counters.txt --output-format csv \
  -d prof_pmc_rank_${OMPI_COMM_WORLD_RANK:-0} -- ./cg_gpu src/Dubcova2.pm rccl
```

`FETCH_SIZE`+`WRITE_SIZE` ÷ kernel time = achieved HBM bandwidth per kernel — or
let [rocprof-compute](rocprof-compute.md) do the math.

## 4. Instruction-level: Advanced Thread Trace (ATT)

**Advanced Thread Trace** records wavefront execution at the **ISA-instruction**
granularity on selected compute units: a per-instruction hotspot/hit map, stall
reasons, VALU/VMEM issue behaviour, and occupancy for a single kernel — the tool
for asking *why* the SpMV is bandwidth-bound rather than just *that* it is.

ATT produces an enormous amount of data, so you **must** target it:

- **One kernel** — `--kernel-include-regex` (here the `rocsparse` SpMV, e.g.
  `csrmv`), optionally `--att-consecutive-kernels N`.
- **One rank** — run `-n 1`; the compute per rank is representative.
- **A few CUs / SIMDs** — `--att-target-cu` (default 1), `--att-simd-select`
  (default `0xF`), `--att-shader-engine-mask` (default `0x1`).

MI300A is **gfx942 (gfx9)**, so the gfx9-only options apply: `--att-perfcounters`
and `--att-activity 8` (AMD's recommended period).

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
> `Fatal error: rocprof-trace-decoder library path not found`. **Verified
> decode-to-ISA on MI300A with `rocm/7.13.0`** (decoder `0.1.7`). Modes 1–3 need no
> decoder.

## Viewing the results remotely

- **Perfetto traces** (`*.pftrace` from step 2): open at
  <https://ui.perfetto.dev>. On a compute node with no outbound browser, start a
  graphical session and run a browser there:
  - `man aac6_vnc` — TurboVNC desktop, then open the trace in Firefox/Chromium
  - `man aac6_novnc` — the same desktop in your local browser
  - `man aac6_x11` — `ssh -X` and launch a single browser window
- **CSV / counter output** (steps 1, 3) is text — inspect with `column -s, -t` or a
  spreadsheet.
- **ATT viewer** (step 4): load the `ui_output_agent_*` folder in the ROCm ATT
  viewer inside the VNC desktop.

## See also

- [rocprof-compute](rocprof-compute.md) — turn these counters into a roofline
- [rocprofiler-systems](rocprofiler-systems.md) — full host+GPU timeline
- [TAU](tau.md) / [HPCToolkit](hpctoolkit.md) — the MPI communication rocprofv3 can't see
