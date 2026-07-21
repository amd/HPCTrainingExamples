# ThreadTrace: minimal HIP GEMM + `rocprofv3 --att`

Three **FP64 square matrix multiply** kernels over the same `N×N` problem, with intentionally different performance:

| Kernel | Role |
| --- | --- |
| `matmul_naive` | One output per thread; inner `k` loop reads `A` and `B` from global memory every iteration. |
| `matmul_mid` | Same structure as naive, but `__restrict__` pointers and partial inner-loop unrolling (still global-memory bound). |
| `matmul_fast` | **16×16 LDS-tiled** GEMM: staging through `__shared__`, `__syncthreads`, padded tiles to reduce bank conflicts. |

Constants live at the top of [`main.hip`](main.hip): `kN`, `kBlock` (naive/mid), `kTile` (LDS tiled kernel). Correctness uses a small **CPU spot-check** on random matrix elements.

The project is built with **`-g -gdwarf-4`** so rocprofv3 ATT / `stats_*.csv` can map instructions back to **source lines** (see AMD doc below).

## Build

```bash
module load rocm/7.14.0-gfx94X   # or your ROCm 7.14 gfx94X module
make clean && make
```

Override GPU arch: `make ARCH=gfx942` (default in `Makefile`).

## Run (always under `srun`)

```bash
make run
```

## Thread trace with rocprofv3

Official guide: [Using thread trace](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html) (develop branch; your installed ROCm may differ). A short excerpt is in [`docs/using-thread-trace.md`](docs/using-thread-trace.md).

**Default-style collection** (run `matmul` under Slurm; example inside `salloc`):

```bash
rocprofv3 --att -d -- srun -n 1 ./matmul
```

**AMD Instinct (SQ activity streamed into ATT buffer):**

```bash
rocprofv3 --att --att-activity 8 -d -- srun -n 1 ./matmul
```

From login without an interactive shell on the node, pass your allocation:

```bash
rocprofv3 --att --att-activity 8 -d -- srun -n 1 --jobid="${myjob}" ./matmul
```

**Trace one kernel** (HIP names are mangled; substrings like the following usually work—confirm with one `--kernel-trace` run if needed):

```bash
rocprofv3 --att --att-activity 8 -d --kernel-include-regex matmul_fast -- srun -n 1 ./matmul
```

Replace `matmul_fast` with `matmul_naive` or `matmul_mid` to compare instruction/stall profiles.

### Outputs

After the run, rocprofv3 typically leaves:

- `stats_*.csv` — per-instruction summary (latency, stall, idle; **Source** column when built with `-g`).
- `ui_output_agent_*_dispatch_*` — JSON for **ROCprof Compute Viewer**.
- Raw `.att` / `.out` — SQ thread trace and code objects (see AMD doc).

Add these paths to `.gitignore` (already listed) so they are not committed by mistake.

## Viewing results

Open the generated `ui_output_agent_*` directory in **ROCprof Compute Viewer** (see ROCm / rocprof-compute documentation for the version that matches your install).
