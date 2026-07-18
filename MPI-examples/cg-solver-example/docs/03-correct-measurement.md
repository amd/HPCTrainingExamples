# 3. Measuring performance correctly

This is the heart of the tutorial. Before you compare *any* two communication approaches, the measurement
itself has to be trustworthy. Five things, in priority order:

1. **Affinity** — bind each rank to its GPU-local CPU cores. (Biggest effect by far.)
2. **Reproducibility** — fix the RHS seed so every run solves the same system.
3. **Warm-up & repeats** — discard the first run; report the minimum (or median) of several.
4. **Isolate comm vs compute** — never compare a single "solve time" number blind.
5. **Verify correctness** — same iteration count and residual, or the comparison is meaningless.

Get these wrong and you will "measure" the scheduler, the RNG, or a cold cache instead of the transport.

---

## 3.1 Affinity — the make-or-break factor

On MI300A each GPU is local to one NUMA node (**GPU _i_ ↔ NUMA node _i_**). If a rank runs on cores that are
*remote* from its GPU — or worse, if all ranks are crammed onto a handful of shared threads — every kernel
launch, `hipDeviceSynchronize`, and MPI progress call pays a latency penalty.

How bad? On AAC6, the exact same binary and matrix, 4 ranks:

| launch | CPU affinity | staged | isend | rccl | alltoallv |
|--------|--------------|--------|-------|------|-----------|
| `--bind-to none`, **not** exclusive | 4 hwthreads shared by all ranks (remote) | 9.27 s | 10.58 s | 7.39 s | 8.46 s |
| `--exclusive` + `gpu_bind.sh` | each rank → its GPU-local NUMA (24 cores) | **0.15 s** | **0.05 s** | **0.05 s** | **0.05 s** |

That is a **~100× difference** — dwarfing every transport, copy-engine, or ROCm-version effect in the rest of
this tutorial. **If you take away one thing, take this: fix affinity first.**

### Do it with a wrapper

Two ready-made per-rank wrappers ship with `CG-GPU/`:

```bash
# numactl-based, 4-GPU NUMA binding (OpenMPI):
mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm isend

# taskset-based, 8-GPU MI300A binding (used by run_test_7.13.sh):
mpirun -n 8 --bind-to none bash set_affinity_mi300a.sh ./cg_gpu src/Dubcova2.pm isend
```

Both set `ROCR_VISIBLE_DEVICES=<local_rank>` (each rank sees exactly its own GPU) and pin the rank to that
GPU's local cores (`gpu_bind.sh` via `numactl --cpunodebind/--membind`; `set_affinity_mi300a.sh` via
`taskset`). Use `--bind-to none` on the `mpirun`/`srun` side so the wrapper owns all pinning.

### Verify it, don't assume it

```bash
mpicxx -O2 -std=c++17 check_affinity.cpp -o check_affinity \
       -I$ROCM_PATH/include -L$ROCM_PATH/lib -lamdhip64
mpirun -n 4 ./check_affinity
```

This prints each rank's selected GPU **PCI bus id** and CPU mask and flags any two ranks that collide on a GPU.

### Two affinity traps

- **`srun` without PMIx** launches each task as its own MPI world of size 1, so every task computes
  `rank % num_gpus = 0` and they **all pile onto GPU 0**. Use `srun --mpi=pmix` or a binding wrapper.
- **The cliff is a property of the launch, not the machine.** On an HPE Cray EX system an `--exclusive` `srun`
  allocation already binds each rank to a distinct, largely NUMA-local core set, so explicit binding adds only
  **~3–8 %** there (see [`STUDY_REPORT_PrgEnv-amd.md`](../CG-GPU/STUDY_REPORT_PrgEnv-amd.md) §3). Never assume;
  measure bound vs. unbound on *your* system once, then standardize on the bound launch.

---

## 3.2 Reproducibility — fix the RHS seed

By default the right-hand side is seeded from wall-clock time, so each run solves a slightly different system
and converges in a different number of iterations. You cannot compare methods whose iteration counts differ.

Fix the seed:

```bash
CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm isend   # via env
mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm isend 12345           # via argv[3]
```

The seed is echoed as `RHS seed: 12345` at the top of the output. With it fixed, **all seven methods converge in
exactly the same iteration count with the same residual** (e.g. 172 iterations, final residual 1.815e-4). Now a
difference in solve time is a difference in *transport*, not in *how much work* was done.

---

## 3.3 Warm-up and repeats

The first run of a freshly built binary pays one-time costs: library initialization, JIT/code-object load,
allocator warm-up, and RCCL communicator bring-up. These are not what you want to compare.

- **Discard the first run** (or treat it as warm-up).
- **Repeat 3–5 times** and report the **minimum** (cleanest, least noise) — or the **median ± spread** if you
  want to characterize variability. For the fast GPU-Aware variants the run-to-run spread is comparable to the
  method-to-method difference, so a single run can rank two methods backwards.

The sweep scripts already do this. For example `sweep_sdma.sh` keeps the minimum-solve run over `REPEATS`:

```bash
REPEATS=5 CG_SEED=12345 ...   # every sweep script honours these
```

---

## 3.4 Isolate communication from compute

A single "solve time" number hides the thing you are trying to measure. The instrumented solver reports a
breakdown (max across ranks, accumulated over the CG loop only):

```
CG solve time:      0.0513 s  (0.0003 s/iter)
  comm total:       0.0282 s  (55.0% of solve)
  halo exchange:    0.0142 s  (0.0001 s/iter, 27.6%)   ← the part you choose the transport for
  dot allreduce:    0.0140 s  (27.3%)                   ← MPI_Allreduce for the two dot products
  compute (rest):   0.0231 s  (45.0%)                   ← rocSPARSE SpMV + rocBLAS + launch overhead
```

- **`halo exchange`** = staging copies (D↔H) + GPU gather/pack + the MPI/RCCL `send`/`recv`/`wait`/`alltoallv`
  calls. This is what differs between the seven variants.
- **`dot allreduce`** = the global reductions. Independent of the halo transport.
- **`compute (rest)`** = `solve − comm total`. rocSPARSE/rocBLAS kernels + launch latency. Independent of the
  halo transport — so if it moves when you change *only* the transport, that is your noise floor talking.

Two definitions and a caveat you must keep in mind:

- The timers cover the **CG loop only** — matrix read, GPU upload, and setup are excluded (they are one-time).
- For the **overlap variants** (`isend`, `rccl`, `alltoallv`) the `halo exchange` figure is mostly the
  *exposed wait*: the on-proc SpMV runs concurrently with the in-flight messages, so this is **not** the raw
  wire time. It is the *cost you actually pay*, which is what you want for choosing a transport — just don't
  mistake it for a bandwidth measurement.

---

## 3.5 Verify correctness every time

Before trusting a timing, check the run actually solved the problem: same **iteration count** and **final
residual** as the CPU reference (for the same seed). The bundled `test_cg_gpu.sh` does this automatically —
it builds, runs every variant, asserts convergence to a tolerance, and prints the timing summary; it exits 0
gracefully on a node with no GPU.

```bash
# inside a GPU allocation, with binding:
GPU_BIND=./gpu_bind.sh CG_SEED=12345 ./test_cg_gpu.sh
# or submit:
sbatch submit_cg_gpu_test.sbatch
```

---

## The measurement recipe (checklist)

```
[ ] Exclusive allocation (own whole NUMA nodes)
[ ] Per-rank GPU+CPU binding wrapper (gpu_bind.sh / set_affinity_mi300a.sh)
[ ] Verified binding once with check_affinity
[ ] Fixed CG_SEED
[ ] Warm-up run discarded
[ ] 3–5 repeats, report min (or median ± spread)
[ ] Read the comm/compute breakdown, not just solve time
[ ] Confirmed identical iteration count + residual across methods
```

Only once every box is ticked are the comparisons in the next chapters meaningful.

Next: [4. Comparing communication variants →](04-communication-variants.md)
