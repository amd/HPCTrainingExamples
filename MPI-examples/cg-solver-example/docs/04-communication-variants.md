# 4. Comparing communication variants

With the measurement under control (Chapter 3), you can now compare the five transports *fairly*. The design of
the examples makes this a set of clean, controlled experiments: pairs of variants that differ in exactly one
thing.

## The five variants and what each isolates

| method | transport | buffers | pairs with… | to isolate |
|--------|-----------|---------|-------------|------------|
| `staged` | Isend/Irecv | host | `isend` | the D↔H staging copies |
| `isend` | Isend/Irecv | GPU (GPU-Aware) | `staged` | value of GPU-Aware point-to-point |
| `alltoallv_staged` | Alltoallv | host | `alltoallv` | the D↔H staging copies (collective) |
| `alltoallv` | Alltoallv | GPU (GPU-Aware) | `alltoallv_staged` | value of GPU-Aware collective |
| `rccl` | ncclSend/Recv | GPU (RCCL) | `isend`/`alltoallv` | GPU-native vs GPU-Aware MPI |

Two controlled contrasts fall out immediately:

- **`staged` vs `isend`** and **`alltoallv_staged` vs `alltoallv`** each isolate the **PCIe/staging round-trip**
  (device→host before the send, host→device after the receive) that GPU-Aware MPI removes.
- **`isend` vs `alltoallv` vs `rccl`** compares three *GPU-resident* transports head-to-head.

## Run the comparison

```bash
cd CG-GPU && make
for m in staged isend rccl alltoallv_staged alltoallv; do
  echo "=== $m ==="
  CG_SEED=12345 mpirun -n 4 ./gpu_bind.sh ./cg_gpu src/Dubcova2.pm $m
done
```

Or use the harness that builds once and sweeps all methods with warm-up + repeats:

```bash
sbatch submit_cg_gpu_test.sbatch          # or: GPU_BIND=./gpu_bind.sh ./test_cg_gpu.sh
```

## Results (AAC6, ROCm 6.4.1, 4 ranks, seed 12345, min of runs)

Times are seconds, max across ranks, CG loop only:

| method | solve | comm total | halo exch | dot allreduce | compute | comm % |
|--------|-------|-----------|-----------|---------------|---------|--------|
| staged            | 0.1477 | 0.1243 | 0.1095 | 0.0148 | 0.0234 | 84 % |
| isend             | 0.0513 | 0.0282 | 0.0142 | 0.0140 | 0.0231 | 55 % |
| rccl              | 0.0505 | 0.0252 | 0.0160 | 0.0092 | 0.0253 | 50 % |
| alltoallv_staged  | 0.0823 | 0.0560 | 0.0422 | 0.0138 | 0.0263 | 68 % |
| alltoallv         | 0.0512 | 0.0250 | 0.0129 | 0.0121 | 0.0262 | 49 % |

## How to read it

- **Staging is the dominant cost when present.** `staged` halo (0.110 s) vs `isend` (0.014 s) is a ~7× gap that
  is *purely* the D→H + H→D copies GPU-Aware MPI eliminates. `alltoallv_staged` (0.042 s) vs `alltoallv`
  (0.013 s) shows the same round-trip on the collective path. **Takeaway: if you have GPU-Aware MPI, use it —
  this is the single largest transport-level win.**
- **The three GPU-resident transports are within noise of each other** here (~0.013–0.016 s halo). On one node,
  4 GPUs, over XGMI, `isend`, `alltoallv`, and `rccl` have no decisive winner. Do not over-interpret a 1 ms
  gap that is smaller than your run-to-run spread — this is exactly why Chapter 3 insists on repeats.
- **`compute` is essentially constant across methods** (~0.023–0.026 s). Good — it *should* be, because you
  changed only the transport. If your `compute` column moves a lot between methods, your measurement is noisy
  (revisit affinity/warm-up), not the transport.
- **`dot allreduce` is a real, transport-independent cost** (0.009–0.015 s, up to ~27 % of solve): two
  latency-bound scalar allreduces per iteration, each gated by a `hipDeviceSynchronize`. No choice of halo
  transport touches it — it is a separate optimization axis (Chapter 7).

## The subtlety: overlap means "halo" is exposed wait, not wire time

For `isend`/`rccl`/`alltoallv` the on-proc SpMV runs while messages are in flight, so the measured `halo
exchange` is the part of the transfer that *couldn't* be hidden behind compute. That is the right number for
"which transport costs me less," but it means:

- a faster raw transport can show a *larger* exposed halo if it leaves less compute to overlap with, and
- at larger per-GPU problem sizes (more on-proc work to hide behind), the exposed halo of the overlap variants
  shrinks further relative to `staged`.

To measure *raw* wire time instead, you would disable overlap or use a profiler trace (Chapter 7).

## Does the ranking hold on a different MPI stack?

Yes. On HPE Cray EX with **cray-mpich** (ROCm 7.0.3), the same qualitative picture holds — staging is expensive,
the GPU-resident transports are comparable, and cray-mpich's GPU-Aware path is on par with (marginally better
than) GPU-Aware OpenMPI:

| method | solve | comm total | halo exch | dot allreduce | compute | comm % |
|--------|-------|-----------|-----------|---------------|---------|--------|
| staged            | 0.1722 | 0.1064 | 0.0862 | 0.0202 | 0.0657 | 62 % |
| isend             | 0.0973 | 0.0182 | 0.0146 | 0.0036 | 0.0791 | 19 % |
| rccl              | 0.0926 | 0.0129 | 0.0099 | 0.0030 | 0.0796 | 14 % |
| alltoallv_staged  | 0.1261 | 0.0547 | 0.0439 | 0.0107 | 0.0714 | 43 % |
| alltoallv         | 0.0971 | 0.0191 | 0.0156 | 0.0035 | 0.0779 | 20 % |

*(from [`STUDY_REPORT_PrgEnv-amd.md`](../CG-GPU/STUDY_REPORT_PrgEnv-amd.md) §4). Note the `comm %` is much lower
here — not because communication got faster, but because compute got slower on ROCm 7.x. That cross-cutting
effect is Chapter 6.)*

## Caveats

Single node, single matrix, single rank count, latency-bound regime. Absolute deltas between the fast variants
are small; treat cross-method differences as **indicative**, and always report the measurement conditions with
the numbers. Strong/weak scaling and larger per-GPU sizes (where a transport's bandwidth, not its latency,
dominates) are the natural next experiments.

Next: [5. Tuning the copy engines (SDMA vs blit) →](05-sdma-vs-blit.md)
