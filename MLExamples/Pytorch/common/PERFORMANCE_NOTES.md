# Cross-cutting performance notes (MI300A)

Runtime findings that apply to **all three** distributed examples (`imagenet`,
`minGPT-ddp`, `FSDP2`), measured on AAC6 `PPAC_MI300A_SPX` (4× MI300A, SPX). They
answer the recurring questions: *which ROCm/PyTorch should I use, does the wheel
matter, and does CPU/NUMA affinity help?* For the zero-copy input-staging
(`--migrate`) results, see [`README.md`](README.md).

All numbers are 4-GPU unless noted: RCCL all-reduce **busbw** (GB/s), imagenet
resnet50 b=128 (**img/s**), minGPT gpt2-small (**tok/s**), FSDP2 16L/dim1024
(**tok/s**).

## 1. ROCm / PyTorch version and packaging

### Wheel vs. site module is a wash — at matched ROCm

Comparing the pip wheel `torch 2.9.1+rocm6.4` against the site module
`pytorch/2.12.0` built on **the same ROCm 6.4** isolates packaging (the wheel's
generic bundled rocBLAS/RCCL vs. the site-built libraries):

| Metric | Wheel `2.9.1+rocm6.4` | Module `2.12.0`+`rocm/6.4.3` | Δ |
|---|---|---|---|
| RCCL busbw 64 MB | 222.5 | 222.9 | ~0% |
| RCCL busbw 256 MB | 241.3 | 242.3 | ~0% |
| imagenet | 3840 | 3846 | ~0% |
| minGPT | 157832 | 158754 | +0.6% |
| FSDP2 | 105305 | 104555 | −0.7% |

Everything is within 0.7%. **Packaging makes no meaningful performance
difference on this system** — pick the module for convenience (system MPI/RCCL
integration, no pip/venv management), not for speed.

### The real lever is the ROCm version

Same PyTorch (`pytorch/2.12.0` module), only the ROCm runtime changes:

| Metric | `rocm/6.4.3` | `rocm/7.2.3` | effect |
|---|---|---|---|
| RCCL busbw 64 MB | 222.9 | 231.8 | 7.2.3 **+4%** |
| RCCL busbw 256 MB | 242.3 | 246.8 | 7.2.3 **+2%** |
| imagenet | 3846 | 3956 | 7.2.3 **+3%** |
| minGPT | 158754 | 121756 | 7.2.3 **−23%** |
| FSDP2 | 104555 | 90725 | 7.2.3 **−13%** |

- **ROCm 6.4 is much faster on GEMM-bound transformers** (minGPT +30% / FSDP2
  +15% relative to 7.2.3) due to a rocBLAS GEMM regression in 7.x — the same
  effect seen in the CG-GPU solver study.
- **ROCm 7.2.3 is modestly faster on communication and convolution** (RCCL busbw
  +2–4%, conv-bound imagenet +3%).
- **PyTorch version is noise:** module 2.9.1 vs 2.12.0 on the same ROCm differ by
  <0.7% everywhere.

> Earlier notes attributed the transformer gap to "wheel vs. site RCCL." That was
> a **mis-attribution** — the wheel simply bundles ROCm 6.4; at matched ROCm the
> wheel and module are identical (table above). The gap is purely the ROCm
> version.

### TunableOp closes the GEMM gap on 7.2.3

The `*_tunableop_enabled` PyTorch modules autotune the GEMMs and recover — and in
the sweep exceeded — the ROCm 6.4 transformer throughput, while keeping 7.2.3's
comm/conv advantage. It pays a one-time tuning cost on the first steps (minutes),
so warm up before timing.

### Recommendation

| Situation | Use |
|---|---|
| GEMM-bound transformer, need speed now | `rocm/7.2.3` + a `*_tunableop_enabled` module (best of both), or `rocm/6.4.3` |
| Comm-/conv-bound (RCCL, convnets) | `rocm/7.2.3` |
| Reproducibility / no pip management | site **module** (wheel gives no speed benefit) |

Sample any combo with `module load rocm/<ver> openmpi pytorch/<ver>`.

## 2. CPU / NUMA affinity and placement

The node has **4 NUMA nodes, each = 1 GPU + 24 cores/48 threads + 128 GB local
HBM**, with a clean 1:1 `GPU[i] → node i` map (remote NUMA distance 32 vs. 10
local). Binding each rank to its GPU's node (`numactl --cpunodebind=i
--membind=i`) vs. leaving ranks unbound:

| Workload | free | bind | Δ |
|---|---|---|---|
| RCCL busbw 64 MB | 232.9 | 232.8 | ~0% |
| RCCL busbw 256 MB | 247.9 | 246.3 | −0.7% |
| imagenet (on-GPU input) | 3991 | 3971 | −0.5% |
| imagenet `--host-copy` | 3924 | 3916 | −0.2% |
| imagenet `--migrate` | 3895 | 3977 | +2.1% |
| minGPT | 121780 | 121942 | +0.1% |
| FSDP2 | 90984 | 90861 | −0.1% |

**Conclusion: affinity does not help these GPU-resident workloads** (all within
±1–2% run-to-run noise). Why, and why it's MI300A-specific:

- The steps are **GPU-compute-bound**; the CPU only launches kernels, and RCCL
  moves data **GPU→GPU over xGMI**, never touching CPU/NUMA — hence perfectly
  flat busbw.
- On the APU, "host" memory **is** the GPU's HBM; with `HSA_XNACK=1`/managed
  memory the pages migrate to the accessing GPU regardless of first-touch node,
  so placement is largely self-correcting (unlike a discrete-GPU host, where a
  remote-socket staging buffer pays a real NUMA/PCIe penalty every copy).
- The one positive signal is `--migrate`'s managed host buffer (+2%, near noise),
  the expected direction for host-buffer locality.

**Where affinity *does* matter** (keep the tooling for these): CPU-heavy input
pipelines (real image `DataLoader` decode/augment), **host-staged RCCL fallback**
(`NCCL_P2P_DISABLE=1`, routing collectives through host SHM), **multi-node** runs
(NIC↔NUMA locality), and **CPU-offload** optimizers (ZeRO-offload).

### Enabling it

All three sweep drivers accept `AFFINITY=1`, which prepends
[`affinity_launcher.py`](affinity_launcher.py) (re-execs each rank under
`numactl` using the 1:1 GPU→NUMA map, falling back to a pure-Python CPU pin if
`numactl` is absent):

```bash
AFFINITY=1 GPUS="2 4 8" ./rccl_scaling_sweep.sh          # minGPT / FSDP2
AFFINITY=1 GPUS="1 2 4" ./ddp_bench_sweep.sh             # imagenet
```

Override the map for other partitions/CPX modes with
`AFFINITY_NODES="0,1,2,3"` (indexed by `LOCAL_RANK`). Use it directly, too:

```bash
torchrun --standalone --nproc_per_node=4 ../common/affinity_launcher.py \
  fsdp2_bench.py --n-layers 16
```

## 3. `torch.compile` (`--compile`)

Graph capture + kernel fusion; the first (warm-up) step pays a one-time compile
cost (the benchmarks warm the compiled graph before timing). Modest, consistent
gains (2 GPUs shown):

| Example | baseline | `--compile` | Δ |
|---|---|---|---|
| imagenet resnet50 | 2013 img/s | 2128 | +5.7% |
| minGPT gpt2-small | 60182 tok/s | 62640 | +4.1% |
| FSDP2 transformer | 43586 tok/s | 43850 | +0.6% |

Stacks with `--amp` and (for transformers) with TunableOp. Pass through the
sweeps with `OPTS="--compile"` (or `OPTS="--amp --compile"`).
