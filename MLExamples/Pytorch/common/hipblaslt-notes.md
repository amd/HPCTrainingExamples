# hipBLASLt on the PyTorch examples: `hipblaslt/patched` and a bf16 gotcha

Shared notes for [`imagenet`](../imagenet/README.md), [`minGPT-ddp`](../minGPT-ddp/README.md),
and [`FSDP2`](../FSDP2/README.md). **hipBLASLt** is the GEMM (matrix-multiply)
backend rocBLAS/PyTorch use for many `--amp`/`--mixed-precision` (fp16/bf16) ops.

> **Verified on AAC6 `PPAC_MI300A_SPX` (MI300A), ROCm 7.2.3, PyTorch 2.12.0, 2 GPUs.**
> All numbers below are from A/B runs on one node.

## 1. Does `hipblaslt/patched` change performance here? No.

On ROCm 7.2.x a `hipblaslt/patched` module is available. It does **not** replace
`libhipblaslt.so`; it sets `HIPBLASLT_TENSILE_LIBPATH` to a **Tensile kernel
overlay** whose stated scope is narrow:

> *"hipBLASLt heuristic-regression overlay … Restores perf on MI300A SPX and CPX
> for **skinny fp16 GEMMs**."*

So it is a targeted micro-fix for specific skinny fp16 GEMM shapes, not a general
speed-up. A/B on the same node (stock vs patched) shows **no measurable difference**
on these workloads:

| Test | stock | patched | Δ |
|------|-------|---------|---|
| ResNet-50 `--amp` (img/s, 2 GPU) | 244 | 245 | none (noise) |
| square bf16 GEMM 4k / 8k (TFLOP/s) | 286 / 375 | 286 / 376 | none |
| skinny fp16 GEMM (8 shapes) | — | — | identical within noise |

Reasons: ResNet is conv-bound (MIOpen), and the transformer/GEMM shapes these
examples hit are not the specific skinny fp16 shapes the overlay targets. torch
loads the stock `libhipblaslt.so.1.2.70203` either way.

**Guidance:** loading `hipblaslt/patched` is harmless and correct where a ROCm build
provides it (it helps GEMM-inference workloads dominated by those skinny shapes),
but expect **no change** on imagenet / minGPT-ddp / FSDP2. This mirrors the CG
solver finding (`MPI-examples/cg-solver-example/CG-GPU/STUDY_REPORT.md`, §5.3).

## 2. bf16 transformer hang on ROCm 7.2.x — force rocBLAS

Separately (and **not** fixed by `hipblaslt/patched`), on **ROCm 7.2.x** the
**bf16 transformer** path (`minGPT-ddp --amp`, `FSDP2 --mixed-precision`) can
**stall for minutes in hipBLASLt** on the first GEMMs. fp32 runs fine, and ResNet
`--amp` is unaffected — it is specific to the transformer bf16 GEMM shapes on the
7.2.x hipBLASLt. (bf16 works normally on ROCm 6.4.3 — see the READMEs' measured
bf16 results.)

Workaround — route GEMMs to rocBLAS instead of hipBLASLt:

```bash
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

Measured on minGPT-ddp (n_layer=12, n_embd=768, bf16, 2 GPU):

| config | result |
|--------|--------|
| bf16 via **hipBLASLt** (default, 7.2.3) | **hangs** (>300 s) |
| bf16 via **rocBLAS** (`TORCH_BLAS_PREFER_HIPBLASLT=0`) | 137,645 tok/s |
| fp32 | 60,723 tok/s |

So with the env var set, bf16 both runs and is ~2.3× faster than fp32, as expected.
Prefer running the transformer bf16 sweeps on ROCm 6.4.3, or set
`TORCH_BLAS_PREFER_HIPBLASLT=0` on 7.2.x.
