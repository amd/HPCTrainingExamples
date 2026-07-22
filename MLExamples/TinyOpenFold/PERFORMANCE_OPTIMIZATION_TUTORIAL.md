# TinyOpenFold: Complete Performance Optimization Tutorial

**Learn GPU optimization by progressively improving AlphaFold 2 Evoformer performance**

This tutorial demonstrates the complete GPU optimization pipeline from baseline PyTorch through kernel fusion and custom Triton kernels, with kernel fusion (V2) delivering the peak **~2.4x speedup** and `torch.compile` adding an orthogonal ~1.6–1.9x on top of the baseline. All numbers are measured on AMD Instinct MI300X / ROCm 7.13.

---

## Table of Contents
1. [Tutorial Overview](#tutorial-overview)
2. [Environment Setup](#environment-setup)
3. [Stage 1: Baseline (V1)](#stage-1-baseline-v1---pure-pytorch)
4. [Stage 2: Kernel Fusion (V2)](#stage-2-kernel-fusion-v2---pytorch-level-optimization)
5. [Stage 3: Custom Kernels (V3)](#stage-3-custom-triton-kernels-v3---gpu-level-optimization)
6. [Performance Analysis](#performance-analysis)
7. [Lessons Learned](#lessons-learned)

---

## Tutorial Overview

### What You'll Learn

This tutorial covers the complete optimization pipeline from profiling to implementation. You'll start by establishing baseline performance metrics with clean PyTorch code, then apply high-level kernel fusion optimizations without writing custom GPU code. Next, you'll drop down to low-level custom Triton kernels for maximum performance. Throughout the journey, you'll learn profiling techniques to identify bottlenecks at each stage and develop the analytical skills to understand exactly where speedups come from.

### Performance Journey

Measured on **AMD Instinct MI300X, ROCm 7.13 (PyTorch 2.11 / Triton 3.6)**, small
problem (seq-len 64, 16 MSA, batch 4), training samples/sec:

```
Version 1 (Baseline)   →   Version 2 (Fused)   →   Version 3 (Triton)
   100 samples/sec          241 samples/sec         92 samples/sec
      1.00x                    2.4x                    0.9x
  [Pure PyTorch]          [Kernel Fusion]        [Custom Triton kernels]
```

**Two independent optimization axes.** Kernel *fusion* (V2) is the biggest single
lever — ~2.4x with no memory cost. `torch.compile` (inductor auto-fusion) is an
orthogonal lever available on **all three** versions and roughly doubles V1 and V3:

```
              eager      + torch.compile
V1 baseline   100 s/s        164 s/s      (1.6x)
V2 fused      241 s/s        235 s/s      (already fused — compile adds nothing)
V3 triton      92 s/s        175 s/s      (1.9x — compiled V3 edges past compiled V1)
```

**How to read this tutorial:** V2 (fusion) is the performance peak and the main
lesson. V3's lesson is different — how to write *correct, gradient-safe* custom
Triton kernels and combine them with `torch.compile`; eager V3 sits near the V1
baseline (the model is GEMM-dominated, so the memory-bound ops V3 customizes are a
small slice), but compiled V3 overtakes compiled V1. Numbers below are ROCm 7.13
measurements and will vary by hardware and stack.

> **Note on the earlier "2.0x from custom kernels" figure:** that came from an
> initial V3 whose raw Triton kernels bypassed autograd (no gradients flowed, so
> the backward pass did almost no work). With correct gradients the honest result
> is the table above. See Stage 3 for details.

### Problem Sizes (Small & Medium for best demonstration)

| Size | Seq Length | MSA Seqs | Batch | Memory | Best For |
|------|------------|----------|-------|--------|----------|
| **Small** | 64 | 16 | 4 | ~196 MB | Quick demos; V2 fusion ~2.4x, compile stacks on V1/V3 |
| **Medium** | 128 | 32 | 2 | ~209 MB | Realistic workload; V2 fusion ~1.9x |

---

## Environment Setup

### Option A: ROCm 7.2 (module + PyTorch rocm7.1 nightly)

```bash
# Load required modules
module load python/3.12 rocm/7.2 libffi/3.3

# Navigate to TinyOpenFold
cd /mnt/thera/data/incoming/asimishr/aiml_prof/HPCTrainingExamples/MLExamples/TinyOpenFold

# Activate virtual environment
source venvOF/bin/activate

# Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected**: `GPU: AMD Instinct MI300X`

### Option B: ROCm 7.13 (TheRock nightly — PyTorch 2.11 + Triton 3.6)

Newer alpha stack for gfx942 (MI300X) / gfx950 (MI355X): PyTorch 2.11 + Triton 3.6. ROCm 7.13 is
alpha-only and not on the pytorch.org nightly index. There are two ways to get the ROCm 7.13
runtime — pick whichever matches your site:

- **B1 — TheRock pip wheels (self-contained).** The ROCm runtime is pulled in as a wheel dependency
  of `torch` from AMD's [TheRock](https://github.com/ROCm/TheRock) multi-arch pip channel. No system
  ROCm module needed.
- **B2 — ROCm 7.13 as a system module.** If your cluster provides a `rocm/7.13` module, `module load`
  it and install only the Python packages (`torch`, `triton`, …) into a venv on top of it.

Both install the Python packages into a venv; they differ only in where the ROCm runtime comes from.

**B1 — TheRock pip wheels:**

```bash
# Python 3.14 from the module system
module load python/3.14

cd HPCTrainingExamples/MLExamples/TinyOpenFold
python3 -m venv venv713 && source venv713/bin/activate
pip3 install --upgrade pip

# PyTorch 2.11 + Triton 3.6 for ROCm 7.13 (torch pulls the ROCm libraries in automatically)
pip3 install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    torch torchvision torchaudio triton
pip3 install deepspeed && pip3 install -r setup/requirements.txt

# Verify the stack and GPU
python3 -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '|', torch.cuda.get_device_name(0))"
```

**B2 — ROCm 7.13 system module (if available at your site):**

```bash
module load rocm/7.13 python/3.14        # module names may differ per cluster
python3 -m venv venv713 && source venv713/bin/activate
pip3 install --upgrade pip

# Match the torch/triton wheels to the module's ROCm version. Use the pytorch.org
# ROCm index if a matching build exists, or your site's provided wheels.
pip3 install torch torchvision torchaudio triton
pip3 install deepspeed && pip3 install -r setup/requirements.txt

python3 -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '|', torch.cuda.get_device_name(0))"
```

**Expected**: `torch 2.11.0+rocm7.13.0a... | triton 3.6.0+rocm7.13.0a... | AMD Instinct MI300X`

Notes for Option B:
- **B1 (pip wheels):** no `LD_LIBRARY_PATH` / `libcaffe2_nvrtc.so` fix is needed — the TheRock wheels
  bundle their own runtime.
- **B2 (module):** the ROCm runtime comes from the module, so you may need `ROCM_PATH` /
  `LD_LIBRARY_PATH` set up by the module (usually automatic on `module load`).
- The ROCm hardware profilers (`rocprofv3`, `rocprof-compute`, `rocprof-sys`) come with the module
  (B2) but are *not* in the TheRock training wheels (B1). For B1, install the full TheRock SDK
  (`rocm[libraries,devel,device-gfx942]==7.13.*`).

### `torch.compile` — an optimization axis available on every version

All three versions accept `--enable-torch-compile`, which wraps the model in
`torch.compile` (PyTorch's inductor backend generates and fuses Triton kernels
automatically). It is **orthogonal** to the manual optimizations: it stacks on top
of V1's plain PyTorch, V2's fusion, and V3's custom kernels.

```bash
# Works on all three (add to any run command):
python3 tiny_openfold_v1.py --seq-len 64 --num-seqs 16 --batch-size 4 --enable-torch-compile
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py --enable-all-fusion --enable-torch-compile
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py --enable-torch-compile
```

- First step is slow (compilation); measured steps after warm-up are what count.
- Biggest wins on V1 (~1.6x) and V3 (~1.9x). V2 is already hand-fused, so compile
  adds little on top.
- For V3, the custom Triton LayerNorm still genuinely runs under compile (inductor
  calls out to it); inductor fuses the surrounding graph.

---

## Stage 1: Baseline (V1) - Pure PyTorch

### Objective
Establish baseline performance with clean, readable PyTorch implementation.

### Characteristics

The baseline prioritizes clarity over performance. The code is clean and well-documented, using only standard PyTorch operations that anyone can understand. However, it's completely unoptimized—each operation launches a separate GPU kernel with no fusion. This means kernel launch overhead dominates execution time, especially for small workloads.

### Run Small Problem

```bash
cd version1_pytorch_baseline

python3 tiny_openfold_v1.py \
    --seq-len 64 \
    --num-seqs 16 \
    --batch-size 4 \
    --num-blocks 4 \
    --num-steps 30 \
    --device 0
```

### Expected Output

```
================================================================================
TINY OPENFOLD - VERSION 1: PYTORCH BASELINE
================================================================================

Model Configuration:
   MSA dimension: 64
   Pair dimension: 128
   Evoformer blocks: 4
   Total parameters: 2,653,760
   Model size: 10.6 MB (FP32)

Training Configuration:
   Training steps: 30
   Batch size: 4

======================================================================
Step   0/50 | Loss: 33.06 | Speed: 100.1 samples/sec | Memory:  195.7 MB | Time:  40.0ms
Step  10/50 | Loss: 33.25 | Speed: 100.1 samples/sec | Memory:  195.7 MB | Time:  40.0ms
Step  20/50 | Loss: 33.45 | Speed: 100.1 samples/sec | Memory:  195.7 MB | Time:  40.0ms
======================================================================

Performance Summary:
   Average training speed: 100.1 samples/sec
   Average batch time: 40.0 ms
   Average forward time: 15.4 ms
   Average backward time: 21.0 ms
   Average optimizer time: 4.1 ms
   Peak memory usage: 195.7 MB
```

*(Measured on MI300X / ROCm 7.13. With `--enable-torch-compile` this rises to ~164
samples/sec / 24.4 ms.)*

### Key Metrics (Small Problem)

| Metric | Value | Notes |
|--------|-------|-------|
| Speed | **100.1 samples/sec** | Baseline reference (eager) |
| Batch time | **40.0 ms** | Total time per step |
| Forward | 15.4 ms | ~38% of batch time |
| **Backward** | **21.0 ms** | **~53% of batch time** (main bottleneck) |
| Optimizer | 4.1 ms | ~10% of batch time |
| Memory | 195.7 MB | Peak allocation |

### Bottleneck Analysis

**Profile with PyTorch Profiler:**

```bash
python3 tiny_openfold_v1.py \
    --seq-len 64 --num-seqs 16 --batch-size 4 \
    --enable-pytorch-profiler \
    --profile-dir ./profiles_v1_small \
    --device 0

# View results
tensorboard --logdir ./profiles_v1_small
```

**What to look for in the profiler traces:**

You'll notice multiple attention kernels where Q, K, and V are computed as separate operations instead of being fused. Triangle operations dominate the backward pass due to their O(N³) complexity. You'll also see significant kernel launch overhead from many small, short-lived kernel calls.

**Optional: ROCm System-Level Profiling**

For deeper insights into GPU utilization and kernel behavior, use rocprof-sys:

```bash
# Profile GPU kernels and API calls
./run_rocprof_sys.sh

# Results show: kernel launch frequency, memory transfers, GPU occupancy
# Look for: many short-lived kernels, poor occupancy on small operations
```

### Run Medium Problem

```bash
python3 tiny_openfold_v1.py \
    --seq-len 128 \
    --num-seqs 32 \
    --batch-size 2 \
    --num-blocks 4 \
    --num-steps 30 \
    --device 0
```

### Key Metrics (Medium Problem)

| Metric | Value | Notes |
|--------|-------|-------|
| Speed | **47.4 samples/sec** | ~half of small (larger per-sample work) |
| Batch time | **42.2 ms** | Similar to small! (batch size = 2 vs 4) |
| Memory | 208.9 MB | Scales with sequence length² |

*(With `--enable-torch-compile`: ~73 samples/sec / 27.4 ms.)*

### Stage 1 Summary

**Baseline Established:**

We now have reference numbers for both problem sizes. The small problem runs at ~100 samples/sec with 40.0 ms per batch, while the medium problem achieves ~47 samples/sec at 42.2 ms per batch (eager, MI300X / ROCm 7.13).

**Bottlenecks Identified:**

Profiling reveals where optimization will have the most impact. The backward pass dominates (~53% of batch time), with many small kernel launches for attention operations creating overhead. Triangle operations are the most compute-intensive due to their cubic complexity, and run as rocBLAS GEMMs.

**Next Step**: Apply kernel fusion to reduce launch overhead

---

## Stage 2: Kernel Fusion (V2) - PyTorch-Level Optimization

### Objective
Reduce kernel launch overhead by fusing operations at the PyTorch level.

### Optimizations Applied

#### 1. MSA QKV Fusion
**Before (V1)**:
```python
q = self.q_proj(msa)  # Kernel 1
k = self.k_proj(msa)  # Kernel 2
v = self.v_proj(msa)  # Kernel 3
```

**After (V2)**:
```python
qkv = self.qkv_proj(msa)  # Single fused kernel
q, k, v = qkv.chunk(3, dim=-1)
```

**Benefit**: 3 kernels → 1 kernel

#### 2. Flash Attention
**Before (V1)**:
```python
# Standard attention: O(N²) memory
scores = torch.matmul(q, k.transpose(-2, -1))
attn_weights = softmax(scores / sqrt(d_k))
output = torch.matmul(attn_weights, v)
```

**After (V2)**:
```python
# Flash Attention: O(N) memory, fused kernel
output = F.scaled_dot_product_attention(q, k, v)
```

**Benefit**: Memory-efficient, fewer kernels, better cache utilization

#### 3. Triangle Gate/Proj Fusion
**Before (V1)**:
```python
left = self.left_proj(pair)      # Kernel 1
right = self.right_proj(pair)    # Kernel 2
left_gate = sigmoid(self.left_gate_proj(pair))   # Kernel 3
right_gate = sigmoid(self.right_gate_proj(pair)) # Kernel 4
```

**After (V2)**:
```python
# Fused gate and projection
combined = self.fused_gate_proj(pair)  # Single kernel
left, right, left_gate, right_gate = combined.chunk(4, dim=-1)
left_gate = sigmoid(left_gate)
right_gate = sigmoid(right_gate)
```

**Benefit**: 4 kernels → 2 kernels

### Run Small Problem (V2)

```bash
cd ../version2_pytorch_fused

ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py \
    --seq-len 64 \
    --num-seqs 16 \
    --batch-size 4 \
    --num-blocks 4 \
    --num-steps 30
```

### Expected Output

```
================================================================================
TINY OPENFOLD - VERSION 2: PYTORCH FUSED
================================================================================

Fusion Optimizations:
   MSA QKV Fusion: Enabled
   Triangle QKV Fusion: Enabled
   Flash Attention: Enabled
   Triangle Gate/Proj Fusion: Enabled
   Kernel Reduction: 80.0% (48 fewer kernels)

======================================================================
Step   0/50 | Loss: 33.06 | Speed: 240.8 samples/sec | Memory:  195.6 MB | Time:  16.6ms
Step  10/50 | Loss: 33.25 | Speed: 240.8 samples/sec | Memory:  195.6 MB | Time:  16.6ms
Step  20/50 | Loss: 33.45 | Speed: 240.8 samples/sec | Memory:  195.6 MB | Time:  16.6ms
======================================================================

Performance Summary V2:
   Average training speed: 240.8 samples/sec  [+141% vs V1]
   Average batch time: 16.6 ms                [-59% vs V1]
   Peak memory usage: 195.6 MB                [Same as V1]
```

### V1 → V2 Improvement (Small Problem, MI300X / ROCm 7.13)

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| Speed | 100.1 s/s | 240.8 s/s | **+141%** ⚡⚡⚡ |
| Batch time | 40.0 ms | 16.6 ms | **-59%** |
| Memory | 195.7 MB | 195.6 MB | **No change** |

**Key Insight**: Kernel fusion (QKV + gate/proj + Flash Attention) cuts launch
overhead across both forward and backward — a ~2.4x throughput gain with no memory
cost. `torch.compile` adds almost nothing on top of V2 because it's already fused.

### Run Medium Problem (V2)

```bash
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py \
    --seq-len 128 \
    --num-seqs 32 \
    --batch-size 2 \
    --num-blocks 4 \
    --num-steps 30
```

### V1 → V2 Improvement (Medium Problem)

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| Speed | 47.4 s/s | 88.8 s/s | **+87%** ⚡⚡ |
| Batch time | 42.2 ms | 22.5 ms | **-47%** |
| Memory | 208.9 MB | 208.9 MB | **No change** |

### Ablation Study: Which Fusion Helps Most?

Test individual optimizations to understand their contribution:

```bash
cd version2_pytorch_fused

# Baseline (all fusions disabled)
echo "=== Baseline (all fusions off) ==="
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py \
    --seq-len 64 --num-seqs 16 --batch-size 4 --num-steps 20 \
    --disable-all-fusion | grep "Average training speed"

# Only MSA QKV fusion
echo "=== Only MSA QKV fusion ==="
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py \
    --seq-len 64 --num-seqs 16 --batch-size 4 --num-steps 20 \
    --disable-all-fusion --enable-qkv-fusion-msa | grep "Average training speed"

# Only Flash Attention
echo "=== Only Flash Attention ==="
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py \
    --seq-len 64 --num-seqs 16 --batch-size 4 --num-steps 20 \
    --disable-all-fusion --enable-flash-attention | grep "Average training speed"

# Only Triangle fusion
echo "=== Only Triangle fusion ==="
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py \
    --seq-len 64 --num-seqs 16 --batch-size 4 --num-steps 20 \
    --disable-all-fusion --enable-triangle-fusion | grep "Average training speed"

# All fusions (default)
echo "=== All fusions enabled ==="
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py \
    --seq-len 64 --num-seqs 16 --batch-size 4 --num-steps 20 | grep "Average training speed"
```

**Expected Results:**

Each optimization contributes differently. Starting from the `--disable-all-fusion`
baseline (which behaves like V1), enabling fusions individually each adds a modest
gain (Flash Attention typically the largest single contributor), and enabling them
all together is where the big jump happens — from ~100 samples/sec (no fusion) to
~240 samples/sec (all fusions) at the small size on MI300X / ROCm 7.13. Exact
per-fusion percentages vary run to run; the point is that the fusions are synergistic.

**Key Learning**: Flash Attention provides the biggest single benefit, but combined optimizations are synergistic.

### Verify Fusion Impact with ROCm Profilers

Now that we've fused kernels, let's verify the improvements at the hardware level:

```bash
cd version2_pytorch_fused

# Kernel-level profiling with rocprofv3
./run_rocprofv3.sh

# Hardware counter analysis with rocprof-compute
./run_rocprof_compute.sh

# Compare kernel counts: V1 vs V2
# V1: ~240 kernel launches per step
# V2: ~48 kernel launches per step (80% reduction!)
```

**Key metrics to check:**
- **Kernel count**: Should see dramatic reduction in total kernel launches
- **Memory bandwidth**: Flash Attention should reduce HBM traffic by 50-80%
- **Occupancy**: Fused kernels should show better GPU utilization

### Stage 2 Summary

**Achievements:**

Kernel fusion delivers the tutorial's biggest gains without increasing memory usage.
The small problem improved from ~100 to ~241 samples/sec (**~2.4x**), and the medium
problem from ~47 to ~89 samples/sec (**~1.9x**), with ~80% fewer kernel launches and
no memory overhead. This is the performance peak of the three versions.

**Remaining Bottlenecks:**

The remaining time is in the triangle/transition **GEMMs**, which are already handled
by rocBLAS and which fusion doesn't touch. That's the key context for Stage 3: the
next lever (custom kernels) targets memory-bound ops that are *not* the bottleneck
here, so it teaches kernel authoring more than it moves the clock.

**Next Step**: Write custom Triton kernels (V3) — and learn where they do and don't help

---

## Stage 3: Custom Triton Kernels (V3) - GPU-Level Optimization

### Objective
Learn to write **correct, gradient-safe** custom Triton kernels and integrate them
into a real training loop — and understand *when* custom kernels help versus when
PyTorch's native/compiled kernels already win.

> **What V3 is (and isn't).** V3 is the "author your own GPU kernels" lesson, not
> the performance peak — that's V2. On this GEMM-dominated model at educational
> sizes, the memory-bound ops V3 customizes (LayerNorm, attention) are a small
> fraction of runtime, so eager V3 lands near the V1 baseline. Its payoff shows up
> **with `torch.compile`**, where compiled V3 overtakes compiled V1. The lesson is
> as much about *measuring honestly* as about writing kernels.

### Correctness first: three things V3 gets right

Custom kernels are only useful if they're numerically correct and differentiable.
V3's kernels are validated by `test_correctness.py` (run it — expect **4/4**):

1. **LayerNorm numerics.** The Triton LayerNorm matches `torch.nn.LayerNorm` to
   ~1e-6. (A subtle bug to avoid: masked out-of-range lanes must be excluded from
   the variance reduction with `tl.where`, or they inflate the variance.)
2. **Gradients actually flow.** A raw `@triton.jit` kernel that writes into a fresh
   output tensor is *opaque to autograd* — no parameter behind it gets a gradient.
   V3 wraps its kernels in `torch.autograd.Function` with an explicit backward, so
   `loss.backward()` trains every parameter.
3. **Pair bias is used.** MSA row attention is genuinely pair-biased (via a
   bias-aware fused attention), not silently dropped.

### Triton Optimizations

#### 1. Custom LayerNorm Kernel
**Why optimize?** Standard LayerNorm is memory-bound and makes multiple passes through data.

**Triton Implementation**:
```python
@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_elements, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Fused LayerNorm: compute mean, variance, normalize, and scale in one pass.
    
    Memory optimization:
    - Two passes through input (statistics + normalize)
    - Mean/variance computed in registers
    - Immediate normalization and scaling
    """
    # Load block of data
    block_id = tl.program_id(0)
    offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    
    # Compute statistics in registers
    mean = tl.sum(x, axis=0) / n_elements
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_elements
    
    # Normalize and scale
    rstd = 1 / tl.sqrt(var + eps)
    weight = tl.load(weight_ptr + offset, mask=mask, other=1.0)
    output = (x - mean) * rstd * weight
    
    # Store result
    tl.store(output_ptr + offset, output, mask=mask)
```

**Benefits:**

Custom implementation beats PyTorch's generic approach. Instead of 3+ separate kernel launches, we execute everything in a single kernel. Data stays in cache and registers rather than being written back to main memory between operations. Memory access patterns are hand-optimized for sequential reads and writes.

#### 2. Flash Attention for MSA (Triton)
**Why optimize?** MSA operations dominate forward/backward passes.

**Key Optimizations:**

MSA attention is memory-bound, so we focus on reducing data movement. Tiled computation allows us to fit working sets in shared memory, dramatically reducing expensive HBM (main memory) traffic. The implementation is specifically optimized for ROCm/AMD GPUs, taking advantage of architectural features like LDS (local data share).

#### 3. Flash Attention for Triangles (Triton)
**Why optimize?** Triangle operations are O(N³) and very expensive.

**Key Optimizations:**

Triangle operations have O(N³) complexity, making backward pass optimization critical. We use a custom tiling strategy designed specifically for the pair representation's access patterns. Memory transfers are minimized by reusing data across tiles. The backward pass gets special attention since it's the biggest bottleneck—custom gradient implementations avoid PyTorch's generic autograd overhead.

### Run Small Problem (V3)

```bash
cd ../version3_triton

ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py \
    --seq-len 64 \
    --num-seqs 16 \
    --batch-size 4 \
    --num-blocks 4 \
    --num-steps 30
```

### Expected Output

```
================================================================================
TINY OPENFOLD - VERSION 3: TRITON CUSTOM KERNELS
================================================================================

Triton Kernel Performance:
   Custom kernels active: LayerNorm (Triton), Flash Attention (SDPA), QKV fusion

Running warmup steps to compile Triton kernels...
Warmup complete. Starting measured training loop...

======================================================================
Step   0/50 | Loss: 33.06 | Speed:  91.5 samples/sec | Memory:  195.6 MB | Time:  43.7ms
Step  10/50 | Loss: 33.25 | Speed:  91.6 samples/sec | Memory:  195.6 MB | Time:  43.7ms
======================================================================

Performance Summary V3 (eager):
   Average training speed: 91.5 samples/sec   [~0.9x vs V1 eager]
   Average batch time: 43.7 ms
   Final loss: 33.21                          [matches V1 — numerically correct]
   Peak memory usage: 195.6 MB
```

### V1 → V2 → V3 Progression (Small Problem, ROCm 7.13 / MI300X)

**Eager (no `torch.compile`):**

| Metric | V1 | V2 | V3 | V1→V2 | V1→V3 |
|--------|----|----|----|-------|-------|
| **Speed** | 100 s/s | **241 s/s** | 92 s/s | **+141%** ⚡ | -8% |
| Batch time | 40.0 ms | **16.6 ms** | 43.7 ms | -59% | +9% |
| Memory | 195.7 MB | 195.6 MB | 195.6 MB | ~0% | ~0% |

V2 (fusion) is the clear performance win. Eager V3 sits slightly *below* V1: the
model's runtime is dominated by triangle/transition GEMMs (identical rocBLAS calls
in every version), so customizing the memory-bound LayerNorm/attention moves the
needle little, and the custom-kernel autograd wrappers add a small per-call
overhead at this size.

**With `--enable-torch-compile`:**

| Metric | V1 | V2 | V3 |
|--------|----|----|----|
| **Speed** | 164 s/s | 235 s/s | **175 s/s** |
| Batch time | 24.4 ms | 17.1 ms | 22.8 ms |
| vs its own eager | **1.6x** | ~1.0x | **1.9x** |

`torch.compile` roughly doubles V1 and V3. Compiled V3 (175 s/s) **edges past
compiled V1 (164 s/s)** — the custom Triton LayerNorm plus QKV fusion and SDPA
finally pay off once inductor fuses the surrounding graph. V2 is already hand-fused,
so compile adds nothing.

**🎯 Key lesson**: On a GEMM-dominated model at small sizes, hand-written
memory-bound kernels barely move total runtime on their own — but they compose with
`torch.compile` to overtake the baseline. Always measure against the *compiled*
baseline, not just eager.

### Run Medium Problem (V3)

```bash
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py \
    --seq-len 128 \
    --num-seqs 32 \
    --batch-size 2 \
    --num-blocks 4 \
    --num-steps 30
```

### V1 → V2 → V3 Progression (Medium Problem, seq-len 128 / 32 MSA / batch 2)

**Eager:**

| Metric | V1 | V2 | V3 | V1→V2 | V1→V3 |
|--------|----|----|----|-------|-------|
| **Speed** | 47.4 s/s | **88.8 s/s** | 45.0 s/s | **+87%** ⚡ | -5% |
| Batch time | 42.2 ms | **22.5 ms** | 44.5 ms | -47% | +5% |

**With `--enable-torch-compile`:**

| Metric | V1 | V2 | V3 |
|--------|----|----|----|
| **Speed** | 73.1 s/s | 88.7 s/s | **76.9 s/s** |
| Batch time | 27.4 ms | 22.6 ms | 26.0 ms |
| vs its own eager | **1.5x** | ~1.0x | **1.7x** |

Same pattern as small: V2 fusion is the peak; compiled V3 (76.9 s/s) edges past
compiled V1 (73.1 s/s). The crossover holds across problem sizes.

### Why eager V3 ≈ V1 (and where the custom kernels *do* help)

This model spends most of its time in **triangle and transition GEMMs**, which are
identical rocBLAS calls in all three versions — V3 doesn't touch them. The ops V3
customizes (LayerNorm, attention) are memory-bound but a small slice of total
runtime, so replacing them barely moves the needle on its own. Two forces cancel out
in eager mode:

- **Wins:** QKV fusion (3 GEMMs → 1) and PyTorch's fused SDPA (a real flash-attention
  kernel) cut launch overhead.
- **Costs:** each custom Triton kernel is wrapped in a `torch.autograd.Function`, and
  that Python wrapper adds ~0.1 ms/call. Across ~36 LayerNorm calls per step that
  overhead roughly matches the fusion savings.

`torch.compile` tips the balance: inductor fuses the graph around the custom kernels
and eliminates the per-call Python overhead, so compiled V3 pulls ahead of compiled
V1. The custom Triton LayerNorm still runs under compile (verified: inductor calls
out to it rather than replacing it).

### Analyze Triton Kernel Performance

Verify that custom Triton kernels are actually faster at the hardware level:

```bash
cd version3_triton

# Profile Triton kernel efficiency
./run_rocprof_compute.sh

# System-level view of Triton kernels
./run_rocprof_sys.sh
```

**What to verify:**
- **Custom LayerNorm**: Single kernel vs 3+ PyTorch kernels, better register usage
- **Flash Attention**: Reduced HBM bandwidth (memory-bound → compute-bound)
- **Triangle kernels**: Improved cache hit rate, minimized memory traffic
- **Overall occupancy**: Higher GPU utilization compared to V1/V2

**Pro tip**: Compare rocprof-compute outputs between V2 and V3 to see memory bandwidth reduction—this is where Triton shines.

### Stage 3 Summary

**What V3 achieves:**

Correct, gradient-safe custom Triton kernels integrated into a real training loop.
`test_correctness.py` passes 4/4 (LayerNorm numerics, MSA attention, full-model
forward, and gradient flow), and V3's loss curve matches the V1 baseline — the
kernels are numerically faithful, not just fast-looking.

**Performance (ROCm 7.13 / MI300X):**

- Eager V3 lands near the V1 baseline (0.9–0.95x) — the model is GEMM-dominated, so
  customizing memory-bound ops has limited headroom, and autograd-wrapper overhead
  offsets the fusion wins at these sizes.
- With `torch.compile`, V3 reaches ~1.7–1.9x over eager V1 and edges past compiled
  V1. This is the regime where writing custom kernels pays off.
- V2 (kernel fusion) remains the outright performance peak (~2.4x small / ~1.9x
  medium) with no memory cost.

**Trade-offs & the real lesson:**

Custom kernels add real complexity (GPU programming, explicit autograd backward) and,
on this workload, don't beat native/compiled kernels on their own — the compute is in
GEMMs neither version customizes. The takeaway is a professional habit: **profile to
find where time actually goes, measure against the compiled baseline, and reach for
custom kernels only when a memory-bound op is genuinely the bottleneck.**

---

## Performance Analysis

### Complete Comparison Table

All numbers measured on **AMD Instinct MI300X, ROCm 7.13** (PyTorch 2.11 / Triton
3.6), average training speed in samples/sec.

#### Small Problem (64 residues, 16 MSA, batch=4)

```
Mode              V1 Baseline    V2 Fused      V3 Triton
──────────────────────────────────────────────────────────────
eager (s/s)          100           241            92
+torch.compile       164           235           175
peak memory (MB)     196           196           196
──────────────────────────────────────────────────────────────
Best:  V2 fused (241 s/s, 2.4x over eager V1)
```

#### Medium Problem (128 residues, 32 MSA, batch=2)

```
Mode              V1 Baseline    V2 Fused      V3 Triton
──────────────────────────────────────────────────────────────
eager (s/s)          47            89            45
+torch.compile       73            89            77
──────────────────────────────────────────────────────────────
Best:  V2 fused (89 s/s, ~1.9x over eager V1)
```

### Where Do the Gains Come From?

Two orthogonal levers, plus one that doesn't pay off here:

1. **Kernel fusion (V2) — the big win.** Fusing QKV projections, gate/proj, and
   using Flash Attention cuts kernel-launch overhead dramatically: **~2.4x** (small)
   and **~1.9x** (medium) over eager V1, with no extra memory. This is the lesson to
   internalize.
2. **`torch.compile` — a free, stacking win.** Inductor auto-fusion gives ~1.6x on
   V1 and ~1.9x on V3. It adds ~nothing to V2 because V2 is already hand-fused —
   a nice demonstration that manual fusion and compiler fusion target the same
   overhead.
3. **Custom Triton kernels (V3) — not a standalone win on this model.** The runtime
   is dominated by triangle/transition GEMMs (identical rocBLAS in every version),
   so hand-writing the memory-bound LayerNorm/attention kernels can't move the total
   much. Eager V3 ≈ V1; only *with* `torch.compile` does V3 edge past V1.

### Memory

Memory is essentially flat across all three versions (~196 MB small, ~209 MB
medium) — none of these optimizations trades memory for speed at these sizes.

---

## Lessons Learned

### 1. Optimization Strategy

Always optimize incrementally and profile before you act. Start with a clean baseline
(V1), profile to find where time *actually* goes, then apply the cheapest effective
lever first. Here that order is: **kernel fusion (V2) → `torch.compile` → custom
kernels (V3)**. Fusion gave the biggest win for the least code; custom kernels were
the most effort and, on this model, the smallest standalone payoff. Don't jump
straight to writing Triton.

### 2. Know What Dominates Your Runtime

The single most important measurement here: this model is **GEMM-dominated**
(triangle multiplications, transitions). Those GEMMs are identical rocBLAS calls in
every version, so no amount of hand-writing memory-bound kernels changes them. Custom
kernels only help when a *memory-bound* op is genuinely your bottleneck — profile to
confirm that before investing in Triton.

### 3. Measure Against the Compiled Baseline

`torch.compile` roughly doubles the plain baseline for free. If you compare your
hand-optimized version only against *eager* V1, you can fool yourself — eager V3 looks
like ~V1, but the honest comparison is compiled-vs-compiled. Always benchmark both
your change and the baseline with the same compilation settings.

### 4. Correctness Is Not Optional

V3's first implementation *looked* 2x faster — because its raw Triton kernels bypassed
autograd and computed no gradients, so the backward pass did almost no work. Fast but
wrong is worthless. `test_correctness.py` (numerics + gradient flow) is what caught
it. Validate every custom kernel against the baseline, including that gradients flow.

### 5. Problem Size Shifts the Balance

Smaller problems are launch-overhead-bound (favoring fusion/compile); larger problems
shift toward compute (where the GEMMs dominate even more). The crossover where
compiled V3 overtakes compiled V1 held across sizes here, but the margin is small —
your mileage depends on hardware, sizes, and stack.

---

## Quick Reference Commands

### Complete Tutorial Run (All 3 Versions, Both Sizes)

```bash
# Automated tutorial script
bash optimization_tutorial.sh
```

**Duration**: ~30 seconds  
**Output**: Complete progression V1 → V2 → V3 for small and medium

### Manual Individual Runs

```bash
# Small Problem
## V1 Baseline
cd version1_pytorch_baseline
python3 tiny_openfold_v1.py --seq-len 64 --num-seqs 16 --batch-size 4 --device 0

## V2 Fused
cd ../version2_pytorch_fused
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py --seq-len 64 --num-seqs 16 --batch-size 4

## V3 Triton
cd ../version3_triton
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py --seq-len 64 --num-seqs 16 --batch-size 4

# Medium Problem - same commands but:
# --seq-len 128 --num-seqs 32 --batch-size 2
```

---

## Profiling Cheat Sheet

Quick reference for ROCm profiling tools across all versions:

| Tool | What It Shows | When to Use | Command |
|------|---------------|-------------|---------|
| **PyTorch Profiler** | High-level PyTorch ops, kernel names | Initial bottleneck identification | `--enable-pytorch-profiler` |
| **rocprof-sys** | System-level GPU trace, kernel timeline | Overall GPU utilization, kernel patterns | `./run_rocprof_sys.sh` |
| **rocprofv3** | Detailed kernel metrics, launch counts | Verify fusion, count kernel launches | `./run_rocprofv3.sh` |
| **rocprof-compute** | Hardware counters, memory bandwidth | Memory bottlenecks, cache efficiency | `./run_rocprof_compute.sh` |

**Typical workflow**: Start with PyTorch Profiler → rocprof-sys for overview → rocprof-compute for memory analysis → rocprofv3 for kernel details.

---

## Next Steps: Advanced Optimizations

### 1. Torch Compile — the orthogonal lever (all versions)
```bash
# Stacks on top of any version; biggest wins on V1 and V3
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py \
    --seq-len 128 --num-seqs 32 --batch-size 2 --enable-torch-compile
```
**Measured (MI300X / ROCm 7.13)**: ~1.6x on V1, ~1.7–1.9x on V3, ~1.0x on the
already-fused V2.

### 2. Combine fusion + compile (V2)
```bash
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v2.py \
    --seq-len 128 --num-seqs 32 --batch-size 2 --enable-all-fusion --enable-torch-compile
```
**Note**: V2 is already hand-fused, so compile adds little — a useful demonstration
that manual and compiler fusion target the same launch overhead.

### 3. Multi-GPU (V1)
```bash
# DataParallel scaling lives in V1 (see version1_pytorch_baseline/README.md)
ROCR_VISIBLE_DEVICES=0,1,2,3 python3 tiny_openfold_v1.py \
    --seq-len 128 --num-seqs 32 --batch-size 8
```
**Note**: Multi-GPU (`nn.DataParallel`) is implemented in V1 only; V2/V3 are
single-GPU. See the V1 scaling scripts for measured efficiency.

---

## Summary: What You Learned

You now have a complete mental model of GPU optimization. You learned how to establish reference performance through baseline measurement, identify bottlenecks systematically using profiling tools, and apply high-level PyTorch kernel fusion optimizations. You progressed to low-level GPU programming with custom Triton kernels, developed skills in performance analysis to understand where speedups actually come from, and learned to evaluate trade-offs between memory usage, speed, complexity, and maintainability.

**Final Achievement**: **~2.4x speedup** on small workloads from kernel fusion (V2),
plus an orthogonal **~1.6–1.9x** from `torch.compile` on the baseline and Triton
versions — and, just as importantly, the judgment to know *which* lever to pull.
You've seen that fusion beats hand-written kernels on this GEMM-dominated model, that
`torch.compile` is nearly free throughput, and that a "fast" custom kernel is
worthless if it isn't numerically correct and gradient-safe. That measurement
discipline — profile first, compare against the compiled baseline, verify
correctness — is the transferable skill.
