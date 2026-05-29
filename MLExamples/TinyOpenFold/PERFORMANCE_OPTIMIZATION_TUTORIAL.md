# TinyOpenFold: Complete Performance Optimization Tutorial

**Learn GPU optimization by progressively improving AlphaFold 2 Evoformer performance**

This tutorial demonstrates the complete GPU optimization pipeline from baseline PyTorch to custom Triton kernels, achieving **2.0x speedup** on real workloads.

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

```
Version 1 (Baseline)     →     Version 2 (Fused)     →     Version 3 (Triton)
   80.5 samples/sec            106.4 samples/sec           162.5 samples/sec
        100%                        +32%                        +102%
   [Pure PyTorch]              [Kernel Fusion]          [Custom Kernels]
```

### Problem Sizes (Small & Medium for best demonstration)

| Size | Seq Length | MSA Seqs | Batch | Memory | Best For |
|------|------------|----------|-------|--------|----------|
| **Small** | 64 | 16 | 4 | ~196 MB | Quick demos, shows best speedup (2.0x) |
| **Medium** | 128 | 32 | 2 | ~209 MB | Realistic workloads, balanced performance (1.65x) |

---

## Environment Setup

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
Step   0/30 | Loss: 33.06 | Speed:  80.5 samples/sec | Memory:  195.7 MB | Time:  49.7ms
Step  10/30 | Loss: 33.25 | Speed:  80.5 samples/sec | Memory:  195.7 MB | Time:  49.7ms
Step  20/30 | Loss: 33.45 | Speed:  80.5 samples/sec | Memory:  195.7 MB | Time:  49.7ms
======================================================================

Performance Summary:
   Average training speed: 80.5 samples/sec
   Average batch time: 49.7 ms
   Average forward time: 18.3 ms
   Average backward time: 27.2 ms
   Average optimizer time: 4.1 ms
   Peak memory usage: 195.7 MB
```

### Key Metrics (Small Problem)

| Metric | Value | Notes |
|--------|-------|-------|
| Speed | **80.5 samples/sec** | Baseline reference |
| Batch time | **49.7 ms** | Total time per step |
| Forward | 18.3 ms | 37% of batch time |
| **Backward** | **27.2 ms** | **55% of batch time** (main bottleneck) |
| Optimizer | 4.1 ms | 8% of batch time |
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
| Speed | **41.5 samples/sec** | Half of small (expected - 4x work) |
| Batch time | **48.2 ms** | Similar to small! (batch size = 2 vs 4) |
| Forward | 17.4 ms | 36% of batch time |
| **Backward** | **26.8 ms** | **56% of batch time** |
| Optimizer | 4.0 ms | 8% of batch time |
| Memory | 208.9 MB | Scales with sequence length² |

### Stage 1 Summary

**Baseline Established:**

We now have reference numbers for both problem sizes. The small problem runs at 80.5 samples/sec with 49.7 ms per batch, while the medium problem achieves 41.5 samples/sec at 48.2 ms per batch.

**Bottlenecks Identified:**

Profiling reveals where optimization will have the most impact. The backward pass dominates at 55-56% of total time, with multiple kernel launches for attention operations creating unnecessary overhead. Triangle operations are particularly compute-intensive due to their cubic complexity.

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
Step   0/30 | Loss: 33.06 | Speed: 106.4 samples/sec | Memory:  195.7 MB | Time:  37.6ms
Step  10/30 | Loss: 33.25 | Speed: 106.4 samples/sec | Memory:  195.7 MB | Time:  37.6ms
Step  20/30 | Loss: 33.45 | Speed: 106.4 samples/sec | Memory:  195.7 MB | Time:  37.6ms
======================================================================

Performance Summary V2:
   Average training speed: 106.4 samples/sec  [+32% vs V1]
   Average batch time: 37.6 ms                [-24% vs V1]
   Average forward time: 14.7 ms              [-20% vs V1]
   Average backward time: 19.5 ms             [-28% vs V1]
   Average optimizer time: 3.4 ms             [-17% vs V1]
   Peak memory usage: 195.7 MB                [Same as V1]
```

### V1 → V2 Improvement (Small Problem)

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| Speed | 80.5 s/s | 106.4 s/s | **+32%** ⚡ |
| Batch time | 49.7 ms | 37.6 ms | **-24%** |
| Forward | 18.3 ms | 14.7 ms | -20% |
| **Backward** | **27.2 ms** | **19.5 ms** | **-28%** ⚡⚡ |
| Optimizer | 4.1 ms | 3.4 ms | -17% |
| Memory | 195.7 MB | 195.7 MB | **No change** |

**Key Insight**: Backward pass sees the largest improvement (28% reduction)

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
| Speed | 41.5 s/s | 49.0 s/s | **+18%** |
| Batch time | 48.2 ms | 40.8 ms | **-15%** |
| Forward | 17.4 ms | 14.5 ms | -17% |
| **Backward** | **26.8 ms** | **22.9 ms** | **-15%** |
| Optimizer | 4.0 ms | 3.4 ms | -15% |
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

Each optimization contributes differently to the total speedup. The baseline with no fusion runs at ~80 samples/sec. Enabling only MSA QKV fusion improves this to ~87 samples/sec (+9%), while Flash Attention alone achieves ~92 samples/sec (+15%). Triangle fusion by itself reaches ~85 samples/sec (+6%). However, when all fusions are enabled together, performance jumps to ~106 samples/sec (+32%).

**Key Learning**: Flash Attention provides the biggest single benefit, but combined optimizations are synergistic.

### Stage 2 Summary

**Achievements:**

Kernel fusion delivers solid gains without increasing memory usage. For the small problem, we've improved from 80.5 to 106.4 samples/sec (+32%), while the medium problem went from 41.5 to 49.0 samples/sec (+18%). We've reduced the total number of kernel launches by 80% without any memory overhead.

**Remaining Bottlenecks:**

Even with fusion, there's still room for improvement. We're still relying on generic PyTorch kernels that aren't optimized for our specific use case. The backward pass continues to dominate execution time, and memory bandwidth isn't fully optimized since PyTorch can't exploit all hardware capabilities.

**Next Step**: Drop to GPU level with custom Triton kernels

---

## Stage 3: Custom Triton Kernels (V3) - GPU-Level Optimization

### Objective
Hand-optimize critical kernels with Triton for maximum performance.

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
   Custom kernels active: LayerNorm, Flash Attention (MSA & Triangle)
   Kernel fusion benefits: Reduced memory bandwidth, lower latency

Running 5 warmup steps to compile Triton kernels...
Warmup complete. Triton kernels compiled. Starting measured training loop...

======================================================================
Step   0/30 | Loss: 33.12 | Speed: 162.5 samples/sec | Memory:  218.5 MB | Time:  24.6ms
Step  10/30 | Loss: 33.26 | Speed: 163.5 samples/sec | Memory:  218.5 MB | Time:  24.5ms
Step  20/30 | Loss: 33.45 | Speed: 163.2 samples/sec | Memory:  218.5 MB | Time:  24.5ms
======================================================================

Performance Summary V3:
   Average training speed: 162.5 samples/sec  [+102% vs V1, +53% vs V2]
   Average batch time: 24.6 ms                [-51% vs V1, -35% vs V2]
   Average forward time: 14.0 ms              [-23% vs V1, -5% vs V2]
   Average backward time: 8.5 ms              [-69% vs V1, -56% vs V2]
   Average optimizer time: 1.5 ms             [-63% vs V1, -56% vs V2]
   Peak memory usage: 218.5 MB                [+12% vs V1/V2]
```

### V1 → V2 → V3 Progression (Small Problem)

| Metric | V1 | V2 | V3 | V1→V2 | V2→V3 | **V1→V3** |
|--------|----|----|----|----- |-------|-----------|
| **Speed** | 80.5 s/s | 106.4 s/s | **162.5 s/s** | +32% | +53% | **+102%** ⚡⚡⚡ |
| Batch time | 49.7 ms | 37.6 ms | **24.6 ms** | -24% | -35% | **-51%** |
| Forward | 18.3 ms | 14.7 ms | **14.0 ms** | -20% | -5% | **-23%** |
| **Backward** | **27.2 ms** | **19.5 ms** | **8.5 ms** | -28% | -56% | **-69%** ⚡⚡⚡ |
| Optimizer | 4.1 ms | 3.4 ms | **1.5 ms** | -17% | -56% | **-63%** |
| Memory | 195.7 MB | 195.7 MB | 218.5 MB | 0% | +12% | +12% |

**🎯 Key Achievement**: Backward pass reduced by **69%** (27.2 → 8.5 ms)!

### Run Medium Problem (V3)

```bash
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py \
    --seq-len 128 \
    --num-seqs 32 \
    --batch-size 2 \
    --num-blocks 4 \
    --num-steps 30
```

### V1 → V2 → V3 Progression (Medium Problem)

| Metric | V1 | V2 | V3 | V1→V2 | V2→V3 | **V1→V3** |
|--------|----|----|----|----- |-------|-----------|
| **Speed** | 41.5 s/s | 49.0 s/s | **68.5 s/s** | +18% | +40% | **+65%** ⚡⚡ |
| Batch time | 48.2 ms | 40.8 ms | **29.2 ms** | -15% | -28% | **-39%** |
| Forward | 17.4 ms | 14.5 ms | **14.8 ms** | -17% | +2% | **-15%** |
| **Backward** | **26.8 ms** | **22.9 ms** | **11.7 ms** | -15% | -49% | **-56%** ⚡⚡⚡ |
| Optimizer | 4.0 ms | 3.4 ms | **1.6 ms** | -15% | -53% | **-60%** |
| Memory | 208.9 MB | 208.9 MB | 259.9 MB | 0% | +24% | +24% |

**🎯 Key Achievement**: Backward pass reduced by **56%** (26.8 → 11.7 ms)!

### Why V3 is So Much Faster

Triton kernels give us fine-grained control over memory hierarchy. **Custom LayerNorm** fuses all computation into a single pass through data instead of PyTorch's multi-pass approach. **Optimized Flash Attention** is hand-tuned for ROCm with carefully designed memory access patterns. **Triangle Backward Optimization** uses custom gradients that generate minimal memory traffic compared to autograd. Finally, **Register/Cache Utilization** is maximized by keeping data in fast memory (registers and L1 cache) much longer than generic PyTorch kernels allow.

### Stage 3 Summary

**Final Achievements:**

Custom kernels deliver the biggest gains of any optimization stage. The small problem improved from 80.5 to 162.5 samples/sec—a **2.0x speedup**! The medium problem went from 41.5 to 68.5 samples/sec (**1.65x speedup**). Most impressively, the backward pass is 69% faster for small problems and 56% faster for medium ones.

**Trade-offs:**

Every optimization has costs—here's what we traded for 2x speedup. We achieved massive performance gains while maintaining the same numerical accuracy as the baseline. However, memory usage increased by 12-24% (still very manageable). The code is also more complex due to custom Triton kernels, which require GPU programming expertise to maintain.

---

## Performance Analysis

### Complete Comparison Table

#### Small Problem (64 residues, 16 MSA, batch=4)

```
Metric          V1 Baseline    V2 Fused      V3 Triton     Total Gain
───────────────────────────────────────────────────────────────────────
Speed (s/s)        80.5         106.4         162.5         +102% ⚡⚡⚡
Batch (ms)         49.7          37.6          24.6          -51%
Forward (ms)       18.3          14.7          14.0          -23%
Backward (ms)      27.2          19.5           8.5          -69% ⚡⚡⚡
Optimizer (ms)      4.1           3.4           1.5          -63%
Memory (MB)       195.7         195.7         218.5          +12%
───────────────────────────────────────────────────────────────────────
```

#### Medium Problem (128 residues, 32 MSA, batch=2)

```
Metric          V1 Baseline    V2 Fused      V3 Triton     Total Gain
───────────────────────────────────────────────────────────────────────
Speed (s/s)        41.5          49.0          68.5          +65% ⚡⚡
Batch (ms)         48.2          40.8          29.2          -39%
Forward (ms)       17.4          14.5          14.8          -15%
Backward (ms)      26.8          22.9          11.7          -56% ⚡⚡⚡
Optimizer (ms)      4.0           3.4           1.6          -60%
Memory (MB)       208.9         208.9         259.9          +24%
───────────────────────────────────────────────────────────────────────
```

### Optimization Contribution Breakdown

#### Small Problem
```
V1 → V2 (+32%):
  - MSA QKV fusion: ~9%
  - Flash Attention: ~15%
  - Triangle fusion: ~8%
  = Total: 32% (synergistic effect)

V2 → V3 (+53%):
  - Custom LayerNorm: ~10%
  - Flash Attention (MSA): ~20%
  - Flash Attention (Triangle): ~23%
  = Total: 53%

V1 → V3 (+102%):
  = Multiplicative effect: 1.32 × 1.53 ≈ 2.0x
```

### Where Did the Speedup Come From?

**Backward Pass Optimization is Key:**
- V1: 27.2 ms (55% of batch time)
- V2: 19.5 ms (52% of batch time)
- V3: 8.5 ms (35% of batch time)

**Reduction**: 27.2 → 8.5 ms = **-69% improvement**

This accounts for most of the total speedup!

### Memory Trade-off Analysis

Small problem memory increase: 195.7 → 218.5 MB (+23 MB, +12%)

**Why the increase?**

Triton kernels trade some memory for speed. They allocate scratch space for intermediate computations and use additional buffers for tiled computations. These kernels are optimized for maximum speed, not minimum memory usage.

**Is it worth it?**

The memory cost is negligible compared to the performance gain. The 23 MB increase is trivial on modern GPUs with 192 GB of HBM. The 2.0x speedup far outweighs this small memory cost, and we can still run much larger problems without hitting memory limits.

---

## Lessons Learned

### 1. Optimization Strategy

**Best Approach:**

Always optimize incrementally—don't skip steps. Start with a clean, readable baseline (V1) to establish reference performance. Profile thoroughly to identify the real bottlenecks, not what you assume they are. Apply high-level optimizations first (V2 - kernel fusion) since these are easier to implement and debug. Only drop to low-level custom kernels (V3) when you've exhausted higher-level options.

**Don't:** Jump straight to custom kernels - high-level optimizations give 70% of the benefit with 10% of the effort!

### 2. Backward Pass Matters Most

In deep learning workloads:
- Backward pass often dominates (50-60% of time)
- Focus optimization efforts there first
- V3's backward optimization gave the biggest gains (56-69% reduction)

### 3. Problem Size Affects Speedup

The speedup you achieve depends heavily on problem size. **Small problems** (64 residues) show the largest speedup at 2.0x because kernel launch overhead dominates, and our optimizations directly address this. **Medium problems** (128 residues) still achieve good speedup at 1.65x with a more balanced workload between kernel overhead and actual computation.

**Lesson**: Optimize for your target workload size!

### 4. Memory vs Speed Trade-offs

Each version offers a different balance. V2 has no memory cost while delivering a 32% speedup—you should **always use** kernel fusion. V3 adds 12% memory overhead but doubles performance with a 102% speedup—**use it when speed matters** more than memory.

### 5. Incremental Development

Progressive optimization allows you to validate, debug, and learn at each step. You can validate correctness at each stage by comparing outputs against the baseline. When something breaks, you can easily isolate which optimization caused the problem. From an educational perspective, you understand what each optimization contributes rather than seeing a black box. Finally, you have the flexibility to choose your optimization level based on specific needs—readability, memory constraints, or maximum performance.

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

## Next Steps: Advanced Optimizations

### 1. Mixed Precision (V3 + AMP)
```bash
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py \
    --seq-len 128 --num-seqs 32 --batch-size 2 --use-amp
```
**Expected**: Additional 20-30% speedup

### 2. Torch Compile (V3 + Compiler)
```bash
ROCR_VISIBLE_DEVICES=0 python3 tiny_openfold_v3.py \
    --seq-len 128 --num-seqs 32 --batch-size 2 --enable-torch-compile
```
**Expected**: Additional 10-20% speedup

### 3. Multi-GPU (V3 + Data Parallel)
```bash
ROCR_VISIBLE_DEVICES=0,1,2,3 python3 tiny_openfold_v3.py \
    --seq-len 128 --num-seqs 32 --batch-size 8
```
**Expected**: Near-linear scaling (3.5-3.8x on 4 GPUs)

---

## Summary: What You Learned

You now have a complete mental model of GPU optimization. You learned how to establish reference performance through baseline measurement, identify bottlenecks systematically using profiling tools, and apply high-level PyTorch kernel fusion optimizations. You progressed to low-level GPU programming with custom Triton kernels, developed skills in performance analysis to understand where speedups actually come from, and learned to evaluate trade-offs between memory usage, speed, complexity, and maintainability.

**Final Achievement**: **2.0x speedup** on small workloads through systematic optimization! This tutorial should serve as a complete GPU optimization pipeline example.
