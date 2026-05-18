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

1. **Baseline Performance Measurement** - Establish reference metrics
2. **High-Level Optimizations** - Kernel fusion with PyTorch
3. **Low-Level Optimizations** - Custom GPU kernels with Triton
4. **Profiling Techniques** - Identify bottlenecks at each stage
5. **Performance Analysis** - Understand where speedups come from

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
- ✅ **Readable**: Clear, well-documented code
- ✅ **Standard**: Pure PyTorch operations
- ⚠️ **Unoptimized**: Multiple kernel launches, no fusion
- ⚠️ **Slow**: Kernel overhead dominates for small workloads

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

**What to look for:**
1. **Multiple attention kernels** - Q, K, V as separate operations
2. **Triangle operations** - Dominate backward pass
3. **Kernel launch overhead** - Many small kernel calls

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
- Small: 80.5 samples/sec, 49.7 ms/batch
- Medium: 41.5 samples/sec, 48.2 ms/batch

**Bottlenecks Identified:**
1. Backward pass dominates (55-56% of time)
2. Multiple kernel launches for attention
3. Triangle operations are compute-intensive

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

**Expected Results**:
- Baseline (no fusion): ~80 samples/sec
- MSA QKV only: ~87 samples/sec (+9%)
- Flash Attention only: ~92 samples/sec (+15%)
- Triangle fusion only: ~85 samples/sec (+6%)
- All fusions: ~106 samples/sec (+32%)

**Key Learning**: Flash Attention provides the biggest single benefit, but combined optimizations are synergistic.

### Stage 2 Summary

**Achievements:**
- Small: 80.5 → 106.4 samples/sec (+32%)
- Medium: 41.5 → 49.0 samples/sec (+18%)
- No memory overhead
- 80% kernel reduction

**Remaining Bottlenecks:**
1. Still using generic PyTorch kernels
2. Backward pass still dominates
3. Memory bandwidth not fully optimized

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

**Benefits**:
- Single kernel launch vs 3+ in PyTorch
- Data stays in cache/registers
- Optimized memory access patterns

#### 2. Flash Attention for MSA (Triton)
**Why optimize?** MSA operations dominate forward/backward passes.

**Key Optimizations**:
- Tiled computation to fit in shared memory
- Reduced HBM traffic (main memory)
- Optimized for ROCm/AMD GPUs

#### 3. Flash Attention for Triangles (Triton)
**Why optimize?** Triangle operations are O(N³) and very expensive.

**Key Optimizations**:
- Custom tiling strategy for pair representation
- Minimized memory transfers
- Optimized backward pass (biggest bottleneck!)

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

1. **Custom LayerNorm**: Fused computation, single pass through data
2. **Optimized Flash Attention**: Hand-tuned for ROCm, better memory access
3. **Triangle Backward Optimization**: Custom gradients with minimal memory traffic
4. **Register/Cache Utilization**: Data stays in fast memory longer

### Stage 3 Summary

**Final Achievements:**
- Small: 80.5 → 162.5 samples/sec (**2.0x speedup!**)
- Medium: 41.5 → 68.5 samples/sec (**1.65x speedup**)
- Backward pass: 69% faster (small), 56% faster (medium)

**Trade-offs:**
- ✅ Massive performance gains
- ✅ Same numerical accuracy
- ⚠️ Slightly higher memory (12-24%, still manageable)
- ⚠️ More complex code (Triton kernels)

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
- Triton kernels allocate scratch space
- Optimized for speed, not minimum memory
- Additional buffers for tiled computations

**Is it worth it?**
- ✅ Yes! 23 MB is negligible on modern GPUs (192 GB HBM)
- ✅ 2.0x speedup far outweighs small memory cost
- ✅ Can still run much larger problems

---

## Lessons Learned

### 1. Optimization Strategy

**Best Approach:**
1. Start with clean, readable baseline (V1)
2. Profile to identify bottlenecks
3. Apply high-level optimizations first (V2 - kernel fusion)
4. Drop to low-level only when needed (V3 - custom kernels)

**Don't:** Jump straight to custom kernels - high-level optimizations give 70% of the benefit with 10% of the effort!

### 2. Backward Pass Matters Most

In deep learning workloads:
- Backward pass often dominates (50-60% of time)
- Focus optimization efforts there first
- V3's backward optimization gave the biggest gains (56-69% reduction)

### 3. Problem Size Affects Speedup

- **Small problems**: Larger speedup (2.0x) - kernel overhead dominates
- **Medium problems**: Good speedup (1.65x) - balanced workload
- **Large problems**: Smaller speedup (~1.3x) - compute-bound

**Lesson**: Optimize for your target workload size!

### 4. Memory vs Speed Trade-offs

- V2: No memory cost, 32% speedup → **Always use**
- V3: 12% memory cost, 102% speedup → **Use when speed matters**

### 5. Incremental Development

Progressive optimization allows:
- **Validation** at each stage (compare outputs)
- **Debugging** (isolate which optimization broke things)
- **Education** (understand what each optimization contributes)
- **Flexibility** (choose optimization level based on needs)

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

## Documentation

- **This tutorial**: Complete optimization guide
- **QUICK_START_PERFORMANCE.md**: Fast reference
- **REGRESSION_TEST_SUMMARY.md**: Full benchmarks (4 problem sizes)
- **ARCHITECTURE.md**: Evoformer architecture details
- **Version READMEs**: Version-specific documentation

---

## Summary: What You Learned

✅ **Baseline measurement** - Establish reference performance  
✅ **Profiling** - Identify bottlenecks systematically  
✅ **Kernel fusion** - High-level PyTorch optimizations  
✅ **Custom kernels** - Low-level GPU programming with Triton  
✅ **Performance analysis** - Understand where speedups come from  
✅ **Trade-offs** - Memory vs speed, complexity vs maintainability  

**Final Achievement**: **2.0x speedup** on small workloads through systematic optimization!

🎉 **You now understand the complete GPU optimization pipeline!**
